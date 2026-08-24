"""M10: LoRA fuse-on-stream — parse formats, bitwise gate, quantized shards.

The correctness gate is the same as the rest of the engine: streamed output
with a LoRA fused after dequant must equal a fully-materialized reference
whose weights had ``W += scale * up @ down`` applied. Compression=None is
bitwise; fp8 is within the existing streamed quant tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tests.conftest import ToyDiT, write_checkpoint

from aircanvas.cli import main
from aircanvas.lora import LoraError, LoraOverlay, parse_lora_state_dict, resolve_lora_path
from aircanvas.sharding import quant
from aircanvas.sharding.manifest import Manifest
from aircanvas.sharding.splitter import split_model
from aircanvas.streaming.engine import StreamingEngine

RANK = 2
DIM = 8
N_STEPS = 3


def _peft_pair(mod: str, down: torch.Tensor, up: torch.Tensor) -> dict[str, torch.Tensor]:
    return {f"{mod}.lora_A.weight": down, f"{mod}.lora_B.weight": up}


def _write_lora(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    save_file(tensors, str(path))
    return path


@pytest.fixture()
def toy(tmp_path: Path) -> tuple[ToyDiT, Path, Path]:
    torch.manual_seed(13)
    reference = ToyDiT(dim=DIM).eval()
    tensors = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    checkpoint = write_checkpoint(
        tmp_path / "toy", tensors, sharded=False, class_name="ToyTransformer2DModel"
    )
    cache = tmp_path / "cache"
    split_model(str(checkpoint), cache_dir=cache, compute_dtype=None)
    return reference, cache, tmp_path


def meta_toy() -> ToyDiT:
    with torch.device("meta"):
        return ToyDiT(dim=DIM).eval()


def _fuse_copy(reference: ToyDiT, name: str, down: torch.Tensor, up: torch.Tensor, scale: float):
    fused = ToyDiT(dim=DIM).eval()
    fused.load_state_dict(reference.state_dict())
    *path, leaf = name.split(".")
    mod = fused
    for part in path:
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    with torch.no_grad():
        getattr(mod, leaf).addmm_(up, down, alpha=scale)
    return fused


# -- parser ----------------------------------------------------------------


def test_parse_peft_keys() -> None:
    down = torch.randn(RANK, DIM)
    up = torch.randn(DIM, RANK)
    deltas, n, te = parse_lora_state_dict(_peft_pair("blocks.0.lin1", down, up))
    assert n == 1 and te == 0
    delta = deltas["blocks.0.lin1.weight"]
    assert torch.equal(delta.down, down)
    assert torch.equal(delta.up, up)
    assert delta.rank == RANK
    assert delta.scale == pytest.approx(1.0)  # alpha defaults to rank


def test_parse_transformer_prefix_and_kohya() -> None:
    down = torch.randn(RANK, DIM)
    up = torch.randn(DIM, RANK)
    peft, _, _ = parse_lora_state_dict(_peft_pair("transformer.blocks.1.lin2", down, up))
    assert "blocks.1.lin2.weight" in peft

    nested, _, _ = parse_lora_state_dict(
        _peft_pair("base_model.model.transformer.blocks.1.lin2", down, up)
    )
    assert "blocks.1.lin2.weight" in nested

    kohya, _, _ = parse_lora_state_dict(
        {
            "lora_unet_blocks_2_lin1.lora_down.weight": down,
            "lora_unet_blocks_2_lin1.lora_up.weight": up,
            "lora_unet_blocks_2_lin1.alpha": torch.tensor(4.0),
        }
    )
    delta = kohya["blocks.2.lin1.weight"]
    assert delta.alpha == pytest.approx(4.0)
    assert delta.scale == pytest.approx(4.0 / RANK)


def test_parse_peft_default_adapter_infix() -> None:
    down = torch.randn(RANK, DIM)
    up = torch.randn(DIM, RANK)
    deltas, _, _ = parse_lora_state_dict(
        {
            "blocks.0.lin1.lora_A.default.weight": down,
            "blocks.0.lin1.lora_B.default.weight": up,
        }
    )
    assert "blocks.0.lin1.weight" in deltas


def test_parse_skips_text_encoder_keys() -> None:
    down = torch.randn(RANK, DIM)
    up = torch.randn(DIM, RANK)
    state = {
        **_peft_pair("blocks.0.lin1", down, up),
        "text_encoder.encoder.layer.0.self_attn.q_proj.lora_A.weight": down,
        "text_encoder.encoder.layer.0.self_attn.q_proj.lora_B.weight": up,
        "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": down,
        "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": up,
    }
    deltas, _, te = parse_lora_state_dict(state)
    assert "blocks.0.lin1.weight" in deltas
    assert te >= 4
    assert all("text_encoder" not in k for k in deltas)


def test_parse_empty_raises() -> None:
    with pytest.raises(LoraError, match="No DiT LoRA pairs"):
        parse_lora_state_dict({"unrelated.weight": torch.ones(2, 2)})


def test_resolve_local_file_and_rejects_bin(tmp_path: Path) -> None:
    pair = _peft_pair("m", torch.ones(1, 2), torch.ones(2, 1))
    lora = _write_lora(tmp_path / "x.safetensors", pair)
    assert resolve_lora_path(lora) == lora
    bin_path = tmp_path / "x.bin"
    bin_path.write_bytes(b"nope")
    with pytest.raises(LoraError, match="safetensors"):
        resolve_lora_path(bin_path)


def test_apply_shape_mismatch_raises(tmp_path: Path) -> None:
    down = torch.randn(RANK, DIM)
    up = torch.randn(DIM, RANK)
    lora_path = _write_lora(tmp_path / "bad.safetensors", _peft_pair("blocks.0.lin1", down, up))
    overlay = LoraOverlay()
    overlay.load(lora_path)
    wrong = {"blocks.0.lin1.weight": torch.zeros(DIM + 1, DIM)}
    with pytest.raises(LoraError, match="shape mismatch"):
        overlay.apply(wrong)


# -- engine gate -----------------------------------------------------------


def test_streamed_lora_equals_fused_reference(toy) -> None:
    reference, cache, tmp_path = toy
    down = torch.randn(RANK, DIM, generator=torch.Generator().manual_seed(2))
    up = torch.randn(DIM, RANK, generator=torch.Generator().manual_seed(3))
    lora_path = _write_lora(tmp_path / "style.safetensors", _peft_pair("blocks.0.lin1", down, up))

    fused = _fuse_copy(reference, "blocks.0.lin1.weight", down, up, scale=1.0)
    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        expected = fused(x)
        plain = reference(x)

    overlay = LoraOverlay()
    overlay.load(lora_path, scale=1.0)

    streamed = meta_toy()
    with (
        StreamingEngine(streamed, Manifest.load(cache), cache, device="cpu", lora=overlay),
        torch.no_grad(),
    ):
        out = streamed(x)

    assert not torch.equal(plain, expected), "LoRA must actually change the output"
    assert torch.equal(out, expected)
    assert overlay.adapter_names == ("style",)


def test_prefetched_lora_equals_fused_reference(toy) -> None:
    """Second+ forwards take the prefetch path; fusion must not double-apply."""
    reference, cache, tmp_path = toy
    down = torch.randn(RANK, DIM, generator=torch.Generator().manual_seed(2))
    up = torch.randn(DIM, RANK, generator=torch.Generator().manual_seed(3))
    lora_path = _write_lora(tmp_path / "style.safetensors", _peft_pair("blocks.0.lin1", down, up))
    fused = _fuse_copy(reference, "blocks.0.lin1.weight", down, up, scale=1.0)
    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        expected = fused(x)

    overlay = LoraOverlay()
    overlay.load(lora_path)
    streamed = meta_toy()
    engine = StreamingEngine(streamed, Manifest.load(cache), cache, device="cpu", lora=overlay)
    with engine, torch.no_grad():
        for _ in range(N_STEPS):
            assert torch.equal(streamed(x), expected)
    n = len(Manifest.load(cache).blocks)
    assert engine.stats["prefetch_hits"] == (N_STEPS - 1) * n - 1


def test_lora_on_resident_embedder(toy) -> None:
    """x_embedder lives in resident.safetensors — fusion must still land."""
    reference, cache, tmp_path = toy
    down = torch.randn(RANK, DIM, generator=torch.Generator().manual_seed(5))
    up = torch.randn(DIM, RANK, generator=torch.Generator().manual_seed(6))
    lora_path = _write_lora(tmp_path / "emb.safetensors", _peft_pair("x_embedder", down, up))
    fused = _fuse_copy(reference, "x_embedder.weight", down, up, scale=1.0)
    x = torch.randn(1, DIM, generator=torch.Generator().manual_seed(7))
    overlay = LoraOverlay()
    overlay.load(lora_path)
    streamed = meta_toy()
    with (
        StreamingEngine(streamed, Manifest.load(cache), cache, device="cpu", lora=overlay),
        torch.no_grad(),
    ):
        out = streamed(x)
        expected = fused(x)
    assert torch.equal(out, expected)


def test_lora_on_pinned_resident_block(toy) -> None:
    """A LoRA targeting a block the budget solver pinned must still fuse once."""
    reference, cache, tmp_path = toy
    down = torch.randn(RANK, DIM, generator=torch.Generator().manual_seed(17))
    up = torch.randn(DIM, RANK, generator=torch.Generator().manual_seed(18))
    lora_path = _write_lora(tmp_path / "pin.safetensors", _peft_pair("blocks.0.lin1", down, up))
    fused = _fuse_copy(reference, "blocks.0.lin1.weight", down, up, scale=1.0)
    overlay = LoraOverlay()
    overlay.load(lora_path)
    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(19))
    streamed = meta_toy()
    with (
        StreamingEngine(
            streamed, Manifest.load(cache), cache, device="cpu", lora=overlay, resident_blocks=2
        ),
        torch.no_grad(),
    ):
        with torch.no_grad():
            expected = fused(x)
        assert torch.equal(streamed(x), expected)
        assert torch.equal(streamed(x), expected)  # still fused, not double-applied


def test_scale_zero_is_identity(toy) -> None:
    reference, cache, tmp_path = toy
    down = torch.randn(RANK, DIM)
    up = torch.randn(DIM, RANK)
    lora_path = _write_lora(tmp_path / "z.safetensors", _peft_pair("blocks.1.lin1", down, up))
    overlay = LoraOverlay()
    name = overlay.load(lora_path, scale=0.0)
    overlay.set_scale(name, 0.0)

    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        expected = reference(x)
    streamed = meta_toy()
    with (
        StreamingEngine(streamed, Manifest.load(cache), cache, device="cpu", lora=overlay),
        torch.no_grad(),
    ):
        assert torch.equal(streamed(x), expected)


def test_two_loras_compose_additively(toy) -> None:
    reference, cache, tmp_path = toy
    d1 = torch.randn(RANK, DIM, generator=torch.Generator().manual_seed(9))
    u1 = torch.randn(DIM, RANK, generator=torch.Generator().manual_seed(10))
    d2 = torch.randn(RANK, DIM, generator=torch.Generator().manual_seed(11))
    u2 = torch.randn(DIM, RANK, generator=torch.Generator().manual_seed(12))
    p1 = _write_lora(tmp_path / "a.safetensors", _peft_pair("blocks.0.lin1", d1, u1))
    p2 = _write_lora(tmp_path / "b.safetensors", _peft_pair("blocks.0.lin1", d2, u2))

    fused = ToyDiT(dim=DIM).eval()
    fused.load_state_dict(reference.state_dict())
    with torch.no_grad():
        fused.blocks[0].lin1.weight.addmm_(u1, d1, alpha=0.5)
        fused.blocks[0].lin1.weight.addmm_(u2, d2, alpha=1.0)

    overlay = LoraOverlay()
    overlay.load(p1, scale=0.5, adapter_name="a")
    overlay.load(p2, scale=1.0, adapter_name="b")

    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(13))
    streamed = meta_toy()
    with (
        StreamingEngine(streamed, Manifest.load(cache), cache, device="cpu", lora=overlay),
        torch.no_grad(),
    ):
        assert torch.equal(streamed(x), fused(x))

    overlay.unload("a")
    assert overlay.adapter_names == ("b",)
    overlay.unload()
    assert not overlay


def test_kohya_keys_stream(toy) -> None:
    reference, cache, tmp_path = toy
    down = torch.randn(RANK, DIM, generator=torch.Generator().manual_seed(14))
    up = torch.randn(DIM, RANK, generator=torch.Generator().manual_seed(15))
    lora_path = _write_lora(
        tmp_path / "kohya.safetensors",
        {
            "lora_unet_blocks_3_lin2.lora_down.weight": down,
            "lora_unet_blocks_3_lin2.lora_up.weight": up,
        },
    )
    fused = _fuse_copy(reference, "blocks.3.lin2.weight", down, up, scale=1.0)
    overlay = LoraOverlay()
    overlay.load(lora_path)

    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(16))
    streamed = meta_toy()
    with (
        StreamingEngine(streamed, Manifest.load(cache), cache, device="cpu", lora=overlay),
        torch.no_grad(),
    ):
        assert torch.equal(streamed(x), fused(x))


def test_fp8_shards_still_take_lora(tmp_path: Path) -> None:
    """Fusion happens AFTER dequant: streamed+LoRA == dequant-then-fuse.

    Comparing against a full-precision fused reference would mix quant error
    with the adapter; the honest gate is the decompressed shard cache plus
    the same addmm_, which is exactly what the engine does.
    """
    torch.manual_seed(23)
    reference = ToyDiT(dim=64, n_blocks=6).eval()
    tensors = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    checkpoint = write_checkpoint(
        tmp_path / "toy", tensors, sharded=False, class_name="ToyTransformer2DModel"
    )
    cache = tmp_path / "fp8"
    manifest = split_model(
        str(checkpoint), cache_dir=cache, compression="fp8", compute_dtype="float32"
    )
    down = torch.randn(RANK, 64, generator=torch.Generator().manual_seed(24)) * 0.05
    up = torch.randn(64, RANK, generator=torch.Generator().manual_seed(25)) * 0.05
    lora_path = _write_lora(tmp_path / "s.safetensors", _peft_pair("blocks.0.lin1", down, up))

    dequant: dict[str, torch.Tensor] = {}
    for fname in manifest.shard_files():
        dequant.update(
            quant.decompress_state_dict(
                load_file(str(cache / fname)),
                compression="fp8",
                compute_dtype=torch.float32,
            )
        )
    fused = ToyDiT(dim=64, n_blocks=6).eval()
    fused.load_state_dict(dequant)
    with torch.no_grad():
        fused.blocks[0].lin1.weight.addmm_(up, down, alpha=1.0)

    overlay = LoraOverlay()
    overlay.load(lora_path)
    with torch.device("meta"):
        streamed = ToyDiT(dim=64, n_blocks=6).eval()
    x = torch.randn(2, 64, generator=torch.Generator().manual_seed(26))
    with StreamingEngine(streamed, manifest, cache, device="cpu", lora=overlay), torch.no_grad():
        out = streamed(x)
        expected = fused(x)
    assert torch.equal(out, expected)


def test_unload_unknown_adapter_raises() -> None:
    overlay = LoraOverlay()
    with pytest.raises(LoraError, match="No LoRA adapter"):
        overlay.unload("missing")


def test_run_help_lists_lora(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--lora" in out
    assert "--lora-scale" in out


def test_run_lora_scale_count_mismatch() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "repo",
                "-p",
                "hi",
                "--lora",
                "a",
                "--lora",
                "b",
                "--lora-scale",
                "0.5",
                "--lora-scale",
                "0.6",
                "--lora-scale",
                "0.7",
            ]
        )
    assert exc.value.code != 0
