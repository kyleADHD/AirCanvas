"""M4: fp8 shard storage — round-trip bounds, alignment, streamed tolerance.

Three things have to hold for fp8 shards to be usable:

1. the numeric error is bounded and small (float8_e4m3fn with a per-tensor
   scale is a ~2^-4 relative half-ulp code, so ~6% worst case per weight);
2. the mixed-dtype files stay byte-ALIGNED, or `prefetch.ShardHeader` refuses
   them and every fp8 run silently degrades to synchronous loads;
3. the two decompress implementations — the allocating one on the engine's
   synchronous path and the pre-allocated one in the prefetch pipeline —
   agree exactly, otherwise step 1 and step 2 of a denoise loop would differ.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tests.conftest import ToyDiT, write_checkpoint

from aircanvas.sharding import quant
from aircanvas.sharding.manifest import Manifest
from aircanvas.sharding.splitter import split_model
from aircanvas.streaming.engine import StreamingEngine
from aircanvas.streaming.prefetch import DecompressPlan, ShardHeader

DIM = 64
N_BLOCKS = 6
N_STEPS = 3

# float8_e4m3fn: 3 mantissa bits => half-ulp relative error <= 2^-4 for normals.
FP8_REL_TOLERANCE = 1 / 16
# End-to-end through 6 residual blocks; calibrated with margin (measured ~2e-2).
STREAMED_REL_TOLERANCE = 0.08


@pytest.fixture()
def toy_fp8(tmp_path: Path) -> tuple[ToyDiT, Manifest, Path, Manifest, Path]:
    """A toy DiT split twice: once uncompressed, once fp8."""
    torch.manual_seed(23)
    reference = ToyDiT(dim=DIM, n_blocks=N_BLOCKS).eval()
    tensors = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    checkpoint = write_checkpoint(
        tmp_path / "toy", tensors, sharded=False, class_name="ToyTransformer2DModel"
    )
    plain = split_model(str(checkpoint), cache_dir=tmp_path / "plain", compute_dtype="float32")
    fp8 = split_model(
        str(checkpoint), cache_dir=tmp_path / "fp8", compression="fp8", compute_dtype="float32"
    )
    return reference, plain, tmp_path / "plain", fp8, tmp_path / "fp8"


def meta_toy() -> ToyDiT:
    with torch.device("meta"):
        return ToyDiT(dim=DIM, n_blocks=N_BLOCKS).eval()


# -- unit: the codec -------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e3])
def test_fp8_roundtrip_error_bound(scale: float) -> None:
    """Error is bounded relative to amax at every magnitude — that is what the
    per-tensor scale buys over a bare cast."""
    t = torch.randn(128, 256, generator=torch.Generator().manual_seed(1)) * scale
    stored = quant.compress_tensor("blocks.0.attn.to_q.weight", t, "fp8")
    back = quant.decompress_state_dict(stored, compression="fp8", compute_dtype=torch.float32)[
        "blocks.0.attn.to_q.weight"
    ]

    assert stored["blocks.0.attn.to_q.weight"].dtype is torch.float8_e4m3fn
    assert back.dtype is torch.float32
    amax = t.abs().max()
    assert (back - t).abs().max() <= FP8_REL_TOLERANCE * amax
    # And the relative error per *element* is bounded too, away from zero.
    big = t.abs() > amax * 1e-2
    assert ((back[big] - t[big]).abs() / t[big].abs()).max() <= FP8_REL_TOLERANCE


def test_fp8_halves_the_bytes() -> None:
    t = torch.randn(64, 64)
    stored = quant.compress_tensor("blocks.0.lin1.weight", t, "fp8")
    payload = stored["blocks.0.lin1.weight"]
    assert payload.numel() * payload.element_size() == t.numel()  # 1 byte per element


def test_skip_list_and_shape_rules() -> None:
    w = torch.randn(32, 32)
    assert quant.is_quantizable("blocks.0.attn.to_q.weight", w, "fp8")
    for skipped in (
        "blocks.0.norm1.linear.weight",
        "blocks.0.modulation.weight",
        "blocks.0.adaln_single.weight",
        "x_embedder.weight",
        "pos_embed.proj.weight",
    ):
        assert not quant.is_quantizable(skipped, w, "fp8"), skipped
    assert not quant.is_quantizable("blocks.0.lin2.bias", torch.randn(32), "fp8")  # 1-D
    assert not quant.is_quantizable("blocks.0.ids", torch.zeros(4, 4, dtype=torch.int64), "fp8")
    assert not quant.is_quantizable("blocks.0.w", w, None)


def test_zero_tensor_scale_is_one() -> None:
    stored = quant.compress_tensor("blocks.0.w.weight", torch.zeros(8, 8), "fp8")
    assert float(stored["blocks.0.w.weight" + quant.SCALE_SUFFIX]) == 1.0
    back = quant.decompress_state_dict(stored, compression="fp8", compute_dtype=torch.float32)
    assert torch.equal(back["blocks.0.w.weight"], torch.zeros(8, 8))


def test_scale_is_zero_dim() -> None:
    """A 1-element 1-D scale would promote `mul_` to float32 and raise; only a
    0-dim operand keeps the in-place result in the compute dtype."""
    stored = quant.compress_tensor("blocks.0.w.weight", torch.randn(8, 8), "fp8")
    scale = stored["blocks.0.w.weight" + quant.SCALE_SUFFIX]
    assert scale.dim() == 0 and scale.dtype is torch.float32
    dst = torch.empty(8, 8, dtype=torch.bfloat16)
    dst.copy_(stored["blocks.0.w.weight"])
    dst.mul_(scale)  # must not raise
    assert dst.dtype is torch.bfloat16


def test_order_for_alignment_is_descending_element_size() -> None:
    tensors = {
        "payload": torch.zeros(3, dtype=torch.float8_e4m3fn),
        "scale": torch.zeros((), dtype=torch.float32),
        "kept": torch.zeros(5, dtype=torch.bfloat16),
    }
    assert list(quant.order_for_alignment(tensors)) == ["scale", "kept", "payload"]


def test_decompress_none_is_identity() -> None:
    tensors = {"a": torch.randn(4, 4)}
    out = quant.decompress_state_dict(tensors, compression=None, compute_dtype=torch.bfloat16)
    assert out["a"] is tensors["a"]  # no copy, no cast — the bitwise gate depends on it


# -- the shard files -------------------------------------------------------


def test_fp8_shards_are_aligned_and_smaller(toy_fp8) -> None:
    _, plain, plain_dir, fp8, fp8_dir = toy_fp8
    assert fp8.compression == "fp8"
    assert plain.compression is None

    for shard in fp8.blocks:
        header = ShardHeader.parse(fp8_dir / shard.file)
        assert header.aligned, f"{shard.file} would be rejected by the prefetcher"
        dtypes = {m.dtype for m in header.tensors}
        assert torch.float8_e4m3fn in dtypes and torch.float32 in dtypes  # mixed, on purpose

    assert fp8.disk_bytes() < plain.disk_bytes()
    # Materialised size is unchanged: fp8 is a disk-traffic cut, not a VRAM cut.
    assert sum(b.materialized_bytes for b in fp8.blocks) == sum(
        b.materialized_bytes for b in plain.blocks
    )


def test_resident_shard_is_never_compressed(toy_fp8) -> None:
    _, _, _, fp8, fp8_dir = toy_fp8
    resident = load_file(fp8_dir / fp8.resident_file)
    assert resident, "resident shard should not be empty"
    assert all(not quant.is_scale_key(k) for k in resident)
    assert all(t.dtype is torch.float32 for t in resident.values())


def test_misaligned_shard_is_detected(tmp_path: Path) -> None:
    """Write a deliberately bad layout and confirm the reader rejects it — this
    is the invariant the splitter verifies before dropping a .done marker."""
    from safetensors.torch import save_file

    path = tmp_path / "bad.safetensors"
    # 1-byte tensor of odd length first => the bf16 tensor starts on an odd byte.
    save_file(
        {
            "a": torch.zeros(3, dtype=torch.uint8),
            "b": torch.zeros(4, dtype=torch.bfloat16),
        },
        str(path),
    )
    header = ShardHeader.parse(path)
    if header.tensors[0].name == "a":  # safetensors kept our (bad) order
        assert not header.aligned
    else:  # safetensors reordered it for us; the good layout must be aligned
        assert header.aligned


# -- the two decompress implementations must agree -------------------------


def test_decompress_plan_matches_reference_decompress(toy_fp8) -> None:
    _, _, _, fp8, fp8_dir = toy_fp8
    for shard in fp8.blocks:
        path = fp8_dir / shard.file
        header = ShardHeader.parse(path)
        plan = DecompressPlan.build(header, torch.float32)

        raw = torch.empty(header.blob_size, dtype=torch.uint8)
        header.read_blob_into(raw)
        out = torch.empty(plan.out_bytes, dtype=torch.uint8)
        plan.run(raw, out)
        got = plan.views(out)

        expected = quant.decompress_state_dict(
            load_file(path), compression="fp8", compute_dtype=torch.float32
        )
        assert set(got) == set(expected)
        for name in expected:
            assert torch.equal(got[name], expected[name]), name


# -- end to end ------------------------------------------------------------


def _run_engine(cache: Path, manifest: Manifest, x: torch.Tensor, device: str) -> torch.Tensor:
    model = meta_toy()
    outs = []
    with StreamingEngine(model, manifest, cache, device=device) as engine, torch.no_grad():
        for _ in range(N_STEPS):
            outs.append(model(x).clone())
    assert engine.stats["prefetch_hits"] > 0, "prefetch never armed — test is not exercising it"
    for later in outs[1:]:
        # Sync-loaded step 1 and prefetched steps 2+ must agree exactly, or the
        # two decompress implementations have drifted apart.
        assert torch.equal(outs[0], later)
    return outs[0]


def test_streamed_fp8_within_tolerance_of_reference(toy_fp8) -> None:
    reference, plain, plain_dir, fp8, fp8_dir = toy_fp8
    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        expected = reference(x)

    exact = _run_engine(plain_dir, plain, x, "cpu")
    assert torch.equal(exact, expected), "compression=None must stay bitwise-exact"

    approx = _run_engine(fp8_dir, fp8, x, "cpu")
    rel = (approx - expected).abs().max() / expected.abs().max()
    assert rel < STREAMED_REL_TOLERANCE, f"fp8 drifted {rel:.4f} from the reference"
    assert not torch.equal(approx, expected), "fp8 should be approximate, not exact"


@pytest.mark.gpu
def test_streamed_fp8_on_cuda(toy_fp8) -> None:
    reference, _, _, fp8, fp8_dir = toy_fp8
    device = torch.device("cuda:0")
    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(9)).to(device)
    with torch.no_grad():
        expected = reference.to(device)(x)

    approx = _run_engine(fp8_dir, fp8, x, "cuda:0")
    torch.cuda.synchronize()
    rel = (approx - expected).abs().max() / expected.abs().max()
    assert rel < STREAMED_REL_TOLERANCE, f"fp8 drifted {rel:.4f} from the reference"


@pytest.mark.gpu
def test_fp8_allocates_a_raw_staging_pool(toy_fp8) -> None:
    """ADR #8: the compressed path needs a second device pool, and its cost has
    to show up in the stats so a report is honest about VRAM."""
    _, plain, plain_dir, fp8, fp8_dir = toy_fp8
    x = torch.randn(1, DIM).cuda()

    stats = {}
    for name, manifest, cache in (("plain", plain, plain_dir), ("fp8", fp8, fp8_dir)):
        model = meta_toy()
        with StreamingEngine(model, manifest, cache, device="cuda:0") as engine, torch.no_grad():
            for _ in range(N_STEPS):
                model(x)
        stats[name] = engine.stats

    assert stats["fp8"]["slot_bytes"] > stats["plain"]["slot_bytes"]
    assert stats["fp8"]["bytes_loaded"] < stats["plain"]["bytes_loaded"]  # less disk traffic
