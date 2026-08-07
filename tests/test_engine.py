"""M2: the correctness gate — streamed output must be BITWISE-equal to a
fully-materialized reference (CLAUDE.md hard rule)."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tests.conftest import ToyDiT, write_checkpoint

from aircanvas.sharding.manifest import Manifest
from aircanvas.sharding.splitter import split_model
from aircanvas.streaming.engine import StreamingEngine, StreamingError


@pytest.fixture()
def toy(tmp_path: Path) -> tuple[ToyDiT, Path, Manifest, Path]:
    torch.manual_seed(7)
    reference = ToyDiT().eval()
    tensors = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    checkpoint = write_checkpoint(
        tmp_path / "toy", tensors, sharded=False, class_name="ToyTransformer2DModel"
    )
    cache = tmp_path / "cache"
    manifest = split_model(str(checkpoint), cache_dir=cache, compute_dtype=None)
    return reference, checkpoint, manifest, cache


def meta_toy() -> ToyDiT:
    with torch.device("meta"):
        return ToyDiT().eval()


def test_streamed_equals_reference_bitwise(toy) -> None:
    reference, _, manifest, cache = toy
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = reference(x)

    streamed = meta_toy()
    with StreamingEngine(streamed, manifest, cache, device="cpu") as engine, torch.no_grad():
        out1 = streamed(x)
        out2 = streamed(x)  # second pass re-streams every block

    assert torch.equal(out1, expected)
    assert torch.equal(out2, expected)
    assert engine.schedule == tuple(b.name for b in manifest.blocks)
    assert engine.stats["block_loads"] == 2 * len(manifest.blocks)


def test_blocks_are_evicted_after_forward(toy) -> None:
    _, _, manifest, cache = toy
    streamed = meta_toy()
    with StreamingEngine(streamed, manifest, cache, device="cpu"), torch.no_grad():
        streamed(torch.randn(1, 8))
        for name, p in streamed.named_parameters():
            if name.startswith("blocks."):
                assert p.is_meta, f"{name} not evicted"
            else:
                assert not p.is_meta, f"resident {name} should stay bound"


def test_incomplete_cache_raises(toy) -> None:
    _, _, manifest, cache = toy
    (cache / manifest.blocks[0].file).unlink()
    with pytest.raises(StreamingError, match="incomplete"):
        StreamingEngine(meta_toy(), manifest, cache, device="cpu")


def test_shard_module_mismatch_raises(toy) -> None:
    _, _, manifest, cache = toy
    shard_path = cache / manifest.blocks[2].file
    # .clone() detaches from the mmap — Windows can't rewrite a mapped file
    tensors = {k: v.clone() for k, v in load_file(shard_path).items()}
    tensors.pop(sorted(tensors)[0])  # drop one tensor from the shard
    save_file(tensors, str(shard_path))
    streamed = meta_toy()
    with (
        StreamingEngine(streamed, manifest, cache, device="cpu"),
        torch.no_grad(),
        pytest.raises(StreamingError, match="missing="),
    ):
        streamed(torch.randn(1, 8))


def test_uncovered_meta_param_raises(toy) -> None:
    _, _, manifest, cache = toy

    class ToyDiTExtra(ToyDiT):
        def __init__(self) -> None:
            super().__init__()
            self.extra_head = torch.nn.Linear(8, 8)

    with torch.device("meta"):
        model = ToyDiTExtra()
    with pytest.raises(StreamingError, match="extra_head"):
        StreamingEngine(model, manifest, cache, device="cpu")


def test_manifest_block_missing_from_model_raises(toy) -> None:
    _, _, manifest, cache = toy
    with torch.device("meta"):
        small = ToyDiT(n_blocks=4)  # manifest expects 6 blocks
    with pytest.raises(StreamingError, match="does not exist in the model"):
        StreamingEngine(small, manifest, cache, device="cpu")


def test_close_removes_hooks(toy) -> None:
    _, _, manifest, cache = toy
    streamed = meta_toy()
    engine = StreamingEngine(streamed, manifest, cache, device="cpu")
    engine.close()
    with pytest.raises(RuntimeError), torch.no_grad():
        streamed(torch.randn(1, 8))  # blocks are meta and nothing streams them


def test_close_releases_all_weights(toy) -> None:
    """Default close() must leave NOTHING bound — a fresh engine per call
    means kept residents are a pure cross-engine VRAM leak (the first
    Qwen-Image 20B run died on exactly this)."""
    _, _, manifest, cache = toy
    streamed = meta_toy()
    engine = StreamingEngine(streamed, manifest, cache, device="cpu")
    with torch.no_grad():
        streamed(torch.randn(1, 8))
    engine.close()
    still_bound = [n for n, p in streamed.named_parameters() if not p.is_meta]
    assert still_bound == [], still_bound

    # opt-out keeps residents for engine-free reuse of the bound model
    streamed2 = meta_toy()
    engine2 = StreamingEngine(streamed2, manifest, cache, device="cpu")
    engine2.close(release_weights=False)
    assert any(not p.is_meta for _, p in streamed2.named_parameters())


@pytest.mark.gpu
def test_streamed_equals_reference_on_cuda(toy) -> None:
    reference, _, manifest, cache = toy
    device = torch.device("cuda:0")
    reference = reference.to(device)
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(1)).to(device)
    with torch.no_grad():
        expected = reference(x)

    streamed = meta_toy()
    with StreamingEngine(streamed, manifest, cache, device=device) as engine, torch.no_grad():
        out = streamed(x)

    assert torch.equal(out, expected)
    assert engine.stats["block_loads"] == len(manifest.blocks)
