"""M3: prefetch pipeline — same bitwise gate as M2, now with the ring live."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tests.conftest import ToyDiT, write_checkpoint

from aircanvas.config import StreamConfig
from aircanvas.sharding.manifest import Manifest
from aircanvas.sharding.splitter import split_model
from aircanvas.streaming.engine import StreamingEngine
from aircanvas.streaming.prefetch import Prefetcher, PrefetchError, ShardHeader

N_STEPS = 3


@pytest.fixture()
def toy(tmp_path: Path) -> tuple[ToyDiT, Manifest, Path]:
    torch.manual_seed(11)
    reference = ToyDiT().eval()
    tensors = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    checkpoint = write_checkpoint(
        tmp_path / "toy", tensors, sharded=False, class_name="ToyTransformer2DModel"
    )
    manifest = split_model(str(checkpoint), cache_dir=tmp_path / "cache", compute_dtype=None)
    return reference, manifest, tmp_path / "cache"


def test_shard_header_views_match_load_file(toy) -> None:
    _, manifest, cache = toy
    path = cache / manifest.blocks[0].file
    header = ShardHeader.parse(path)
    assert header.aligned
    buf = torch.empty(header.blob_size, dtype=torch.uint8)
    header.read_blob_into(buf)
    views = header.build_views(buf)
    expected = load_file(path)
    assert set(views) == set(expected)
    for name in expected:
        assert torch.equal(views[name], expected[name]), name


def test_prefetched_equals_reference_bitwise(toy) -> None:
    reference, manifest, cache = toy
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = reference(x)

    streamed = ToyDiT().to("meta").eval()
    with StreamingEngine(streamed, manifest, cache, device="cpu") as engine, torch.no_grad():
        for _ in range(N_STEPS):
            assert torch.equal(streamed(x), expected)

    n = len(manifest.blocks)
    # Step 1 fully synchronous (recording); step 2 starts with one sync load
    # (the arming block), everything after is prefetched.
    assert engine.stats["sync_loads"] == n + 1
    assert engine.stats["prefetch_hits"] == (N_STEPS - 1) * n - 1
    assert engine.stats["block_loads"] == N_STEPS * n


def test_prefetch_with_minimal_ring(toy) -> None:
    reference, manifest, cache = toy
    x = torch.randn(1, 8, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        expected = reference(x)
    streamed = ToyDiT().to("meta").eval()
    config = StreamConfig(gpu_slots=2, ring_depth=1)
    with (
        StreamingEngine(streamed, manifest, cache, device="cpu", config=config),
        torch.no_grad(),
    ):
        for _ in range(N_STEPS):
            assert torch.equal(streamed(x), expected)


def test_prefetch_disabled(toy) -> None:
    reference, manifest, cache = toy
    x = torch.randn(1, 8, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        expected = reference(x)
    streamed = ToyDiT().to("meta").eval()
    with (
        StreamingEngine(streamed, manifest, cache, device="cpu", prefetch=False) as engine,
        torch.no_grad(),
    ):
        for _ in range(2):
            assert torch.equal(streamed(x), expected)
    assert engine.stats["prefetch_hits"] == 0
    assert engine.stats["sync_loads"] == 2 * len(manifest.blocks)


def test_bad_config_raises(toy) -> None:
    _, manifest, cache = toy
    with pytest.raises(PrefetchError, match="gpu_slots"):
        Prefetcher(
            manifest,
            cache,
            tuple(b.name for b in manifest.blocks),
            start_index=0,
            device=torch.device("cpu"),
            config=StreamConfig(gpu_slots=1, ring_depth=0),
        )


def test_report_smoke(toy) -> None:
    _, manifest, cache = toy
    streamed = ToyDiT().to("meta").eval()
    with StreamingEngine(streamed, manifest, cache, device="cpu") as engine, torch.no_grad():
        streamed(torch.randn(1, 8))
    text = engine.report()
    assert "prefetched" in text and "synchronous" in text


@pytest.mark.gpu
def test_prefetched_equals_reference_on_cuda(toy) -> None:
    reference, manifest, cache = toy
    device = torch.device("cuda:0")
    reference = reference.to(device)
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(5)).to(device)
    with torch.no_grad():
        expected = reference(x)

    streamed = ToyDiT().to("meta").eval()
    with StreamingEngine(streamed, manifest, cache, device=device) as engine, torch.no_grad():
        for _ in range(N_STEPS):
            out = streamed(x)
            torch.cuda.synchronize()
            assert torch.equal(out, expected)

    n = len(manifest.blocks)
    assert engine.stats["prefetch_hits"] == (N_STEPS - 1) * n - 1
