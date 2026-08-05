"""M5: NF4 shard compression.

The GPU test is the important one: the streamed model (sync path on step 1,
DecompressPlan/prefetch path on steps 2+) must produce BITWISE-identical
outputs to a reference model loaded from allocating dequantization — the two
decompress paths must agree exactly, or step 1 and step 2 of a denoise loop
would differ.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tests.conftest import ToyDiT, write_checkpoint

from aircanvas.sharding import quant
from aircanvas.sharding.manifest import Manifest
from aircanvas.sharding.splitter import split_model
from aircanvas.streaming.engine import StreamingEngine, StreamingError
from aircanvas.streaming.prefetch import ShardHeader

BNB_MISSING = importlib.util.find_spec("bitsandbytes") is None
N_STEPS = 3


def test_nf4_requires_bitsandbytes(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)  # import -> ImportError
    with pytest.raises(quant.QuantError, match=r"aircanvas\[nf4\]"):
        quant._require_bnb()


def test_nf4_engine_requires_cuda(tmp_path: Path) -> None:
    manifest = Manifest(
        source="toy",
        revision=None,
        subfolder="transformer",
        model_class="ToyTransformer2DModel",
        adapter="generic",
        compression="nf4",
        compute_dtype="bfloat16",
    )
    with torch.device("meta"):
        model = ToyDiT()
    with pytest.raises(StreamingError, match="requires a CUDA device"):
        StreamingEngine(model, manifest, tmp_path, device="cpu")


def test_compress_tensor_rejects_nf4() -> None:
    with pytest.raises(quant.QuantError, match="compress_state_dict"):
        quant.compress_tensor("blocks.0.lin1.weight", torch.randn(8, 8), "nf4")


@pytest.fixture()
def nf4_split(tmp_path: Path) -> tuple[ToyDiT, Manifest, Path]:
    torch.manual_seed(21)
    reference = ToyDiT().eval()
    tensors = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    checkpoint = write_checkpoint(
        tmp_path / "toy", tensors, sharded=False, class_name="ToyTransformer2DModel"
    )
    cache = tmp_path / "cache"
    manifest = split_model(
        str(checkpoint), cache_dir=cache, compression="nf4", compute_dtype="bfloat16"
    )
    return reference, manifest, cache


@pytest.mark.gpu
@pytest.mark.skipif(BNB_MISSING, reason="bitsandbytes not installed")
class TestNF4OnGPU:
    def _dequant_reference(self, manifest: Manifest, cache: Path) -> ToyDiT:
        state: dict[str, torch.Tensor] = {}
        for shard in manifest.blocks:
            path = cache / shard.file
            state.update(
                quant.decompress_state_dict(
                    load_file(path, device="cuda"),
                    compression="nf4",
                    compute_dtype=torch.bfloat16,
                    metadata=ShardHeader.parse(path).metadata,
                )
            )
        state.update(load_file(cache / manifest.resident_file, device="cuda"))
        ref = ToyDiT().to(device="cuda", dtype=torch.bfloat16).eval()
        ref.load_state_dict(state)
        return ref

    def test_shards_aligned_and_small(self, nf4_split) -> None:
        _, manifest, cache = nf4_split
        blob = 0
        for shard in manifest.blocks:
            header = ShardHeader.parse(cache / shard.file)
            assert header.aligned, shard.file
            assert any(quant.is_nf4_absmax_key(m.name) for m in header.tensors)
            blob += header.blob_size  # file size is header-dominated on toy shards
        materialized = sum(b.materialized_bytes for b in manifest.blocks)
        assert blob < 0.45 * materialized  # ~0.28x for the payload + bf16 leftovers

    def test_streamed_matches_dequant_reference_bitwise(self, nf4_split) -> None:
        original, manifest, cache = nf4_split
        dequant_ref = self._dequant_reference(manifest, cache)
        x = torch.randn(2, 8, generator=torch.Generator().manual_seed(6)).to(
            device="cuda", dtype=torch.bfloat16
        )
        with torch.no_grad():
            expected = dequant_ref(x)

        with torch.device("meta"):
            streamed = ToyDiT().eval()
        with (
            StreamingEngine(streamed, manifest, cache, device="cuda") as engine,
            torch.no_grad(),
        ):
            for step in range(N_STEPS):
                out = streamed(x)
                torch.cuda.synchronize()
                # sync path (step 1) and DecompressPlan path (steps 2+) must
                # agree exactly with allocating dequantization.
                assert torch.equal(out, expected), f"step {step}"
        assert engine.stats["prefetch_hits"] == (N_STEPS - 1) * len(manifest.blocks) - 1

        # loose sanity vs the ORIGINAL weights: nf4 is lossy but not broken
        with torch.no_grad():
            original_out = original.to(device="cuda", dtype=torch.bfloat16)(x)
        rel = (original_out.float() - expected.float()).norm() / original_out.float().norm()
        assert rel < 0.2, f"nf4 relative error {rel:.3f} looks broken"
