"""GGUF-as-source splitting (M9 phase 1).

The GGUF is dequantized at split time into the existing codecs, so the
streaming hot path is untouched — but the split result must still match a
reference model built from the same dequantized tensors, bitwise where the
transform chain is exact (F32 source, no compression)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")

from gguf import GGMLQuantizationType, GGUFWriter, quants  # noqa: E402
from safetensors import safe_open  # noqa: E402
from tests.conftest import ToyDiT  # noqa: E402

from aircanvas.sharding.gguf_source import (  # noqa: E402
    DiffusersGGUFCheckpoint,
    GGUFCheckpoint,
    GGUFError,
    resolve_gguf_path,
)
from aircanvas.sharding.manifest import DONE_SUFFIX, Manifest  # noqa: E402
from aircanvas.sharding.splitter import shard_cache_dir, split_model  # noqa: E402
from aircanvas.streaming.engine import StreamingEngine  # noqa: E402


def write_gguf(
    path: Path,
    tensors: dict[str, np.ndarray],
    quantize: dict[str, GGMLQuantizationType] | None = None,
) -> Path:
    w = GGUFWriter(str(path), arch="aircanvas-test")
    for name, arr in tensors.items():
        qt = (quantize or {}).get(name)
        if qt is None:
            w.add_tensor(name, arr)
        else:
            q = quants.quantize(arr, qt)
            w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def write_config_only(root: Path, class_name: str) -> Path:
    """A source dir holding ONLY transformer/config.json — proving a GGUF
    split never needs the original weights."""
    src = root / "transformer"
    src.mkdir(parents=True)
    (src / "config.json").write_text(json.dumps({"_class_name": class_name}))
    return root


@pytest.fixture()
def toy_gguf(tmp_path: Path) -> tuple[ToyDiT, Path, Path]:
    torch.manual_seed(11)
    reference = ToyDiT().eval()
    tensors = {k: v.detach().numpy() for k, v in reference.state_dict().items()}
    source = write_config_only(tmp_path / "src", "ToyTransformer2DModel")
    gpath = write_gguf(tmp_path / "toy.gguf", tensors)
    return reference, source, gpath


def test_gguf_split_streams_bitwise(toy_gguf, tmp_path: Path) -> None:
    reference, source, gpath = toy_gguf
    cache = tmp_path / "cache"
    manifest = split_model(str(source), cache_dir=cache, compute_dtype=None, gguf_file=str(gpath))
    assert manifest.gguf_file == str(gpath)
    assert manifest.model_class == "ToyTransformer2DModel"
    assert Manifest.load(cache).gguf_file == str(gpath)

    with torch.device("meta"):
        streamed = ToyDiT().eval()
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        expected = reference(x)
    with StreamingEngine(streamed, manifest, cache, device="cpu"), torch.no_grad():
        out = streamed(x)
    assert torch.equal(out, expected)


def test_quantized_tensor_fidelity_matches_gguf_reference(tmp_path: Path) -> None:
    """Q8_0/F16 payloads must land in shards exactly as gguf's own dequant
    (then dtype cast) produces them — no extra transform sneaks in."""
    rng = np.random.default_rng(5)
    arrs = {
        f"blocks.{i}.lin.weight": rng.standard_normal((8, 32), dtype=np.float32)
        for i in range(4)  # generic adapter needs >= 4 contiguous blocks
    }
    arrs["x_embedder.weight"] = rng.standard_normal((8, 32), dtype=np.float32)
    qmap = {
        "blocks.0.lin.weight": GGMLQuantizationType.Q8_0,
        "blocks.1.lin.weight": GGMLQuantizationType.F16,
    }
    gpath = write_gguf(tmp_path / "q.gguf", arrs, qmap)
    source = write_config_only(tmp_path / "src", "ToyTransformer2DModel")
    cache = tmp_path / "cache"
    manifest = split_model(
        str(source), cache_dir=cache, compute_dtype="bfloat16", gguf_file=str(gpath)
    )

    reader = gguf.GGUFReader(str(gpath))
    reference = {
        t.name: torch.from_numpy(
            np.ascontiguousarray(quants.dequantize(t.data, t.tensor_type)).copy()
        ).to(torch.bfloat16)
        for t in reader.tensors
    }
    for shard in manifest.shard_files():
        with safe_open(cache / shard, framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118 — safe_open handles aren't dicts
                assert torch.equal(f.get_tensor(name), reference[name]), name


def test_fp8_compression_from_gguf(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    arrs = {
        f"blocks.{i}.lin.weight": rng.standard_normal((8, 32), dtype=np.float32)
        for i in range(4)  # generic adapter needs >= 4 contiguous blocks
    }
    arrs["x_embedder.weight"] = rng.standard_normal((8, 32), dtype=np.float32)
    gpath = write_gguf(tmp_path / "q.gguf", arrs)
    source = write_config_only(tmp_path / "src", "ToyTransformer2DModel")
    cache = tmp_path / "cache"
    manifest = split_model(
        str(source),
        cache_dir=cache,
        compression="fp8",
        compute_dtype="bfloat16",
        gguf_file=str(gpath),
    )
    assert manifest.compression == "fp8"
    with safe_open(cache / manifest.blocks[0].file, framework="pt", device="cpu") as f:
        names = list(f.keys())
        payload = f.get_tensor("blocks.0.lin.weight")
        scale = f.get_tensor("blocks.0.lin.weight.__ac_scale")
    assert payload.dtype == torch.float8_e4m3fn, names
    recon = payload.to(torch.float32) * scale.to(torch.float32)
    original = torch.from_numpy(arrs["blocks.0.lin.weight"])
    assert torch.allclose(recon, original, atol=0.12, rtol=0.12)


def test_gguf_caches_never_collide(tmp_path: Path) -> None:
    plain = shard_cache_dir("org/model", "transformer", "fp8", "bfloat16")
    a = shard_cache_dir("org/model", "transformer", "fp8", "bfloat16", "m-Q4_K_S.gguf")
    b = shard_cache_dir("org/model", "transformer", "fp8", "bfloat16", "repo/x:m-Q8_0.gguf")
    assert len({plain, a, b}) == 3
    assert "gguf" in a.name and "gguf" in b.name


def test_gguf_resume_repairs_missing_shard(toy_gguf, tmp_path: Path) -> None:
    _, source, gpath = toy_gguf
    cache = tmp_path / "cache"
    manifest = split_model(str(source), cache_dir=cache, compute_dtype=None, gguf_file=str(gpath))
    victim = manifest.blocks[1].file
    (cache / (victim + DONE_SUFFIX)).unlink()
    repaired = split_model(str(source), cache_dir=cache, compute_dtype=None, gguf_file=str(gpath))
    assert repaired.is_complete(cache)


def test_gguf_and_plain_cache_mismatch_raises(toy_gguf, tmp_path: Path) -> None:
    _, source, gpath = toy_gguf
    cache = tmp_path / "cache"
    split_model(str(source), cache_dir=cache, compute_dtype=None, gguf_file=str(gpath))
    with pytest.raises(ValueError, match="gguf"):
        split_model(str(source), cache_dir=cache, compute_dtype=None)


def test_resolve_rejects_missing_paths_and_windows_drives() -> None:
    with pytest.raises(GGUFError, match="neither an existing file"):
        resolve_gguf_path("definitely-missing.gguf")
    # A non-existent Windows path must NOT be parsed as repo_id:filename.
    with pytest.raises(GGUFError, match="neither an existing file"):
        resolve_gguf_path(r"C:\models\missing.gguf")


def test_foreign_layout_without_diffusers_class_raises(tmp_path: Path) -> None:
    """Names that no adapter recognizes need diffusers' single-file converter;
    a class diffusers doesn't ship gets a clear error, not a crash."""
    gpath = write_gguf(
        tmp_path / "alien.gguf",
        {"alien_stack.0.qkv.weight": np.zeros((8, 32), dtype=np.float32)},
    )
    with pytest.raises(GGUFError, match="from_single_file"):
        DiffusersGGUFCheckpoint(
            gpath, "ToyTransformer2DModel", "unused", "transformer", None, None, None
        )


def test_direct_reader_reports_names_and_sizes(toy_gguf) -> None:
    _, _, gpath = toy_gguf
    ckpt = GGUFCheckpoint(gpath)
    names = ckpt.tensor_names()
    assert "x_embedder.weight" in names
    # f32 target: every toy tensor materializes at 4 bytes/element
    expected = sum(t.numel() * 4 for t in ToyDiT().state_dict().values())
    assert ckpt.materialized_bytes(None) == expected
