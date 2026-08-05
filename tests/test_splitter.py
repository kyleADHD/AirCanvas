"""M1: end-to-end split on toy checkpoints (CPU, no network)."""

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from aircanvas.sharding.manifest import DONE_SUFFIX, Manifest
from aircanvas.sharding.splitter import NotEnoughSpaceError, split_model


def read_shard(path: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in list(f.keys()):
            out[name] = f.get_tensor(name)
    return out


def assert_cache_matches_source(
    cache: Path, manifest: Manifest, tensors: dict[str, torch.Tensor]
) -> None:
    reassembled: dict[str, torch.Tensor] = {}
    for shard in manifest.blocks:
        got = read_shard(cache / shard.file)
        assert set(got), f"empty shard {shard.file}"
        assert all(n.startswith(shard.name + ".") for n in got)
        reassembled.update(got)
    reassembled.update(read_shard(cache / manifest.resident_file))
    assert set(reassembled) == set(tensors)
    for name, t in tensors.items():
        assert torch.equal(reassembled[name], t), name


@pytest.mark.parametrize("fixture", ["flux_checkpoint", "flux_checkpoint_sharded"])
def test_split_roundtrip(fixture: str, request, flux_tensors, tmp_path: Path) -> None:
    checkpoint = request.getfixturevalue(fixture)
    cache = tmp_path / "cache"
    manifest = split_model(str(checkpoint), cache_dir=cache, compute_dtype=None)
    assert manifest.model_class == "FluxTransformer2DModel"
    assert manifest.adapter == "flux"
    assert len(manifest.blocks) == 10
    assert manifest.blocks[0].name == "transformer_blocks.0"
    assert manifest.is_complete(cache)
    assert_cache_matches_source(cache, manifest, flux_tensors)


def test_split_casts_to_bfloat16(flux_checkpoint, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    manifest = split_model(str(flux_checkpoint), cache_dir=cache, compute_dtype="bfloat16")
    assert manifest.compute_dtype == "bfloat16"
    got = read_shard(cache / manifest.blocks[0].file)
    assert all(t.dtype == torch.bfloat16 for t in got.values())


def test_split_reuses_complete_cache(flux_checkpoint, tmp_path: Path, caplog) -> None:
    cache = tmp_path / "cache"
    first = split_model(str(flux_checkpoint), cache_dir=cache, compute_dtype=None)
    mtimes = {f: (cache / f).stat().st_mtime_ns for f in first.shard_files()}
    second = split_model(str(flux_checkpoint), cache_dir=cache, compute_dtype=None)
    assert second == first
    assert {f: (cache / f).stat().st_mtime_ns for f in second.shard_files()} == mtimes


def test_split_repairs_missing_shard(flux_checkpoint, flux_tensors, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    manifest = split_model(str(flux_checkpoint), cache_dir=cache, compute_dtype=None)
    victim = manifest.blocks[3]
    (cache / victim.file).unlink()
    (cache / (victim.file + DONE_SUFFIX)).unlink()
    assert not manifest.is_complete(cache)
    repaired = split_model(str(flux_checkpoint), cache_dir=cache, compute_dtype=None)
    assert repaired.is_complete(cache)
    assert_cache_matches_source(cache, repaired, flux_tensors)


def test_split_incompatible_cache_raises(flux_checkpoint, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    split_model(str(flux_checkpoint), cache_dir=cache, compute_dtype=None)
    with pytest.raises(ValueError, match="different cache_dir"):
        split_model(str(flux_checkpoint), cache_dir=cache, compute_dtype="bfloat16")


def test_preflight_raises_when_disk_full(flux_checkpoint, tmp_path: Path, monkeypatch) -> None:
    import shutil

    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        "aircanvas.sharding.splitter.shutil.disk_usage",
        lambda _: usage._replace(free=0),
    )
    with pytest.raises(NotEnoughSpaceError, match="another drive"):
        split_model(str(flux_checkpoint), cache_dir=tmp_path / "cache", compute_dtype=None)


def test_compression_not_implemented_yet(flux_checkpoint, tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="M4"):
        split_model(str(flux_checkpoint), cache_dir=tmp_path / "c", compression="fp8")


def test_manifest_roundtrip(flux_checkpoint, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    manifest = split_model(str(flux_checkpoint), cache_dir=cache, compute_dtype=None)
    assert Manifest.load(cache) == manifest


def test_cli_split(flux_checkpoint, tmp_path: Path, capsys) -> None:
    from aircanvas.cli import main

    rc = main(
        [
            "split",
            str(flux_checkpoint),
            "--cache-dir",
            str(tmp_path / "cli_cache"),
            "--compute-dtype",
            "source",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Split complete" in out
    assert "10 blocks" in out
