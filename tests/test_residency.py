"""M4: budget solver + hardware probes, against synthetic hardware profiles.

These are the tests that keep `aircanvas doctor` and every OOM message honest:
the solver is pure arithmetic over a manifest and a HardwareProfile, so it can
be checked exactly without a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aircanvas.sharding.manifest import BlockShard, Manifest
from aircanvas.streaming.residency import (
    RAM_HEADROOM_BYTES,
    VRAM_HEADROOM_BYTES,
    InsufficientVRAMError,
    ResidencyPlan,
    Workload,
    solve,
)
from aircanvas.utils import hw

GB = 1_000_000_000
MB = 1_000_000


def make_manifest(
    n_blocks: int = 10,
    block_bytes: int = 680 * MB,
    *,
    compression=None,
    ratio: float = 1.0,
    resident_bytes: int = 500 * MB,
    first_block_bytes: int | None = None,
) -> Manifest:
    sizes = [block_bytes] * n_blocks
    if first_block_bytes is not None:
        sizes[0] = first_block_bytes
    return Manifest(
        source="synthetic",
        revision=None,
        subfolder="transformer",
        model_class="SyntheticTransformer2DModel",
        adapter="generic",
        compression=compression,
        compute_dtype="bfloat16",
        blocks=tuple(
            BlockShard(
                name=f"blocks.{i}",
                file=f"block_{i:04d}.safetensors",
                n_bytes=int(size * ratio),
                load_bytes=size,
            )
            for i, size in enumerate(sizes)
        ),
        resident_bytes=resident_bytes,
        resident_load_bytes=resident_bytes,
    )


def profile(vram: int, ram: int = 16 * GB, bw: float = 3.5e9) -> hw.HardwareProfile:
    return hw.HardwareProfile(vram_bytes=vram, ram_bytes=ram, disk_bw_bytes_s=bw)


SMALL_WORKLOAD = Workload(steps=28, activation_bytes=1 * GB)


# -- the waterfall ---------------------------------------------------------


# fixed reserve for make_manifest() defaults: 1 GB activations + 500 MB
# non-block resident + headroom; plus a 2 x 680 MB slot pool = ~3.25 GB.
TIGHT_VRAM = int(3.3 * GB)


def test_tight_vram_streams_everything() -> None:
    manifest = make_manifest()
    plan = solve(manifest, profile(TIGHT_VRAM), SMALL_WORKLOAD)
    assert plan.resident_blocks == 0
    assert plan.streamed_blocks == 10
    assert plan.slot_bytes == 2 * 680 * MB
    assert plan.vram_planned_bytes <= TIGHT_VRAM
    assert any("No VRAM left for resident blocks" in w for w in plan.warnings)


def test_spare_vram_becomes_resident_blocks() -> None:
    manifest = make_manifest()
    tight = solve(manifest, profile(TIGHT_VRAM), SMALL_WORKLOAD)
    roomy = solve(manifest, profile(12 * GB), SMALL_WORKLOAD)
    assert roomy.resident_blocks > tight.resident_blocks
    assert roomy.step_read_bytes < tight.step_read_bytes
    assert roomy.vram_planned_bytes <= 12 * GB


def test_everything_resident_when_vram_is_huge() -> None:
    manifest = make_manifest(n_blocks=4, block_bytes=100 * MB)
    plan = solve(manifest, profile(64 * GB), SMALL_WORKLOAD)
    assert plan.resident_blocks == 4
    assert plan.streamed_blocks == 0
    assert plan.step_read_bytes == 0


def test_plan_never_exceeds_the_budget() -> None:
    manifest = make_manifest(n_blocks=20, block_bytes=300 * MB)
    for vram_gb in range(3, 25):
        plan = solve(manifest, profile(vram_gb * GB), SMALL_WORKLOAD)
        assert plan.vram_planned_bytes <= vram_gb * GB, vram_gb


def test_residency_is_not_monotonic_but_the_scan_finds_the_best() -> None:
    """One huge block at the front: pinning it SHRINKS the slot pool, so a
    first-miss break would wrongly stop at R=0."""
    manifest = make_manifest(
        n_blocks=6, block_bytes=100 * MB, first_block_bytes=3 * GB, resident_bytes=0
    )
    budget = int(1 * GB + VRAM_HEADROOM_BYTES + 3.3 * GB)
    plan = solve(manifest, profile(budget), SMALL_WORKLOAD)
    assert plan.resident_blocks >= 1
    assert plan.slot_bytes == 2 * 100 * MB


def test_insufficient_vram_reports_the_attempted_plan() -> None:
    manifest = make_manifest(n_blocks=40, block_bytes=700 * MB)
    with pytest.raises(InsufficientVRAMError) as excinfo:
        solve(manifest, profile(1 * GB), SMALL_WORKLOAD)
    message = str(excinfo.value)
    assert "Attempted plan" in message
    assert "slot pool" in message and "activations est." in message


def test_max_resident_blocks_caps_the_waterfall() -> None:
    manifest = make_manifest()
    plan = solve(manifest, profile(64 * GB), SMALL_WORKLOAD, max_resident_blocks=3)
    assert plan.resident_blocks == 3
    assert plan.streamed_blocks == 7


# -- compression changes the arithmetic ------------------------------------


def test_compressed_cache_reads_less_but_needs_a_staging_pool() -> None:
    plain = make_manifest()
    fp8 = make_manifest(compression="fp8", ratio=0.5)
    # Same residency on both sides, so only the compression differs.
    p_plan = solve(plain, profile(8 * GB), SMALL_WORKLOAD, max_resident_blocks=0)
    f_plan = solve(fp8, profile(8 * GB), SMALL_WORKLOAD, max_resident_blocks=0)
    assert f_plan.step_read_bytes == p_plan.step_read_bytes // 2
    # ADR #8: 2 x compressed staging on top of the materialised slots.
    assert f_plan.slot_bytes == p_plan.slot_bytes + 2 * 340 * MB


def test_uncompressed_large_model_warns_about_traffic() -> None:
    manifest = make_manifest(n_blocks=40, block_bytes=700 * MB)
    plan = solve(manifest, profile(4 * GB), SMALL_WORKLOAD)
    assert any("compression='fp8'" in w for w in plan.warnings)


# -- RAM waterfall ---------------------------------------------------------


def test_low_ram_shrinks_the_ring_and_warns() -> None:
    manifest = make_manifest(n_blocks=10, block_bytes=700 * MB)
    plan = solve(manifest, profile(4 * GB, ram=RAM_HEADROOM_BYTES + 800 * MB), SMALL_WORKLOAD)
    assert plan.ring_depth == 1
    assert plan.pinned_bytes == 700 * MB
    assert any("Pinned ring reduced" in w for w in plan.warnings)


def test_plentiful_ram_sizes_the_shard_cache() -> None:
    manifest = make_manifest(n_blocks=10, block_bytes=200 * MB)
    plan = solve(manifest, profile(4 * GB, ram=64 * GB), SMALL_WORKLOAD, max_resident_blocks=0)
    assert plan.ram_cache_bytes == plan.step_read_bytes
    assert any("M8" in w for w in plan.warnings)
    assert "promoted after first read" in plan.describe()  # tier is live (M8)


# -- warnings --------------------------------------------------------------


def test_sata_disk_warns() -> None:
    plan = solve(make_manifest(), profile(4 * GB, bw=0.5e9), SMALL_WORKLOAD)
    assert any("SATA-class disk" in w for w in plan.warnings)


def test_distilled_model_warns() -> None:
    plan = solve(make_manifest(), profile(4 * GB), Workload(steps=4, activation_bytes=1 * GB))
    assert any("distilled" in w for w in plan.warnings)


# -- workload model --------------------------------------------------------


def test_activation_estimate_scales_with_tokens() -> None:
    small = Workload(tokens=4608, hidden=3072).estimate_activation_bytes()
    big = Workload(tokens=75_600, hidden=5120).estimate_activation_bytes()
    assert big > small
    # RESEARCH.md §4 measures ~2.9 GB for Wan 14B @720p; stay in the right order.
    assert 2 * GB < big < 4 * GB


def test_activation_estimate_has_a_floor() -> None:
    assert Workload(tokens=1, hidden=1).estimate_activation_bytes() >= 256 << 20


def test_explicit_activation_bytes_wins() -> None:
    assert Workload(tokens=99999, activation_bytes=123).estimate_activation_bytes() == 123


def test_adapter_token_count() -> None:
    from aircanvas.adapters import GenericAdapter
    from aircanvas.adapters.flux import FluxAdapter

    assert GenericAdapter().token_count(1024, 1024) == 64 * 64
    assert FluxAdapter().token_count(1024, 1024) == 64 * 64 + 512
    assert GenericAdapter().token_count(832, 480, frames=21) == 52 * 30 * 21


def test_stream_config_round_trip() -> None:
    plan = ResidencyPlan(resident_blocks=2, ring_depth=4, lookahead=2, ram_cache_bytes=0)
    config = plan.stream_config()
    assert config.ring_depth == 4 and config.gpu_slots == 2


def test_empty_manifest_rejected() -> None:
    manifest = make_manifest(n_blocks=0)
    with pytest.raises(ValueError, match="no blocks"):
        solve(manifest, profile(4 * GB), SMALL_WORKLOAD)


# -- hardware probes -------------------------------------------------------


def test_ram_probe_is_plausible() -> None:
    free, total = hw.free_ram_bytes(), hw.total_ram_bytes()
    assert 0 < free <= total
    assert total > 1 * GB  # no supported machine has less


def test_vram_probe_matches_cuda_availability() -> None:
    if hw.cuda_available():
        assert hw.total_vram_bytes() > 0
        assert 0 <= hw.free_vram_bytes() <= hw.total_vram_bytes()
    else:
        assert hw.free_vram_bytes() == 0 and hw.total_vram_bytes() == 0


def test_disk_probe_falls_back_without_a_sample(tmp_path: Path) -> None:
    assert hw.probe_disk_bandwidth(tmp_path / "nope.safetensors") == hw.DEFAULT_DISK_BW


def test_disk_probe_ignores_tiny_samples(tmp_path: Path) -> None:
    """A 1 MB read measures the page cache; it must not be cached as truth."""
    sample = tmp_path / "small.bin"
    sample.write_bytes(b"\0" * (1 << 20))
    assert hw.probe_disk_bandwidth(sample, cache_dir=tmp_path) == hw.DEFAULT_DISK_BW
    assert not (tmp_path / hw.BW_SIDECAR).exists()


def test_disk_probe_uses_the_sidecar(tmp_path: Path) -> None:
    (tmp_path / hw.BW_SIDECAR).write_text('{"bytes_per_s": 1234.0}', encoding="utf-8")
    sample = tmp_path / "small.bin"
    sample.write_bytes(b"\0" * 1024)
    assert hw.probe_disk_bandwidth(sample, cache_dir=tmp_path) == 1234.0


def test_disk_probe_survives_a_corrupt_sidecar(tmp_path: Path) -> None:
    (tmp_path / hw.BW_SIDECAR).write_text("not json", encoding="utf-8")
    assert hw.probe_disk_bandwidth(tmp_path / "nope.bin", cache_dir=tmp_path) == hw.DEFAULT_DISK_BW


def test_free_disk_bytes_walks_up_to_an_existing_parent(tmp_path: Path) -> None:
    assert hw.free_disk_bytes(tmp_path / "does" / "not" / "exist") > 0


def test_hardware_profile_probe_describes_itself() -> None:
    text = hw.HardwareProfile.probe().describe()
    assert "VRAM" in text and "RAM" in text and "disk read" in text


@pytest.mark.skipif(sys.platform != "win32", reason="GlobalMemoryStatusEx is Windows-only")
def test_windows_ram_uses_ctypes_not_psutil() -> None:
    free, total = hw._windows_ram()
    assert 0 < free <= total
