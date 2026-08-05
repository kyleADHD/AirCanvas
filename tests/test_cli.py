"""M4: `aircanvas doctor` — hardware probe + runnable-model table.

Doctor's verdicts run the REAL budget solver against synthetic manifests
shaped like each known model, so its answers cannot drift away from what an
actual run would plan.
"""

from __future__ import annotations

import pytest

from aircanvas.cli import KNOWN_MODELS, _synthetic_manifest, _verdict, main
from aircanvas.utils.hw import HardwareProfile

GB = 1_000_000_000


def profile(vram: int, ram: int = 32 * GB, bw: float = 3.5e9) -> HardwareProfile:
    return HardwareProfile(vram_bytes=vram, ram_bytes=ram, disk_bw_bytes_s=bw, device="cuda")


def test_doctor_runs_and_lists_every_known_model(capsys) -> None:
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "AirCanvas" in out and "doctor" in out
    assert "VRAM" in out and "RAM" in out and "disk read" in out
    for model in KNOWN_MODELS:
        assert model.name in out


def test_doctor_notes_gated_repos(capsys) -> None:
    main(["doctor"])
    out = capsys.readouterr().out
    for model in KNOWN_MODELS:
        if model.gated:
            assert model.name in out and ("gated" in out)


@pytest.mark.parametrize("model", KNOWN_MODELS, ids=lambda m: m.name)
def test_synthetic_manifest_matches_the_model(model) -> None:
    manifest = _synthetic_manifest(model, "fp8")
    assert len(manifest.blocks) == model.n_blocks
    assert manifest.compression == "fp8"
    # fp8 shrinks disk bytes but never the materialised (VRAM) footprint.
    plain = _synthetic_manifest(model, None)
    assert manifest.disk_bytes() < plain.disk_bytes()
    assert sum(b.materialized_bytes for b in manifest.blocks) == sum(
        b.materialized_bytes for b in plain.blocks
    )


def test_verdicts_improve_with_more_vram() -> None:
    """The table has to be monotone, or it is telling users nonsense."""
    flux = next(m for m in KNOWN_MODELS if m.name == "FLUX.1-dev")
    tiny = _verdict(flux, profile(2 * GB), "fp8")
    small = _verdict(flux, profile(6 * GB), "fp8")
    big = _verdict(flux, profile(24 * GB), "fp8")
    assert tiny == "no"
    assert small.startswith(("yes", "tight"))
    assert big.startswith("yes")


def test_fp8_is_never_worse_than_uncompressed() -> None:
    for model in KNOWN_MODELS:
        fp8 = _verdict(model, profile(6 * GB), "fp8")
        raw = _verdict(model, profile(6 * GB), None)
        if raw == "no":
            continue
        assert fp8 != "no", f"{model.name}: fp8 refused where bf16 was accepted"


def test_qwen_image_needs_streaming_but_fits_a_6gb_card() -> None:
    """The M5 headline claim, checked against the solver rather than a slide."""
    qwen = next(m for m in KNOWN_MODELS if m.name == "Qwen-Image")
    assert _verdict(qwen, profile(6 * GB), "fp8").startswith(("yes", "tight"))


def test_probe_disk_flag_is_accepted(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("aircanvas.config.cache_root", lambda: tmp_path)
    assert main(["doctor", "--probe-disk"]) == 0
    out = capsys.readouterr().out
    assert "no shard cache to probe yet" in out or "measured" in out
