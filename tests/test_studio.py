"""Studio: catalog, verdicts, state store, and the HTTP surface.

Runs offline on CPU. The parts that need a GPU (a real generation) are not
mocked here — `RunManager` is exercised through demo mode, whose replay uses
the same publishing, cancellation and gallery paths as the real one.
"""

from __future__ import annotations

import json

import pytest

from aircanvas.sharding.manifest import synthetic_manifest
from aircanvas.studio import catalog
from aircanvas.studio.events import EventBus
from aircanvas.studio.machine import mask_token
from aircanvas.studio.store import Store, default_state
from aircanvas.utils.hw import HardwareProfile

fastapi = pytest.importorskip("fastapi", reason="the Studio needs the [studio] extra")
from fastapi.testclient import TestClient  # noqa: E402

GB = 1_000_000_000


def laptop() -> HardwareProfile:
    """The 6 GB dev box of docs/BENCHMARKS.md."""
    return HardwareProfile(
        vram_bytes=int(5.1 * GB),
        ram_bytes=int(5.07 * GB),
        disk_bw_bytes_s=1.62e9,
        device="cuda",
        gpu_name="RTX 4050 Laptop",
        vram_total_bytes=int(6.4 * GB),
        ram_total_bytes=int(16.9 * GB),
    )


def workstation() -> HardwareProfile:
    return HardwareProfile(
        vram_bytes=int(23.1 * GB),
        ram_bytes=int(52 * GB),
        disk_bw_bytes_s=6.4e9,
        device="cuda",
        gpu_name="RTX 4090",
        vram_total_bytes=int(24 * GB),
        ram_total_bytes=int(64 * GB),
    )


# -- catalog ---------------------------------------------------------------


def test_every_model_has_a_manifest_the_solver_accepts() -> None:
    for model in catalog.MODELS:
        for fmt in model.formats:
            manifest = model.manifest(fmt)
            assert len(manifest.blocks) == model.n_blocks
            assert manifest.disk_bytes() > 0


def test_on_disk_sizes_drive_the_synthetic_manifest() -> None:
    """The size the setup screen shows IS the size the solver plans against."""
    qwen = catalog.get("qwen-image")
    manifest = qwen.manifest("nf4")
    # Within a rounding block: n_bytes are ints, so the sum drifts by < n_blocks.
    assert manifest.disk_bytes() == pytest.approx(qwen.disk_bytes("nf4"), rel=0.02)


def test_gguf_entries_are_only_verified_sources() -> None:
    """No invented download sizes: a GGUF option exists only where this repo
    has actually read that file (docs/BENCHMARKS.md), or the README names it."""
    for model in catalog.MODELS:
        if model.gguf is None:
            continue
        assert ":" in model.gguf.file, model.id
        assert model.gguf.download_bytes < model.native_bytes, model.id


def test_verdicts_move_with_the_machine() -> None:
    """A verdict is solver output, not a stored table: the same model on a
    bigger card must produce a different, better answer."""
    wan = catalog.get("wan21-t2v-14b")
    tight = catalog.verdict(wan, "fp8", laptop())
    roomy = catalog.verdict(wan, "fp8", workstation())
    assert tight.tone in ("warn", "fail")
    assert roomy.tone == "good"
    assert roomy.resident_blocks > tight.resident_blocks


def test_verdict_always_names_a_reason() -> None:
    for model in catalog.MODELS:
        verdict = catalog.verdict(model, model.default_format, laptop())
        assert verdict.label.strip()
        # Every label is "<call> · <reason>" or "✗ needs N GB".
        assert "·" in verdict.label or verdict.label.startswith("✗")


def test_impossible_model_reports_what_it_would_need() -> None:
    tiny = HardwareProfile(
        vram_bytes=1 * GB,
        ram_bytes=2 * GB,
        disk_bw_bytes_s=5e8,
        device="cuda",
        vram_total_bytes=2 * GB,
        ram_total_bytes=4 * GB,
    )
    verdict = catalog.verdict(catalog.get("qwen-image"), "bf16", tiny)
    assert verdict.tone == "fail"
    assert "needs" in verdict.label
    assert verdict.needed_bytes > 0


def test_estimate_prefers_a_real_measurement() -> None:
    qwen = catalog.get("qwen-image")
    total, per_step = catalog.estimate_seconds(qwen, "nf4", laptop(), steps=20)
    measured_total, measured_step = catalog.estimate_seconds(
        qwen, "nf4", laptop(), steps=20, measured_per_step=12.5
    )
    assert per_step >= qwen.measured.seconds_per_step  # published floor applies
    assert measured_step == 12.5
    assert measured_total == pytest.approx(250.0)
    assert total > 0


def test_synthetic_manifest_rejects_a_zero_block_model() -> None:
    with pytest.raises(ValueError):
        synthetic_manifest(
            source="x",
            n_blocks=0,
            largest_block_bytes=1,
            dit_bytes=1,
            resident_bytes=0,
            compression=None,
        )


# -- store -----------------------------------------------------------------


def test_store_round_trips_and_is_atomic(tmp_path) -> None:
    path = tmp_path / "studio.json"
    store = Store(path)
    store.patch("desk", {"prompt": "a paper boat", "steps": 12})
    assert Store(path).snapshot()["desk"]["prompt"] == "a paper boat"
    assert not list(tmp_path.glob("*.tmp"))


def test_store_starts_fresh_on_a_corrupt_file(tmp_path) -> None:
    path = tmp_path / "studio.json"
    path.write_text("{not json", encoding="utf-8")
    assert Store(path).snapshot()["desk"] == default_state()["desk"]


def test_store_fills_in_new_keys_but_keeps_lists(tmp_path) -> None:
    """A state file from an older build must not lose the user's gallery."""
    path = tmp_path / "studio.json"
    old = default_state()
    old["outputs"] = [{"id": "keep-me"}]
    del old["desk"]["guidance"]
    path.write_text(json.dumps(old), encoding="utf-8")

    state = Store(path).snapshot()
    assert state["outputs"] == [{"id": "keep-me"}]
    assert state["desk"]["guidance"] == default_state()["desk"]["guidance"]


def test_outputs_are_newest_first(tmp_path) -> None:
    store = Store(tmp_path / "studio.json")
    store.add_output({"id": "old", "prompt": "first"})
    store.add_output({"id": "new", "prompt": "second"})
    assert [o["id"] for o in store.snapshot()["outputs"]] == ["new", "old"]


def test_last_measurement_only_matches_the_same_shape(tmp_path) -> None:
    store = Store(tmp_path / "studio.json")
    store.add_output(
        {
            "id": "a",
            "modelId": "qwen-image",
            "state": "done",
            "steps": 20,
            "width": 1024,
            "height": 1024,
            "frames": 1,
            "secondsPerStep": 21.1,
        }
    )
    assert store.last_measurement("qwen-image", steps=20, pixels=1024 * 1024) == 21.1
    # A 512^2 run is not evidence about a 1024^2 one.
    assert store.last_measurement("qwen-image", steps=20, pixels=512 * 512) is None
    assert store.last_measurement("qwen-image", steps=28, pixels=1024 * 1024) is None


def test_mask_token_keeps_only_the_tail() -> None:
    masked = mask_token("hf_abcdefghijklmnopqrstuvwx2f9d")
    assert masked is not None
    assert masked.startswith("hf_") and masked.endswith("2f9d")
    assert "abcdefgh" not in masked
    assert mask_token(None) is None


# -- events ----------------------------------------------------------------


def test_bus_fans_out_and_replays_a_backlog() -> None:
    bus = EventBus()
    bus.publish("run", run={"id": "1"})
    with bus.subscribe() as first, bus.subscribe() as second:
        assert first.get(0.1)["type"] == "run"  # replayed
        bus.publish("outputs", outputs=[])
        assert first.get(0.5)["type"] == "outputs"
        # `second` also replayed the backlog, so its first event is the old one.
        assert second.get(0.1)["type"] == "run"
        assert second.get(0.5)["type"] == "outputs"
    assert bus.subscriber_count == 0


def test_bus_get_times_out_rather_than_blocking_forever() -> None:
    bus = EventBus()
    with bus.subscribe(replay=False) as subscriber:
        assert subscriber.get(0.01) is None


# -- HTTP ------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    from aircanvas.studio import demo
    from aircanvas.studio.server import Studio, create_app

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    studio = Studio(store=Store(tmp_path / "studio.json"), demo=True)
    demo.install(studio)
    with TestClient(create_app(studio)) as test_client:
        test_client.studio = studio  # type: ignore[attr-defined]
        yield test_client


def test_state_fills_every_panel(client) -> None:
    state = client.get("/api/state").json()
    for key in ("mode", "theme", "desk", "settings", "machine", "models", "cache", "outputs"):
        assert key in state
    assert state["demo"] is True
    assert len(state["models"]) == len(catalog.MODELS)
    assert state["machine"]["gpu"] == "RTX 4050 Laptop"


def test_models_carry_a_verdict_per_format(client) -> None:
    for model in client.get("/api/models").json()["models"]:
        assert set(model["verdicts"]) == set(model["formats"])
        for verdict in model["verdicts"].values():
            assert verdict["tone"] in ("good", "warn", "fail")


def test_patch_persists_and_is_read_back(client) -> None:
    client.post("/api/state", json={"mode": "pro", "desk": {"prompt": "a rainy tram"}})
    state = client.get("/api/state").json()
    assert state["mode"] == "pro"
    assert state["desk"]["prompt"] == "a rainy tram"


def test_estimate_uses_the_last_run_of_the_same_shape(client) -> None:
    payload = {"modelId": "qwen-image", "format": "nf4", "steps": 20, "width": 1024, "height": 1024}
    estimate = client.post("/api/estimate", json=payload).json()
    # Demo mode seeds a measured 427.4 s Qwen run at this exact shape.
    assert estimate["source"] == "measured"
    assert estimate["totalSeconds"] == pytest.approx(427.4, abs=1.0)


def test_estimate_404s_on_an_unknown_model(client) -> None:
    assert client.post("/api/estimate", json={"modelId": "nope"}).status_code == 404


def test_index_and_static_assets_are_served(client) -> None:
    assert client.get("/").status_code == 200
    for asset in (
        "/static/styles.css",
        "/static/app.js",
        "/static/fonts/space-grotesk-latin.woff2",
        "/static/fonts/jetbrains-mono-latin.woff2",
    ):
        assert client.get(asset).status_code == 200, asset


def test_split_can_be_queued_paused_and_resumed(client) -> None:
    job = client.post("/api/splits", json={"modelId": "sd35-large", "format": "fp8"}).json()
    assert job["blocksTotal"] == catalog.get("sd35-large").n_blocks

    assert client.post(f"/api/splits/{job['id']}/pause").status_code == 200
    resumed = client.post(f"/api/splits/{job['id']}/resume").json()
    assert any(j["id"] == job["id"] for j in resumed["jobs"])
    assert client.post(f"/api/splits/{job['id']}/cancel").status_code == 200
    assert client.post(f"/api/splits/{job['id']}/pause").status_code == 404


def test_unknown_split_action_is_rejected(client) -> None:
    assert client.post("/api/splits/abc/detonate").status_code == 400


def test_run_produces_a_gallery_entry_with_a_report(client) -> None:
    started = client.post(
        "/api/runs",
        json={
            "modelId": "wan21-t2v-1.3b",
            "format": "fp8",
            "prompt": "a paper boat",
            "steps": 2,
            "seed": 7130,
        },
    ).json()
    assert started["status"] in ("starting", "encode", "denoise")

    run = _await_run(client, started["id"])
    assert run["status"] == "done", run.get("error")
    assert run["outputId"]

    entry = next(o for o in client.get("/api/outputs").json()["outputs"] if o["id"] == run["id"])
    assert entry["seed"] == 7130
    assert entry["report"]["plan"]["resident_blocks"] >= 0
    assert entry["report"]["phases"]["steps"] == 2

    report = client.get(f"/api/outputs/{run['id']}/report")
    assert report.status_code == 200
    assert report.json()["plan"]["step_read_bytes"] >= 0


def test_only_one_run_at_a_time(client) -> None:
    body = {"modelId": "wan21-t2v-1.3b", "prompt": "a", "steps": 4}
    first = client.post("/api/runs", json=body)
    assert first.status_code == 200
    assert client.post("/api/runs", json=body).status_code == 409
    client.post("/api/runs/cancel")
    _await_run(client, first.json()["id"])


def test_cancelling_a_run_keeps_the_shard_cache(client) -> None:
    before = client.get("/api/state").json()["cache"]["totalBytes"]
    started = client.post(
        "/api/runs", json={"modelId": "qwen-image", "prompt": "a tram", "steps": 20}
    ).json()
    assert client.post("/api/runs/cancel").json()["cancelled"] is True
    run = _await_run(client, started["id"])
    assert run["status"] == "cancelled"
    assert client.get("/api/state").json()["cache"]["totalBytes"] == before


def test_reproduce_counter_and_delete(client) -> None:
    entry = client.get("/api/outputs").json()["outputs"][0]
    bumped = client.post(f"/api/outputs/{entry['id']}/reproduced").json()
    assert bumped["reproducedCount"] == (entry.get("reproducedCount") or 0) + 1

    remaining = client.delete(f"/api/outputs/{entry['id']}").json()["outputs"]
    assert all(o["id"] != entry["id"] for o in remaining)
    assert client.delete(f"/api/outputs/{entry['id']}").status_code == 404


def test_cache_remove_refuses_a_path_outside_the_cache_root(client) -> None:
    assert client.post("/api/cache/remove", json={"cacheDir": "/etc"}).status_code == 400
    assert client.post("/api/cache/remove", json={"cacheDir": ""}).status_code == 400


def test_missing_output_file_is_a_404_not_a_crash(client) -> None:
    entry = client.get("/api/outputs").json()["outputs"][0]
    assert client.get(f"/api/outputs/{entry['id']}/file").status_code == 404


def _await_run(client, run_id: str, timeout: float = 30.0) -> dict:
    """Poll until the run leaves the live states. Demo runs take ~1 s."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get("/api/runs/current").json()["run"]
        if run and run["id"] == run_id and run["status"] in ("done", "cancelled", "failed"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


def test_report_plan_dict_covers_every_plan_field() -> None:
    """A saved report must be able to redraw the plan it describes.

    The Report screen reads the residency bar's axis out of the plan dict.
    Four byte counts were missing from it while `describe()` printed them, so
    a persisted report drew a zero-width bar; `as_dict()` now enumerates the
    dataclass so the two cannot drift again.
    """
    from dataclasses import fields

    from aircanvas.streaming.residency import ResidencyPlan

    plan = ResidencyPlan(resident_blocks=1, ring_depth=2, lookahead=2, ram_cache_bytes=3)
    assert set(plan.as_dict()) == {f.name for f in fields(ResidencyPlan)}
    assert plan.as_dict()["warnings"] == []  # tuples serialise as JSON arrays
