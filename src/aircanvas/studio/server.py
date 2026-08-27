"""The local HTTP surface: one JSON API, one event stream, one static app.

Bound to 127.0.0.1 by default and never authenticated, because it is a desktop
app that happens to be drawn by a browser — the same posture as a Jupyter
server started by hand. Everything it exposes is already readable by whoever
is at the keyboard: their own shard cache, their own hardware, their own
outputs. `--host` exists for the "run it on the box under the desk" case and
says what it means in the CLI help.

Route shape follows the screens rather than the internals: `/api/state` is one
GET that fills every panel on first paint, and `/api/events` streams the
deltas after that. A screen never polls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from aircanvas.studio import catalog, machine
from aircanvas.studio.events import EventBus
from aircanvas.studio.jobs import RunManager, SplitManager
from aircanvas.studio.store import QUALITY_STEPS, SHAPES, Store

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
KEEPALIVE_SECONDS = 15.0


class Studio:
    """Everything one Studio process owns. The demo swaps parts of it out."""

    def __init__(self, *, store: Store | None = None, demo: bool = False) -> None:
        self.bus = EventBus()
        self.store = store or Store()
        self.splits = SplitManager(self.bus, self.store)
        self.runs = RunManager(self.bus, self.store)
        self.demo = demo
        #: Set by demo mode to a declared machine instead of a probed one.
        self.probe_override: machine.MachineProbe | None = None
        #: Likewise for what is on disk, so the demo shows a populated cache.
        self.installed_override: list[machine.InstalledModel] | None = None
        self._probe: machine.MachineProbe | None = None

    def installed(self) -> list[machine.InstalledModel]:
        if self.installed_override is not None:
            return self.installed_override
        return machine.installed_models()

    def cache(self) -> dict[str, Any]:
        if self.installed_override is None:
            return machine.cache_summary()
        models = self.installed_override
        probe = self.probe()
        return {
            "path": probe.cache_path,
            "totalBytes": sum(m.disk_bytes for m in models),
            "freeBytes": probe.cache_free_bytes,
            "models": [m.as_dict() for m in models],
        }

    # -- machine -----------------------------------------------------------

    def probe(self, *, force: bool = False, probe_disk: bool = False) -> machine.MachineProbe:
        """The cached machine reading, re-read on demand.

        Cached because every verdict chip on the setup screen re-solves against
        it and probing CUDA per chip would be silly; re-read on `force` because
        "another app released VRAM" is exactly when a user presses Re-probe.
        """
        if self.probe_override is not None:
            return self.probe_override
        if self._probe is None or force or probe_disk:
            self._probe = machine.probe(probe_disk=probe_disk)
        return self._probe

    # -- the one payload every screen starts from --------------------------

    def state(self) -> dict[str, Any]:
        probe = self.probe()
        state = self.store.snapshot()
        return {
            "mode": state.get("mode", "simple"),
            "theme": state.get("theme", "dark"),
            "seenWelcome": state.get("seenWelcome", False),
            "desk": state.get("desk", {}),
            "settings": state.get("settings", {}),
            "setup": state.get("setup", {}),
            "outputs": state.get("outputs", []),
            "machine": probe.as_dict(),
            "models": self.models(),
            "installed": [m.as_dict() for m in self.installed()],
            "cache": self.cache(),
            "huggingFace": machine.hf_account(),
            "splits": self.splits.jobs(),
            "run": self.runs.current(),
            "shapes": {name: {"width": w, "height": h} for name, (w, h) in SHAPES.items()},
            "quality": dict(QUALITY_STEPS),
            "demo": self.demo,
            "versions": versions(),
        }

    def models(self) -> list[dict[str, Any]]:
        """The catalog, with a verdict per available shard format.

        Verdicts are recomputed here on every call rather than cached, so a
        re-probe or another app releasing VRAM moves every chip — the handoff's
        rule that a verdict is solver output, not a stored table.
        """
        hardware = self.probe().hardware()
        records = self.installed()
        installed = {m.source: m for m in records}

        # Anything split locally that the table does not list gets a row of its
        # own, built from its manifest. Without this a cache made by
        # `aircanvas split ./my-model` shows up in Settings but cannot be
        # selected on the Desk, which is a strange thing for a UI to do.
        catalog.LOCAL.clear()
        known = {m.repo_id for m in catalog.MODELS}
        for record in records:
            if record.source in known or not record.complete:
                continue
            local = catalog.from_installed(record)
            if local is not None:
                catalog.LOCAL[local.id] = local

        rows: list[dict[str, Any]] = []
        for model in (*catalog.MODELS, *catalog.LOCAL.values()):
            verdicts = {
                fmt: catalog.verdict(model, fmt, hardware).as_dict() for fmt in model.formats
            }
            here = installed.get(model.repo_id)
            rows.append(
                {
                    "id": model.id,
                    "name": model.name,
                    "repoId": model.repo_id,
                    "kind": model.kind,
                    "params": model.params,
                    "blocks": model.n_blocks,
                    "nativeBytes": model.native_bytes,
                    "defaultFormat": model.default_format,
                    "formats": list(model.formats),
                    "onDisk": {fmt: model.disk_bytes(fmt) for fmt in model.formats},
                    "gated": model.gated,
                    "note": model.note,
                    "warnings": list(model.warnings),
                    "verdicts": verdicts,
                    "gguf": (
                        {
                            "file": model.gguf.file,
                            "downloadBytes": model.gguf.download_bytes,
                            "quant": model.gguf.quant,
                            "verified": model.gguf.verified,
                        }
                        if model.gguf
                        else None
                    ),
                    "measured": (
                        {
                            "secondsPerStep": model.measured.seconds_per_step,
                            "steps": model.measured.steps,
                            "note": model.measured.note,
                            "residentRatio": model.measured.resident_ratio,
                        }
                        if model.measured
                        else None
                    ),
                    "simple": (
                        {
                            "headline": model.simple.headline,
                            "quality": model.simple.quality,
                            "tone": model.simple.tone,
                            "unit": model.simple.unit,
                        }
                        if model.simple
                        else None
                    ),
                    "shape": {
                        "width": model.width,
                        "height": model.height,
                        "frames": model.frames,
                        "fps": model.fps,
                        "steps": model.steps,
                        "guidance": model.guidance,
                    },
                    "installed": here.as_dict() if here else None,
                    "local": model.local,
                }
            )
        return rows

    def estimate(self, params: dict[str, Any]) -> dict[str, Any]:
        model = catalog.get(str(params["modelId"]))
        fmt = str(params.get("format") or model.default_format)
        steps = int(params.get("steps") or model.steps)
        width = int(params.get("width") or model.width)
        height = int(params.get("height") or model.height)
        frames = max(1, int(params.get("frames") or model.frames))
        probe = self.probe()
        measured = self.store.last_measurement(
            model.id, steps=steps, pixels=width * height * frames
        )
        if probe.device != "cuda" and measured is None:
            # No CUDA and nothing measured here: the only per-step figures we
            # have came off a GPU. Report no estimate rather than one of those.
            return {"totalSeconds": 0.0, "secondsPerStep": 0.0, "source": "cpu"}
        total, per_step = catalog.estimate_seconds(
            model,
            fmt,
            probe.hardware(),
            steps=steps,
            measured_per_step=measured,
        )
        return {
            "totalSeconds": round(total, 1),
            "secondsPerStep": round(per_step, 2),
            # The Desk says "~7 min" for a solver bound and "7 min" for a
            # measurement of this exact shape. The screens need to know which.
            "source": "measured" if measured else ("published" if model.measured else "solver"),
        }


def versions() -> dict[str, str]:
    """`aircanvas 0.2.0 · torch 2.6 · diffusers 0.39` in the Settings footer."""
    from aircanvas import __version__

    out = {"aircanvas": __version__}
    for name in ("torch", "diffusers"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "?"))
        except ImportError:
            continue
    return out


def create_app(studio: Studio | None = None) -> Any:
    """Build the FastAPI application around one `Studio`."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as e:  # pragma: no cover - the extra is documented
        raise ImportError(
            "AirCanvas Studio needs FastAPI and uvicorn: pip install 'aircanvas[studio]'"
        ) from e

    app_state = studio or Studio()
    app = FastAPI(title="AirCanvas Studio", docs_url=None, redoc_url=None)
    app.state.studio = app_state

    @app.get("/")
    def index() -> Any:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    def get_state() -> Any:
        return app_state.state()

    @app.post("/api/state")
    def patch_state(payload: dict[str, Any]) -> Any:
        for section in ("desk", "settings", "setup"):
            if section in payload and isinstance(payload[section], dict):
                app_state.store.patch(section, payload[section])
        for key in ("mode", "theme", "seenWelcome"):
            if key in payload:
                app_state.store.set(key, payload[key])
        state = app_state.state()
        app_state.bus.publish("state", state=state)
        return state

    @app.get("/api/machine")
    def get_machine(probe_disk: bool = False) -> Any:
        probe = app_state.probe(force=True, probe_disk=probe_disk)
        payload = {"machine": probe.as_dict(), "models": app_state.models()}
        app_state.bus.publish("machine", **payload)
        return payload

    @app.get("/api/models")
    def get_models() -> Any:
        return {"models": app_state.models()}

    @app.post("/api/estimate")
    def post_estimate(payload: dict[str, Any]) -> Any:
        try:
            return app_state.estimate(payload)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    # -- splits ------------------------------------------------------------

    @app.post("/api/splits")
    def post_split(payload: dict[str, Any]) -> Any:
        try:
            job = app_state.splits.enqueue(
                str(payload["modelId"]),
                str(payload.get("format") or catalog.get(str(payload["modelId"])).default_format),
                from_gguf=bool(payload.get("fromGguf")),
            )
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        return job.as_dict()

    @app.post("/api/splits/{job_id}/{action}")
    def act_on_split(job_id: str, action: str) -> Any:
        actions = {
            "pause": app_state.splits.pause,
            "resume": app_state.splits.resume,
            "cancel": app_state.splits.cancel,
        }
        if action not in actions:
            raise HTTPException(400, f"Unknown split action {action!r}")
        if not actions[action](job_id):
            raise HTTPException(404, f"No split job {job_id!r} to {action}")
        return {"jobs": app_state.splits.jobs()}

    @app.post("/api/splits/pause-all")
    def pause_all_splits() -> Any:
        app_state.splits.pause_all()
        return {"jobs": app_state.splits.jobs()}

    # -- runs --------------------------------------------------------------

    @app.post("/api/runs")
    def post_run(payload: dict[str, Any]) -> Any:
        try:
            run = app_state.runs.start(payload)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return run.as_dict()

    @app.post("/api/runs/cancel")
    def cancel_run() -> Any:
        return {"cancelled": app_state.runs.cancel()}

    @app.get("/api/runs/current")
    def current_run() -> Any:
        return {"run": app_state.runs.current()}

    # -- outputs -----------------------------------------------------------

    @app.get("/api/outputs")
    def get_outputs() -> Any:
        return {"outputs": app_state.store.snapshot().get("outputs", [])}

    @app.get("/api/outputs/{output_id}/file")
    def get_output_file(output_id: str) -> Any:
        entry = app_state.store.get_output(output_id)
        name = (entry or {}).get("file")
        if not entry or not name:
            raise HTTPException(404, "No file for that output")
        path = app_state.store.outputs_dir() / str(name)
        if not path.is_file():
            raise HTTPException(404, f"{path.name} is no longer on disk")
        return FileResponse(path)

    @app.get("/api/outputs/{output_id}/report")
    def get_output_report(output_id: str) -> Any:
        entry = app_state.store.get_output(output_id)
        if not entry:
            raise HTTPException(404, "No such output")
        body = json.dumps(entry.get("report") or {}, indent=2)
        return JSONResponse(
            content=json.loads(body),
            headers={"content-disposition": f'attachment; filename="run-{output_id}.json"'},
        )

    @app.delete("/api/outputs/{output_id}")
    def delete_output(output_id: str) -> Any:
        entry = app_state.store.get_output(output_id)
        if entry and entry.get("file"):
            (app_state.store.outputs_dir() / str(entry["file"])).unlink(missing_ok=True)
        if not app_state.store.remove_output(output_id):
            raise HTTPException(404, "No such output")
        outputs = app_state.store.snapshot().get("outputs", [])
        app_state.bus.publish("outputs", outputs=outputs)
        return {"outputs": outputs}

    @app.post("/api/outputs/{output_id}/reproduced")
    def mark_reproduced(output_id: str) -> Any:
        entry = app_state.store.get_output(output_id)
        if not entry:
            raise HTTPException(404, "No such output")
        updated = app_state.store.update_output(
            output_id, {"reproducedCount": int(entry.get("reproducedCount") or 0) + 1}
        )
        app_state.bus.publish("outputs", outputs=app_state.store.snapshot().get("outputs", []))
        return updated or {}

    # -- cache -------------------------------------------------------------

    @app.post("/api/cache/remove")
    def remove_cache(payload: dict[str, Any]) -> Any:
        target = Path(str(payload.get("cacheDir") or ""))
        root = Path(machine.cache_summary()["path"])  # type: ignore[arg-type]
        if not target.is_dir() or root not in target.parents:
            raise HTTPException(400, "That path is not a shard cache under the AirCanvas root")
        shutil.rmtree(target)
        summary = machine.cache_summary()
        app_state.bus.publish("cache", cache=summary, models=app_state.models())
        return summary

    # -- events ------------------------------------------------------------

    @app.get("/api/events")
    async def events() -> Any:
        async def stream() -> Any:
            with app_state.bus.subscribe() as subscriber:
                yield _sse("state", app_state.state())
                while True:
                    event = await asyncio.to_thread(subscriber.get, KEEPALIVE_SECONDS)
                    if event is None:
                        yield ": keep-alive\n\n"
                        continue
                    yield _sse(event["type"], event)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-store", "x-accel-buffering": "no"},
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _sse(kind: str, payload: dict[str, Any]) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8760,
    open_browser: bool = True,
    demo: bool = False,
    log_level: str = "warning",
) -> None:
    """Run the Studio until interrupted."""
    import uvicorn

    studio = Studio(demo=demo)
    if demo:
        from aircanvas.studio import demo as demo_mod

        demo_mod.install(studio)

    app = create_app(studio)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    print(f"AirCanvas Studio on {url}")
    print(f"  shard cache  {studio.probe().cache_path}")
    if demo:
        print("  demo mode    replaying measured runs from docs/BENCHMARKS.md; nothing is loaded")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level=log_level)
