"""Splits and generations, run off the request thread with live telemetry.

Two managers, one shape: a job owns a worker thread, publishes an immutable
snapshot of itself on every meaningful change, and is cancellable at a safe
boundary.

`SplitManager` runs one split at a time (they are disk-bound; two would just
halve each other) and queues the rest. Cancellation raises out of the
splitter's per-block progress callback, which stops it at a block boundary —
every shard already written keeps its `.done` marker, so "pause" and "resume"
are the same operation the splitter already supports.

`RunManager` runs one generation at a time. Phase and step events arrive
through `runtime.progress.RunObserver`; block counters are polled from
`AirPipeline.live_stats()` at 5 Hz, because a callback per block load would
cost more than the telemetry is worth. Cancellation raises from the diffusers
step callback, unwinding the denoise loop; the orchestrator's own `finally`
closes the engines, so the shard cache is untouched — which is exactly what
the Cancel copy promises.

Nothing in here invents a number. Where a value is not available (no CUDA, a
pipeline that takes no step callback, a download whose size the Hub did not
report) the field is absent and the UI says so rather than showing a plausible
default.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aircanvas.config import Compression
from aircanvas.studio import catalog, machine
from aircanvas.studio.events import EventBus
from aircanvas.studio.store import Store, new_id

logger = logging.getLogger(__name__)

#: Telemetry poll interval. The handoff caps the ticker at 10 Hz and the GB /
#: GB-s readouts at 2 Hz so digits stay readable; 5 Hz feeds both comfortably.
POLL_SECONDS = 0.2


class Cancelled(Exception):
    """Raised inside a worker to unwind it at a safe boundary."""


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


@dataclass
class SplitJob:
    """One model on its way from the Hub to a streamable shard cache."""

    id: str
    model_id: str
    name: str
    fmt: str
    from_gguf: bool
    repo_id: str
    blocks_total: int
    state: str = "queued"  # queued|downloading|splitting|done|paused|failed
    blocks_done: int = 0
    bytes_downloaded: int = 0
    bytes_expected: int = 0
    bytes_written: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    block_times: list[float] = field(default_factory=list)
    cache_dir: str | None = None
    error: str | None = None

    @property
    def blocks_per_second(self) -> float | None:
        """Measured from the last few blocks, or None before there are any."""
        recent = self.block_times[-8:]
        if len(recent) < 2:
            return None
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else None

    @property
    def eta_seconds(self) -> float | None:
        rate = self.blocks_per_second
        remaining = self.blocks_total - self.blocks_done
        return remaining / rate if rate and remaining > 0 else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "modelId": self.model_id,
            "name": self.name,
            "format": self.fmt,
            "fromGguf": self.from_gguf,
            "repoId": self.repo_id,
            "state": self.state,
            "blocksDone": self.blocks_done,
            "blocksTotal": self.blocks_total,
            "bytesDownloaded": self.bytes_downloaded,
            "bytesExpected": self.bytes_expected,
            "bytesWritten": self.bytes_written,
            "blocksPerSecond": self.blocks_per_second,
            "etaSeconds": self.eta_seconds,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "cacheDir": self.cache_dir,
            "error": self.error,
        }


class SplitManager:
    """A serial queue of splits, with per-block progress and resume."""

    def __init__(self, bus: EventBus, store: Store) -> None:
        self._bus = bus
        self._store = store
        self._lock = threading.RLock()
        self._jobs: dict[str, SplitJob] = {}
        self._order: list[str] = []
        self._stop: set[str] = set()
        self._worker: threading.Thread | None = None

    # -- queue -------------------------------------------------------------

    def enqueue(self, model_id: str, fmt: str, *, from_gguf: bool = False) -> SplitJob:
        model = catalog.get(model_id)
        job = SplitJob(
            id=new_id(),
            model_id=model_id,
            name=model.name,
            fmt=fmt,
            from_gguf=from_gguf and model.gguf is not None,
            repo_id=model.repo_id,
            blocks_total=model.n_blocks,
            bytes_expected=model.download_bytes(from_gguf=from_gguf and model.gguf is not None),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._publish(job)
        self._ensure_worker()
        return job

    def jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._jobs[i].as_dict() for i in self._order if i in self._jobs]

    def pause(self, job_id: str) -> bool:
        """Stop a job at its next block boundary; completed shards survive."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in ("done", "failed"):
                return False
            self._stop.add(job_id)
            if job.state == "queued":
                job.state = "paused"
        self._publish(self._jobs[job_id])
        return True

    def pause_all(self) -> None:
        for job_id in list(self._jobs):
            self.pause(job_id)

    def resume(self, job_id: str) -> bool:
        """Undo a pause, including one the worker has not noticed yet.

        Pausing an in-flight job only sets a flag: the split stops at its next
        block boundary, so for a moment the job is still `splitting` with a
        pause pending. Resuming in that window has to clear the request rather
        than fail, or a quick pause-then-resume leaves the job stopped with the
        UI insisting it is running. `_drain` closes the other half of the race.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state == "done":
                return False
            self._stop.discard(job_id)
            if job.state in ("paused", "failed"):
                job.state = "queued"
            job.error = None
        self._publish(self._jobs[job_id])
        self._ensure_worker()
        return True

    def cancel(self, job_id: str) -> bool:
        paused = self.pause(job_id)
        with self._lock:
            self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)
        self._bus.publish("splits", jobs=self.jobs())
        return paused

    # -- worker ------------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._drain, name="aircanvas-split", daemon=True)
            self._worker.start()

    def _next(self) -> SplitJob | None:
        with self._lock:
            for job_id in self._order:
                job = self._jobs.get(job_id)
                if job is not None and job.state == "queued":
                    return job
        return None

    def _drain(self) -> None:
        while True:
            job = self._next()
            if job is None:
                return
            try:
                self._run(job)
            except Cancelled:
                # A resume that arrived while the worker was unwinding cleared
                # the flag: re-queue instead of parking a job the user has
                # already asked to continue.
                job.state = "paused" if job.id in self._stop else "queued"
                self._publish(job)
            except Exception as e:  # noqa: BLE001 — a failed split must not kill the queue
                logger.exception("Split failed for %s", job.model_id)
                job.state = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.finished_at = time.time()
                self._publish(job)

    def _run(self, job: SplitJob) -> None:
        from aircanvas.sharding.splitter import shard_cache_dir, split_model

        model = catalog.get(job.model_id)
        compression = _compression(job.fmt)
        gguf_file = model.gguf.file if job.from_gguf and model.gguf is not None else None
        cache_dir = shard_cache_dir(
            model.repo_id, model.subfolder, compression, "bfloat16", gguf_file
        )
        job.cache_dir = str(cache_dir)
        job.started_at = time.time()
        job.state = "downloading"
        self._publish(job)

        # A GGUF split fetches from the quant's repo, not the model's, so that
        # is the cache directory whose bytes the download watcher must count.
        fetch_repo = gguf_file.split(":")[0] if gguf_file else model.repo_id
        watcher = threading.Thread(
            target=self._watch_download,
            args=(job, fetch_repo),
            name="aircanvas-split-dl",
            daemon=True,
        )
        watcher.start()

        def progress(done: int, total: int, _block: str) -> None:
            if job.id in self._stop:
                raise Cancelled(job.id)
            job.state = "splitting"
            job.blocks_done = done
            job.blocks_total = total
            job.block_times.append(time.perf_counter())
            job.bytes_written = _dir_bytes(cache_dir)
            self._publish(job)

        split_model(
            model.repo_id,
            cache_dir=cache_dir,
            compression=compression,
            subfolder=model.subfolder,
            compute_dtype="bfloat16",
            hf_token=machine.hf_token(),
            gguf_file=gguf_file,
            progress=progress,
        )
        job.state = "done"
        job.finished_at = time.time()
        job.bytes_written = _dir_bytes(cache_dir)
        self._publish(job)
        self._bus.publish("installed", models=[m.as_dict() for m in machine.installed_models()])

    def _watch_download(self, job: SplitJob, repo_id: str) -> None:
        """Report real fetched bytes while the Hub download runs.

        huggingface_hub does not hand us a byte callback, but it does write the
        blobs — including `.incomplete` partials — into a cache we can stat.
        That is a measurement, not an estimate, which is the only kind of
        progress this UI is allowed to show.
        """
        while job.state == "downloading":
            fetched, _partial = machine.repo_cache_bytes(repo_id)
            if fetched != job.bytes_downloaded:
                job.bytes_downloaded = fetched
                self._publish(job)
            time.sleep(1.0)

    def _publish(self, job: SplitJob) -> None:
        self._bus.publish("split", job=job.as_dict(), jobs=self.jobs())


def _dir_bytes(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return 0
    for entry in path.iterdir():
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:  # pragma: no cover - file vanished mid-scan
            continue
    return total


# --------------------------------------------------------------------------
# Generations
# --------------------------------------------------------------------------


@dataclass
class RunState:
    """Everything the generating screens display, and nothing they don't."""

    id: str
    model_id: str
    model_name: str
    fmt: str
    prompt: str
    seed: int | None
    steps: int
    width: int
    height: int
    frames: int
    guidance: float
    lora_count: int
    kind: str = "image"
    status: str = "starting"  # starting|encode|denoise|decode|done|cancelled|failed
    phase: str = "encode"
    phase_seconds: dict[str, float] = field(default_factory=dict)
    step: int = 0
    step_started_at: float = 0.0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    block: int = 0
    blocks: int = 0
    bytes_streamed: int = 0
    prefetch_pct: float | None = None
    stalls: int = 0
    stall_seconds: float = 0.0
    queue_depth: int = 0
    ring_blocks: int = 0
    pinned_bytes: int = 0
    vram_in_use_bytes: int | None = None
    vram_total_bytes: int = 0
    disk_read_bytes_s: float | None = None
    resident_blocks: int = 0
    #: On-disk bytes per streamed block, so the ticker's sparkline can draw the
    #: model's real shard profile instead of a decorative waveform.
    block_bytes: list[int] = field(default_factory=list)
    preview_step: int | None = None
    preview_url: str | None = None
    output_id: str | None = None
    error: str | None = None
    plan: dict[str, Any] | None = None
    report: dict[str, Any] | None = None

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    @property
    def step_elapsed(self) -> float:
        return time.time() - self.step_started_at if self.step_started_at else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "modelId": self.model_id,
            "modelName": self.model_name,
            "format": self.fmt,
            "kind": self.kind,
            "prompt": self.prompt,
            "seed": self.seed,
            "steps": self.steps,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "guidance": self.guidance,
            "loraCount": self.lora_count,
            "status": self.status,
            "phase": self.phase,
            "phaseSeconds": dict(self.phase_seconds),
            "step": self.step,
            "stepElapsed": round(self.step_elapsed, 2),
            "elapsed": round(self.elapsed, 2),
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "block": self.block,
            "blocks": self.blocks,
            "bytesStreamed": self.bytes_streamed,
            "prefetchPct": self.prefetch_pct,
            "stalls": self.stalls,
            "stallSeconds": round(self.stall_seconds, 2),
            "queueDepth": self.queue_depth,
            "ringBlocks": self.ring_blocks,
            "pinnedBytes": self.pinned_bytes,
            "residentBlocks": self.resident_blocks,
            "blockBytes": list(self.block_bytes),
            "vramInUseBytes": self.vram_in_use_bytes,
            "vramTotalBytes": self.vram_total_bytes,
            "diskReadBytesPerSecond": self.disk_read_bytes_s,
            "previewStep": self.preview_step,
            "previewUrl": self.preview_url,
            "outputId": self.output_id,
            "error": self.error,
            "plan": self.plan,
            "report": self.report,
        }


class RunManager:
    """One generation at a time, with the machine shown while it happens."""

    def __init__(self, bus: EventBus, store: Store) -> None:
        self._bus = bus
        self._store = store
        self._lock = threading.RLock()
        self._run: RunState | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: RunState | None = None

    # -- api ---------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def current(self) -> dict[str, Any] | None:
        run = self._run or self._last
        return run.as_dict() if run else None

    def start(self, params: Mapping[str, Any]) -> RunState:
        with self._lock:
            if self.busy:
                raise RuntimeError("A generation is already running")
            model = catalog.get(str(params["modelId"]))
            fmt = str(params.get("format") or model.default_format)
            run = RunState(
                id=new_id(),
                model_id=model.id,
                model_name=model.name,
                fmt=fmt,
                kind=model.kind,
                prompt=str(params.get("prompt") or ""),
                seed=_as_int(params.get("seed")),
                steps=int(params.get("steps") or model.steps),
                width=int(params.get("width") or model.width),
                height=int(params.get("height") or model.height),
                frames=int(params.get("frames") or model.frames),
                guidance=float(params.get("guidance", model.guidance)),
                lora_count=len(list(params.get("loras") or [])),
                blocks=model.n_blocks,
            )
            self._run = run
            self._cancel.clear()
            self._thread = threading.Thread(
                target=self._work,
                args=(run, dict(params)),
                name="aircanvas-run",
                daemon=True,
            )
            self._thread.start()
        self._publish(run)
        return run

    def cancel(self) -> bool:
        if not self.busy:
            return False
        self._cancel.set()
        return True

    # -- observer (runtime.progress.RunObserver) ---------------------------

    def phase_started(self, phase: str) -> None:
        run = self._run
        if run is None:
            return
        run.phase = phase
        run.status = phase
        if phase == "denoise":
            run.step_started_at = time.time()
        self._publish(run)

    def phase_finished(self, phase: str, seconds: float) -> None:
        run = self._run
        if run is None:
            return
        run.phase_seconds[phase] = seconds
        self._publish(run)

    def step(self, index: int, total: int) -> None:
        run = self._run
        if run is None:
            return
        run.step = index
        run.steps = total or run.steps
        run.step_started_at = time.time()
        self._publish(run)

    # -- worker ------------------------------------------------------------

    def _work(self, run: RunState, params: dict[str, Any]) -> None:
        pipe = None
        sampler: threading.Thread | None = None
        try:
            pipe = self._build(run, params)
            sampler = threading.Thread(
                target=self._sample, args=(run, pipe), name="aircanvas-telemetry", daemon=True
            )
            sampler.start()
            result = self._generate(run, pipe, params)
            self._finish(run, pipe, result, params)
        except Cancelled:
            run.status = "cancelled"
            run.finished_at = time.time()
            self._publish(run)
        except Exception as e:  # noqa: BLE001 — surface it, never crash the server
            logger.exception("Generation failed")
            run.status = "failed"
            run.error = _error_text(e)
            run.finished_at = time.time()
            self._publish(run)
        finally:
            with self._lock:
                self._last = run
                self._run = None
            if sampler is not None:
                sampler.join(timeout=1.0)
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:  # noqa: BLE001 - teardown is best-effort
                    logger.debug("Pipeline close failed", exc_info=True)
            self._publish(run)

    def _build(self, run: RunState, params: dict[str, Any]) -> Any:
        from aircanvas.api import AirPipeline

        model = catalog.get(run.model_id)
        settings = self._store.snapshot().get("settings", {})
        caps = params.get("caps") or {}
        compression = _compression(run.fmt)
        pipe = AirPipeline.from_pretrained(
            model.repo_id,
            compression=compression,
            subfolder=model.subfolder,
            # A locally-split model names its own cache: its source path may no
            # longer exist (split-once-delete-original is the whole point), so
            # the default location derived from the source must not be trusted.
            shard_cache=model.cache_dir,
            vram_budget=_as_int(caps.get("vramBytes")) or "auto",
            ram_budget=_as_int(caps.get("ramBytes")) or "auto",
            prefetch=bool(settings.get("backgroundPrefetch", True)),
            cache_embeddings=bool(settings.get("embeddingCache", True)),
            hf_token=machine.hf_token(),
            gguf_file=model.gguf.file if params.get("fromGguf") and model.gguf else None,
        )
        for lora in params.get("loras") or []:
            source = lora.get("source") or lora.get("id")
            if source:
                pipe.load_lora(str(source), scale=float(lora.get("scale", 1.0)))
        # The ticker counts what STREAMS. On a card big enough to hold the
        # whole model that is zero, and saying "block 0/57" would misreport a
        # run that is going perfectly — the screen says "all resident" instead.
        run.blocks = pipe.plan.streamed_blocks
        run.block_bytes = [b.n_bytes for b in pipe.manifest.blocks[pipe.plan.resident_blocks :]]
        run.resident_blocks = pipe.plan.resident_blocks
        run.ring_blocks = pipe.plan.ring_depth
        run.pinned_bytes = pipe.plan.pinned_bytes
        run.vram_total_bytes = pipe.hardware.vram_total_bytes
        run.plan = _plan_dict(pipe.plan)
        self._publish(run)
        return pipe

    def _generate(self, run: RunState, pipe: Any, params: dict[str, Any]) -> Any:
        model = catalog.get(run.model_id)
        kwargs: dict[str, Any] = {
            "num_inference_steps": run.steps,
            "height": run.height,
            "width": run.width,
            "callback_on_step_end": self._step_guard,
        }
        if run.guidance:
            kwargs["guidance_scale"] = run.guidance
        if model.video:
            kwargs["num_frames"] = run.frames
        if run.seed is not None:
            import torch

            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(run.seed)
        negative = params.get("negativePrompt")
        if negative:
            kwargs["negative_prompt"] = str(negative)
        return pipe(run.prompt, observer=self, **kwargs)

    def _step_guard(self, _pipe: Any, _index: int, _timestep: Any, kwargs: dict) -> dict:
        """Cancellation point. Chained by the orchestrator, so it can raise."""
        if self._cancel.is_set():
            raise Cancelled("cancelled by the user")
        return kwargs

    def _sample(self, run: RunState, pipe: Any) -> None:
        """Poll the engine's own counters while the denoise loop runs."""
        import torch

        last_bytes, last_at = 0, time.perf_counter()
        while self._run is run and run.status in ("starting", "encode", "denoise", "decode"):
            stats = pipe.live_stats()
            if stats:
                loads = int(stats.get("block_loads", 0))
                run.block = (loads - 1) % max(1, run.blocks) + 1 if loads else 0
                streamed = int(stats.get("bytes_loaded", 0))
                hits = float(stats.get("prefetch_hits", 0))
                run.prefetch_pct = round(100 * hits / loads, 1) if loads else None
                run.stalls = int(stats.get("sync_loads", 0))
                run.stall_seconds = float(stats.get("sync_load_s", 0.0))
                run.pinned_bytes = int(stats.get("pinned_bytes", run.pinned_bytes))
                now = time.perf_counter()
                if streamed > last_bytes and now > last_at:
                    run.disk_read_bytes_s = (streamed - last_bytes) / (now - last_at)
                    last_bytes, last_at = streamed, now
                run.bytes_streamed = streamed
                run.queue_depth = min(run.ring_blocks, max(0, run.blocks - run.block))
            if torch.cuda.is_available():
                run.vram_in_use_bytes = int(torch.cuda.memory_reserved())
            self._publish(run)
            time.sleep(POLL_SECONDS)

    def _finish(self, run: RunState, pipe: Any, result: Any, params: dict[str, Any]) -> None:
        report = pipe.report(as_dict=True)
        run.report = report
        run.status = "done"
        run.finished_at = time.time()
        stats = report.get("phases", {})
        total = float(stats.get("total_s") or run.elapsed)
        per_step = total / run.steps if run.steps else 0.0

        path, kind = _save_result(result, self._store.outputs_dir(), run.id)
        entry = self._store.add_output(
            {
                "id": run.id,
                "prompt": run.prompt,
                "modelId": run.model_id,
                "modelName": run.model_name,
                "format": run.fmt,
                "seed": run.seed,
                "steps": run.steps,
                "guidance": run.guidance,
                "width": run.width,
                "height": run.height,
                "frames": run.frames,
                "kind": kind,
                "file": path.name if path else None,
                "loras": list(params.get("loras") or []),
                "durationSeconds": round(total, 1),
                "secondsPerStep": round(per_step, 2),
                "state": "done",
                "report": report,
            }
        )
        run.output_id = str(entry["id"])
        self._publish(run)
        self._bus.publish("outputs", outputs=self._store.snapshot().get("outputs", []))

    def _publish(self, run: RunState) -> None:
        self._bus.publish("run", run=run.as_dict())


def _compression(fmt: str) -> Compression:
    """A UI format name as the splitter's `Compression`; bf16 means none."""
    if fmt in ("fp8", "nf4"):
        return fmt  # type: ignore[return-value]
    if fmt == "bf16":
        return None
    raise ValueError(f"Unknown shard format {fmt!r} (expected fp8, nf4 or bf16)")


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _plan_dict(plan: Any) -> dict[str, Any]:
    """The residency plan, as the Report screen's stacked bars need it."""
    return {
        "residentBlocks": plan.resident_blocks,
        "streamedBlocks": plan.streamed_blocks,
        "gpuSlots": plan.gpu_slots,
        "ringDepth": plan.ring_depth,
        "slotBytes": plan.slot_bytes,
        "pinnedBytes": plan.pinned_bytes,
        "activationBytes": plan.activation_bytes,
        "residentBlockBytes": plan.resident_block_bytes,
        "residentShardBytes": plan.resident_shard_bytes,
        "ramCacheBytes": plan.ram_cache_bytes,
        "stepReadBytes": plan.step_read_bytes,
        "stepReadSeconds": plan.step_read_seconds,
        "vramBudgetBytes": plan.vram_budget_bytes,
        "vramPlannedBytes": plan.vram_planned_bytes,
        "warnings": list(plan.warnings),
    }


def _error_text(exc: BaseException) -> str:
    """The message, plus the attempted plan when the solver refused.

    `InsufficientVRAMError` carries `ResidencyPlan.describe()` in its message
    (ARCHITECTURE.md §4) — the most useful thing a failed run can show, so it
    goes straight to the screen rather than being flattened to "failed".
    """
    text = str(exc).strip() or type(exc).__name__
    logger.debug("Run error detail:\n%s", "".join(traceback.format_exception(exc)))
    return text


def _save_result(result: Any, out_dir: Path, run_id: str) -> tuple[Path | None, str]:
    """Persist whatever the pipeline returned; (path, kind)."""
    frames = getattr(result, "frames", None)
    if frames:
        clip = frames[0]
        path = out_dir / f"{run_id}.mp4"
        try:
            from diffusers.utils import export_to_video

            export_to_video(clip, str(path), fps=16)
            return path, "video"
        except Exception as e:  # noqa: BLE001 — video export is an optional extra
            logger.warning("Could not write %s (%s); saving the first frame instead", path, e)
            still = out_dir / f"{run_id}.png"
            clip[0].save(still)
            return still, "image"
    images = getattr(result, "images", None)
    if images:
        path = out_dir / f"{run_id}.png"
        images[0].save(path)
        return path, "image"
    logger.warning("Pipeline returned neither images nor frames; nothing saved")
    return None, "image"
