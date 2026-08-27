"""`aircanvas studio --demo`: the twelve screens without a 6 GB GPU.

Reviewing a UI whose subject is a 7-minute generation should not require a
7-minute generation, a 10 GB download, or the specific laptop the benchmarks
were run on. Demo mode stands in for all three — and does it without inventing
a single number:

- The machine is the box docs/BENCHMARKS.md was measured on (RTX 4050 Laptop,
  6.4 GB VRAM, 16.9 GB RAM at ~70% committed, 1.62 GB/s NVMe), declared here
  as data rather than probed.
- Every verdict, residency plan and per-step disk time is produced by the real
  budget solver against that profile. Nothing is a stored answer, which is
  exactly the property the setup screen is claiming.
- Run and split timings replay measured values (21.1 s/step for Qwen-Image at
  1024², 427.4 s total, 204 GB streamed, 97.5% prefetched) on a compressed
  clock, so a demo run takes seconds.

Two things demo mode does NOT do: produce artwork, and pretend to be real. No
image is rendered — every image slot shows its empty state, because a
stand-in picture is the one lie a "measured, not promised" UI cannot tell —
and every screen carries a demo marker in the status cluster.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aircanvas.streaming.residency import solve
from aircanvas.studio import catalog
from aircanvas.studio.jobs import Cancelled, RunManager, RunState, SplitJob, SplitManager
from aircanvas.studio.machine import InstalledModel, MachineProbe
from aircanvas.studio.store import Store

logger = logging.getLogger(__name__)

GB = 1_000_000_000

#: Wall-clock compression for replayed runs. A 427 s generation plays in ~11 s.
SPEED = 40.0

#: Pacing for a replayed split, in replayed (pre-compression) seconds.
#:
#: Unlike the run figures, these two are NOT measurements — the repo has no
#: recorded download or split rate, and the real screen is driven by the real
#: splitter's per-block callback and the real Hub cache on disk. They exist so
#: the split screen advances at a legible pace, and they are conservative
#: stand-ins: a plain home connection, and the cost of reading, dequantizing,
#: requantizing and writing one block.
DEMO_NETWORK_BYTES_S = 40e6
DEMO_SPLIT_BYTES_S = 50e6

#: The machine every number in docs/BENCHMARKS.md was measured on.
DEV_BOX = MachineProbe(
    gpu="RTX 4050 Laptop",
    device="cuda",
    vram_free_bytes=int(5.1 * GB),
    vram_total_bytes=int(6.4 * GB),
    ram_free_bytes=int(5.07 * GB),  # 16.9 GB at the box's chronic ~70% committed
    ram_total_bytes=int(16.9 * GB),
    disk_bw_bytes_s=1.62e9,
    disk_probed=True,
    disk_probed_at=time.time() - 2 * 3600,
    cache_path=r"C:\Users\kyle\.cache\huggingface\aircanvas",
    cache_free_bytes=318 * GB,
)

#: Prompts from the handoff's gallery table. They are evidence of use, not a
#: showroom — including a queued one that has not run yet.
GALLERY: tuple[dict[str, Any], ...] = (
    {
        "prompt": "a watercolor fox reading a newspaper on a rainy tram, soft morning light",
        "modelId": "qwen-image",
        "seed": 84421,
        "steps": 20,
        "durationSeconds": 427.4,
        "reproducedCount": 3,
        "loras": [
            {"id": "watercolour-ink-v3", "scale": 0.85, "kind": "peft", "sizeBytes": 42_000_000},
            {
                "id": "tram-lighting-warm-v2.safetensors",
                "scale": 0.40,
                "kind": "kohya",
                "sizeBytes": 18_000_000,
            },
        ],
    },
    {
        "prompt": "brass orrery on a chart table, single window light",
        "modelId": "flux1-schnell",
        "seed": 11902,
        "steps": 4,
        "durationSeconds": 129.0,
    },
    {
        "prompt": "cross-section of a seed, botanical plate",
        "modelId": "flux2-klein-9b",
        "seed": 30117,
        "steps": 20,
        "durationSeconds": 88.0,
    },
    {
        "prompt": "a paper boat crossing a puddle at dusk, camera drifting left",
        "modelId": "wan21-t2v-14b",
        "seed": 7130,
        "steps": 20,
        "durationSeconds": 38_160.0,
    },
    {
        "prompt": "lighthouse keeper's ledger, ink and rain",
        "modelId": "qwen-image",
        "seed": 55810,
        "steps": 20,
        "durationSeconds": 441.0,
    },
    {
        "prompt": "empty tram at 5 a.m., long exposure",
        "modelId": "qwen-image",
        "seed": 84422,
        "steps": 20,
        "durationSeconds": 430.0,
    },
    {
        "prompt": "curtain lifting in a hot room, handheld",
        "modelId": "wan21-t2v-1.3b",
        "seed": 71204,
        "steps": 20,
        "durationSeconds": 1046.0,
    },
    {
        "prompt": "hand-drawn map of a river delta, faded ink",
        "modelId": "flux1-schnell",
        "seed": 44018,
        "steps": 4,
        "durationSeconds": 131.0,
    },
    {
        "prompt": "worn leather satchel, studio backdrop",
        "modelId": "sd35-large",
        "seed": 90233,
        "steps": 28,
        "durationSeconds": 96.0,
    },
    {
        "prompt": "the same tram, but at noon and completely empty, harsh light",
        "modelId": "qwen-image",
        "seed": None,
        "steps": 20,
        "state": "queued",
    },
)


#: The three shard caches the benchmark machine actually held: the sizes are
#: the measured on-disk figures in docs/BENCHMARKS.md, not round numbers.
INSTALLED = (("wan21-t2v-14b", "fp8"), ("flux1-schnell", "fp8"), ("qwen-image", "nf4"))


def installed_models() -> list[InstalledModel]:
    """What the demo machine has already split, as `machine.installed_models`."""
    out: list[InstalledModel] = []
    for model_id, fmt in INSTALLED:
        model = catalog.get(model_id)
        manifest = model.manifest(fmt)
        out.append(
            InstalledModel(
                cache_dir=f"{DEV_BOX.cache_path}\\{model.repo_id.replace('/', '--')}--{fmt}",
                source=model.repo_id,
                model_class="SyntheticTransformer2DModel",
                adapter="generic",
                compression=None if fmt == "bf16" else fmt,  # type: ignore[arg-type]
                compute_dtype="bfloat16",
                subfolder=model.subfolder,
                n_blocks=model.n_blocks,
                disk_bytes=model.disk_bytes(fmt),
                largest_block_bytes=max(b.n_bytes for b in manifest.blocks),
                complete=True,
                from_gguf=None,
                modified_at=time.time() - 86_400,
            )
        )
    return out


def plan_for(model_id: str, fmt: str, *, steps: int | None = None) -> Any:
    """The real residency plan for `model_id` on the benchmark machine."""
    model = catalog.get(model_id)
    return solve(model.manifest(fmt), DEV_BOX.hardware(), model.workload(steps=steps))


def report_for(model_id: str, fmt: str, *, steps: int, seconds: float) -> dict[str, Any]:
    """A run report built from the measured total and the solved plan.

    The phase split follows the measurement (encode and decode are seconds, a
    denoise is minutes); the IO figures follow from the plan's own
    `step_read_bytes`, so 20 Qwen steps report the 204 GB the benchmark
    recorded because that is what 10.2 GB per step over 20 steps is.
    """
    model = catalog.get(model_id)
    plan = plan_for(model_id, fmt, steps=steps)
    encode_s, decode_s = 2.4, 3.4
    denoise_s = max(0.0, seconds - encode_s - decode_s)
    loads = plan.streamed_blocks * steps
    prefetched = round(loads * 0.975)
    return {
        "model": model.repo_id,
        "model_class": "SyntheticTransformer2DModel",
        "adapter": "generic",
        "compression": None if fmt == "bf16" else fmt,
        "compute_dtype": "bfloat16",
        "device": "cuda",
        "phases": {
            "encode_s": encode_s,
            "denoise_s": round(denoise_s, 1),
            "decode_s": decode_s,
            "total_s": round(seconds, 1),
            "steps": steps,
            "encode_cached": False,
            "engine": {
                "block_loads": loads,
                "bytes_loaded": plan.step_read_bytes * steps,
                "prefetch_hits": prefetched,
                "sync_loads": loads - prefetched,
                "sync_load_s": 1.9,
                "prefetch_wait_s": 0.0,
                "resident_blocks": plan.resident_blocks,
                "resident_block_bytes": plan.resident_block_bytes,
                "slot_bytes": plan.slot_bytes,
                "pinned_bytes": plan.pinned_bytes,
            },
        },
        "plan": {
            "resident_blocks": plan.resident_blocks,
            "streamed_blocks": plan.streamed_blocks,
            "gpu_slots": plan.gpu_slots,
            "ring_depth": plan.ring_depth,
            "lookahead": plan.lookahead,
            "slot_bytes": plan.slot_bytes,
            "pinned_bytes": plan.pinned_bytes,
            "activation_bytes": plan.activation_bytes,
            "resident_block_bytes": plan.resident_block_bytes,
            "resident_shard_bytes": plan.resident_shard_bytes,
            "ram_cache_bytes": plan.ram_cache_bytes,
            "step_read_bytes": plan.step_read_bytes,
            "step_read_seconds": plan.step_read_seconds,
            "vram_budget_bytes": plan.vram_budget_bytes,
            "vram_planned_bytes": plan.vram_planned_bytes,
            "warnings": list(plan.warnings),
        },
        "hardware": {
            "device": "cuda",
            "gpu": DEV_BOX.gpu,
            "vram_free_bytes": DEV_BOX.vram_free_bytes,
            "vram_total_bytes": DEV_BOX.vram_total_bytes,
            "ram_free_bytes": DEV_BOX.ram_free_bytes,
            "ram_total_bytes": DEV_BOX.ram_total_bytes,
            "disk_bw_bytes_s": DEV_BOX.disk_bw_bytes_s,
        },
    }


def seed(store: Store) -> None:
    """Fill an empty store with the sample desk and gallery."""
    state = store.snapshot()
    if state.get("outputs"):
        return
    qwen = catalog.get("qwen-image")
    store.patch(
        "desk",
        {
            "modelId": qwen.id,
            "format": qwen.default_format,
            "prompt": GALLERY[0]["prompt"],
            "steps": 20,
            "seed": 84421,
            "guidance": 3.5,
            "width": 1024,
            "height": 1024,
            "loras": GALLERY[0]["loras"],
        },
    )
    store.set("seenWelcome", True)
    store.set("mode", "pro")
    for sample in reversed(GALLERY):
        model = catalog.get(str(sample["modelId"]))
        steps = int(sample.get("steps") or model.steps)
        seconds = float(sample.get("durationSeconds") or 0.0)
        queued = sample.get("state") == "queued"
        store.add_output(
            {
                "prompt": sample["prompt"],
                "modelId": model.id,
                "modelName": model.name,
                "format": model.default_format,
                "seed": sample.get("seed"),
                "steps": steps,
                "guidance": model.guidance,
                "width": model.width,
                "height": model.height,
                "frames": model.frames,
                "kind": model.kind,
                "file": None,
                "loras": sample.get("loras") or [],
                "durationSeconds": seconds or None,
                "secondsPerStep": round(seconds / steps, 2) if seconds else None,
                "reproducedCount": int(sample.get("reproducedCount") or 0),
                "state": "queued" if queued else "done",
                "report": (
                    None
                    if queued
                    else report_for(model.id, model.default_format, steps=steps, seconds=seconds)
                ),
            }
        )


class DemoSplitManager(SplitManager):
    """Replays a split at the rate this disk could actually write it."""

    def _run(self, job: SplitJob) -> None:
        model = catalog.get(job.model_id)
        per_block = model.disk_bytes(job.fmt) / max(1, model.n_blocks)
        seconds_per_block = per_block / DEMO_SPLIT_BYTES_S
        job.cache_dir = f"{DEV_BOX.cache_path}\\{model.repo_id.replace('/', '--')}"
        job.started_at = time.time()

        job.state = "downloading"
        self._publish(job)
        expected = job.bytes_expected
        chunks = 40
        for i in range(1, chunks + 1):
            if job.id in self._stop:
                raise Cancelled(job.id)
            job.bytes_downloaded = int(expected * i / chunks)
            self._publish(job)
            time.sleep((expected / chunks) / DEMO_NETWORK_BYTES_S / SPEED)

        job.state = "splitting"
        for block in range(1, model.n_blocks + 1):
            if job.id in self._stop:
                raise Cancelled(job.id)
            time.sleep(seconds_per_block / SPEED)
            job.blocks_done = block
            # Stamp REPLAYED time, not wall time, so the blocks/s and eta the
            # screen derives are the ones a real split would show rather than
            # the compressed-clock ones.
            job.block_times.append(time.perf_counter() * SPEED)
            job.bytes_written = int(per_block * block)
            self._publish(job)
        job.state = "done"
        job.finished_at = time.time()
        self._publish(job)


class DemoRunManager(RunManager):
    """Replays a measured generation on a compressed clock."""

    def _work(self, run: RunState, params: dict[str, Any]) -> None:
        try:
            self._replay(run, params)
        except Cancelled:
            run.status = "cancelled"
            run.finished_at = time.time()
        finally:
            with self._lock:
                self._last = run
                self._run = None
            self._publish(run)

    def _replay(self, run: RunState, params: dict[str, Any]) -> None:
        model = catalog.get(run.model_id)
        plan = plan_for(run.model_id, run.fmt, steps=run.steps)
        measured = model.measured.seconds_per_step if model.measured else plan.step_read_seconds
        run.blocks = plan.streamed_blocks or model.n_blocks
        manifest = model.manifest(run.fmt)
        run.block_bytes = [b.n_bytes for b in manifest.blocks[plan.resident_blocks :]]
        run.resident_blocks = plan.resident_blocks
        run.ring_blocks = plan.ring_depth
        run.pinned_bytes = plan.pinned_bytes
        run.vram_total_bytes = DEV_BOX.vram_total_bytes
        run.plan = {
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

        self.phase_started("encode")
        self._sleep(2.4)
        self.phase_finished("encode", 2.4)

        self.phase_started("denoise")
        per_block = measured / max(1, run.blocks)
        for step in range(1, run.steps + 1):
            self.step(step, run.steps)
            for block in range(1, run.blocks + 1):
                self._sleep(per_block)
                run.block = block
                run.bytes_streamed += plan.step_read_bytes // max(1, run.blocks)
                run.prefetch_pct = 97.5
                run.queue_depth = min(plan.ring_depth, run.blocks - block)
                run.disk_read_bytes_s = DEV_BOX.disk_bw_bytes_s
                run.vram_in_use_bytes = plan.vram_planned_bytes
                self._publish(run)
        self.phase_finished("denoise", measured * run.steps)

        self.phase_started("decode")
        self._sleep(3.4)
        self.phase_finished("decode", 3.4)

        total = 2.4 + measured * run.steps + 3.4
        run.status = "done"
        run.finished_at = time.time()
        run.report = report_for(run.model_id, run.fmt, steps=run.steps, seconds=total)
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
                "kind": run.kind,
                "file": None,
                "loras": list(params.get("loras") or []),
                "durationSeconds": round(total, 1),
                "secondsPerStep": round(measured, 2),
                "state": "done",
                "report": run.report,
            }
        )
        run.output_id = str(entry["id"])
        self._publish(run)
        self._bus.publish("outputs", outputs=self._store.snapshot().get("outputs", []))

    def _sleep(self, seconds: float) -> None:
        """Sleep `seconds` of replayed time, in slices, watching for cancel."""
        deadline = time.perf_counter() + seconds / SPEED
        while time.perf_counter() < deadline:
            if self._cancel.is_set():
                raise Cancelled("cancelled by the user")
            time.sleep(min(0.05, max(0.0, deadline - time.perf_counter())))


def install(studio: Any) -> None:
    """Point a `Studio` at the benchmark machine and the replaying managers."""
    studio.demo = True
    studio.probe_override = DEV_BOX
    studio.installed_override = installed_models()
    studio.splits = DemoSplitManager(studio.bus, studio.store)
    studio.runs = DemoRunManager(studio.bus, studio.store)
    seed(studio.store)
    logger.info("Demo mode: %s profile, measured runs replayed at %.0fx", DEV_BOX.gpu, SPEED)
