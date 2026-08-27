"""The model table the setup screen is built from, and its verdicts.

Two rules shape this module.

**Verdicts are computed, never tabulated.** "Can this machine run Qwen-Image
at nf4?" is answered by `streaming.residency.solve` against a manifest shaped
like the real model (`sharding.manifest.synthetic_manifest`) — the same solver
that runs at generation time. Change the machine, the shard format or the free
VRAM and every chip moves, because nothing here is a stored answer.

**Numbers keep their provenance.** Every size and timing below is either
measured on the dev machine (docs/BENCHMARKS.md), published by the model's own
repo, or derived arithmetically from one of those — and says which. Nothing is
a plausible-looking guess: a UI whose whole personality is "measured, not
promised" cannot ship invented figures. Where the repo has not verified a
quantized source for a model, that model simply has no GGUF option rather than
a made-up download size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aircanvas.config import Compression
from aircanvas.sharding.manifest import Manifest, synthetic_manifest
from aircanvas.streaming.residency import InsufficientVRAMError, Workload, solve
from aircanvas.utils.hw import HardwareProfile

logger = logging.getLogger(__name__)

GB = 1_000_000_000
MB = 1_000_000

Kind = Literal["image", "video"]
Format = Literal["fp8", "nf4", "bf16"]

#: Slack above the minimum viable plan below which a model is only "tight".
#: Same constant as `aircanvas doctor` uses, for the same reason: the waterfall
#: spends every spare byte on resident blocks, so the CHOSEN plan always looks
#: full and only the minimum plan reveals real headroom.
COMFORT_MARGIN = 1 * GB


@dataclass(frozen=True)
class GgufSource:
    """A quantized checkpoint this repo has actually read end-to-end."""

    file: str  # "repo_id:filename", as `aircanvas split --gguf-file` takes it
    download_bytes: int
    quant: str  # "Q4_K_S", "Q8_0"
    verified: bool = False  # proven bitwise by benchmarks/verify_gguf_lora.py


@dataclass(frozen=True)
class Measured:
    """A real run on the dev box (RTX 4050 Laptop 6 GB), docs/BENCHMARKS.md."""

    seconds_per_step: float
    steps: int
    note: str = ""
    #: Streamed vs fully-resident step time, where both were measured.
    resident_ratio: float | None = None


@dataclass(frozen=True)
class SimpleFacing:
    """What Simple mode is allowed to say about a model.

    Simple copy never names a format, a bandwidth or a shard count (handoff
    "Copy rules"), so the plain-words quality note lives here rather than being
    derived from the compression in the UI.
    """

    headline: str  # "Best quality images"
    quality: str  # "Near-identical quality"
    tone: Literal["good", "warn", "plain"]
    unit: str = "per image"


@dataclass(frozen=True)
class StudioModel:
    """One row of the setup screen, with everything both modes need."""

    id: str
    name: str
    repo_id: str
    kind: Kind
    params: str  # display only: "20.4B"
    n_blocks: int
    dit_bytes: int  # the DiT in bf16 — what gets sharded
    largest_block_bytes: int  # sizes the GPU slot pool
    resident_bytes: int  # embedders/final proj, never streamed
    native_bytes: int  # whole checkpoint: "needs ~41 GB natively"
    hidden: int
    tokens: int  # default workload, drives the activation reserve
    steps: int
    on_disk: dict[str, int]  # format -> bytes after the split
    formats: tuple[str, ...] = ("fp8", "nf4", "bf16")
    default_format: str = "fp8"
    subfolder: str = "transformer"
    gguf: GgufSource | None = None
    gated: bool = False
    note: str = ""
    measured: Measured | None = None
    simple: SimpleFacing | None = None
    #: Default generation shape, so the Desk opens on something that fits.
    width: int = 1024
    height: int = 1024
    frames: int = 1
    fps: int = 16
    guidance: float = 3.5
    warnings: tuple[str, ...] = field(default=())
    #: Set for a model discovered on disk: the shard cache's own manifest.
    real_manifest: Manifest | None = None
    #: Local shard cache this row came from, if it was not in the table.
    cache_dir: str | None = None

    @property
    def video(self) -> bool:
        return self.kind == "video"

    @property
    def local(self) -> bool:
        return self.cache_dir is not None

    def disk_bytes(self, fmt: str) -> int:
        """Bytes the shard cache occupies in `fmt`.

        Measured or published where we have it; otherwise the DiT scaled by the
        format's documented payload share (README "Model compression"). fp8
        keeps norms, modulation and every 1-D param at bf16, so the real cut is
        ~0.6x rather than the 0.5x "fp8 is half of bf16" suggests.
        """
        if fmt in self.on_disk:
            return self.on_disk[fmt]
        return int(self.dit_bytes * {"fp8": 0.6, "nf4": 0.3, "bf16": 1.0}.get(fmt, 1.0))

    def download_bytes(self, *, from_gguf: bool) -> int:
        """What still has to come over the network before a split can start."""
        if from_gguf and self.gguf is not None:
            return self.gguf.download_bytes
        return self.native_bytes

    def manifest(self, fmt: str) -> Manifest:
        """A manifest to plan against.

        For a model already on disk this is its REAL manifest — actual
        per-block bytes, actual resident set — so the verdict and the estimate
        describe the shards that will be streamed rather than a model-shaped
        approximation of them. Only a model that has not been split yet needs
        the synthetic one.
        """
        if self.real_manifest is not None and self.real_manifest.compression == (
            None if fmt == "bf16" else fmt
        ):
            return self.real_manifest
        compression: Compression = None if fmt == "bf16" else fmt  # type: ignore[assignment]
        return synthetic_manifest(
            source=self.repo_id,
            n_blocks=self.n_blocks,
            largest_block_bytes=self.largest_block_bytes,
            dit_bytes=self.dit_bytes,
            resident_bytes=self.resident_bytes,
            compression=compression,
            compressed_ratio=self.disk_bytes(fmt) / self.dit_bytes,
        )

    def workload(self, *, steps: int | None = None) -> Workload:
        return Workload(steps=steps or self.steps, tokens=self.tokens, hidden=self.hidden)


# --------------------------------------------------------------------------
# The table.
#
# dit_bytes / largest_block_bytes / resident_bytes / hidden / tokens come from
# docs/RESEARCH.md §3-§4 (the same values `aircanvas doctor` plans against).
# on_disk sizes and `measured` timings come from docs/BENCHMARKS.md. Entries
# marked "derived" have no measurement yet and are computed from the published
# parameter count at 2 bytes/param; they are honest estimates and the UI shows
# them as such by never labelling an unmeasured model with a measured time.
# --------------------------------------------------------------------------

MODELS: tuple[StudioModel, ...] = (
    StudioModel(
        id="flux1-schnell",
        name="FLUX.1-schnell",
        repo_id="black-forest-labs/FLUX.1-schnell",
        kind="image",
        params="12B",
        n_blocks=57,
        dit_bytes=int(23.8 * GB),
        largest_block_bytes=680 * MB,
        resident_bytes=1 * GB,
        native_bytes=34 * GB,
        hidden=3072,
        tokens=4608,
        steps=4,
        on_disk={"fp8": int(11.9 * GB)},
        measured=Measured(32.3, 4, "1024^2 4-step: 129 s warm, 385 s cold"),
        simple=SimpleFacing("Fastest images", "Slightly softer detail", "warn"),
        note="4-step distilled: streaming overhead is visible",
        guidance=0.0,
    ),
    StudioModel(
        id="flux1-dev",
        name="FLUX.1-dev",
        repo_id="black-forest-labs/FLUX.1-dev",
        kind="image",
        params="12B",
        n_blocks=57,
        dit_bytes=int(23.8 * GB),
        largest_block_bytes=680 * MB,
        resident_bytes=1 * GB,
        native_bytes=34 * GB,
        hidden=3072,
        tokens=4608,
        steps=28,
        on_disk={"fp8": int(11.9 * GB)},
        gguf=GgufSource("city96/FLUX.1-dev-gguf:flux1-dev-Q4_K_S.gguf", int(6.9 * GB), "Q4_K_S"),
        gated=True,
        measured=Measured(14.1, 28, "~7 min/image at 28 steps"),
        simple=SimpleFacing("Detailed images", "Full quality", "plain"),
        note="gated repo: accept the licence on the Hub first",
    ),
    StudioModel(
        id="qwen-image",
        name="Qwen-Image",
        repo_id="Qwen/Qwen-Image",
        kind="image",
        params="20.4B",
        n_blocks=60,
        dit_bytes=int(41.0 * GB),
        largest_block_bytes=680 * MB,
        resident_bytes=1 * GB,
        native_bytes=41 * GB,
        hidden=3072,
        tokens=4608,
        steps=20,
        on_disk={"nf4": int(10.2 * GB)},
        default_format="nf4",
        measured=Measured(21.1, 20, "1024^2 20-step: 427.4 s; 50-step: 1034 s"),
        simple=SimpleFacing("Best quality images", "Near-identical quality", "good"),
        note="the 16.6 GB Qwen2.5-VL text encoder runs load-run-evict",
    ),
    # The two FLUX.2 klein sizes. Every figure below is read from the model's
    # own files: block counts and per-block bytes come from instantiating the
    # published transformer config on meta (which reproduces the checkpoint
    # size to the byte), native size is the sum of the repo's safetensors, and
    # the GGUF size is the file's own length on the Hub.
    StudioModel(
        id="flux2-klein-4b",
        name="FLUX.2-klein-4B",
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        kind="image",
        params="3.9B",
        n_blocks=25,  # 5 double-stream + 20 single-stream
        dit_bytes=7_751_109_744,
        largest_block_bytes=491 * MB,  # a double-stream block; singles are 245 MB
        resident_bytes=390 * MB,
        native_bytes=15_964_000_000,  # DiT 7.75 + Qwen3 TE 8.04 + VAE 0.17
        hidden=3072,  # 24 heads x 128
        tokens=4608,  # 1024^2 -> 4096 image + 512 text
        steps=20,
        on_disk={},
        gguf=GgufSource(
            "unsloth/FLUX.2-klein-4B-GGUF:flux-2-klein-4b-Q4_K_S.gguf",
            2_583_077_440,
            "Q4_K_S",
        ),
        simple=SimpleFacing("Balanced images", "Near-identical quality", "good"),
        note="Apache-2.0 and ungated; distilled, so no CFG and one forward per step",
    ),
    StudioModel(
        id="flux2-klein-9b",
        name="FLUX.2-klein-9B",
        repo_id="black-forest-labs/FLUX.2-klein-9B",
        kind="image",
        params="9.1B",
        n_blocks=32,  # 8 double-stream + 24 single-stream
        dit_bytes=18_157_185_168,
        largest_block_bytes=872 * MB,  # a double-stream block; singles are 436 MB
        resident_bytes=709 * MB,
        native_bytes=34_706_000_000,  # DiT 18.16 + Qwen3 TE 16.38 + VAE 0.17
        hidden=4096,  # 32 heads x 128
        tokens=4608,
        steps=20,
        on_disk={},
        gguf=GgufSource(
            "unsloth/FLUX.2-klein-9B-GGUF:flux-2-klein-9b-Q4_K_S.gguf",
            int(5.8 * GB),  # docs/BENCHMARKS.md, the size that harness downloads
            "Q4_K_S",
            verified=True,
        ),
        gated=True,
        simple=SimpleFacing("Best quality images", "Near-identical quality", "good"),
        note="Q4_K_S source verified bitwise by benchmarks/verify_gguf_lora.py",
    ),
    StudioModel(
        id="sd35-large",
        name="SD3.5 Large",
        repo_id="stabilityai/stable-diffusion-3.5-large",
        kind="image",
        params="8B",
        n_blocks=38,
        dit_bytes=int(16.5 * GB),
        largest_block_bytes=420 * MB,
        resident_bytes=1 * GB,
        native_bytes=int(20.0 * GB),
        hidden=2432,
        tokens=4608,
        steps=28,
        on_disk={"fp8": int(7.9 * GB)},
        gated=True,
        simple=SimpleFacing("Photographic images", "Full quality", "plain"),
        note="gated repo: accept the licence on the Hub first",
    ),
    StudioModel(
        id="wan21-t2v-1.3b",
        name="Wan 2.1 T2V",
        repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        kind="video",
        params="1.3B",
        # Derived: 1.3B x 2 bytes = 2.6 GB; the 1.5 GB fp8 cache below is
        # measured, and the Q8_0 GGUF is the size benchmarks/verify downloads.
        n_blocks=30,
        dit_bytes=int(2.6 * GB),
        largest_block_bytes=100 * MB,
        resident_bytes=400 * MB,
        native_bytes=8 * GB,
        hidden=1536,
        tokens=24_000,
        steps=20,
        on_disk={"fp8": int(1.5 * GB)},
        gguf=GgufSource(
            "city96/Wan2.1-T2V-1.3B-gguf:wan2.1-t2v-1.3b-Q8_0.gguf",
            int(1.5 * GB),
            "Q8_0",
            verified=True,
        ),
        measured=Measured(
            49.4, 20, "480x832x81f 20-step: 1046 s incl. decode", resident_ratio=1.02
        ),
        simple=SimpleFacing("Short video clips", "Full quality", "plain", unit="per 5-second clip"),
        width=832,
        height=480,
        frames=81,
        guidance=5.0,
    ),
    StudioModel(
        id="wan21-t2v-14b",
        name="Wan 2.1 T2V",
        repo_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        kind="video",
        params="14B",
        n_blocks=40,
        dit_bytes=int(29.1 * GB),
        largest_block_bytes=700 * MB,
        resident_bytes=1 * GB,
        native_bytes=80 * GB,
        hidden=5120,
        tokens=24_000,
        steps=20,
        on_disk={"fp8": int(20.7 * GB)},
        measured=Measured(1890.0, 20, "480x832x81f 20-step: 10.6 h, 562 GB streamed"),
        simple=SimpleFacing("Best video clips", "Full quality", "plain", unit="per 5-second clip"),
        width=832,
        height=480,
        frames=81,
        guidance=5.0,
        warnings=("IO-bound on a laptop NVMe under memory pressure: hours, not minutes.",),
    ),
    StudioModel(
        id="hunyuan-video",
        name="HunyuanVideo",
        repo_id="hunyuanvideo-community/HunyuanVideo",
        kind="video",
        params="13B",
        n_blocks=60,
        dit_bytes=int(25.7 * GB),
        largest_block_bytes=620 * MB,
        resident_bytes=1 * GB,
        native_bytes=60 * GB,
        hidden=3072,
        tokens=33_000,
        steps=30,
        on_disk={},
        simple=SimpleFacing("Long video clips", "Full quality", "plain", unit="per clip"),
        note="the 15 GB Llava text encoder must run on the CPU",
        width=848,
        height=480,
        frames=61,
        guidance=6.0,
    ),
)

BY_ID: dict[str, StudioModel] = {m.id: m for m in MODELS}

#: Models discovered on this machine's disk that the table does not list —
#: anything split with `aircanvas split <path-or-repo>`. Refreshed by the
#: Studio whenever it scans the cache (server.Studio.models), and looked up by
#: `get()` so a run started from the Desk resolves the same way a catalog
#: model does. Process-local by design: it describes THIS box's disk.
LOCAL: dict[str, StudioModel] = {}


def get(model_id: str) -> StudioModel:
    model = BY_ID.get(model_id) or LOCAL.get(model_id)
    if model is None:
        raise KeyError(f"Unknown model {model_id!r}")
    return model


def local_id(cache_dir: str) -> str:
    """Stable id for a shard cache, so a reload keeps the Desk's selection."""
    import hashlib

    return "local-" + hashlib.sha256(cache_dir.encode("utf-8")).hexdigest()[:10]


def from_installed(record: Any) -> StudioModel | None:
    """A catalog row for a shard cache that is not in the table.

    The Studio can generate from anything already split — `aircanvas split
    ./my-model` should not leave a cache the UI can see in Settings but not
    select on the Desk. Every number comes from the manifest itself, so the
    verdict is solved against the real shards; the fields a manifest cannot
    know (a marketing parameter count, a native download size, a plain-words
    quality note) are simply absent rather than guessed.
    """
    from aircanvas.sharding.manifest import Manifest

    try:
        manifest = Manifest.load(Path(record.cache_dir))
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning("Ignoring shard cache %s (%s)", record.cache_dir, e)
        return None
    if not manifest.blocks:
        return None

    fmt = manifest.compression or "bf16"
    # Caches written before sources were recorded absolute may hold a relative
    # path; resolve it against the cwd rather than letting it reach the Hub.
    source = str(manifest.source)
    if Path(source).is_dir():
        source = str(Path(source).resolve())
    name = Path(source).name or source
    materialized = sum(b.materialized_bytes for b in manifest.blocks)
    video = "video" in manifest.adapter or manifest.adapter in ("wan", "cogvideox")
    return StudioModel(
        id=local_id(record.cache_dir),
        name=name,
        repo_id=source,
        kind="video" if video else "image",
        # Parameters are inferred from bf16 bytes; below a rounding
        # threshold the figure would read "0.0B", so say nothing.
        params=f"{materialized / 2e9:.1f}B" if materialized >= 1e8 else "",
        n_blocks=len(manifest.blocks),
        dit_bytes=materialized,
        largest_block_bytes=max(b.materialized_bytes for b in manifest.blocks),
        resident_bytes=manifest.resident_materialized_bytes,
        native_bytes=0,  # already on disk; nothing left to download
        hidden=Workload.hidden,
        tokens=Workload.tokens,
        steps=20,
        on_disk={fmt: record.disk_bytes},
        formats=(fmt,),  # the only format it is actually split into
        default_format=fmt,
        subfolder=manifest.subfolder,
        note=f"split locally from {source}",
        real_manifest=manifest,
        cache_dir=record.cache_dir,
    )


# -- verdicts ---------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """One chip: a call, the reason for it, and the plan behind both."""

    tone: Literal["good", "warn", "fail"]
    label: str
    resident_blocks: int = 0
    streamed_blocks: int = 0
    seconds_per_step: float = 0.0
    headroom_bytes: int = 0
    needed_bytes: int = 0
    ring_depth: int = 0
    step_read_bytes: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "tone": self.tone,
            "label": self.label,
            "residentBlocks": self.resident_blocks,
            "streamedBlocks": self.streamed_blocks,
            "secondsPerStep": round(self.seconds_per_step, 2),
            "headroomBytes": self.headroom_bytes,
            "neededBytes": self.needed_bytes,
            "ringDepth": self.ring_depth,
            "stepReadBytes": self.step_read_bytes,
        }


def _duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 90:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def verdict(
    model: StudioModel,
    fmt: str,
    hardware: HardwareProfile,
    *,
    steps: int | None = None,
) -> Verdict:
    """Run the real budget solver and say what it found, with the reason.

    A verdict always names why: headroom when it is tight, the disk time per
    step when it runs, the VRAM it would need when it does not. "Slack" is
    measured against the MINIMUM plan (nothing resident) rather than the chosen
    one — the waterfall deliberately spends every spare byte, so the chosen
    plan looks equally full on a 6 GB laptop and a 24 GB workstation.
    """
    manifest = model.manifest(fmt)
    workload = model.workload(steps=steps)
    try:
        minimal = solve(manifest, hardware, workload, max_resident_blocks=0)
        plan = solve(manifest, hardware, workload)
    except InsufficientVRAMError:
        floor = (
            workload.estimate_activation_bytes()
            + manifest.resident_materialized_bytes
            + 2 * model.largest_block_bytes
        )
        return Verdict("fail", f"✗ needs {floor / GB:.0f} GB", needed_bytes=floor)

    headroom = minimal.vram_budget_bytes - minimal.vram_planned_bytes
    if hardware.device != "cuda":
        # The memory answer is still real — on the CPU path device memory IS
        # system RAM, and the solver plans against it. The *time* answer is
        # not: every per-step figure this module knows was measured on a GPU,
        # so naming one here would be a promise this box cannot keep.
        return Verdict(
            "warn",
            "runs on CPU · not fast",
            resident_blocks=plan.resident_blocks,
            streamed_blocks=plan.streamed_blocks,
            headroom_bytes=headroom,
            ring_depth=plan.ring_depth,
            step_read_bytes=plan.step_read_bytes,
        )
    io_per_step = plan.step_read_seconds
    common = {
        "resident_blocks": plan.resident_blocks,
        "streamed_blocks": plan.streamed_blocks,
        "seconds_per_step": io_per_step,
        "headroom_bytes": headroom,
        "ring_depth": plan.ring_depth,
        "step_read_bytes": plan.step_read_bytes,
    }
    if headroom <= COMFORT_MARGIN:
        return Verdict("warn", f"tight · {headroom / GB:.1f} GB headroom", **common)  # type: ignore[arg-type]
    if io_per_step * max(1, workload.steps) > 3600:
        return Verdict("warn", f"tight · {_duration(io_per_step)}/step on this disk", **common)  # type: ignore[arg-type]
    if plan.streamed_blocks == 0:
        return Verdict("good", "✓ runs · fully resident", **common)  # type: ignore[arg-type]
    if model.measured is not None and model.measured.resident_ratio is not None:
        ratio = model.measured.resident_ratio
        return Verdict("good", f"✓ runs · {ratio:.2f}× resident", **common)  # type: ignore[arg-type]
    return Verdict("good", f"✓ runs · ~{_duration(io_per_step)} IO/step", **common)  # type: ignore[arg-type]


def estimate_seconds(
    model: StudioModel,
    fmt: str,
    hardware: HardwareProfile,
    *,
    steps: int | None = None,
    measured_per_step: float | None = None,
) -> tuple[float, float]:
    """(total seconds, seconds per step) for one generation.

    `measured_per_step` — the last real run of this model at this shape, when
    the gallery has one — always wins: nothing beats a measurement of the
    thing itself. Otherwise the solver's disk time per step is the floor, and
    a published measurement from docs/BENCHMARKS.md raises it to include
    compute, scaled by how this disk compares to the one it was measured on.
    """
    steps = steps or model.steps
    if measured_per_step is not None and measured_per_step > 0:
        return measured_per_step * steps, measured_per_step

    try:
        plan = solve(model.manifest(fmt), hardware, model.workload(steps=steps))
    except InsufficientVRAMError:
        return 0.0, 0.0
    per_step = plan.step_read_seconds
    if model.measured is not None:
        # The published figure is one machine's total per step (compute + IO).
        # Its IO share scales with this disk; its compute share does not, and
        # we cannot separate them, so take the larger of the two bounds rather
        # than pretending to a precision we do not have.
        per_step = max(per_step, model.measured.seconds_per_step)
    return per_step * steps, per_step
