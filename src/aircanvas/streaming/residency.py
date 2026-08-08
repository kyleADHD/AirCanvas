"""Budget solver: turn (hardware, manifest, workload) into a residency plan (M4).

Waterfall policy (ARCHITECTURE.md §3.2), not an ILP:

1. Reserve activations (adapter workload model) + the non-block resident shard
   + a fragmentation/context headroom.
2. Leftover VRAM -> slot pool + permanently resident blocks, taken from the
   FRONT of the manifest. Slot size depends on which blocks still stream, so
   the two are solved together: for each candidate R we recompute the pool from
   the tail and keep the largest R that fits. n_blocks <= ~60, so scanning is
   cheaper than being clever.
3. Leftover RAM -> pinned ring first, then the shard RAM-cache size.
4. Ring depth from what RAM allows, capped at MAX_RING_DEPTH.
5. Warnings: SATA-class disk (~0.5 GB/s), distilled few-step models (streaming
   overhead visible — prefer the RAM tier), uncompressed cache on a slow disk,
   VRAM below the hard floor.

Every OOM path prints the plan that was attempted (ARCHITECTURE.md §4): see
`InsufficientVRAMError`, which carries `ResidencyPlan.describe()`.

**Scope note.** `ram_cache_bytes` is live as of M8: the prefetcher promotes
raw shard blobs into pageable RAM after their first disk read, up to this
budget — with the whole streamed set cached, disk is touched exactly once per
run and repeat steps are PCIe-bound instead of NVMe-bound.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aircanvas.config import StreamConfig
from aircanvas.sharding.manifest import Manifest
from aircanvas.utils.hw import SATA_DISK_BW, HardwareProfile

logger = logging.getLogger(__name__)

#: CUDA context, allocator fragmentation and small transient buffers.
VRAM_HEADROOM_BYTES = 384 << 20
#: Leave the OS room to breathe; pinning every free page is how you hang Windows.
RAM_HEADROOM_BYTES = 1 << 30
#: Floor for the activation estimate (latents, embeddings, attention scratch).
MIN_ACTIVATION_BYTES = 256 << 20
#: Bytes of transient activation per token per hidden unit, in compute dtype.
#: Calibrated against RESEARCH.md §4: Wan 14B @720p (75.6k tokens, hidden 5120)
#: measures ~2.9 GB of FFN transient + residual copies => ~3.75 -> 4.
ACTIVATION_TOKENS_FACTOR = 4
MAX_RING_DEPTH = 4
#: At or below this many steps a model is distilled enough that per-step
#: streaming overhead stops hiding behind compute (RESEARCH.md §2).
DISTILLED_STEPS = 8


class InsufficientVRAMError(RuntimeError):
    """Not even zero resident blocks fit. Carries the attempted plan."""


@dataclass(frozen=True)
class Workload:
    """What the caller is about to generate; drives the activation reserve."""

    steps: int = 30
    tokens: int = 4608  # FLUX @1024^2: 4096 image + 512 text
    hidden: int = 3072
    dtype_bytes: int = 2
    activation_bytes: int | None = None  # explicit override, skips the estimate

    def estimate_activation_bytes(self) -> int:
        if self.activation_bytes is not None:
            return self.activation_bytes
        est = self.tokens * self.hidden * self.dtype_bytes * ACTIVATION_TOKENS_FACTOR
        return max(MIN_ACTIVATION_BYTES, est)

    @property
    def distilled(self) -> bool:
        return self.steps <= DISTILLED_STEPS


@dataclass(frozen=True)
class ResidencyPlan:
    """The memory plan a run will execute, and enough context to debug an OOM."""

    resident_blocks: int
    ring_depth: int
    lookahead: int
    ram_cache_bytes: int
    gpu_slots: int = 2
    slot_bytes: int = 0
    pinned_bytes: int = 0
    activation_bytes: int = 0
    resident_shard_bytes: int = 0
    resident_block_bytes: int = 0
    vram_budget_bytes: int = 0
    vram_planned_bytes: int = 0
    streamed_blocks: int = 0
    step_read_bytes: int = 0
    step_read_seconds: float = 0.0
    warnings: tuple[str, ...] = field(default=())

    def stream_config(self) -> StreamConfig:
        return StreamConfig(gpu_slots=self.gpu_slots, ring_depth=self.ring_depth)

    def describe(self) -> str:
        gb = 1e9
        lines = [
            f"resident blocks    {self.resident_blocks} ({self.resident_block_bytes / gb:.2f} GB)",
            f"streamed blocks    {self.streamed_blocks} "
            f"({self.step_read_bytes / gb:.2f} GB read/step, "
            f"~{self.step_read_seconds:.2f}s of disk time/step)",
            f"slot pool          {self.gpu_slots} slots, {self.slot_bytes / gb:.2f} GB",
            f"pinned ring        depth {self.ring_depth} (lookahead {self.lookahead}), "
            f"{self.pinned_bytes / gb:.2f} GB",
            f"activations est.   {self.activation_bytes / gb:.2f} GB",
            f"non-block resident {self.resident_shard_bytes / gb:.2f} GB",
            f"VRAM planned       {self.vram_planned_bytes / gb:.2f} GB "
            f"of {self.vram_budget_bytes / gb:.2f} GB budget",
            f"RAM shard cache    {self.ram_cache_bytes / gb:.2f} GB (promoted after first read)",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _pool_bytes(
    manifest: Manifest, streamed: list, gpu_slots: int, raw_slots: int
) -> tuple[int, int]:
    """(device slot bytes, max on-disk block bytes) for a given streamed tail."""
    if not streamed:
        return 0, 0
    max_disk = max(b.n_bytes for b in streamed)
    max_load = max(b.materialized_bytes for b in streamed)
    slot = gpu_slots * max_load
    if manifest.compression is not None:
        slot += raw_slots * max_disk  # compressed staging pool (ADR #8)
    return slot, max_disk


def solve(
    manifest: Manifest,
    hardware: HardwareProfile,
    workload: Workload | None = None,
    *,
    config: StreamConfig | None = None,
    vram_budget: int | None = None,
    ram_budget: int | None = None,
    max_resident_blocks: int | None = None,
) -> ResidencyPlan:
    """Waterfall-solve the residency plan. Raises InsufficientVRAMError if even
    a fully streamed run does not fit."""
    workload = workload or Workload()
    config = config or StreamConfig()
    blocks = list(manifest.blocks)
    if not blocks:
        raise ValueError("Manifest has no blocks to plan for")

    # On a CPU run the "device" memory IS system RAM, so the same pool pays for
    # slots, residents AND the ring — which is why it is subtracted below rather
    # than counted twice. (CPU is the correctness path, not the fast path.)
    device_is_host = hardware.device != "cuda"
    ram = hardware.ram_bytes if ram_budget is None else ram_budget
    if vram_budget is not None:
        vram = vram_budget
    else:
        vram = ram if device_is_host else hardware.vram_bytes
    activations = workload.estimate_activation_bytes()
    fixed = activations + manifest.resident_materialized_bytes + VRAM_HEADROOM_BYTES

    # -- 2. VRAM waterfall: largest R whose pool + residents still fit --------
    # Scanned exhaustively rather than stopped at the first miss: the slot pool
    # SHRINKS as R grows for two-species models (FLUX's big blocks are at the
    # front), so total VRAM is not monotonic in R.
    best: tuple[int, int, int] | None = None  # (R, slot_bytes, total)
    r_limit = len(blocks) if max_resident_blocks is None else min(len(blocks), max_resident_blocks)
    for r in range(r_limit + 1):
        slot_bytes, _ = _pool_bytes(manifest, blocks[r:], config.gpu_slots, config.raw_slots)
        resident_bytes = sum(b.materialized_bytes for b in blocks[:r])
        total = fixed + slot_bytes + resident_bytes
        if total <= vram:
            best = (r, slot_bytes, total)
    if best is None:
        slot_bytes, _ = _pool_bytes(manifest, blocks, config.gpu_slots, config.raw_slots)
        attempted = ResidencyPlan(
            resident_blocks=0,
            ring_depth=config.ring_depth,
            lookahead=min(config.ring_depth, config.gpu_slots),
            ram_cache_bytes=0,
            gpu_slots=config.gpu_slots,
            slot_bytes=slot_bytes,
            activation_bytes=activations,
            resident_shard_bytes=manifest.resident_materialized_bytes,
            streamed_blocks=len(blocks),
            step_read_bytes=manifest.disk_bytes(),
            vram_budget_bytes=vram,
            vram_planned_bytes=fixed + slot_bytes,
        )
        raise InsufficientVRAMError(
            f"{(fixed + slot_bytes) / 1e9:.2f} GB needed but only {vram / 1e9:.2f} GB of VRAM "
            f"is available, with every block streamed. Attempted plan:\n" + attempted.describe()
        )

    resident_blocks, slot_bytes, planned = best
    streamed = blocks[resident_blocks:]
    _, max_disk = _pool_bytes(manifest, streamed, config.gpu_slots, config.raw_slots)

    # -- 3/4. RAM waterfall: pinned ring first, then the shard cache ---------
    ram_for_buffers = max(0, ram - RAM_HEADROOM_BYTES - (planned if device_is_host else 0))
    ring_depth = config.ring_depth
    if max_disk > 0:
        affordable = int(ram_for_buffers // max_disk)
        ring_depth = max(1, min(MAX_RING_DEPTH, config.ring_depth, affordable))
    pinned_bytes = ring_depth * max_disk
    streamed_disk = sum(b.n_bytes for b in streamed)
    ram_cache_bytes = min(streamed_disk, max(0, ram_for_buffers - pinned_bytes))

    step_read = streamed_disk
    step_seconds = step_read / hardware.disk_bw_bytes_s if hardware.disk_bw_bytes_s > 0 else 0.0

    # -- 5. warnings ---------------------------------------------------------
    warnings: list[str] = []
    if hardware.disk_bw_bytes_s < SATA_DISK_BW:
        warnings.append(
            f"SATA-class disk detected ({hardware.disk_bw_bytes_s / 1e9:.2f} GB/s): expect "
            f"visible streaming overhead; a RAM cache or fewer steps will help more than VRAM."
        )
    if workload.distilled:
        warnings.append(
            f"{workload.steps}-step (distilled) model: only ~"
            f"{step_seconds:.2f}s of disk time per step can hide behind compute, and there are "
            f"few steps to amortise the first, uncached pass over."
        )
    if manifest.compression is None and step_read > 8e9:
        warnings.append(
            f"Uncompressed shard cache reads {step_read / 1e9:.1f} GB per step; "
            f"compression='fp8' roughly halves that."
        )
    if resident_blocks == 0 and len(blocks) > 1:
        warnings.append(
            "No VRAM left for resident blocks — every block streams every step. "
            "This is the intended low-VRAM path, just the slowest one."
        )
    if ram_cache_bytes >= streamed_disk and streamed_disk > 0:
        warnings.append(
            f"RAM would hold all {streamed_disk / 1e9:.2f} GB of streamed shards; the "
            f"auto-promotion tier that would exploit that lands in M8 (docs/ROADMAP.md)."
        )
    if ring_depth < config.ring_depth:
        warnings.append(
            f"Pinned ring reduced to depth {ring_depth} (from {config.ring_depth}) to fit "
            f"{ram / 1e9:.1f} GB of free RAM."
        )

    plan = ResidencyPlan(
        resident_blocks=resident_blocks,
        ring_depth=ring_depth,
        lookahead=min(ring_depth, config.gpu_slots),
        ram_cache_bytes=ram_cache_bytes,
        gpu_slots=config.gpu_slots,
        slot_bytes=slot_bytes,
        pinned_bytes=pinned_bytes,
        activation_bytes=activations,
        resident_shard_bytes=manifest.resident_materialized_bytes,
        resident_block_bytes=sum(b.materialized_bytes for b in blocks[:resident_blocks]),
        vram_budget_bytes=vram,
        vram_planned_bytes=planned,
        streamed_blocks=len(streamed),
        step_read_bytes=step_read,
        step_read_seconds=step_seconds,
        warnings=tuple(warnings),
    )
    logger.info(
        "Residency plan: %d resident / %d streamed blocks, %.2f GB slots, ring %d",
        plan.resident_blocks,
        plan.streamed_blocks,
        plan.slot_bytes / 1e9,
        plan.ring_depth,
    )
    return plan
