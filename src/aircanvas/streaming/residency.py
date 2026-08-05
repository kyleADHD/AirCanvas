"""Budget solver: turn (hardware, manifest, workload) into a residency plan (M4).

Waterfall policy (ARCHITECTURE.md §3.2), not an ILP:
1. Reserve activations (adapter workload model) + GPU slot pool + CUDA context.
2. Leftover VRAM -> permanently resident blocks (front of schedule first).
3. Leftover RAM  -> pinned shard cache size (MRU promotion).
4. Ring depth / lookahead from disk-bw vs per-block compute estimate.
5. Warnings: SATA-class disk (~0.5 GB/s), distilled few-step models (streaming
   overhead visible — prefer RAM tier), budget below hard floor.

Every OOM path must print the plan that was attempted (ARCHITECTURE.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidencyPlan:
    resident_blocks: int
    ring_depth: int
    lookahead: int
    ram_cache_bytes: int
    warnings: tuple[str, ...] = ()


def solve(*args: object, **kwargs: object) -> ResidencyPlan:
    raise NotImplementedError("M4")
