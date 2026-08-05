"""StreamingEngine: hook-based just-in-time weight binding (M2).

Mechanism (ARCHITECTURE.md §3.2):
- DiT instantiated via from_config under init_empty_weights (meta device).
- Per-block forward pre-hook: wait on the block's GPU-ready CUDA event, bind
  weights into the module. Post-hook: release the GPU slot to the pool, notify
  the prefetcher.
- GPU SLOT POOL (ADR #5): 2-3 reusable weight buffers sized to the largest
  block. Never allocate/free GPU or pinned memory in the per-block hot loop;
  no gc.collect()/empty_cache() per block (AirLLM's measured overhead).
- First forward RECORDS the block execution schedule; later steps prefetch
  against it (ADR #7 — the denoise loop is deterministic).
- Resident blocks (budget solver output) are loaded once and never evicted.
- expert_groups (Wan 2.2): the scheduler tells the engine which expert is
  active per timestep; only that expert's shards enter the schedule.

Correctness gate: streamed output must be bitwise-equal to full-VRAM at
compression=None (tests/integration/test_equivalence.py).
"""

from __future__ import annotations


class StreamingEngine:
    def __init__(self) -> None:
        raise NotImplementedError("M2")
