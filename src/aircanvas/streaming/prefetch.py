"""Three-stage prefetch pipeline (M3).

  NVMe --io threads--> pinned CPU ring --dedicated CUDA copy stream--> GPU slot

- Stage 1: small thread pool reads shard files into a fixed ring of pinned
  buffers (allocated ONCE at startup — pinned alloc is slow on Windows).
- Stage 2: copy_(non_blocking=True) on a dedicated stream; records a per-block
  'ready' CUDA event; fp8 upcast / nf4 dequant runs on GPU right after the copy.
- Stage 3: compute (default stream) waits on the ready event in the pre-hook.
- Lookahead K over the recorded schedule; ring depth and K come from the
  budget solver.
- RAM shard cache tier: when ram_budget allows, completed reads are retained
  (whole model in RAM => disk touched only on first step).
"""

from __future__ import annotations


class PrefetchPipeline:
    def __init__(self) -> None:
        raise NotImplementedError("M3")
