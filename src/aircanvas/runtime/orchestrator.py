"""Phase orchestration behind AirPipeline (M4).

Phase 1 ENCODE: per text encoder — load to GPU (whole, or group-offloaded if
  it alone exceeds budget: HunyuanVideo's 15 GB Llava, HiDream's 4-TE stack),
  run once, cache embeddings, evict. Optional pure-CPU TE mode. Embeddings for
  empty/negative prompts cached on disk.
Phase 2 DENOISE: DiT handed to StreamingEngine; the stock diffusers pipeline
  loop runs unmodified (we own placement only, ADR #6).
Phase 3 DECODE: tiled VAE decode via runtime.vae.
"""

from __future__ import annotations
