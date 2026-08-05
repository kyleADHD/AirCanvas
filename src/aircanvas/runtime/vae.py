"""VAE decode with bounded memory (M4 image passthrough, M6 Wan tiling).

- Models with native support: passthrough to enable_tiling()/enable_slicing().
- AutoencoderKLWan: diffusers only frame-chunks via feat_cache (CACHE_T=2) —
  NO spatial tiling upstream. We ship spatial tile decode with causal-cache-
  aware overlap blending. This is AirCanvas's upstream-worthy contribution
  (ROADMAP M6).
"""

from __future__ import annotations
