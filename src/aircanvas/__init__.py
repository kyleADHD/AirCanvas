"""AirCanvas: run 20B+ image & video diffusion models on 4-8 GB GPUs.

Streams quantized per-block weight shards from disk just-in-time during the
denoise loop. See docs/ARCHITECTURE.md for the full design.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]

# Public API (exported once implemented, M4):
# from aircanvas.api import AirPipeline
