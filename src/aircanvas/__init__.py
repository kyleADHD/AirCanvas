"""AirCanvas: run 20B+ image & video diffusion models on 4-8 GB GPUs.

Streams quantized per-block weight shards from disk just-in-time during the
denoise loop. See docs/ARCHITECTURE.md for the full design.
"""

__version__ = "0.0.1"

# Imported after __version__ so the splitter's `from aircanvas import __version__`
# resolves while this module is still initialising.
from aircanvas.api import AirPipeline  # noqa: E402

__all__ = ["AirPipeline", "__version__"]
