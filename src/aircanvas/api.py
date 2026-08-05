"""Public API: AirPipeline.

M4 milestone. Wraps a stock diffusers pipeline and owns ONLY device/memory
placement (ARCHITECTURE.md ADR #6): text encoders load-run-evict, DiT handed
to the StreamingEngine, VAE decode tiled.
"""

from __future__ import annotations

from aircanvas.config import BudgetConfig, Compression


class AirPipeline:
    """Phase-aware wrapper around a diffusers DiffusionPipeline.

    Usage (target API)::

        pipe = AirPipeline.from_pretrained("Qwen/Qwen-Image", vram_budget="6GB")
        image = pipe("a prompt", num_inference_steps=30).images[0]
        pipe.report()  # io / h2d / dequant / compute breakdown
    """

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        vram_budget: str | int = "auto",
        ram_budget: str | int = "auto",
        compression: Compression = "fp8",
        shard_cache: str | None = None,
        **pipeline_kwargs: object,
    ) -> AirPipeline:
        """Resolve budgets, find-or-create the shard cache, build the wrapped pipeline."""
        raise NotImplementedError("M4")

    def __call__(self, prompt: str, **kwargs: object) -> object:
        raise NotImplementedError("M4")

    def report(self) -> None:
        """Print where time went last run and the residency plan used."""
        raise NotImplementedError("M4")

    _budget: BudgetConfig
