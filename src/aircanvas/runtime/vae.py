"""VAE decode with bounded memory (M4 image passthrough, M6 Wan tiling).

Phase 3 of ARCHITECTURE.md §3.3. The VAE runs ONCE per output, so — like the
text encoders and unlike the DiT — it should not occupy VRAM for the whole
denoise loop. `vae_on_demand` keeps it on the CPU until diffusers actually
calls `decode`, moves it across for that one call, and moves it back. It also
times the call, which is how `pipe.report()` gets a real decode number without
us reimplementing any part of the pipeline's latent unpacking or denormalising
(ADR #6: we own placement, diffusers owns the maths).

- Models with native support: passthrough to enable_tiling()/enable_slicing().
- AutoencoderKLWan: diffusers only frame-chunks via feat_cache (CACHE_T=2) —
  NO spatial tiling upstream. We ship spatial tile decode with causal-cache-
  aware overlap blending. This is AirCanvas's upstream-worthy contribution
  (ROADMAP M6); `wan_tiled_decode` below is the placeholder for it.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from aircanvas.utils.memory import clean_memory

logger = logging.getLogger(__name__)

#: VAE classes diffusers cannot spatially tile today (RESEARCH.md §4).
NO_UPSTREAM_TILING = ("AutoencoderKLWan",)


def configure_vae(vae: nn.Module | None, *, tiling: bool = True, slicing: bool = True) -> list[str]:
    """Turn on whatever bounded-memory decode the model supports natively.

    Returns the feature names actually enabled, so callers can report honestly
    instead of assuming tiling happened.
    """
    enabled: list[str] = []
    if vae is None:
        return enabled
    if tiling and callable(getattr(vae, "enable_tiling", None)):
        vae.enable_tiling()
        enabled.append("tiling")
    if slicing and callable(getattr(vae, "enable_slicing", None)):
        vae.enable_slicing()
        enabled.append("slicing")
    if type(vae).__name__ in NO_UPSTREAM_TILING and "tiling" not in enabled:
        logger.warning(
            "%s has no spatial tiling in diffusers; decode memory is unbounded in resolution "
            "until AirCanvas ships tiled Wan-VAE decode (ROADMAP M6).",
            type(vae).__name__,
        )
    return enabled


@contextlib.contextmanager
def vae_on_demand(
    pipe: Any, device: torch.device, stats: dict[str, float], *, evict: bool = True
) -> Iterator[None]:
    """Keep the VAE off `device` until decode, then put it back.

    Wraps the bound `decode` with an instance attribute (shadowing the class
    method) and removes it on exit, so nothing about the pipeline object
    survives the context.
    """
    vae = getattr(pipe, "vae", None)
    if not isinstance(vae, nn.Module) or not callable(getattr(vae, "decode", None)):
        yield
        return

    original = vae.decode

    def timed_decode(*args: object, **kwargs: object) -> object:
        start = time.perf_counter()
        vae.to(device)
        try:
            return original(*args, **kwargs)
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stats["decode_s"] = stats.get("decode_s", 0.0) + (time.perf_counter() - start)
            if evict:
                vae.to("cpu")

    vae.decode = timed_decode  # type: ignore[method-assign]
    try:
        yield
    finally:
        try:
            del vae.decode  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - defensive
            vae.decode = original  # type: ignore[method-assign]
        if evict:
            clean_memory()


def wan_tiled_decode(*args: object, **kwargs: object) -> torch.Tensor:
    """Spatially tiled AutoencoderKLWan decode — ROADMAP M6.

    Deliberately not stubbed as a silent passthrough: a caller that reaches
    here on a 720p video would OOM in a way that looks like our bug.
    """
    raise NotImplementedError(
        "Tiled Wan-VAE decode lands in M6 (docs/ROADMAP.md). Until then, decode Wan latents "
        "at a resolution the unbounded upstream decoder can hold."
    )
