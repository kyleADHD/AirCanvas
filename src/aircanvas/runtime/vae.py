"""VAE decode with bounded memory.

Phase 3 of ARCHITECTURE.md §3.3. The VAE runs ONCE per output, so — like the
text encoders and unlike the DiT — it should not occupy VRAM for the whole
denoise loop. `vae_on_demand` keeps it on the CPU until diffusers actually
calls `decode`, moves it across for that one call, and moves it back. It also
times the call, which is how `pipe.report()` gets a real decode number without
us reimplementing any part of the pipeline's latent unpacking or denormalising
(ADR #6: we own placement, diffusers owns the maths).

Passthrough to enable_tiling()/enable_slicing() covers every current target:
as of diffusers 0.39 that INCLUDES AutoencoderKLWan, which gained spatial
tiling upstream after our research snapshot — the custom Wan tiling this
module once planned (old ROADMAP M6) is happily obsolete. configure_vae's
duck-typing needs no per-model knowledge; it reports what it actually enabled
so a future untileable VAE shows up in the report rather than OOMing silently.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from aircanvas.runtime.progress import RunObserver, notify
from aircanvas.utils.memory import clean_memory

logger = logging.getLogger(__name__)


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
    if tiling and "tiling" not in enabled:
        logger.warning(
            "%s exposes no enable_tiling(); decode memory is unbounded in resolution.",
            type(vae).__name__,
        )
    return enabled


@contextlib.contextmanager
def vae_on_demand(
    pipe: Any,
    device: torch.device,
    stats: dict[str, float],
    *,
    evict: bool = True,
    observer: RunObserver | None = None,
) -> Iterator[None]:
    """Keep the VAE off `device` until decode, then put it back.

    Wraps the bound `decode` with an instance attribute (shadowing the class
    method) and removes it on exit, so nothing about the pipeline object
    survives the context. Decode is the only phase that runs *inside* the
    pipeline call, so this wrapper is also where a live UI learns the phase
    changed (`observer`).
    """
    vae = getattr(pipe, "vae", None)
    if not isinstance(vae, nn.Module) or not callable(getattr(vae, "decode", None)):
        yield
        return

    original = vae.decode

    def timed_decode(*args: object, **kwargs: object) -> object:
        start = time.perf_counter()
        notify(observer, "phase_started", "decode")
        vae.to(device)
        try:
            return original(*args, **kwargs)
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            stats["decode_s"] = stats.get("decode_s", 0.0) + elapsed
            notify(observer, "phase_finished", "decode", elapsed)
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
