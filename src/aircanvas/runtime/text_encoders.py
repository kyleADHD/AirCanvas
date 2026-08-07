"""Text-encoder strategies: load-run-evict, with a disk embedding cache (M4).

Phase 1 of ARCHITECTURE.md §3.3. Text encoders run ONCE per prompt while the
DiT runs every step, so they get the opposite treatment from the DiT: pull the
whole encoder onto the compute device, run it, cache the embeddings, push it
straight back to CPU. Peak VRAM is then max(TE, DiT-slot-pool) instead of
their sum, and the DiT never competes with a 5–16 GB encoder for space.

We call the *stock* pipeline's own `encode_prompt` (ADR #6 — prompt handling
is diffusers' job, not ours) and only decide where the weights live while it
runs. Mapping its return tuple onto `__call__` kwargs is the one piece of
per-family knowledge involved, and it lives in the adapter
(`ModelAdapter.encode_prompt_outputs`); anything the pipeline's signature does
not accept is dropped rather than guessed at.

Quirks to keep in mind (RESEARCH.md §7):
- T5's forward does its own dtype casting — it breaks naive fp8 *storage*, but
  TEs are never sharded or quantized here, so this path is unaffected.
- SD3.5's T5 is droppable entirely (quality tradeoff, saves ~9.5 GB) — a
  later adapter knob, not wired in M4.
- Qwen-Image's TE is a full Qwen2.5-VL run with a task system prompt (M5).
- HunyuanVideo's 15 GB Llava does not fit a 6 GB card even alone; it needs
  group-offloading or `text_encoder_device="cpu"` (M7).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from aircanvas.adapters import ModelAdapter
from aircanvas.utils.hw import free_vram_bytes
from aircanvas.utils.memory import clean_memory

logger = logging.getLogger(__name__)

#: Prefix used to discover text-encoder components on a stock pipeline.
TE_PREFIX = "text_encoder"

#: Fraction of free VRAM the encoders may claim before falling back to CPU —
#: activations, allocator fragmentation and the CUDA context need the rest.
TE_VRAM_FRACTION = 0.8


@dataclass
class EncodedPrompt:
    """Embedding kwargs to hand to the pipeline call, plus how they got there."""

    kwargs: dict[str, torch.Tensor] = field(default_factory=dict)
    seconds: float = 0.0
    cached: bool = False
    encoders_run: tuple[str, ...] = ()


def text_encoder_names(pipe: object) -> tuple[str, ...]:
    """Names of the pipeline's text-encoder modules, in declaration order."""
    components = getattr(pipe, "components", {}) or {}
    return tuple(
        name
        for name in components
        if name.startswith(TE_PREFIX) and isinstance(getattr(pipe, name, None), nn.Module)
    )


def _is_dispatched(module: nn.Module) -> bool:
    """True for accelerate device_map models — they manage their own placement
    and `.to()` on them is an error. Used to load an 11 GB UMT5 across
    GPU+CPU+disk on a 16 GB machine where a plain CPU load segfaults."""
    return getattr(module, "hf_device_map", None) is not None


def _move(pipe: object, names: Iterable[str], device: torch.device | str) -> None:
    for name in names:
        module = getattr(pipe, name, None)
        if isinstance(module, nn.Module) and not _is_dispatched(module):
            module.to(device)


def _te_param_bytes(pipe: object, names: Iterable[str]) -> int:
    total = 0
    for name in names:
        module = getattr(pipe, name, None)
        if isinstance(module, nn.Module):
            total += sum(p.numel() * p.element_size() for p in module.parameters())
            total += sum(b.numel() * b.element_size() for b in module.buffers())
    return total


def resolve_te_device(
    pipe: object,
    names: Iterable[str],
    device: torch.device,
    requested: torch.device | str | None = None,
) -> torch.device:
    """Where to RUN the encoders (embeddings always land on `device` after).

    'auto' (the default): the compute device when every encoder fits in free
    VRAM together, else CPU — slow but one-time per prompt, and the embedding
    cache makes repeats free. FLUX's 9.1 GB T5-XXL on a 6 GB card is the
    motivating case: without this the whole run dies in phase 1.
    """
    if requested is not None and requested != "auto":
        return torch.device(requested)
    if any(isinstance(m := getattr(pipe, n, None), nn.Module) and _is_dispatched(m) for n in names):
        return device  # dispatched encoders stay put; inputs go to the compute device
    if device.type != "cuda":
        return device
    need = _te_param_bytes(pipe, names)
    free = free_vram_bytes(device)
    if need <= free * TE_VRAM_FRACTION:
        return device
    logger.warning(
        "Text encoders need %.1f GB but only %.1f GB VRAM is free — encoding on CPU "
        "(one-time per prompt; cached after)",
        need / 1e9,
        free / 1e9,
    )
    return torch.device("cpu")


def _filter_kwargs(fn: object, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only what `fn` actually accepts (diffusers signatures drift)."""
    try:
        params = inspect.signature(fn).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return dict(candidate)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidate)
    return {k: v for k, v in candidate.items() if k in params}


def _cache_key(parts: Mapping[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def _load_cached(path: Path) -> dict[str, torch.Tensor] | None:
    if not path.is_file():
        return None
    try:
        from safetensors.torch import load_file

        return load_file(path)
    except Exception as e:  # noqa: BLE001 — a bad cache entry must never fail a run
        logger.warning("Ignoring unreadable embedding cache %s (%s)", path.name, e)
        return None


def _save_cached(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    try:
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        save_file({k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}, str(tmp))
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not write embedding cache: %s", e)


def encode_and_evict(
    pipe: Any,
    adapter: ModelAdapter,
    *,
    device: torch.device,
    prompt: str | list[str],
    negative_prompt: str | list[str] | None = None,
    encode_kwargs: Mapping[str, Any] | None = None,
    cache_dir: Path | None = None,
    te_device: torch.device | str | None = None,
) -> EncodedPrompt:
    """Run the pipeline's text encoders (see resolve_te_device), then evict
    them to CPU; returned embeddings always live on `device`.

    Returns the subset of embedding kwargs the pipeline's `__call__` accepts.
    On a disk-cache hit the encoders are never touched at all — which is the
    whole point for the empty/negative prompts that dominate repeat runs.
    """
    encode_kwargs = dict(encode_kwargs or {})
    names = text_encoder_names(pipe)
    candidates = {n: None for n in adapter.encode_prompt_outputs}
    candidates.update({f"negative_{n}": None for n in adapter.encode_prompt_outputs})
    accepted = set(_filter_kwargs(pipe.__call__, candidates))

    key = None
    if cache_dir is not None:
        key = _cache_key(
            {
                "adapter": adapter.key,
                "pipeline": type(pipe).__name__,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "encode_kwargs": encode_kwargs,
                "outputs": adapter.encode_prompt_outputs,
            }
        )
        hit = _load_cached(cache_dir / f"{key}.safetensors")
        if hit is not None:
            logger.info("Text-encoder cache hit — skipping encoder load entirely")
            return EncodedPrompt(
                kwargs={k: v.to(device) for k, v in hit.items()}, cached=True, seconds=0.0
            )

    start = time.perf_counter()
    run_device = resolve_te_device(pipe, names, device, te_device)
    _move(pipe, names, run_device)
    try:
        with torch.no_grad():
            out = _run_encode(pipe, adapter, run_device, prompt, encode_kwargs, prefix="")
            if negative_prompt is not None:
                out.update(
                    _run_encode(
                        pipe,
                        adapter,
                        run_device,
                        negative_prompt,
                        encode_kwargs,
                        prefix="negative_",
                    )
                )
    finally:
        _move(pipe, names, "cpu")
        clean_memory()
    seconds = time.perf_counter() - start

    kwargs = {
        k: v.to(device) for k, v in out.items() if k in accepted and isinstance(v, torch.Tensor)
    }
    dropped = sorted(set(out) - set(kwargs))
    if dropped:
        logger.debug("Dropped encode_prompt outputs not accepted by __call__: %s", dropped)
    if not kwargs:
        raise RuntimeError(
            f"{type(pipe).__name__}.encode_prompt produced nothing the pipeline accepts "
            f"(adapter {adapter.key!r} expects {adapter.encode_prompt_outputs}). "
            f"Set encode_prompt_outputs on the adapter for this family."
        )
    if key is not None and cache_dir is not None:
        _save_cached(cache_dir / f"{key}.safetensors", kwargs)
    logger.info("Encoded prompt in %.2fs with %s, encoders evicted", seconds, list(names))
    return EncodedPrompt(kwargs=kwargs, seconds=seconds, cached=False, encoders_run=names)


def _run_encode(
    pipe: Any,
    adapter: ModelAdapter,
    device: torch.device,
    prompt: str | list[str],
    encode_kwargs: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    # No setdefault("prompt", ...) fallback here on purpose: if a pipeline names
    # its first argument something else, a clear TypeError beats us guessing.
    call_kwargs = _filter_kwargs(
        pipe.encode_prompt, {"prompt": prompt, "device": device, **encode_kwargs}
    )
    outputs = pipe.encode_prompt(**call_kwargs)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    names = adapter.encode_prompt_outputs
    if len(outputs) != len(names):
        logger.warning(
            "%s.encode_prompt returned %d values but adapter %r names %d (%s) — "
            "mapping positionally as far as they go",
            type(pipe).__name__,
            len(outputs),
            adapter.key,
            len(names),
            names,
        )
    return {
        f"{prefix}{n}": v
        for n, v in zip(names, outputs, strict=False)
        if isinstance(v, torch.Tensor)
    }
