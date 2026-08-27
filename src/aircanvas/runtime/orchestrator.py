"""Phase orchestration behind AirPipeline (M4).

Phase 1 ENCODE: per text encoder — load to GPU, run once, cache embeddings,
  evict (runtime/text_encoders.py). Optional pure-CPU TE mode. Embeddings for
  empty/negative prompts cached on disk.
Phase 2 DENOISE: DiT handed to StreamingEngine; the stock diffusers pipeline
  loop runs unmodified (we own placement only, ADR #6).
Phase 3 DECODE: VAE pulled onto the device for the single decode call and
  pushed back (runtime/vae.py).

Two placement details are worth calling out because they look like hacks and
are not:

- **`_execution_device` override (ADR #9).** diffusers derives the device it
  puts latents/timesteps on from the first component that has one. Our DiT is
  on `meta` and the TEs/VAE sit on the CPU between phases, so that derivation
  gives "cpu" and the whole run silently falls off the GPU. We therefore bind
  `_execution_device` to the device we planned for, via a throwaway subclass
  (the attribute is a read-only property, so an instance attribute cannot
  shadow it). AirLLM patches `.device` for exactly the same reason
  (RESEARCH.md §1).
- **Meta instantiation via the pipeline's own config.** The DiT is built with
  `accelerate.init_empty_weights(include_buffers=False)`: parameters land on
  meta (zero memory, the engine binds them from shards) while non-persistent
  buffers — rotary tables and friends, which are computed at __init__ and
  never appear in a checkpoint — are built for real and moved across.

A StreamingEngine is built and torn down per `generate()` rather than kept
alive, because the residency plan is re-solved per call (ADR #12) and a stale
plan would size the slot pool for the wrong resolution. The cost is re-reading
the resident blocks once per image (~0.6 s for a 2 GB resident set on NVMe,
well under 1% of a FLUX-class generation); if that ever matters, cache the
engine keyed on the plan rather than weakening the re-solve.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from aircanvas.adapters import ModelAdapter
from aircanvas.config import StreamConfig
from aircanvas.lora import LoraOverlay
from aircanvas.runtime.progress import RunObserver, notify
from aircanvas.runtime.text_encoders import EncodedPrompt, encode_and_evict, text_encoder_names
from aircanvas.runtime.vae import configure_vae, vae_on_demand
from aircanvas.sharding.manifest import Manifest
from aircanvas.streaming.engine import StreamingEngine
from aircanvas.streaming.residency import ResidencyPlan
from aircanvas.utils.memory import clean_memory

logger = logging.getLogger(__name__)


class OrchestrationError(RuntimeError):
    """The wrapped pipeline does not expose what phase-aware execution needs."""


@dataclass
class PhaseStats:
    """Wall time per phase of one generation, plus the engine's own breakdown."""

    encode_s: float = 0.0
    denoise_s: float = 0.0
    decode_s: float = 0.0
    total_s: float = 0.0
    steps: int = 0
    encode_cached: bool = False
    engine: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "encode_s": self.encode_s,
            "denoise_s": self.denoise_s,
            "decode_s": self.decode_s,
            "total_s": self.total_s,
            "steps": self.steps,
            "encode_cached": self.encode_cached,
            "engine": dict(self.engine),
        }


def meta_transformer(model_cls: type, config: Mapping[str, Any], device: torch.device) -> nn.Module:
    """Instantiate a diffusers DiT with meta parameters and real buffers."""
    try:
        from accelerate import init_empty_weights
    except ImportError as e:  # pragma: no cover - accelerate is a declared dep
        raise OrchestrationError(
            "accelerate is required for meta-device model instantiation (pip install accelerate)."
        ) from e

    with init_empty_weights(include_buffers=False):
        model = model_cls.from_config(config)
    model.eval()
    # Buffers were built for real on CPU; the engine only binds checkpoint
    # tensors, so anything not on `device` now would break the forward.
    for name, buf in list(model.named_buffers()):
        if not buf.is_meta and buf.device != device:
            _assign(model, name, buf.to(device))
    return model


def _with_step_callback(
    pipe: Any, call_kwargs: dict[str, Any], observer: RunObserver, steps: int
) -> dict[str, Any]:
    """Add a `callback_on_step_end` that reports each step to `observer`.

    diffusers' own per-step hook is the only honest source of step progress —
    the denoise loop is theirs (ADR #6), and counting block loads instead would
    guess wrong the moment a model streams a different number of blocks per
    step. A caller's own callback is chained, not replaced. Pipelines too old
    to accept the kwarg simply get no step events; every other telemetry
    channel still works.
    """
    import inspect

    try:
        params = inspect.signature(type(pipe).__call__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic __call__
        return call_kwargs
    if "callback_on_step_end" not in params:
        logger.debug("%s takes no callback_on_step_end; no step events", type(pipe).__name__)
        return call_kwargs

    chained = call_kwargs.get("callback_on_step_end")

    def _on_step(
        inner_pipe: Any, index: int, timestep: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        total = int(getattr(inner_pipe, "num_timesteps", 0) or steps)
        notify(observer, "step", index + 1, total)
        if callable(chained):
            result = chained(inner_pipe, index, timestep, kwargs)
            if isinstance(result, dict):
                return result
        return kwargs

    return {**call_kwargs, "callback_on_step_end": _on_step}


def _assign(root: nn.Module, full_name: str, value: torch.Tensor) -> None:
    *path, leaf = full_name.split(".")
    mod = root
    for part in path:
        mod = getattr(mod, part)
    if leaf in mod._buffers:
        mod._buffers[leaf] = value
    else:  # pragma: no cover - only buffers are reassigned here
        setattr(mod, leaf, value)


@contextlib.contextmanager
def force_execution_device(pipe: Any, device: torch.device) -> Iterator[None]:
    """Pin `pipe._execution_device` to `device` for the duration (ADR #9)."""
    original_cls = type(pipe)
    patched = type(
        f"AirCanvas{original_cls.__name__}",
        (original_cls,),
        {
            "_execution_device": property(lambda self: device),
            "__doc__": (
                f"{original_cls.__name__} with AirCanvas-pinned execution device. "
                f"Behaviour is identical; only device derivation is overridden."
            ),
        },
    )
    pipe.__class__ = patched
    try:
        yield
    finally:
        pipe.__class__ = original_cls


def arm_expert_handover(
    pipe: Any,
    extra_manifests: Mapping[str, tuple[Manifest, Path]],
    engines: dict[str, StreamingEngine],
    *,
    device: torch.device,
    config: StreamConfig | None = None,
    prefetch: bool = True,
    lora: LoraOverlay | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Lazy per-timestep expert switching (Wan 2.2's dual DiT, M7).

    The pipeline swaps from `transformer` (high-noise expert) to
    `transformer_2` (low-noise) at a boundary timestep it manages itself — we
    only decide where weights live (ADR #6). Building both engines up front
    would double pool VRAM for the whole run, so the successor's engine is
    built INSIDE its first forward pre-hook: every prior engine is closed
    first (weights, pools, residents all released), and the newcomer inherits
    the full budget. The boundary is one-way, so closed experts stay closed;
    the hook removes itself after firing.
    """
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    for name, (manifest, cache_dir) in extra_manifests.items():
        module = getattr(pipe, name, None)
        if not isinstance(module, nn.Module):
            logger.warning("extra manifest %r has no matching pipeline module — skipped", name)
            continue

        handle_box: dict[str, torch.utils.hooks.RemovableHandle] = {}

        def _handover(
            mod: nn.Module,
            args: tuple,
            kwargs: dict,
            _name: str = name,
            _manifest: Manifest = manifest,
            _cache: Path = cache_dir,
            _box: dict = handle_box,
            _lora: LoraOverlay | None = lora,
        ) -> None:
            _box["h"].remove()
            if _box["h"] in hooks:
                hooks.remove(_box["h"])
            for prior in list(engines.values()):
                prior.close()
            engines.clear()
            clean_memory()
            engines[_name] = StreamingEngine(
                mod,
                _manifest,
                _cache,
                device=device,
                config=config,
                prefetch=prefetch,
                resident_blocks=0,
                lora=_lora,
            )
            logger.info("Expert handover: %s now streams; prior engines released", _name)

        handle = module.register_forward_pre_hook(_handover, with_kwargs=True)
        handle_box["h"] = handle
        hooks.append(handle)
    return hooks


class Orchestrator:
    """Runs one stock pipeline in three explicitly-placed phases."""

    def __init__(
        self,
        pipe: Any,
        *,
        manifest: Manifest,
        cache_dir: Path,
        adapter: ModelAdapter,
        plan: ResidencyPlan,
        device: torch.device,
        embed_cache_dir: Path | None = None,
        prefetch: bool = True,
        text_encoder_device: torch.device | str | None = None,
        extra_manifests: dict[str, tuple[Manifest, Path]] | None = None,
        lora: LoraOverlay | None = None,
    ) -> None:
        if not hasattr(pipe, "encode_prompt"):
            raise OrchestrationError(
                f"{type(pipe).__name__} has no encode_prompt(); AirCanvas cannot run its text "
                f"encoders separately from the denoise loop."
            )
        self.pipe = pipe
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.adapter = adapter
        self.plan = plan
        self.device = device
        self.embed_cache_dir = embed_cache_dir
        self.prefetch = prefetch
        self.text_encoder_device = text_encoder_device
        self.extra_manifests = dict(extra_manifests or {})
        self.lora = lora
        self.stats = PhaseStats()
        # Engines of the generation currently in flight, so a UI can poll block
        # counters live (runtime/progress.py). Empty between runs.
        self.engines: dict[str, StreamingEngine] = {}
        self.vae_features = configure_vae(getattr(pipe, "vae", None))
        logger.info(
            "Orchestrator ready: %s, TEs=%s, VAE=%s",
            type(pipe).__name__,
            list(text_encoder_names(pipe)),
            self.vae_features or ["untiled"],
        )

    def generate(
        self,
        prompt: str | list[str],
        *,
        negative_prompt: str | list[str] | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
        observer: RunObserver | None = None,
        **call_kwargs: Any,
    ) -> Any:
        """Encode -> stream-denoise -> decode, with the phases timed."""
        stats = PhaseStats(steps=int(call_kwargs.get("num_inference_steps", 0) or 0))
        started = time.perf_counter()

        notify(observer, "phase_started", "encode")
        encoded: EncodedPrompt = encode_and_evict(
            self.pipe,
            self.adapter,
            device=self.device,
            prompt=prompt,
            negative_prompt=negative_prompt,
            encode_kwargs=encode_kwargs,
            cache_dir=self.embed_cache_dir,
            te_device=self.text_encoder_device,
        )
        stats.encode_s = encoded.seconds
        stats.encode_cached = encoded.cached
        notify(observer, "phase_finished", "encode", encoded.seconds)

        if observer is not None:
            call_kwargs = _with_step_callback(self.pipe, call_kwargs, observer, stats.steps)

        phase_times: dict[str, float] = {}
        transformer = self.pipe.transformer
        engine = StreamingEngine(
            transformer,
            self.manifest,
            self.cache_dir,
            device=self.device,
            config=self.plan.stream_config(),
            prefetch=self.prefetch,
            resident_blocks=self.plan.resident_blocks,
            ram_cache_bytes=self.plan.ram_cache_bytes,
            lora=self.lora,
        )
        engines: dict[str, StreamingEngine] = {"transformer": engine}
        self.engines = engines
        handover_hooks = arm_expert_handover(
            self.pipe,
            self.extra_manifests,
            engines,
            device=self.device,
            config=self.plan.stream_config(),
            prefetch=self.prefetch,
            lora=self.lora,
        )
        denoise_start = time.perf_counter()
        notify(observer, "phase_started", "denoise")
        try:
            with (
                force_execution_device(self.pipe, self.device),
                vae_on_demand(self.pipe, self.device, phase_times, observer=observer),
                torch.no_grad(),
            ):
                result = self.pipe(**encoded.kwargs, **call_kwargs)
        finally:
            for h in handover_hooks:
                h.remove()
            for eng in engines.values():
                eng.close()
            stats.engine = dict(engine.stats)
            for name, eng in engines.items():
                if name != "transformer":
                    stats.engine[f"{name}_block_loads"] = eng.stats["block_loads"]
                    stats.engine[f"{name}_bytes_loaded"] = eng.stats["bytes_loaded"]
            self.engines = {}
            clean_memory()

        stats.decode_s = phase_times.get("decode_s", 0.0)
        stats.denoise_s = max(0.0, time.perf_counter() - denoise_start - stats.decode_s)
        stats.total_s = time.perf_counter() - started
        self.stats = stats
        notify(observer, "phase_finished", "denoise", stats.denoise_s)
        logger.info(
            "Generated in %.2fs (encode %.2fs, denoise %.2fs, decode %.2fs)",
            stats.total_s,
            stats.encode_s,
            stats.denoise_s,
            stats.decode_s,
        )
        return result
