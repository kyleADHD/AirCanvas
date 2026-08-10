"""Public API: AirPipeline (M4).

Wraps a stock diffusers pipeline and owns ONLY device/memory placement
(ARCHITECTURE.md ADR #6): text encoders load-run-evict, DiT handed to the
StreamingEngine, VAE pulled in for decode only.

    pipe = AirPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                       vram_budget="6GB", compression="fp8")
    image = pipe("a still life", num_inference_steps=28).images[0]
    print(pipe.report())

`from_pretrained` does four things in order:

1. **Find or create the shard cache.** The DiT subfolder is downloaded (weights
   only) and split into per-block shards on first use; later runs reuse it.
   Nothing is ever written inside the repo (CLAUDE.md hard rule).
2. **Probe hardware and solve the budget** (utils/hw.py + streaming/residency.py)
   to get resident-block count, slot pool, and ring depth.
3. **Instantiate the DiT on meta** from the checkpoint's own config, and hand
   that instance to `DiffusionPipeline.from_pretrained` so diffusers loads
   every other component normally and never touches the DiT weights.
4. **Wrap it in an Orchestrator** that places each phase.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from aircanvas.adapters import ModelAdapter, resolve
from aircanvas.config import Compression, cache_root, parse_size, safe_slug
from aircanvas.runtime.orchestrator import Orchestrator, PhaseStats, meta_transformer
from aircanvas.sharding.manifest import Manifest
from aircanvas.sharding.splitter import shard_cache_dir, split_model
from aircanvas.streaming.residency import ResidencyPlan, Workload, solve
from aircanvas.utils.hw import HardwareProfile, free_ram_bytes, free_vram_bytes
from aircanvas.utils.memory import clean_memory

logger = logging.getLogger(__name__)

DEFAULT_COMPUTE_DTYPE = "bfloat16"


def _dtype_bytes(name: str) -> int:
    dtype = getattr(torch, name, None)
    return dtype.itemsize if isinstance(dtype, torch.dtype) else 2


def _require_diffusers() -> Any:
    try:
        import diffusers
    except ImportError as e:  # pragma: no cover - diffusers is a declared dep
        raise ImportError(
            "AirPipeline needs diffusers (pip install diffusers). The splitter and "
            "streaming engine work without it."
        ) from e
    return diffusers


def _extra_transformer_components(
    model_id: str, revision: str | None, token: str | None
) -> tuple[str, ...]:
    """Names of secondary transformer components in the pipeline's
    model_index.json (Wan 2.2 ships `transformer_2`). Empty on any failure —
    a missing index must not break single-transformer models."""
    try:
        local = Path(model_id) / "model_index.json"
        if local.is_file():
            text = local.read_text(encoding="utf-8")
        else:
            from huggingface_hub import hf_hub_download

            text = Path(
                hf_hub_download(model_id, "model_index.json", revision=revision, token=token)
            ).read_text(encoding="utf-8")
        index = json.loads(text)
    except Exception:  # noqa: BLE001 — best-effort probe
        return ()
    return tuple(
        name
        for name, spec in index.items()
        if name.startswith("transformer_") and isinstance(spec, list) and spec[-1]
    )


def _pipeline_allow_patterns(index: dict, passed: set[str]) -> list[str]:
    """Hub-download patterns for ONLY the component folders diffusers will
    actually load. Passing a component instance does NOT stop
    DiffusionPipeline.from_pretrained's snapshot fetch from downloading that
    component's folder — discovered when a freshly deleted 57 GB Wan 14B
    checkpoint started re-downloading underneath a benchmark."""
    patterns = ["model_index.json", "*.json"]
    for name, spec in index.items():
        if name.startswith("_") or not isinstance(spec, list) or not spec[-1]:
            continue
        if name in passed:
            continue  # caller supplied it (a meta instance, or None to skip)
        patterns.append(f"{name}/*")
    return patterns


def _local_pipeline_root(
    model_id: str, passed: set[str], revision: str | None, token: str | None
) -> str:
    """A local directory containing exactly the pipeline pieces we need, or
    `model_id` unchanged (local paths; any probe failure falls back to the
    stock download path)."""
    if Path(model_id).is_dir():
        return model_id
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        index_path = hf_hub_download(model_id, "model_index.json", revision=revision, token=token)
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        root = snapshot_download(
            model_id,
            allow_patterns=_pipeline_allow_patterns(index, passed),
            revision=revision,
            token=token,
        )
        return root
    except Exception as e:  # noqa: BLE001 — fall back to stock behaviour
        logger.warning("Selective pipeline download failed (%s); using stock path", e)
        return model_id


def _resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown compute dtype: {name!r}")
    return dtype


class AirPipeline:
    """Phase-aware wrapper around a diffusers DiffusionPipeline."""

    def __init__(
        self,
        pipeline: Any,
        *,
        manifest: Manifest,
        cache_dir: Path,
        adapter: ModelAdapter,
        plan: ResidencyPlan,
        hardware: HardwareProfile,
        device: torch.device,
        embed_cache_dir: Path | None = None,
        prefetch: bool = True,
        hidden_size: int = 0,
        workload: Workload | None = None,
        vram_budget: int | None = None,
        ram_budget: int | None = None,
        max_resident_blocks: int | None = None,
        text_encoder_device: torch.device | str | None = None,
        extra_manifests: dict[str, tuple[Manifest, Path]] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.adapter = adapter
        self.plan = plan
        self.hardware = hardware
        self.device = device
        self._hidden_size = hidden_size or Workload.hidden
        self._pinned_workload = workload
        self._vram_budget = vram_budget
        self._ram_budget = ram_budget
        self._max_resident_blocks = max_resident_blocks
        self._orchestrator = Orchestrator(
            pipeline,
            manifest=manifest,
            cache_dir=cache_dir,
            adapter=adapter,
            plan=plan,
            device=device,
            embed_cache_dir=embed_cache_dir,
            prefetch=prefetch,
            text_encoder_device=text_encoder_device,
            extra_manifests=extra_manifests,
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        vram_budget: str | int = "auto",
        ram_budget: str | int = "auto",
        compression: Compression = "fp8",
        shard_cache: str | Path | None = None,
        device: str | torch.device | None = None,
        compute_dtype: str | torch.dtype = DEFAULT_COMPUTE_DTYPE,
        subfolder: str = "transformer",
        revision: str | None = None,
        workload: Workload | None = None,
        max_resident_blocks: int | None = None,
        prefetch: bool = True,
        cache_embeddings: bool = True,
        hf_token: str | None = None,
        text_encoder_device: torch.device | str | None = None,
        **pipeline_kwargs: Any,
    ) -> AirPipeline:
        """Resolve budgets, find-or-create the shard cache, build the pipeline."""
        diffusers = _require_diffusers()
        dev = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = _resolve_dtype(compute_dtype)
        dtype_name = str(dtype).removeprefix("torch.")

        # 1. shard cache -------------------------------------------------
        cache_dir = (
            Path(shard_cache)
            if shard_cache is not None
            else shard_cache_dir(model_id, subfolder, compression, dtype_name)
        )
        manifest = split_model(
            model_id,
            cache_dir=cache_dir,
            compression=compression,
            subfolder=subfolder,
            revision=revision,
            compute_dtype=dtype_name,
            hf_token=hf_token,
        )
        adapter = resolve(manifest.model_class)

        # 2. hardware + budget ------------------------------------------
        sample = cache_dir / max(manifest.blocks, key=lambda b: b.n_bytes).file
        hardware = HardwareProfile.probe(dev, sample_file=sample, cache_dir=cache_dir)
        hidden = _hidden_size_hint(manifest, cache_dir, model_id, subfolder)
        vram_bytes = None if vram_budget == "auto" else parse_size(vram_budget)
        ram_bytes = None if ram_budget == "auto" else parse_size(ram_budget)
        plan = solve(
            manifest,
            hardware,
            workload or Workload(hidden=hidden, dtype_bytes=dtype.itemsize),
            vram_budget=vram_bytes,
            ram_budget=ram_bytes,
            max_resident_blocks=max_resident_blocks,
        )
        for warning in plan.warnings:
            logger.warning("%s", warning)

        # 3. meta DiT + stock pipeline ----------------------------------
        model_cls = getattr(diffusers, manifest.model_class, None)
        if model_cls is None:
            raise ImportError(
                f"diffusers {diffusers.__version__} has no {manifest.model_class}; "
                f"upgrade diffusers or split with a matching version."
            )
        config = model_cls.load_config(
            model_id, subfolder=subfolder, revision=revision, token=hf_token
        )
        transformer = meta_transformer(model_cls, config, dev)

        # Per-timestep expert models (Wan 2.2's transformer_2, M7): split each
        # extra transformer subfolder and hand a meta instance to the pipeline
        # so diffusers never loads its 28 GB into RAM. The successor's engine
        # is built lazily at the boundary (orchestrator.arm_expert_handover);
        # explicit shard_cache= applies to the PRIMARY only, extras always use
        # the default cache location.
        extra_manifests: dict[str, tuple[Manifest, Path]] = {}
        for extra_name in _extra_transformer_components(model_id, revision, hf_token):
            if extra_name in pipeline_kwargs:
                continue  # caller supplied it (e.g. =None to skip)
            extra_cache = shard_cache_dir(model_id, extra_name, compression, dtype_name)
            extra_manifest = split_model(
                model_id,
                cache_dir=extra_cache,
                compression=compression,
                subfolder=extra_name,
                revision=revision,
                compute_dtype=dtype_name,
                hf_token=hf_token,
            )
            extra_config = model_cls.load_config(
                model_id, subfolder=extra_name, revision=revision, token=hf_token
            )
            pipeline_kwargs[extra_name] = meta_transformer(model_cls, extra_config, dev)
            extra_manifests[extra_name] = (extra_manifest, extra_cache)

        pipeline_kwargs.setdefault("torch_dtype", dtype)
        passed = {"transformer", *pipeline_kwargs.keys()} - {"torch_dtype"}
        pipeline_root = _local_pipeline_root(model_id, passed, revision, hf_token)
        pipeline = diffusers.DiffusionPipeline.from_pretrained(
            pipeline_root,
            transformer=transformer,
            revision=revision if pipeline_root == model_id else None,
            token=hf_token,
            **pipeline_kwargs,
        )
        clean_memory()

        embed_cache = None
        if cache_embeddings:
            embed_cache = cache_root() / safe_slug(model_id) / "embeddings"
        return cls(
            pipeline,
            manifest=manifest,
            cache_dir=cache_dir,
            adapter=adapter,
            plan=plan,
            hardware=hardware,
            device=dev,
            embed_cache_dir=embed_cache,
            prefetch=prefetch,
            hidden_size=hidden,
            workload=workload,
            vram_budget=vram_bytes,
            ram_budget=ram_bytes,
            max_resident_blocks=max_resident_blocks,
            text_encoder_device=text_encoder_device,
            extra_manifests=extra_manifests,
        )

    # -- use ---------------------------------------------------------------

    def __call__(self, prompt: str | list[str], **kwargs: Any) -> Any:
        """Generate. Extra kwargs go straight to the wrapped pipeline call."""
        self._replan(kwargs)
        return self._orchestrator.generate(prompt, **kwargs)

    def _replan(self, call_kwargs: dict[str, Any]) -> None:
        """Re-solve the budget for the resolution/steps actually requested.

        ARCHITECTURE.md §3.2 lists steps and resolution as solver inputs, and
        they are only known at call time — a plan sized for 512x512 will happily
        OOM at 1024x1024. Free VRAM is re-read too, since whatever else is on
        the card between calls is not ours to spend. A `workload=` passed to
        `from_pretrained` pins the plan and disables this.
        """
        if self._pinned_workload is not None:
            return
        height = call_kwargs.get("height")
        width = call_kwargs.get("width")
        if height is None or width is None:
            return
        # Release the PREVIOUS generation's cached buffers (slot pools, upcast
        # pool, activations) before measuring free VRAM — the caching allocator
        # holds them until empty_cache, and measuring first reads ~3 GB of our
        # own garbage as "used", failing plans that would fit fine.
        clean_memory()
        frames = int(call_kwargs.get("num_frames", 1) or 1)
        workload = Workload(
            steps=int(call_kwargs.get("num_inference_steps", Workload.steps) or Workload.steps),
            tokens=self.adapter.token_count(int(width), int(height), frames),
            hidden=self._hidden_size,
            dtype_bytes=_dtype_bytes(self.manifest.compute_dtype),
        )
        hardware = replace(
            self.hardware,
            vram_bytes=free_vram_bytes(self.device) if self.device.type == "cuda" else 0,
            ram_bytes=free_ram_bytes(),
        )
        plan = solve(
            self.manifest,
            hardware,
            workload,
            vram_budget=self._vram_budget,
            ram_budget=self._ram_budget,
            max_resident_blocks=self._max_resident_blocks,
        )
        for warning in plan.warnings:
            logger.warning("%s", warning)
        self.plan = plan
        self.hardware = hardware
        self._orchestrator.plan = plan

    @property
    def stats(self) -> PhaseStats:
        return self._orchestrator.stats

    def report(self, *, as_dict: bool = False) -> Any:
        """Where the last run's time went, and the plan that produced it."""
        stats = self._orchestrator.stats
        if as_dict:
            return {
                "model": self.manifest.source,
                "model_class": self.manifest.model_class,
                "adapter": self.adapter.key,
                "compression": self.manifest.compression,
                "compute_dtype": self.manifest.compute_dtype,
                "device": str(self.device),
                "phases": stats.as_dict(),
                "plan": {
                    "resident_blocks": self.plan.resident_blocks,
                    "streamed_blocks": self.plan.streamed_blocks,
                    "gpu_slots": self.plan.gpu_slots,
                    "ring_depth": self.plan.ring_depth,
                    "lookahead": self.plan.lookahead,
                    "slot_bytes": self.plan.slot_bytes,
                    "pinned_bytes": self.plan.pinned_bytes,
                    "activation_bytes": self.plan.activation_bytes,
                    "ram_cache_bytes": self.plan.ram_cache_bytes,
                    "step_read_bytes": self.plan.step_read_bytes,
                    "warnings": list(self.plan.warnings),
                },
                "hardware": {
                    "device": self.hardware.device,
                    "gpu": self.hardware.gpu_name,
                    "vram_free_bytes": self.hardware.vram_bytes,
                    "ram_free_bytes": self.hardware.ram_bytes,
                    "disk_bw_bytes_s": self.hardware.disk_bw_bytes_s,
                },
            }

        engine = stats.engine
        io_line = ""
        if engine:
            io_line = (
                f"  block IO         {int(engine.get('block_loads', 0))} loads, "
                f"{engine.get('bytes_loaded', 0) / 1e9:.2f} GB "
                f"({int(engine.get('prefetch_hits', 0))} prefetched, "
                f"{engine.get('prefetch_wait_s', 0.0):.2f}s waiting; "
                f"{int(engine.get('sync_loads', 0))} sync, "
                f"{engine.get('sync_load_s', 0.0):.2f}s)\n"
            )
        return (
            f"AirCanvas {self.manifest.model_class} ({self.adapter.key}) "
            f"compression={self.manifest.compression} on {self.device}\n"
            f"phases (s)         encode {stats.encode_s:.2f}"
            f"{' [cached]' if stats.encode_cached else ''}  "
            f"denoise {stats.denoise_s:.2f}  decode {stats.decode_s:.2f}  "
            f"total {stats.total_s:.2f}\n"
            f"{io_line}"
            f"{self.plan.describe()}"
        )

    def close(self) -> None:
        """Drop references to the wrapped pipeline and free device memory."""
        self.pipeline = None
        self._orchestrator.pipe = None  # the orchestrator holds the other reference
        clean_memory()

    def __enter__(self) -> AirPipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _hidden_size_hint(manifest: Manifest, cache_dir: Path, model_id: str, subfolder: str) -> int:
    """Best-effort DiT hidden size for the activation estimate.

    Read from the shard cache rather than the model config so it costs nothing
    and works offline: the widest 2-D weight in the first block is the FFN
    projection, whose smaller dimension is the model dimension.
    """
    default = Workload.hidden
    try:
        from aircanvas.streaming.prefetch import ShardHeader

        header = ShardHeader.parse(cache_dir / manifest.blocks[0].file)
        dims = [min(m.shape) for m in header.tensors if len(m.shape) == 2]
        return max(dims) if dims else default
    except Exception as e:  # noqa: BLE001 - a hint, never a hard failure
        logger.debug("Could not infer hidden size for %s/%s: %s", model_id, subfolder, e)
        return default
