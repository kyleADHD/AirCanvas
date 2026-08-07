"""StreamingEngine: hook-based just-in-time weight binding (M2 + M3 prefetch).

Mechanism (ARCHITECTURE.md §3.2): the DiT is instantiated on the meta device
(zero memory); the engine binds resident tensors once, then registers forward
pre/post hooks on every block from the manifest. Pre-hook: obtain the block's
weights and bind them into the module. Post-hook: replace them with meta
placeholders so the memory is released.

The FIRST forward loads synchronously and RECORDS the execution schedule
(ADR #7 — the denoise loop is deterministic). From the second forward on, a
Prefetcher (prefetch.py) walks that schedule ahead of execution: disk ->
pinned ring -> GPU slot buffers on a dedicated copy stream, so the pre-hook
usually just waits on a CUDA event and binds views. If execution diverges
from the schedule the engine logs a warning and falls back to synchronous
loads for the rest of the run — never wrong results, only slower.

M4 adds two things:

- **Compressed shard caches.** With `compression="fp8"` the prefetcher hands
  back already-upcast views (prefetch.DecompressPlan); the synchronous
  fallback path decompresses with `quant.decompress_state_dict`, which is
  allowed to allocate because it is explicitly not the hot loop.
- **Permanently resident blocks.** The budget solver (residency.py) spends
  leftover VRAM on the first R blocks of the manifest: they are bound once at
  construction, get no hooks, and never enter the prefetch schedule. R blocks
  resident means R fewer block-reads per step. Taking them off the *front* is
  deliberate — for two-species models like FLUX the front blocks are the big
  ones, so pinning them also shrinks the slot pool the streamed tail needs.

Correctness gate: streamed output must be bitwise-equal to a fully
materialized reference at `compression=None` (tests/test_engine.py,
tests/test_prefetch.py); within quant tolerance for fp8 (tests/test_quant.py).
Assumes each block executes at most once per forward (true for all target
DiTs; weight-tied reuse would need eviction moved to end-of-forward).
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from aircanvas.config import StreamConfig
from aircanvas.sharding import quant
from aircanvas.sharding.manifest import BlockShard, Manifest
from aircanvas.streaming.prefetch import (
    Prefetcher,
    PrefetchError,
    PrefetchMismatch,
    ReadyItem,
    ShardHeader,
)

logger = logging.getLogger(__name__)


class StreamingError(RuntimeError):
    """The model and the shard cache disagree, or the engine is misused."""


def _set_tensor(root: nn.Module, full_name: str, value: torch.Tensor) -> None:
    """Bind `value` as the parameter/buffer at `full_name`, replacing whatever
    is there (typically a meta placeholder)."""
    *path, leaf = full_name.split(".")
    mod = root
    for part in path:
        mod = getattr(mod, part)
    if leaf in mod._parameters:
        mod._parameters[leaf] = nn.Parameter(value, requires_grad=False)
    elif leaf in mod._buffers:
        mod._buffers[leaf] = value
    else:
        raise StreamingError(f"Shard tensor {full_name!r} is not a parameter or buffer")


def _evict_tensor(root: nn.Module, full_name: str, like: torch.Tensor) -> None:
    _set_tensor(root, full_name, torch.empty_like(like, device="meta"))


class StreamingEngine:
    """Streams manifest blocks through a meta-device model just-in-time.

    The engine owns ONLY weight residency for the given module; the caller
    owns model construction (on meta) and the forward/denoise loop.
    """

    def __init__(
        self,
        model: nn.Module,
        manifest: Manifest,
        cache_dir: Path | str,
        device: torch.device | str = "cuda",
        config: StreamConfig | None = None,
        prefetch: bool = True,
        validate: bool = True,
        resident_blocks: int = 0,
    ) -> None:
        self.model = model
        self.manifest = manifest
        self.cache_dir = Path(cache_dir)
        self.device = torch.device(device)
        self.config = config or StreamConfig()
        self._prefetch_enabled = prefetch
        self._prefetcher: Prefetcher | None = None
        self._prefetch_failed = False
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._loaded: dict[str, dict[str, torch.Tensor]] = {}
        self._active: dict[str, ReadyItem] = {}
        self._validated: set[str] = set()
        self._schedule: list[str] = []
        self._recording = True

        if not 0 <= resident_blocks <= len(manifest.blocks):
            raise StreamingError(
                f"resident_blocks={resident_blocks} out of range for a "
                f"{len(manifest.blocks)}-block manifest"
            )
        self._resident_blocks = tuple(manifest.blocks[:resident_blocks])
        self._streamed_blocks = tuple(manifest.blocks[resident_blocks:])
        self._compute_dtype: torch.dtype | None = None
        if manifest.compression is not None:
            dtype = manifest.torch_compute_dtype()
            assert isinstance(dtype, torch.dtype)
            self._compute_dtype = dtype
        if manifest.compression == "nf4" and self.device.type != "cuda":
            raise StreamingError(
                "compression='nf4' requires a CUDA device (bitsandbytes dequantizes on GPU); "
                "re-split with compression='fp8' or None for CPU use"
            )

        self.stats: dict[str, float] = {
            "block_loads": 0,
            "bytes_loaded": 0,
            "sync_loads": 0,
            "prefetch_hits": 0,
            "sync_load_s": 0.0,
            "prefetch_wait_s": 0.0,
            "resident_blocks": len(self._resident_blocks),
            "resident_block_bytes": sum(b.materialized_bytes for b in self._resident_blocks),
            "slot_bytes": 0,
            "pinned_bytes": 0,
        }

        if not manifest.is_complete(self.cache_dir):
            raise StreamingError(
                f"Shard cache {self.cache_dir} is incomplete — re-run `aircanvas split`"
            )
        self._load_resident()
        # transformers models tie embeddings (UMT5: encoder.embed_tokens ->
        # shared). Binding `shared.weight` replaces the Parameter object, which
        # silently breaks the tie — re-tie so the alias points at the bound
        # tensor instead of a meta leftover (AirLLM's exact lesson).
        if callable(getattr(self.model, "tie_weights", None)):
            self.model.tie_weights()
        self._bind_resident_blocks()
        self._adopt_computed_buffers()
        if validate:
            self._validate_coverage()
        self._install_hooks()

    # -- lifecycle ---------------------------------------------------------

    def close(self, release_weights: bool = True) -> None:
        """Remove hooks, stop the prefetcher, and (by default) evict EVERY
        bound weight — residents included — back to meta.

        A new engine re-binds residents from disk in roughly one resident-set
        read (~0.6 s), while keeping them bound leaks the whole resident set
        across engines: the orchestrator builds a fresh engine per generation,
        so call 2 would see call 1's ~2-3 GB still on the card and plan
        itself right into InsufficientVRAMError (found the hard way on the
        first Qwen-Image 20B run). Pass release_weights=False only when
        reusing THIS engine's model for another engine-free forward.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self._prefetcher is not None:
            self._prefetcher.close()
            self._prefetcher = None
        # Drop buffer references so the caller's clean_memory() can actually
        # return slot/ring memory to the driver (close -> gc -> empty_cache).
        self._loaded.clear()
        self._active.clear()
        if release_weights:
            for name, t in list(self.model.named_parameters()) + list(self.model.named_buffers()):
                if not t.is_meta:
                    _evict_tensor(self.model, name, t)

    def __enter__(self) -> StreamingEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def schedule(self) -> tuple[str, ...]:
        """Streamed-block execution order recorded during the first forward."""
        return tuple(self._schedule)

    @property
    def resident_block_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self._resident_blocks)

    def report(self) -> str:
        """Human-readable stats: where block-weight time went."""
        s = self.stats
        lines = [
            f"blocks loaded      {int(s['block_loads'])}  ({s['bytes_loaded'] / 1e9:.2f} GB)",
            f"  prefetched       {int(s['prefetch_hits'])}  (wait {s['prefetch_wait_s']:.2f}s)",
            f"  synchronous      {int(s['sync_loads'])}  (load {s['sync_load_s']:.2f}s)",
        ]
        if s["resident_blocks"]:
            lines.append(
                f"blocks resident    {int(s['resident_blocks'])}  "
                f"({s['resident_block_bytes'] / 1e9:.2f} GB never re-read)"
            )
        if s["slot_bytes"]:
            lines.append(
                f"slot pool          {s['slot_bytes'] / 1e9:.2f} GB device, "
                f"{s['pinned_bytes'] / 1e9:.2f} GB pinned host"
            )
        return "\n".join(lines)

    # -- setup -------------------------------------------------------------

    def _materialize(self, path: Path) -> dict[str, torch.Tensor]:
        """Load one shard file straight onto the device, decompressing if the
        cache is compressed. Allocates — never called from the hot loop."""
        raw = load_file(path, device=str(self.device))
        if self.manifest.compression is None:
            return raw
        assert self._compute_dtype is not None
        metadata = ShardHeader.parse(path).metadata if self.manifest.compression == "nf4" else None
        return quant.decompress_state_dict(
            raw,
            compression=self.manifest.compression,
            compute_dtype=self._compute_dtype,
            metadata=metadata,
        )

    def _load_resident(self) -> None:
        for name, t in self._materialize(self.cache_dir / self.manifest.resident_file).items():
            _set_tensor(self.model, name, t)

    def _bind_resident_blocks(self) -> None:
        """Pin the budget solver's chosen blocks on the device for the whole run."""
        for shard in self._resident_blocks:
            for name, t in self._materialize(self.cache_dir / shard.file).items():
                _set_tensor(self.model, name, t)
        if self._resident_blocks:
            logger.info(
                "Pinned %d block(s) resident on %s (%.2f GB)",
                len(self._resident_blocks),
                self.device,
                self.stats["resident_block_bytes"] / 1e9,
            )

    def _validate_coverage(self) -> None:
        """Every meta tensor left after resident binding must belong to a
        streamed block — anything else would silently run on garbage."""
        prefixes = tuple(b.name + "." for b in self._streamed_blocks)
        uncovered = [
            name
            for name, t in list(self.model.named_parameters()) + list(self.model.named_buffers())
            if t.is_meta and not (prefixes and name.startswith(prefixes))
        ]
        if uncovered:
            raise StreamingError(
                f"Model tensors not covered by the shard cache (resident or blocks): {uncovered}"
            )

    def _adopt_computed_buffers(self) -> None:
        """Move real-but-misplaced tensors to the engine device.

        `init_empty_weights(include_buffers=False)` computes NON-PERSISTENT
        buffers for real at init — they are absent from checkpoints, so
        nothing ever binds them, and they sit wherever meta-init ran (CPU).
        Wan's rope.freqs_cos/freqs_sin are the canonical case: FLUX/Qwen
        derive rope from the input's device per forward and never trip this
        (RESEARCH.md §1 — AirLLM hit the identical issue with inv_freq).
        """
        for name, t in list(self.model.named_buffers()) + list(self.model.named_parameters()):
            if not t.is_meta and t.device != self.device:
                _set_tensor(self.model, name, t.to(self.device))

    def _install_hooks(self) -> None:
        for shard in self._streamed_blocks:
            try:
                module = self.model.get_submodule(shard.name)
            except AttributeError as e:
                raise StreamingError(
                    f"Manifest block {shard.name!r} does not exist in the model "
                    f"(wrong model class or subfolder?)"
                ) from e
            self._hooks.append(
                module.register_forward_pre_hook(partial(self._pre, shard), with_kwargs=True)
            )
            self._hooks.append(
                module.register_forward_hook(partial(self._post, shard), with_kwargs=True)
            )

    # -- hot path ----------------------------------------------------------

    def _pre(self, shard: BlockShard, module: nn.Module, args: tuple, kwargs: dict) -> None:
        tensors: dict[str, torch.Tensor] | None = None

        if self._prefetcher is not None:
            t0 = time.perf_counter()
            try:
                item = self._prefetcher.get(shard.name)
                self.stats["prefetch_wait_s"] += time.perf_counter() - t0
                self.stats["prefetch_hits"] += 1
                self._active[shard.name] = item
                tensors = item.views
            except PrefetchMismatch as e:
                logger.warning(
                    "Prefetch schedule diverged (%s) — synchronous loads from here on", e
                )
                self._prefetcher.close()
                self._prefetcher = None
                self._prefetch_failed = True

        if tensors is None:
            t0 = time.perf_counter()
            tensors = self._materialize(self.cache_dir / shard.file)
            self.stats["sync_load_s"] += time.perf_counter() - t0
            self.stats["sync_loads"] += 1

        if shard.name not in self._validated:
            self._validate_shard(shard, module, tensors)
        for name, t in tensors.items():
            _set_tensor(self.model, name, t)
        self._loaded[shard.name] = tensors
        self.stats["block_loads"] += 1
        self.stats["bytes_loaded"] += shard.n_bytes

        if self._recording:
            self._schedule.append(shard.name)
            if len(self._schedule) == len(self._streamed_blocks):
                self._recording = False
                logger.debug("Recorded schedule of %d blocks", len(self._schedule))
        elif self._prefetch_enabled and self._prefetcher is None and not self._prefetch_failed:
            self._arm_prefetcher(shard.name)

    def _post(
        self, shard: BlockShard, module: nn.Module, args: tuple, kwargs: dict, output: object
    ) -> None:
        tensors = self._loaded.pop(shard.name, None)
        if tensors is None:
            return
        for name, t in tensors.items():
            _evict_tensor(self.model, name, t)
        item = self._active.pop(shard.name, None)
        if item is not None and self._prefetcher is not None:
            self._prefetcher.release(item)

    def _arm_prefetcher(self, current_block: str) -> None:
        """Called at the first block after recording finished: this block was
        just loaded synchronously; the worker starts at the next one."""
        try:
            start = (self._schedule.index(current_block) + 1) % len(self._schedule)
            self._prefetcher = Prefetcher(
                self.manifest,
                self.cache_dir,
                tuple(self._schedule),
                start_index=start,
                device=self.device,
                config=self.config,
            )
            self.stats["slot_bytes"] = self._prefetcher.slot_bytes
            self.stats["pinned_bytes"] = self._prefetcher.pinned_bytes
            logger.info(
                "Prefetcher armed: %d-deep pinned ring, %d device slots (%.2f GB)",
                self.config.ring_depth,
                self.config.gpu_slots,
                self._prefetcher.slot_bytes / 1e9,
            )
        except PrefetchError as e:
            logger.warning("Prefetch unavailable (%s) — using synchronous loads", e)
            self._prefetch_failed = True

    def _validate_shard(
        self, shard: BlockShard, module: nn.Module, tensors: dict[str, torch.Tensor]
    ) -> None:
        expected = {name for name, _ in module.named_parameters(prefix=shard.name)} | {
            name for name, _ in module.named_buffers(prefix=shard.name)
        }
        missing = expected - tensors.keys()
        extra = tensors.keys() - expected
        if missing or extra:
            raise StreamingError(
                f"Shard {shard.file} does not match module {shard.name!r}: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        self._validated.add(shard.name)
