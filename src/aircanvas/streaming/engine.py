"""StreamingEngine: hook-based just-in-time weight binding (M2).

Mechanism (ARCHITECTURE.md §3.2): the DiT is instantiated on the meta device
(zero memory); the engine binds resident tensors once, then registers forward
pre/post hooks on every block from the manifest. Pre-hook: load the block's
shard and bind its tensors into the module. Post-hook: replace them with meta
placeholders so the memory is released. The first forward RECORDS the block
execution schedule (ADR #7 — the denoise loop is deterministic); M3's
prefetcher consumes it.

M2 is synchronous and correctness-first: loads happen inline in the pre-hook.
The equivalence gate (tests/test_engine.py) requires streamed output to be
bitwise-equal to a fully-materialized reference. Assumes each block executes
at most once per forward (true for all target DiTs; weight-tied reuse would
need eviction to move to end-of-forward).

Known M3+ work, deliberately absent here: pinned ring, CUDA copy stream,
GPU slot pool, lookahead, quantized-shard decompression.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from aircanvas.sharding.manifest import BlockShard, Manifest

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
        validate: bool = True,
    ) -> None:
        self.model = model
        self.manifest = manifest
        self.cache_dir = Path(cache_dir)
        self.device = torch.device(device)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._loaded: dict[str, dict[str, torch.Tensor]] = {}
        self._validated: set[str] = set()
        self._schedule: list[str] = []
        self._recording = True
        self.stats = {"block_loads": 0, "bytes_loaded": 0}

        if not manifest.is_complete(self.cache_dir):
            raise StreamingError(
                f"Shard cache {self.cache_dir} is incomplete — re-run `aircanvas split`"
            )
        self._load_resident()
        if validate:
            self._validate_coverage()
        self._install_hooks()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Remove hooks. Streamed blocks stay meta; resident tensors stay bound."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> StreamingEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def schedule(self) -> tuple[str, ...]:
        """Block execution order recorded during the first forward."""
        return tuple(self._schedule)

    # -- setup -------------------------------------------------------------

    def _load_resident(self) -> None:
        tensors = load_file(self.cache_dir / self.manifest.resident_file, device=str(self.device))
        for name, t in tensors.items():
            _set_tensor(self.model, name, t)

    def _validate_coverage(self) -> None:
        """Every meta tensor left after resident binding must belong to a
        streamed block — anything else would silently run on garbage."""
        prefixes = tuple(b.name + "." for b in self.manifest.blocks)
        uncovered = [
            name
            for name, t in list(self.model.named_parameters()) + list(self.model.named_buffers())
            if t.is_meta and not name.startswith(prefixes)
        ]
        if uncovered:
            raise StreamingError(
                f"Model tensors not covered by the shard cache (resident or blocks): {uncovered}"
            )

    def _install_hooks(self) -> None:
        for shard in self.manifest.blocks:
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
        tensors = load_file(self.cache_dir / shard.file, device=str(self.device))
        if shard.name not in self._validated:
            self._validate_shard(shard, module, tensors)
        for name, t in tensors.items():
            _set_tensor(self.model, name, t)
        self._loaded[shard.name] = tensors
        self.stats["block_loads"] += 1
        self.stats["bytes_loaded"] += shard.n_bytes
        if self._recording:
            self._schedule.append(shard.name)
            if len(self._schedule) == len(self.manifest.blocks):
                self._recording = False
                logger.debug("Recorded schedule of %d blocks", len(self._schedule))

    def _post(
        self, shard: BlockShard, module: nn.Module, args: tuple, kwargs: dict, output: object
    ) -> None:
        tensors = self._loaded.pop(shard.name, None)
        if tensors is None:
            return
        for name, t in tensors.items():
            _evict_tensor(self.model, name, t)

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
