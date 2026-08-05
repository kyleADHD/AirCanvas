"""Adapter protocol + generic block discovery (M1).

The splitter works from checkpoint *tensor names* (no model instantiation
needed): any outermost `<list>.<idx>.` prefix whose indices run contiguously
0..N-1 with N >= MIN_BLOCKS is a streamable block list; everything else is
resident. Named adapters pin down execution order and expectations for the
flagship families; GenericAdapter is best-effort for unknown DiTs (execution
order is confirmed at runtime by the engine's schedule recording, M2).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MIN_BLOCKS = 4

# Outermost numbered segment: shortest dotted prefix followed by ".<int>.".
_BLOCK_RE = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*?)\.(\d+)\.")


class AdapterError(RuntimeError):
    """A checkpoint does not match what the adapter expects."""


@dataclass(frozen=True)
class BlockPlan:
    """How to shard one DiT, derived purely from tensor names."""

    block_lists: tuple[str, ...]  # list names in planned execution order
    block_names: tuple[str, ...]  # e.g. ("transformer_blocks.0", ...) flattened
    tensors_by_block: dict[str, tuple[str, ...]] = field(repr=False)
    resident_tensors: tuple[str, ...] = field(repr=False)

    @property
    def n_blocks(self) -> int:
        return len(self.block_names)


def _discover_block_lists(
    names: Sequence[str], known_lists: frozenset[str] = frozenset()
) -> dict[str, dict[int, list[str]]]:
    """Group tensor names by their outermost `<list>.<idx>.` prefix.

    Returns {list_name: {idx: [tensor names]}} in first-appearance order.
    The MIN_BLOCKS threshold only guards *generic* discovery against false
    positives; lists a named adapter declares in `known_lists` qualify at any
    length. Contiguous 0-based indices are required either way; the rest are
    left for the resident set.
    """
    found: dict[str, dict[int, list[str]]] = {}
    for name in names:
        m = _BLOCK_RE.match(name)
        if m:
            found.setdefault(m.group(1), {}).setdefault(int(m.group(2)), []).append(name)
    qualified: dict[str, dict[int, list[str]]] = {}
    for list_name, by_idx in found.items():
        long_enough = len(by_idx) >= MIN_BLOCKS or list_name in known_lists
        if long_enough and set(by_idx) == set(range(len(by_idx))):
            qualified[list_name] = by_idx
    return qualified


class ModelAdapter:
    """Base adapter. Subclasses set `key`, `model_classes`, and (optionally)
    `expected_block_lists` to pin execution order and validate structure.

    M4 adds the two runtime-facing answers from ARCHITECTURE.md §3.4:
    `encode_prompt_outputs` (question 2, component strategy — how a stock
    pipeline's `encode_prompt` return tuple maps onto its `__call__` kwargs, so
    the orchestrator can run text encoders once and evict them) and
    `token_count` (question 4, workload model — latent tokens as a function of
    resolution, which drives the budget solver's activation reserve).
    """

    key: str = "generic"
    model_classes: tuple[str, ...] = ()
    expected_block_lists: tuple[str, ...] | None = None

    #: Positional names of what `pipeline.encode_prompt(...)` returns. The
    #: orchestrator zips these onto the outputs and keeps whichever the
    #: pipeline's `__call__` actually accepts (FLUX's `text_ids`, for example,
    #: is recomputed internally and must be dropped).
    encode_prompt_outputs: tuple[str, ...] = ("prompt_embeds", "pooled_prompt_embeds")
    #: Text tokens added to the image token count (0 = unknown/none).
    text_tokens: int = 0
    #: Guidance-distilled models do not double the batch for CFG.
    guidance_distilled: bool = False
    #: Latent downscale per spatial axis: VAE stride x patch size.
    latent_stride: int = 16

    def block_plan(self, tensor_names: Sequence[str]) -> BlockPlan:
        discovered = _discover_block_lists(tensor_names, frozenset(self.expected_block_lists or ()))
        if not discovered:
            raise AdapterError(
                "No streamable block lists found (need a '<name>.<i>.' ModuleList "
                f"pattern with >= {MIN_BLOCKS} contiguous blocks)."
            )

        if self.expected_block_lists is not None:
            missing = [b for b in self.expected_block_lists if b not in discovered]
            if missing:
                raise AdapterError(
                    f"{self.key}: expected block list(s) {missing} not found in checkpoint "
                    f"(found: {sorted(discovered)})."
                )
            order = list(self.expected_block_lists)
            extras = [b for b in discovered if b not in order]
            if extras:
                logger.warning("%s: unexpected block lists %s appended to plan", self.key, extras)
                order += extras
        else:
            order = list(discovered)  # first-appearance order in the checkpoint

        block_names: list[str] = []
        tensors_by_block: dict[str, tuple[str, ...]] = {}
        claimed: set[str] = set()
        for list_name in order:
            by_idx = discovered[list_name]
            for idx in range(len(by_idx)):
                block = f"{list_name}.{idx}"
                block_names.append(block)
                tensors_by_block[block] = tuple(by_idx[idx])
                claimed.update(by_idx[idx])

        resident = tuple(n for n in tensor_names if n not in claimed)
        return BlockPlan(
            block_lists=tuple(order),
            block_names=tuple(block_names),
            tensors_by_block=tensors_by_block,
            resident_tensors=resident,
        )

    def token_count(self, width: int, height: int, frames: int = 1) -> int:
        """Latent token count for the budget solver's activation estimate.

        Generic model: (H / stride) x (W / stride) x frames image tokens plus
        the family's fixed text-token budget. Video adapters override for
        temporal compression (Wan: frames//4 + 1).
        """
        spatial = max(1, height // self.latent_stride) * max(1, width // self.latent_stride)
        return spatial * max(1, frames) + self.text_tokens


class GenericAdapter(ModelAdapter):
    """Best-effort adapter for unknown DiTs: pure name-based discovery."""


_ADAPTERS: list[type[ModelAdapter]] = []


def register(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    _ADAPTERS.append(cls)
    return cls


def resolve(model_class_name: str) -> ModelAdapter:
    """Map a diffusers model class name to its adapter; generic fallback."""
    for adapter_cls in _ADAPTERS:
        if model_class_name in adapter_cls.model_classes:
            return adapter_cls()
    logger.info("No named adapter for %s; using generic introspection", model_class_name)
    return GenericAdapter()
