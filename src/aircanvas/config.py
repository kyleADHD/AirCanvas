"""Configuration dataclasses shared across sharding, streaming, and runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Compression = Literal["fp8", "nf4"] | None

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*(GB|GIB|MB|MIB)\s*$", re.IGNORECASE)


def parse_size(value: str | int) -> int:
    """Parse a human size like '6GB' into bytes. Ints pass through."""
    if isinstance(value, int):
        return value
    m = _SIZE_RE.match(value)
    if not m:
        raise ValueError(f"Unparseable size: {value!r} (expected e.g. '6GB', '512MB')")
    n, unit = float(m.group(1)), m.group(2).upper()
    scale = {"MB": 10**6, "MIB": 2**20, "GB": 10**9, "GIB": 2**30}[unit]
    return int(n * scale)


@dataclass(frozen=True)
class BudgetConfig:
    """Memory budgets. 'auto' values are resolved by utils.hw probes."""

    vram_bytes: int | Literal["auto"] = "auto"
    ram_bytes: int | Literal["auto"] = "auto"


@dataclass(frozen=True)
class StreamConfig:
    """Streaming-engine knobs; defaults chosen by the budget solver (residency.py)."""

    gpu_slots: int = 2  # reusable GPU weight buffers (double-buffer)
    ring_depth: int = 3  # pinned CPU prefetch buffers
    lookahead: int = 2  # blocks prefetched ahead of the schedule cursor
