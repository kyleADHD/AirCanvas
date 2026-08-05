"""Configuration dataclasses shared across sharding, streaming, and runtime."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
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


def cache_root() -> Path:
    """Root of every AirCanvas on-disk artifact (shards, embeddings, probes).

    Deliberately under the HF cache home and NEVER inside the repo: the dev
    checkout is OneDrive-synced and shard caches are tens of GB (CLAUDE.md
    hard rule).
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "aircanvas"


def safe_slug(source: str) -> str:
    """Filesystem-safe directory name for a repo id or local path."""
    return re.sub(r"[^\w.\-]+", "--", str(source)).strip("-") or "model"


@dataclass(frozen=True)
class BudgetConfig:
    """Memory budgets. 'auto' values are resolved by utils.hw probes."""

    vram_bytes: int | Literal["auto"] = "auto"
    ram_bytes: int | Literal["auto"] = "auto"


@dataclass(frozen=True)
class StreamConfig:
    """Streaming-engine knobs; defaults chosen by the budget solver (residency.py)."""

    gpu_slots: int = 2  # reusable GPU weight buffers (double-buffer)
    ring_depth: int = 3  # pinned CPU buffers; also bounds how far IO reads ahead
    # Staging buffers holding the *compressed* blob before it is upcast into a
    # gpu slot. Only allocated when the shard cache is compressed; released as
    # soon as the upcast is enqueued, so double-buffering is enough (ADR #8).
    raw_slots: int = 2
