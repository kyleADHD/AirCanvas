"""Cross-platform memory hygiene (M2).

HARD RULE (CLAUDE.md): Windows is the primary dev machine. No libc
malloc_trim (AirLLM's trick is Linux-only). The slot-pool design should make
per-block cleanup unnecessary; this module is for phase BOUNDARIES only
(after TE evict, before VAE decode) — never in the per-block hot loop.
"""

from __future__ import annotations

import gc

import torch


def clean_memory() -> None:
    """Phase-boundary cleanup. Cheap, safe on all platforms."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
