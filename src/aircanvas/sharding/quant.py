"""Shard compression, applied at split time (ARCHITECTURE.md ADR #2/#3).

- fp8 (M4, default): store float8_e4m3fn, upcast to compute dtype on GPU at
  load. Zero extra deps, 2x disk-traffic cut. SKIP list: norms, AdaLN/
  modulation, embeddings (tiny and quality-sensitive — stored bf16).
- nf4 (M5): bitsandbytes quantize_nf4 at split, dequantize_nf4 on GPU at load;
  quant state stored as sibling tensors (AirLLM's recipe). 4x traffic cut.
  bitsandbytes is imported lazily; clear error directing to `pip install
  aircanvas[nf4]` when missing.
"""

from __future__ import annotations

import torch

# Parameter-name substrings never quantized (checked against full tensor name).
QUANT_SKIP_SUBSTRINGS: tuple[str, ...] = ("norm", "modulation", "adaln", "embed")


def compress_tensor(name: str, t: torch.Tensor, scheme: str) -> dict[str, torch.Tensor]:
    """Return the tensors to store for `t` (payload + any quant-state siblings)."""
    raise NotImplementedError("M4 (fp8) / M5 (nf4)")


def decompress_on_gpu(
    name: str, stored: dict[str, torch.Tensor], scheme: str, compute_dtype: torch.dtype
) -> torch.Tensor:
    """Reconstruct the compute-dtype tensor on GPU from stored form."""
    raise NotImplementedError("M4 (fp8) / M5 (nf4)")
