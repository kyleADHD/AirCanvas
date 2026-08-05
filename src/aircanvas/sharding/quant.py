"""Shard compression, applied at split time (ARCHITECTURE.md ADR #2/#3).

- fp8 (M4, default): store ``float8_e4m3fn`` payload + a per-tensor fp32 scale,
  upcast to the compute dtype on GPU at load. Zero extra deps, ~2x disk-traffic
  cut. SKIP list: norms, AdaLN/modulation, embeddings (tiny and quality-
  sensitive), plus every 1-D parameter (biases/gains: <0.1% of the bytes, all
  of the sensitivity).
- nf4 (M5): bitsandbytes quantize_nf4 at split, dequantize_nf4 on GPU at load;
  quant state stored as sibling tensors (AirLLM's recipe). 4x traffic cut.
  bitsandbytes is imported lazily; clear error directing to
  ``pip install aircanvas[nf4]`` when missing.

**Storage format (fp8).** For a quantized tensor ``W`` we store two entries in
the block's safetensors file:

    "<name>"                 float8_e4m3fn, shape(W)      payload = W / scale
    "<name>.__ac_scale"      float32, shape ()            scale = amax / 448

Reconstruction is ``payload.to(compute_dtype) * scale`` — one dtype-converting
copy plus one in-place multiply, both expressible against pre-allocated
destination buffers (no allocation in the hot loop, CLAUDE.md hard rule; see
ADR #8 and streaming/prefetch.py).

The scale matters: ``float8_e4m3fn`` denormalises below 2^-6 (0.0156), and DiT
weight matrices routinely have amax well under that, so an unscaled cast (what
diffusers' layerwise casting does) throws away most of the mantissa for such
tensors. Scaling amax to 448 keeps every value in the normal range, where the
error bound is a flat 2^-4 relative half-ulp.

**Byte alignment (CRITICAL).** ``streaming.prefetch.ShardHeader`` builds
zero-copy views into a raw blob, which requires every tensor's data offset to
be a multiple of its element size. Mixed-dtype shards therefore have to be
written largest-element-size-first (4-byte scales, then 2-byte skip-list
tensors, then 1-byte payloads); since every tensor's byte length is a multiple
of its own element size, that ordering alone guarantees alignment with zero
padding. ``order_for_alignment`` implements it and the splitter verifies the
result with the reader's own parser before writing the ``.done`` marker.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from aircanvas.config import Compression

#: Parameter-name substrings never quantized (checked against the full tensor
#: name, case-insensitively).
QUANT_SKIP_SUBSTRINGS: tuple[str, ...] = ("norm", "modulation", "adaln", "embed")

#: Sibling-key suffix for an fp8 tensor's scale. Cannot collide with a real
#: parameter name (no torch module names contain "__ac_").
SCALE_SUFFIX = ".__ac_scale"

FP8_DTYPE = torch.float8_e4m3fn
#: Largest finite magnitude representable in float8_e4m3fn.
FP8_MAX = 448.0


class QuantError(RuntimeError):
    """A compression scheme is unsupported or its dependencies are missing."""


def scale_key(name: str) -> str:
    return name + SCALE_SUFFIX


def is_scale_key(name: str) -> bool:
    return name.endswith(SCALE_SUFFIX)


def payload_name(scale_name: str) -> str:
    return scale_name[: -len(SCALE_SUFFIX)]


def is_quantizable(name: str, t: torch.Tensor, scheme: Compression) -> bool:
    """True iff `t` should be stored compressed under `scheme`.

    Quantize 2-D+ floating-point weight matrices only, and only when the name
    carries none of QUANT_SKIP_SUBSTRINGS. Everything else (biases, 1-D gains,
    norms, modulation projections, embedding tables, non-float buffers) is
    stored verbatim in the compute dtype.
    """
    if scheme is None:
        return False
    if scheme not in ("fp8", "nf4"):
        raise QuantError(f"Unknown compression scheme {scheme!r} (expected 'fp8', 'nf4' or None)")
    if scheme == "nf4":
        raise QuantError("compression='nf4' lands in M5 (see docs/ROADMAP.md)")
    if not t.is_floating_point() or t.element_size() < 2 or t.dim() < 2:
        return False
    lname = name.lower()
    return not any(s in lname for s in QUANT_SKIP_SUBSTRINGS)


def compress_tensor(name: str, t: torch.Tensor, scheme: Compression) -> dict[str, torch.Tensor]:
    """Return the tensors to store for `t` (payload + any quant-state siblings)."""
    if not is_quantizable(name, t, scheme):
        return {name: t}
    # amax in fp32 regardless of the source dtype so the scale is exact.
    amax = t.detach().abs().max().to(torch.float32)
    scale = (amax / FP8_MAX) if float(amax) > 0.0 else torch.tensor(1.0, dtype=torch.float32)
    payload = (t.to(torch.float32) / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return {name: payload, scale_key(name): scale.reshape(())}


def compress_state_dict(
    tensors: Mapping[str, torch.Tensor], scheme: Compression
) -> dict[str, torch.Tensor]:
    """Compress a whole block's tensors, expanding quantized entries in place."""
    if scheme is None:
        return dict(tensors)
    out: dict[str, torch.Tensor] = {}
    for name, t in tensors.items():
        out.update(compress_tensor(name, t, scheme))
    return out


def decompress_state_dict(
    stored: Mapping[str, torch.Tensor],
    *,
    compression: Compression,
    compute_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Reconstruct compute-dtype tensors from stored form (allocating).

    Used on the engine's synchronous fallback path and when binding permanently
    resident blocks — NOT in the per-block hot loop, which goes through
    ``streaming.prefetch.DecompressPlan`` against pre-allocated buffers.

    With ``compression=None`` the mapping is returned unchanged (no copies, no
    casts) so the bitwise-equivalence gate stays exact.
    """
    if compression is None:
        return dict(stored)
    out: dict[str, torch.Tensor] = {}
    for name, t in stored.items():
        if is_scale_key(name):
            continue
        scale = stored.get(scale_key(name))
        if scale is None:
            out[name] = t
            continue
        value = t.to(compute_dtype)
        # 0-dim operand: type promotion keeps the result in `compute_dtype`.
        out[name] = value.mul_(scale.to(value.device).reshape(()))
    return out


def order_for_alignment(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reorder for a safetensors write so every data offset stays aligned.

    Descending element size, then name. Each tensor's byte length is a multiple
    of its own element size, so a descending-size layout never leaves a tensor
    starting at an offset that is not a multiple of its element size — which is
    exactly ``ShardHeader.aligned``'s requirement for zero-copy views.
    """
    return {k: tensors[k] for k in sorted(tensors, key=lambda n: (-tensors[n].element_size(), n))}


def compressed_dtype(stored_dtype: torch.dtype, compute_dtype: torch.dtype) -> torch.dtype:
    """The dtype a stored tensor materialises to on the compute device."""
    return compute_dtype if stored_dtype is FP8_DTYPE else stored_dtype
