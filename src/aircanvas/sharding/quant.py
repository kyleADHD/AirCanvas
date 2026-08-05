"""Shard compression, applied at split time (ARCHITECTURE.md ADR #2/#3).

- fp8 (M4, default): store ``float8_e4m3fn`` payload + a per-tensor fp32 scale,
  upcast to the compute dtype on GPU at load. Zero extra deps, ~2x disk-traffic
  cut. SKIP list: norms, AdaLN/modulation, embeddings (tiny and quality-
  sensitive), plus every 1-D parameter (biases/gains: <0.1% of the bytes, all
  of the sensitivity).
- nf4 (M5): bitsandbytes ``quantize_4bit`` at split, ``dequantize_4bit`` on GPU
  at load; quant state stored as sibling tensors (AirLLM's recipe). ~3.5x
  traffic cut. bitsandbytes is imported lazily; clear error directing to
  ``pip install aircanvas[nf4]`` when missing. CUDA-only, at split AND at load.

**Storage format (nf4).** For a quantized tensor ``W`` (numel n) we store:

    "<name>"                    uint8, shape (n/2,)      two 4-bit codes/byte
    "<name>.__ac_nf4_absmax"    float32, shape (n/64,)   per-64-block absmax

plus the ORIGINAL shape in the safetensors ``__metadata__`` header under
``"nf4shape:<name>"`` (JSON list) — a packed payload cannot carry it, and the
header is available at ShardHeader.parse time with zero extra IO. The NF4 code
table is NOT stored: ``get_4bit_type("nf4")`` reproduces it exactly (verified),
and reconstruction with a rebuilt QuantState is bitwise-identical to
dequantizing with the state ``quantize_4bit`` returned.

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

import json
from collections.abc import Mapping
from typing import Any

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

#: Sibling-key suffix for an nf4 tensor's per-block absmax.
NF4_ABSMAX_SUFFIX = ".__ac_nf4_absmax"
#: ``__metadata__`` key prefix carrying an nf4 payload's original shape.
NF4_META_PREFIX = "nf4shape:"
NF4_BLOCKSIZE = 64


class QuantError(RuntimeError):
    """A compression scheme is unsupported or its dependencies are missing."""


def scale_key(name: str) -> str:
    return name + SCALE_SUFFIX


def is_scale_key(name: str) -> bool:
    return name.endswith(SCALE_SUFFIX)


def payload_name(scale_name: str) -> str:
    return scale_name[: -len(SCALE_SUFFIX)]


def nf4_absmax_key(name: str) -> str:
    return name + NF4_ABSMAX_SUFFIX


def is_nf4_absmax_key(name: str) -> bool:
    return name.endswith(NF4_ABSMAX_SUFFIX)


def is_sibling_key(name: str) -> bool:
    """True for any quant-state sibling (fp8 scale or nf4 absmax)."""
    return is_scale_key(name) or is_nf4_absmax_key(name)


def nf4_shape(metadata: Mapping[str, str] | None, name: str) -> tuple[int, ...] | None:
    """Original shape of an nf4 payload, from shard `__metadata__` (None if not nf4)."""
    if not metadata:
        return None
    raw = metadata.get(NF4_META_PREFIX + name)
    return None if raw is None else tuple(json.loads(raw))


def _require_bnb() -> Any:
    try:
        from bitsandbytes import functional as bnb_functional
    except ImportError as e:
        raise QuantError(
            "compression='nf4' needs bitsandbytes — pip install 'aircanvas[nf4]'"
        ) from e
    return bnb_functional


_NF4_CODE: dict[str, torch.Tensor] = {}


def _nf4_code(device: torch.device) -> torch.Tensor:
    """The 16-entry NF4 code table, cached per device (hot-loop rule: no
    repeated tiny allocations)."""
    key = str(device)
    code = _NF4_CODE.get(key)
    if code is None:
        code = _require_bnb().get_4bit_type("nf4", device=device)
        _NF4_CODE[key] = code
    return code


def nf4_dequantize(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    shape: tuple[int, ...],
    compute_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct a compute-dtype tensor from nf4 payload + absmax (CUDA only).

    With `out=` provided nothing but bnb's kernel launch happens — this is the
    hot-loop path (DecompressPlan.run). QuantState is rebuilt from parts; the
    result is bitwise-identical to dequantizing with the original state.
    """
    bnb = _require_bnb()
    state = bnb.QuantState(
        absmax=absmax,
        shape=torch.Size(shape),
        code=_nf4_code(packed.device),
        blocksize=NF4_BLOCKSIZE,
        quant_type="nf4",
        dtype=compute_dtype,
    )
    return bnb.dequantize_4bit(packed.reshape(-1, 1), state, quant_type="nf4", out=out)


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
    if not t.is_floating_point() or t.element_size() < 2 or t.dim() < 2:
        return False
    lname = name.lower()
    return not any(s in lname for s in QUANT_SKIP_SUBSTRINGS)


def compress_tensor(name: str, t: torch.Tensor, scheme: Compression) -> dict[str, torch.Tensor]:
    """Return the tensors to store for `t` under fp8 (payload + scale sibling).

    nf4 also produces per-tensor __metadata__ and goes through
    `compress_state_dict`, which returns it alongside the tensors.
    """
    if not is_quantizable(name, t, scheme):
        return {name: t}
    if scheme == "nf4":
        raise QuantError("nf4 carries shape metadata — compress via compress_state_dict()")
    # amax in fp32 regardless of the source dtype so the scale is exact.
    amax = t.detach().abs().max().to(torch.float32)
    scale = (amax / FP8_MAX) if float(amax) > 0.0 else torch.tensor(1.0, dtype=torch.float32)
    payload = (t.to(torch.float32) / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return {name: payload, scale_key(name): scale.reshape(())}


def _compress_nf4(name: str, t: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Quantize one tensor on CUDA (split time only), returning CPU tensors +
    the shape metadata entry."""
    if not torch.cuda.is_available():
        raise QuantError(
            "compression='nf4' requires CUDA at split time (bitsandbytes quantizes on GPU)"
        )
    bnb = _require_bnb()
    packed, state = bnb.quantize_4bit(
        t.detach().cuda(), blocksize=NF4_BLOCKSIZE, quant_type="nf4", compress_statistics=False
    )
    stored = {
        name: packed.reshape(-1).cpu(),
        nf4_absmax_key(name): state.absmax.to(torch.float32).cpu(),
    }
    return stored, {NF4_META_PREFIX + name: json.dumps(list(t.shape))}


def compress_state_dict(
    tensors: Mapping[str, torch.Tensor], scheme: Compression
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Compress a whole block's tensors.

    Returns (stored tensors, extra __metadata__ entries). fp8 and None produce
    no metadata; nf4 records each payload's original shape.
    """
    if scheme is None:
        return dict(tensors), {}
    out: dict[str, torch.Tensor] = {}
    meta: dict[str, str] = {}
    for name, t in tensors.items():
        if scheme == "nf4" and is_quantizable(name, t, scheme):
            stored, extra = _compress_nf4(name, t)
            out.update(stored)
            meta.update(extra)
        else:
            out.update(compress_tensor(name, t, scheme))
    return out, meta


def decompress_state_dict(
    stored: Mapping[str, torch.Tensor],
    *,
    compression: Compression,
    compute_dtype: torch.dtype,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Reconstruct compute-dtype tensors from stored form (allocating).

    Used on the engine's synchronous fallback path and when binding permanently
    resident blocks — NOT in the per-block hot loop, which goes through
    ``streaming.prefetch.DecompressPlan`` against pre-allocated buffers.

    With ``compression=None`` the mapping is returned unchanged (no copies, no
    casts) so the bitwise-equivalence gate stays exact. nf4 additionally needs
    the shard's ``__metadata__`` (original shapes) and CUDA-resident tensors.
    """
    if compression is None:
        return dict(stored)
    out: dict[str, torch.Tensor] = {}
    for name, t in stored.items():
        if is_sibling_key(name):
            continue
        shape = nf4_shape(metadata, name)
        if shape is not None:
            absmax = stored.get(nf4_absmax_key(name))
            if absmax is None:
                raise QuantError(f"nf4 payload {name!r} has no absmax sibling in the shard")
            out[name] = nf4_dequantize(t, absmax, shape, compute_dtype)
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


def out_spec(
    name: str,
    stored_dtype: torch.dtype,
    stored_shape: tuple[int, ...],
    metadata: Mapping[str, str] | None,
    compute_dtype: torch.dtype,
) -> tuple[torch.dtype, tuple[int, ...]] | None:
    """(dtype, shape) a stored entry materialises to, or None for quant-state
    siblings that never bind. The single source of truth shared by
    DecompressPlan.build and the splitter's byte accounting."""
    if is_sibling_key(name):
        return None
    shape = nf4_shape(metadata, name)
    if shape is not None:
        return compute_dtype, shape
    if stored_dtype is FP8_DTYPE:
        return compute_dtype, stored_shape
    return stored_dtype, stored_shape
