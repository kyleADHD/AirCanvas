"""GGUF checkpoints as a split source (M9 phase 1).

A GGUF file (Q4_K/Q5_K/Q6_K/Q8_0/F16/...) is dequantized tensor-by-tensor at
split time and resharded into the existing codecs (none/fp8/nf4). The
streaming hot path is untouched: shards produced here are ordinary AirCanvas
shards, already covered by the bitwise equivalence gates. The win lives at
download time — a ~7 GB Q4_K file instead of a 24 GB bf16 checkpoint — which
is exactly where the low-disk/low-RAM audience hurts. Streaming native
K-quant payloads (GPU dequant in the prefetcher) is a later milestone.

Peak RAM on the direct path is one tensor: gguf-py readers are mmap-backed.
The diffusers fallback path (used when tensor names need a layout conversion,
e.g. BFL-style FLUX ggufs) holds the still-quantized checkpoint in RAM —
roughly the .gguf file size, which is the point of quantized sources.

GGUF stores shapes in reverse order; `ReaderTensor.data` is row-major and
`gguf.quants.dequantize` returns float32 in the row-major shape, so the torch
shape is simply ``reversed(tensor.shape)`` — verified empirically, and by the
round-trip tests in tests/test_gguf.py.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from aircanvas.adapters import BlockPlan

logger = logging.getLogger(__name__)

#: GGUF integer tensor types — stored verbatim, never dequantized or cast.
_INT_TYPE_NAMES = frozenset({"I8", "I16", "I32", "I64"})

#: A non-existent path like ``C:\models\x.gguf`` must not be parsed as
#: ``repo_id:filename`` (Windows is the primary dev machine).
_DRIVE_LETTER_RE = re.compile(r"^[A-Za-z]:[/\\]")


class GGUFError(RuntimeError):
    """Anything wrong with a GGUF source: dependency, path, layout, dtype."""


def _require_gguf() -> Any:
    try:
        import gguf
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise GGUFError(
            "Splitting from a GGUF checkpoint needs the 'gguf' package: pip install aircanvas[gguf]"
        ) from e
    return gguf


def resolve_gguf_path(
    gguf_file: str | Path, *, revision: str | None = None, hf_token: str | None = None
) -> Path:
    """Local .gguf path, or download 'repo_id:filename' from the Hub."""
    p = Path(gguf_file)
    if p.is_file():
        return p
    spec = str(gguf_file)
    if ":" in spec and not _DRIVE_LETTER_RE.match(spec):
        repo, fname = spec.split(":", 1)
        from huggingface_hub import hf_hub_download  # lazy: local paths need no hub

        return Path(hf_hub_download(repo, fname, revision=revision, token=hf_token))
    raise GGUFError(
        f"gguf_file {spec!r} is neither an existing file nor 'repo_id:filename' "
        "(e.g. 'city96/FLUX.1-dev-gguf:flux1-dev-Q4_K_S.gguf')"
    )


class TensorSource(Protocol):
    """What the splitter needs from any checkpoint source."""

    def tensor_names(self) -> list[str]: ...

    def materialized_bytes(self, dtype: torch.dtype | None) -> int: ...

    def gather(
        self, names: tuple[str, ...], dtype: torch.dtype | None
    ) -> dict[str, torch.Tensor]: ...


def plan_covers(plan: BlockPlan, names: list[str]) -> bool:
    """True when the adapter's plan actually recognized this tensor layout.

    A GGUF exported with foreign key names (e.g. BFL-style FLUX) produces a
    plan with no blocks, or one that strands most tensors — either means the
    names need conversion before splitting.
    """
    if plan.n_blocks == 0:
        return False
    covered: set[str] = set(plan.resident_tensors)
    for block in plan.block_names:
        covered.update(plan.tensors_by_block[block])
    return len(covered) >= (len(names) + 1) // 2


class GGUFCheckpoint:
    """Direct reader for a GGUF whose names already match the model layout."""

    def __init__(self, path: Path) -> None:
        gguf = _require_gguf()
        self.path = path
        self._gguf = gguf
        self._reader = gguf.GGUFReader(str(path))  # mmap-backed, lazy tensor data
        self._tensors = {t.name: t for t in self._reader.tensors}
        if not self._tensors:
            raise GGUFError(f"{path} contains no tensors")

    def tensor_names(self) -> list[str]:
        return list(self._tensors)

    def materialized_bytes(self, dtype: torch.dtype | None) -> int:
        """Bytes once every tensor is dequantized to the compute dtype."""
        itemsize = 4 if dtype is None else dtype.itemsize
        total = 0
        for t in self._tensors.values():
            n = int(np.prod([int(d) for d in t.shape])) if len(t.shape) else 1
            size = t.data.dtype.itemsize if t.tensor_type.name in _INT_TYPE_NAMES else itemsize
            total += n * size
        return total

    def gather(self, names: tuple[str, ...], dtype: torch.dtype | None) -> dict[str, torch.Tensor]:
        quants = self._gguf.quants
        out: dict[str, torch.Tensor] = {}
        for name in names:
            t = self._tensors[name]
            shape = tuple(int(d) for d in reversed(t.shape))
            if t.tensor_type.name in _INT_TYPE_NAMES:
                arr = t.data
            else:
                try:
                    arr = quants.dequantize(t.data, t.tensor_type)
                except Exception as e:
                    raise GGUFError(
                        f"{self.path.name}: cannot dequantize {name!r} "
                        f"({t.tensor_type.name}) — unsupported GGUF quant type"
                    ) from e
            # Unconditional copy: detaches from the file mmap (F32/F16 pass
            # through as views) and guarantees a writable, contiguous buffer
            # for torch.from_numpy. One tensor at a time — split-time only.
            arr = np.ascontiguousarray(arr).copy()
            if arr.size != int(np.prod(shape or (1,))):
                raise GGUFError(
                    f"{self.path.name}: {name!r} dequantized to {arr.size} elements "
                    f"but the header says shape {shape}"
                )
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).reshape(shape)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            out[name] = tensor
        return out


class DiffusersGGUFCheckpoint:
    """Layout-converting fallback: diffusers' single-file loader parses and
    renames foreign key layouts (e.g. BFL-style FLUX ggufs); we then
    dequantize its parameters one at a time."""

    def __init__(
        self,
        path: Path,
        model_class: str,
        config_source: str,
        subfolder: str,
        revision: str | None,
        hf_token: str | None,
        dtype: torch.dtype | None,
    ) -> None:
        import diffusers

        cls = getattr(diffusers, model_class, None)
        if cls is None or not hasattr(cls, "from_single_file"):
            raise GGUFError(
                f"{path.name}'s tensor names do not match {model_class}'s state dict, "
                f"and diffusers has no {model_class}.from_single_file to convert the "
                "layout. Re-export the GGUF with model-native tensor names."
            )
        from diffusers import GGUFQuantizationConfig

        compute = dtype or torch.float32
        logger.warning(
            "GGUF layout differs from %s — converting via diffusers.from_single_file "
            "(holds the quantized checkpoint in RAM, roughly the .gguf file size)",
            model_class,
        )
        kwargs: dict[str, Any] = {}
        if revision is not None:
            kwargs["revision"] = revision
        if hf_token is not None:
            kwargs["token"] = hf_token
        model = cls.from_single_file(
            str(path),
            config=config_source,
            subfolder=subfolder,
            quantization_config=GGUFQuantizationConfig(compute_dtype=compute),
            torch_dtype=compute,
            **kwargs,
        )
        self.path = path
        self._state: dict[str, torch.Tensor] = dict(model.state_dict())

    def tensor_names(self) -> list[str]:
        return list(self._state)

    def materialized_bytes(self, dtype: torch.dtype | None) -> int:
        from diffusers.quantizers.gguf.utils import GGUFParameter

        itemsize = 4 if dtype is None else dtype.itemsize
        total = 0
        for t in self._state.values():
            quant_type = getattr(t, "quant_type", None)  # set dynamically by diffusers
            if isinstance(t, GGUFParameter) and quant_type is not None:
                # Packed uint8 storage: recover the logical element count from
                # the quant type's (block_size, type_size) ratio.
                from gguf.constants import GGML_QUANT_SIZES

                block_size, type_size = GGML_QUANT_SIZES[quant_type]
                total += (t.numel() * block_size // type_size) * itemsize
            elif t.is_floating_point():
                total += t.numel() * itemsize
            else:
                total += t.numel() * t.element_size()
        return total

    def gather(self, names: tuple[str, ...], dtype: torch.dtype | None) -> dict[str, torch.Tensor]:
        from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor

        out: dict[str, torch.Tensor] = {}
        for name in names:
            t = self._state[name]
            if isinstance(t, GGUFParameter):
                t = dequantize_gguf_tensor(t)
            t = t.detach()
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            out[name] = t.contiguous()
        return out
