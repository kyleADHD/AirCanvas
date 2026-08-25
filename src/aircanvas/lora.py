"""LoRA loading onto streamed (and quantized) DiT shards (M10).

Architecture (ARCHITECTURE.md §3.2, ADR #2): shards stay quantized on disk.
LoRA is fused **after dequant**, in-place, into the compute-dtype views the
engine is about to bind:

    W.addmm_(up, down, alpha=user_scale * alpha / rank)

That is the only composition that works with fp8/NF4 shards: the adapter was
trained against the original bf16/fp16 weights, so it has to meet them after
the codec has undone itself. Because fusion is a GEMM into an already-owned
buffer, the no-allocation-in-the-hot-loop rule (CONTRIBUTING.md) holds —
``down``/``up`` are moved onto the engine device once at engine construction
(``LoraOverlay.materialize``), never in the per-block path.

We do **not** inject PEFT modules. The DiT stays a stock meta-device model
the shard cache already matches (ADR #6: we wrap diffusers, we don't
reimplement it). Text-encoder LoRAs are detected and skipped with a count:
TEs are load-run-evict and not quantized-sharded.

Accepted on-disk formats (all ``.safetensors``):

- PEFT / diffusers: ``{module}.lora_A.weight`` + ``{module}.lora_B.weight``
  (optional ``.default`` adapter infix).
- Kohya / sd-scripts: ``lora_unet_{underscored}.lora_down.weight`` +
  ``lora_up.weight`` + optional ``.alpha``. Underscores are rewritten to dots
  after the prefix is stripped, which is enough for any DiT whose module
  names already match (FLUX kohya packed-QKV layouts go through diffusers'
  converter when it is importable).

Sources: a local file, a directory of safetensors, or a Hub repo id.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

#: Default Hub filenames, tried in order when ``weight_name`` is omitted.
_DEFAULT_WEIGHT_NAMES: tuple[str, ...] = (
    "pytorch_lora_weights.safetensors",
    "diffusion_pytorch_model.safetensors",
)

#: Component prefixes we strip when mapping a LoRA key onto a DiT tensor name.
#: Longest first so ``base_model.model.`` wins over ``model.``.
_STRIP_PREFIXES: tuple[str, ...] = (
    "model.diffusion_model.",
    "base_model.model.",
    "diffusion_model.",
    "base_model.",
    "transformer.",
    "unet.",
    "model.",
)

#: Kohya's UNet/transformer prefixes. Text-encoder prefixes are listed
#: separately so we can skip them rather than silently mis-bind them.
_KOHYA_DIT_PREFIXES: tuple[str, ...] = ("lora_unet_", "lora_transformer_", "lora_dit_")
_TE_KEY_MARKERS: tuple[str, ...] = (
    "text_encoder_2.",
    "text_encoder.",
    "lora_te2_",
    "lora_te1_",
    "lora_te_",
    "clip_l.",
    "clip_g.",
    "t5xxl.",
)

#: Pair-key: everything before the LoRA suffix is the module path.
_PAIR_RE = re.compile(
    r"^(?P<mod>.*?)(?:\.lora_(?P<ab>A|B|down|up)|\.lora\.(?P<dot>down|up))"
    r"(?:\.(?P<adapter>\w+))?\.weight$"
)
_ALPHA_RE = re.compile(r"^(?P<mod>.*?)(?:\.alpha|\.lora_alpha)$")


class LoraError(RuntimeError):
    """A LoRA file can't be read, parsed, or applied."""


def _is_te_key(key: str) -> bool:
    lowered = key.lower()
    return any(m in lowered for m in _TE_KEY_MARKERS)


def _norm_ab(kind: str) -> str:
    return "A" if kind in {"A", "down"} else "B"


def _peel_dit_prefix(mod: str) -> str | None:
    """Strip one leading DiT/UNet prefix. Kohya prefixes also un-underscore."""
    for lead in _KOHYA_DIT_PREFIXES:
        if mod.startswith(lead):
            return mod[len(lead) :].replace("_", ".")
    for lead in _STRIP_PREFIXES:
        if mod.startswith(lead):
            return mod[len(lead) :]
    return None


def _target_names(mod: str) -> tuple[str, ...]:
    """Candidate DiT parameter names this LoRA module should patch.

    Prefixes are peeled iteratively (``transformer.blocks.0`` and
    ``base_model.model.transformer.blocks.0`` both become ``blocks.0``) so a
    single delta is stored under every plausible shard-tensor name.
    """
    candidates = [mod]
    current = mod
    while True:
        peeled = _peel_dit_prefix(current)
        if peeled is None or peeled == current or not peeled:
            break
        candidates.append(peeled)
        current = peeled
    names = [base if base.endswith(".weight") else base + ".weight" for base in candidates]
    return tuple(dict.fromkeys(names))


def _maybe_convert_with_diffusers(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Best-effort FLUX/SD3 kohya → PEFT conversion. Identity on failure.

    Packed-QKV kohya FLUX LoRAs cannot be mapped by underscore rewriting;
    diffusers already has the layout table. Missing converters (older
    diffusers, non-FLUX files) must not fail the load.
    """
    keys = list(state)
    looks_kohya = any(k.startswith(_KOHYA_DIT_PREFIXES) or "lora_down" in k for k in keys)
    looks_flux = any("double_blocks" in k or "single_blocks" in k for k in keys)
    if not (looks_kohya and looks_flux):
        return state
    try:
        from diffusers.loaders.lora_conversion_utils import (  # type: ignore[import-untyped]
            _convert_kohya_flux_lora_to_diffusers,
        )
    except (ImportError, AttributeError):
        logger.info("diffusers kohya-FLUX converter unavailable; using generic key rewrite")
        return state
    try:
        converted = _convert_kohya_flux_lora_to_diffusers(state)
    except Exception as e:  # noqa: BLE001 — converter is best-effort
        logger.warning("diffusers kohya-FLUX conversion failed (%s); using generic rewrite", e)
        return state
    if not isinstance(converted, dict) or not converted:
        return state
    logger.info("Converted %d kohya-FLUX LoRA keys via diffusers", len(converted))
    return converted


def parse_lora_state_dict(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, LoRADelta], int, int]:
    """Turn a raw LoRA state dict into ``{target_weight_name: LoRADelta}``.

    Returns ``(deltas, n_bound, n_skipped_te)``. Keys that look like
    text-encoder LoRAs are counted and dropped. Unknown / unpaired keys are
    logged, not raised — a CivitAI dump with extra lycoris tensors should
    still apply the standard LoRA pairs it does contain.
    """
    state = _maybe_convert_with_diffusers(dict(state))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    alphas: dict[str, float] = {}
    skipped_te = 0
    unused = 0
    for key, tensor in state.items():
        if _is_te_key(key):
            skipped_te += 1
            continue
        am = _ALPHA_RE.match(key)
        if am is not None:
            alphas[am.group("mod")] = float(tensor.reshape(()).item())
            continue
        pm = _PAIR_RE.match(key)
        if pm is None:
            unused += 1
            continue
        kind = pm.group("ab") or pm.group("dot")
        pairs.setdefault(pm.group("mod"), {})[_norm_ab(kind)] = tensor

    deltas: dict[str, LoRADelta] = {}
    unpaired = 0
    for mod, ab in pairs.items():
        a, b = ab.get("A"), ab.get("B")
        if a is None or b is None:
            unpaired += 1
            continue
        a, b = _orient(a, b)
        if a is None or b is None:
            unpaired += 1
            continue
        rank = int(a.shape[0])
        alpha = alphas.get(mod, float(rank))
        delta = LoRADelta(down=a.contiguous(), up=b.contiguous(), alpha=alpha, rank=rank)
        for name in _target_names(mod):
            deltas[name] = delta

    if unused:
        logger.warning("LoRA: ignored %d unrecognized tensor(s)", unused)
    if unpaired:
        logger.warning("LoRA: dropped %d unpaired A/B module(s)", unpaired)
    if skipped_te:
        logger.info(
            "LoRA: skipped %d text-encoder tensor(s) (TEs are load-run-evict, not sharded)",
            skipped_te,
        )
    if not deltas:
        raise LoraError(
            "No DiT LoRA pairs found in the state dict "
            "(expected *.lora_A/lora_B.weight or kohya lora_unet_*.lora_down/up.weight)"
        )
    logger.info("LoRA: %d DiT target(s), %d TE tensor(s) skipped", len(deltas), skipped_te)
    return deltas, len(deltas), skipped_te


def _orient(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Return ``(down [rank, in], up [out, rank])``, or ``(None, None)``.

    Only 2-D Linear LoRAs are in scope (v1 DiT targets). Conv / DoRA / LoHa
    layouts are refused rather than silently mis-applied.
    """
    if a.ndim != 2 or b.ndim != 2:
        return None, None
    # Standard PEFT/kohya: A [rank, in], B [out, rank] — inner dim agrees.
    if a.shape[0] == b.shape[1]:
        return a, b
    if a.shape[1] == b.shape[0]:
        return b, a
    return None, None


def resolve_lora_path(
    source: str | Path,
    *,
    weight_name: str | None = None,
    revision: str | None = None,
    hf_token: str | None = None,
) -> Path:
    """Local ``.safetensors`` / directory, or a Hub repo id."""
    p = Path(source)
    if p.is_file():
        if p.suffix != ".safetensors":
            raise LoraError(f"LoRA file must be .safetensors (got {p.name})")
        return p
    if p.is_dir():
        files = sorted(p.glob("*.safetensors"))
        if weight_name:
            files = [f for f in files if f.name == weight_name]
        if not files:
            raise LoraError(f"No .safetensors LoRA in directory {p}")
        if len(files) > 1 and weight_name is None:
            preview = ", ".join(f.name for f in files[:8])
            raise LoraError(
                f"Directory {p} has {len(files)} .safetensors files; "
                f"pass weight_name= to pick one ({preview})"
            )
        return files[0]

    spec = str(source)
    from huggingface_hub import hf_hub_download, list_repo_files

    names: tuple[str, ...]
    if weight_name:
        names = (weight_name,)
    else:
        try:
            repo_files = list_repo_files(spec, revision=revision, token=hf_token)
        except Exception as e:  # noqa: BLE001
            raise LoraError(
                f"{spec!r} is not a local LoRA file/dir and Hub listing failed: {e}"
            ) from e
        safes = [f for f in repo_files if f.endswith(".safetensors")]
        preferred = [n for n in _DEFAULT_WEIGHT_NAMES if n in safes]
        names = tuple(preferred or safes)
        if not names:
            raise LoraError(f"Hub repo {spec!r} has no .safetensors LoRA files")
    last_err: Exception | None = None
    for name in names:
        try:
            return Path(hf_hub_download(spec, name, revision=revision, token=hf_token))
        except Exception as e:  # noqa: BLE001 — try the next default name
            last_err = e
    raise LoraError(f"Could not download LoRA from {spec!r}: {last_err}") from last_err


def load_lora_deltas(
    source: str | Path,
    *,
    weight_name: str | None = None,
    revision: str | None = None,
    hf_token: str | None = None,
) -> dict[str, LoRADelta]:
    """Load + parse a LoRA, returning the target-name → delta map."""
    path = resolve_lora_path(source, weight_name=weight_name, revision=revision, hf_token=hf_token)
    deltas, _, _ = parse_lora_state_dict(load_file(str(path)))
    return deltas


@dataclass
class LoRADelta:
    """One Linear LoRA pair. ``scale`` is ``user_scale * alpha / rank``."""

    down: torch.Tensor  # [rank, in]
    up: torch.Tensor  # [out, rank]
    alpha: float
    rank: int
    user_scale: float = 1.0

    @property
    def scale(self) -> float:
        if self.rank == 0:
            return 0.0
        return self.user_scale * (self.alpha / self.rank)

    def nbytes(self) -> int:
        return self.down.nbytes + self.up.nbytes


@dataclass
class _Adapter:
    name: str
    deltas: dict[str, LoRADelta]
    source: str


class LoraOverlay:
    """Zero-or-more LoRAs fused into streamed weight views.

    Thread-safety: mutated from the client thread only (``load``/``unload``
    between generations). The prefetch worker never touches this object —
    fusion happens on the engine's compute stream in the pre-hook, after the
    ready event.
    """

    def __init__(self) -> None:
        self._adapters: list[_Adapter] = []
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    def __bool__(self) -> bool:
        return bool(self._adapters)

    def __len__(self) -> int:
        return len(self._adapters)

    @property
    def adapter_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self._adapters)

    def nbytes(self) -> int:
        # Unique tensor objects: the same LoRADelta is stored under several
        # candidate target names.
        seen: set[int] = set()
        total = 0
        for adapter in self._adapters:
            for delta in adapter.deltas.values():
                ident = id(delta)
                if ident in seen:
                    continue
                seen.add(ident)
                total += delta.nbytes()
        return total

    def load(
        self,
        source: str | Path,
        *,
        scale: float = 1.0,
        adapter_name: str | None = None,
        weight_name: str | None = None,
        revision: str | None = None,
        hf_token: str | None = None,
    ) -> str:
        """Parse and register a LoRA. Returns the adapter name."""
        deltas = load_lora_deltas(
            source, weight_name=weight_name, revision=revision, hf_token=hf_token
        )
        for delta in deltas.values():
            delta.user_scale = float(scale)
        name = adapter_name or _default_adapter_name(source, self.adapter_names)
        if name in self.adapter_names:
            self.unload(name)
        # If we have already materialized onto a device, place this adapter
        # there too so the next generate doesn't mix CPU/GPU deltas.
        if self._device is not None and self._dtype is not None:
            _materialize_deltas(deltas, self._device, self._dtype)
        self._adapters.append(_Adapter(name=name, deltas=deltas, source=str(source)))
        logger.info(
            "LoRA adapter %r loaded (%d target(s), scale=%.3f, %.1f MB)",
            name,
            len(deltas),
            scale,
            self.nbytes() / 1e6,
        )
        return name

    def unload(self, adapter_name: str | None = None) -> None:
        """Drop one adapter, or every adapter when ``adapter_name is None``."""
        if adapter_name is None:
            self._adapters.clear()
            return
        kept = [a for a in self._adapters if a.name != adapter_name]
        if len(kept) == len(self._adapters):
            raise LoraError(f"No LoRA adapter named {adapter_name!r} is loaded")
        self._adapters = kept

    def set_scale(self, adapter_name: str, scale: float) -> None:
        for adapter in self._adapters:
            if adapter.name == adapter_name:
                for delta in adapter.deltas.values():
                    delta.user_scale = float(scale)
                return
        raise LoraError(f"No LoRA adapter named {adapter_name!r} is loaded")

    def materialize(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move every A/B onto ``device``/``dtype`` once (engine construction)."""
        for adapter in self._adapters:
            _materialize_deltas(adapter.deltas, device, dtype)
        self._device = device
        self._dtype = dtype

    def apply(self, tensors: Mapping[str, torch.Tensor]) -> int:
        """Fuse every loaded LoRA into ``tensors`` in place. Returns hits.

        Hot-loop contract: ``down``/``up`` are already on the tensor's device
        and dtype (see ``materialize``). ``addmm_`` writes into the existing
        storage — no new allocations.
        """
        if not self._adapters:
            return 0
        hits = 0
        for name, weight in tensors.items():
            if weight.ndim != 2 or not weight.is_floating_point():
                continue
            for adapter in self._adapters:
                delta = adapter.deltas.get(name)
                if delta is None:
                    continue
                scale = delta.scale
                if scale == 0.0:
                    continue
                out, inn = int(weight.shape[0]), int(weight.shape[1])
                if delta.up.shape != (out, delta.rank) or delta.down.shape != (delta.rank, inn):
                    raise LoraError(
                        f"LoRA shape mismatch for {name}: weight {tuple(weight.shape)} vs "
                        f"up {tuple(delta.up.shape)} @ down {tuple(delta.down.shape)} "
                        f"(rank={delta.rank})"
                    )
                # in-place: weight += scale * up @ down
                weight.addmm_(delta.up, delta.down, alpha=scale)
                hits += 1
        return hits


def _materialize_deltas(
    deltas: dict[str, LoRADelta], device: torch.device, dtype: torch.dtype
) -> None:
    seen: set[int] = set()
    for delta in deltas.values():
        ident = id(delta)
        if ident in seen:
            continue
        seen.add(ident)
        if delta.down.device != device or delta.down.dtype != dtype:
            delta.down = delta.down.to(device=device, dtype=dtype)
        if delta.up.device != device or delta.up.dtype != dtype:
            delta.up = delta.up.to(device=device, dtype=dtype)


def _default_adapter_name(source: str | Path, taken: Iterable[str]) -> str:
    p = Path(source)
    base = p.stem if p.suffix == ".safetensors" else p.name or "lora"
    base = re.sub(r"[^\w.\-]+", "-", base).strip("-") or "lora"
    if base not in taken:
        return base
    i = 2
    while f"{base}-{i}" in taken:
        i += 1
    return f"{base}-{i}"
