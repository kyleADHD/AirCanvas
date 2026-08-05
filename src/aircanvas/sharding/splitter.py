"""Split a diffusers DiT checkpoint into per-block shards (M1).

Design (ARCHITECTURE.md §3.1): only the DiT is sharded; text encoders and VAE
keep stock form. Peak RAM during a split is ~one block (~700 MB for 14B-class
models), not one checkpoint shard: tensors are gathered per-block via
mmap-backed `safe_open` seek-reads, never `load_file` on whole shards.

Resumable: each written shard is temp-file -> os.replace -> `.done` marker;
re-running skips ready shards and repairs missing ones. manifest.json is
written last.

`compression=None` is a dtype-cast passthrough (M1); `compression="fp8"` stores
float8_e4m3fn payloads plus per-tensor scales (M4, see quant.py); "nf4" lands
in M5. Compressed shards are written largest-element-size-first and the result
is verified with the *reader's* own header parser (`ShardHeader.aligned`)
before the `.done` marker goes down — a misaligned shard would be rejected at
runtime by the prefetcher, and that is far better caught at split time.

The resident shard (embedders, final norm/proj, modulation tables) is always
stored uncompressed: it is loaded once and stays on the GPU, so compressing it
buys no per-step disk traffic (ADR #2's entire rationale) while risking the
most quality-sensitive tensors in the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from aircanvas import __version__
from aircanvas.adapters import BlockPlan, ModelAdapter, resolve
from aircanvas.config import Compression, cache_root, safe_slug
from aircanvas.sharding import quant
from aircanvas.sharding.manifest import DONE_SUFFIX, MANIFEST_NAME, BlockShard, Manifest

# The writer validates its own output with the reader's parser (no cycle:
# prefetch imports only config + manifest).
from aircanvas.streaming.prefetch import ShardHeader

logger = logging.getLogger(__name__)

_PREFLIGHT_MARGIN_BYTES = 1 << 30  # 1 GiB


class NotEnoughSpaceError(RuntimeError):
    """Raised by the preflight check before any shard is written."""


class MisalignedShardError(RuntimeError):
    """A written shard would be rejected by the prefetcher's zero-copy views."""


def shard_cache_dir(
    source: str,
    subfolder: str = "transformer",
    compression: Compression = None,
    compute_dtype: str | None = "bfloat16",
) -> Path:
    """Default persistent cache location. Deliberately under the HF cache home
    (never the repo/OneDrive — CLAUDE.md hard rule)."""
    tag = compression or (compute_dtype or "source")
    return cache_root() / safe_slug(source) / subfolder / tag


def _torch_dtype(name: str | None) -> torch.dtype | None:
    if name in (None, "none", "source"):
        return None
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {name!r}")
    return dtype


def _resolve_source(
    source: str, subfolder: str, revision: str | None, hf_token: str | None
) -> Path:
    """Local checkpoint dir, or download the subfolder from the Hub."""
    p = Path(source)
    if p.is_dir():
        return p / subfolder if (p / subfolder).is_dir() else p
    from huggingface_hub import snapshot_download  # lazy: local paths need no hub

    local = snapshot_download(
        source,
        allow_patterns=[f"{subfolder}/*"],
        revision=revision,
        token=hf_token,
    )
    src = Path(local) / subfolder
    if not src.is_dir():
        raise FileNotFoundError(f"{source} has no '{subfolder}/' subfolder")
    return src


def _load_weight_map(src_dir: Path) -> dict[str, str]:
    """tensor name -> relative weight filename, from the index json or a
    single-file checkpoint (map synthesized via safetensors header keys)."""
    indexes = sorted(src_dir.glob("*.safetensors.index.json"))
    if indexes:
        data = json.loads(indexes[0].read_text(encoding="utf-8"))
        return dict(data["weight_map"])  # JSON order preserved
    files = sorted(src_dir.glob("*.safetensors"))
    if len(files) == 1:
        with safe_open(files[0], framework="pt", device="cpu") as f:
            names = f.keys()  # not a dict — safe_open handles aren't iterable
        return dict.fromkeys(names, files[0].name)
    if not files:
        raise FileNotFoundError(f"No .safetensors checkpoint found in {src_dir}")
    raise FileNotFoundError(
        f"{src_dir} has {len(files)} weight files but no index json — cannot map tensors"
    )


def _maybe_cast(t: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    """Cast float tensors to the compute dtype; keep everything else verbatim.
    1-byte floats (fp8 payloads) and non-floats (packed/int/scale tensors)
    must never be cast — AirLLM's verbatim lesson (RESEARCH.md §1)."""
    if dtype is None or not t.is_floating_point() or t.element_size() == 1:
        return t
    return t.to(dtype)


def _gather(
    src_dir: Path,
    weight_map: dict[str, str],
    names: tuple[str, ...],
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(weight_map[n], []).append(n)
    out: dict[str, torch.Tensor] = {}
    for fname, ns in by_file.items():
        with safe_open(src_dir / fname, framework="pt", device="cpu") as f:
            for n in ns:
                out[n] = _maybe_cast(f.get_tensor(n), dtype)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _shard_ready(cache_dir: Path, fname: str) -> bool:
    return (cache_dir / fname).is_file() and (cache_dir / (fname + DONE_SUFFIX)).is_file()


def _materialized_bytes(tensors: dict[str, torch.Tensor]) -> int:
    """Bytes these (already compute-dtype) tensors occupy once bound on GPU."""
    return sum(t.numel() * t.element_size() for t in tensors.values())


def _materialized_bytes_of_file(path: Path, compute_dtype: torch.dtype) -> int:
    """Same, recovered from an existing shard (resume path, no manifest yet)."""
    header = ShardHeader.parse(path)
    total = 0
    for m in header.tensors:
        if quant.is_scale_key(m.name):
            continue
        itemsize = quant.compressed_dtype(m.dtype, compute_dtype).itemsize
        total += itemsize * torch.Size(m.shape).numel()
    return total


def _write_shard(
    cache_dir: Path,
    fname: str,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    hash_shards: bool,
    *,
    verify_alignment: bool = False,
) -> tuple[int, str | None]:
    ordered = quant.order_for_alignment({k: v.contiguous() for k, v in tensors.items()})
    tmp = cache_dir / (fname + ".tmp")
    save_file(ordered, str(tmp), metadata=metadata)
    if verify_alignment and not ShardHeader.parse(tmp).aligned:
        tmp.unlink(missing_ok=True)
        raise MisalignedShardError(
            f"{fname}: mixed-dtype layout left a tensor on an unaligned byte offset — "
            f"the prefetcher would refuse this shard. This is an AirCanvas bug."
        )
    final = cache_dir / fname
    os.replace(tmp, final)
    (cache_dir / (fname + DONE_SUFFIX)).touch()
    return final.stat().st_size, (_sha256(final) if hash_shards else None)


def _preflight(cache_dir: Path, src_dir: Path, weight_map: dict[str, str]) -> None:
    needed = sum((src_dir / f).stat().st_size for f in set(weight_map.values()))
    free = shutil.disk_usage(cache_dir).free
    if free < needed + _PREFLIGHT_MARGIN_BYTES:
        raise NotEnoughSpaceError(
            f"Splitting needs ~{needed / 1e9:.1f} GB in {cache_dir} "
            f"but only {free / 1e9:.1f} GB is free. Pass cache_dir= to use another drive."
        )


def split_model(
    source: str,
    *,
    cache_dir: Path | None = None,
    compression: Compression = None,
    subfolder: str = "transformer",
    revision: str | None = None,
    compute_dtype: str | None = "bfloat16",
    hash_shards: bool = False,
    hf_token: str | None = None,
) -> Manifest:
    """Find-or-create the shard cache for `source`; return its manifest."""
    dtype = _torch_dtype(compute_dtype)
    # `_torch_dtype` maps None/"none"/"source" to None, so a real dtype here
    # implies `compute_dtype` was a real name.
    dtype_tag: str = "source" if dtype is None else str(compute_dtype)
    if compression is not None:
        if compression == "nf4":
            raise NotImplementedError("compression='nf4' lands in M5 (see docs/ROADMAP.md)")
        if compression != "fp8":
            raise ValueError(f"Unknown compression {compression!r} (expected 'fp8', 'nf4', None)")
        if dtype is None:
            raise ValueError(
                "compression='fp8' needs an explicit compute_dtype (e.g. 'bfloat16'): the "
                "upcast target must be recorded in the manifest."
            )

    cache_dir = cache_dir or shard_cache_dir(source, subfolder, compression, dtype_tag)
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_dir = _resolve_source(source, subfolder, revision, hf_token)
    config = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    model_class = config["_class_name"]

    if (cache_dir / MANIFEST_NAME).is_file():
        existing = Manifest.load(cache_dir)
        if not existing.compatible_with(
            compression=compression, compute_dtype=dtype_tag, model_class=model_class
        ):
            raise ValueError(
                f"Shard cache {cache_dir} was built with "
                f"(compression={existing.compression}, dtype={existing.compute_dtype}, "
                f"class={existing.model_class}) — pass a different cache_dir or delete it."
            )
        if existing.is_complete(cache_dir):
            logger.info("Reusing complete shard cache at %s", cache_dir)
            return existing
        logger.warning("Shard cache at %s is incomplete — repairing", cache_dir)

    weight_map = _load_weight_map(src_dir)
    adapter: ModelAdapter = resolve(model_class)
    plan: BlockPlan = adapter.block_plan(list(weight_map))
    _preflight(cache_dir, src_dir, weight_map)
    logger.info(
        "Splitting %s (%s): %d blocks in %s -> %s",
        source,
        model_class,
        plan.n_blocks,
        plan.block_lists,
        cache_dir,
    )

    meta_common = {
        "aircanvas": __version__,
        "source": str(source),
        "compression": compression or "none",
        "compute_dtype": dtype_tag,
    }
    blocks: list[BlockShard] = []
    for i, block in enumerate(plan.block_names):
        fname = f"block_{i:04d}.safetensors"
        if _shard_ready(cache_dir, fname):
            path = cache_dir / fname
            blocks.append(
                BlockShard(
                    name=block,
                    file=fname,
                    n_bytes=path.stat().st_size,
                    load_bytes=(
                        _materialized_bytes_of_file(path, dtype) if dtype is not None else None
                    ),
                )
            )
            continue
        tensors = _gather(src_dir, weight_map, plan.tensors_by_block[block], dtype)
        load_bytes = _materialized_bytes(tensors)
        n_bytes, digest = _write_shard(
            cache_dir,
            fname,
            quant.compress_state_dict(tensors, compression),
            {**meta_common, "block": block},
            hash_shards,
            verify_alignment=compression is not None,
        )
        blocks.append(
            BlockShard(
                name=block, file=fname, n_bytes=n_bytes, sha256=digest, load_bytes=load_bytes
            )
        )

    # The resident shard is never compressed (see module docstring).
    resident_fname = "resident.safetensors"
    if _shard_ready(cache_dir, resident_fname):
        resident_bytes = (cache_dir / resident_fname).stat().st_size
        resident_load_bytes = None
    else:
        tensors = _gather(src_dir, weight_map, plan.resident_tensors, dtype)
        resident_load_bytes = _materialized_bytes(tensors)
        resident_bytes, _ = _write_shard(
            cache_dir, resident_fname, tensors, {**meta_common, "block": "resident"}, hash_shards
        )

    manifest = Manifest(
        source=str(source),
        revision=revision,
        subfolder=subfolder,
        model_class=model_class,
        adapter=adapter.key,
        compression=compression,
        compute_dtype=dtype_tag,
        blocks=tuple(blocks),
        resident_file=resident_fname,
        resident_bytes=resident_bytes,
        resident_load_bytes=resident_load_bytes,
    )
    manifest.save(cache_dir)
    disk = manifest.disk_bytes()
    load = sum(b.materialized_bytes for b in blocks)
    logger.info(
        "Split complete: %d blocks (%.2f GB on disk / %.2f GB materialised) + %.2f GB resident "
        "at %s",
        len(blocks),
        disk / 1e9,
        load / 1e9,
        resident_bytes / 1e9,
        cache_dir,
    )
    return manifest
