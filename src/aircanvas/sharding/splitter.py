"""Split a diffusers DiT checkpoint into per-block shards (M1).

Design (ARCHITECTURE.md §3.1): only the DiT is sharded; text encoders and VAE
keep stock form. Peak RAM during a split is ~one block (~700 MB for 14B-class
models), not one checkpoint shard: tensors are gathered per-block via
mmap-backed `safe_open` seek-reads, never `load_file` on whole shards.

Resumable: each written shard is temp-file -> os.replace -> `.done` marker;
re-running skips ready shards and repairs missing ones. manifest.json is
written last.

`compression=None` is a dtype-cast passthrough (M1); `compression="fp8"` stores
float8_e4m3fn payloads plus per-tensor scales (M4, see quant.py); "nf4" stores
bitsandbytes-packed 4-bit payloads plus absmax siblings and needs CUDA at split
AND load time (M5). Compressed shards are written largest-element-size-first and the result
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
from collections.abc import Callable
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
    gguf_file: str | None = None,
) -> Path:
    """Default persistent cache location. Deliberately under the HF cache home
    (never the repo/OneDrive — hard rule, CONTRIBUTING.md). A GGUF source gets
    its own tag: caches split from different quant files must never collide."""
    tag = compression or (compute_dtype or "source")
    if gguf_file is not None:
        stem = Path(str(gguf_file).rsplit(":", 1)[-1]).stem
        tag = f"{tag}-gguf-{safe_slug(stem)}"
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


def _model_class_from_config(
    source: str, subfolder: str, revision: str | None, hf_token: str | None
) -> str:
    """Model class from the checkpoint's config.json — fetched ALONE for GGUF
    sources, so pointing at a 24 GB repo downloads a few KB, not the weights."""
    p = Path(source)
    for cand in (p / subfolder / "config.json", p / "config.json"):
        if cand.is_file():
            text = cand.read_text(encoding="utf-8")
            break
    else:
        from huggingface_hub import hf_hub_download  # lazy: local paths need no hub

        text = Path(
            hf_hub_download(source, f"{subfolder}/config.json", revision=revision, token=hf_token)
        ).read_text(encoding="utf-8")
    config = json.loads(text)
    model_class = config.get("_class_name") or next(iter(config.get("architectures", [])), None)
    if not model_class:
        raise ValueError(f"config.json for {source}/{subfolder} names no model class")
    return str(model_class)


def _load_weight_map(src_dir: Path) -> dict[str, str]:
    """tensor name -> relative weight filename, from the index json or a
    single-file checkpoint (map synthesized via safetensors header keys)."""
    indexes = sorted(src_dir.glob("*.safetensors.index.json"))
    if indexes:
        data = json.loads(indexes[0].read_text(encoding="utf-8"))
        return dict(data["weight_map"])  # JSON order preserved
    files = sorted(src_dir.glob("*.safetensors"))
    if len(files) == 1:
        try:
            with safe_open(files[0], framework="pt", device="cpu") as f:
                names = list(f.keys())  # not a dict — safe_open handles aren't iterable
        except OSError:  # mmap refused (low commit) — header parse needs no mapping
            names = [m.name for m in ShardHeader.parse(files[0]).tensors]
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


def _gather_ranged(
    path: Path, names: list[str], dtype: torch.dtype | None, out: dict[str, torch.Tensor]
) -> None:
    """mmap-free fallback: seek/read exactly the requested tensors' byte ranges.

    Windows refuses CreateFileMapping on a 10-20 GB checkpoint when system
    commit (RAM + pagefile) is tight — OSError 1455, 'the paging file is too
    small' — which is precisely the situation on the low-RAM machines this
    project targets. Peak memory here is one tensor, not one mapping.
    """
    header = ShardHeader.parse(path)
    metas = {m.name: m for m in header.tensors}
    with open(path, "rb", buffering=0) as f:
        for n in sorted(names, key=lambda n: metas[n].start):  # sequential reads
            m = metas[n]
            buf = bytearray(m.end - m.start)
            f.seek(header.data_start + m.start)
            if f.readinto(buf) != len(buf):
                raise OSError(f"Short read for {n} in {path.name}")
            t = torch.frombuffer(buf, dtype=torch.uint8).view(m.dtype).reshape(m.shape)
            out[n] = _maybe_cast(t, dtype)


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
        path = src_dir / fname
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                for n in ns:
                    out[n] = _maybe_cast(f.get_tensor(n), dtype)
        except OSError as e:
            logger.warning("mmap of %s failed (%s) — falling back to ranged reads", fname, e)
            _gather_ranged(path, ns, dtype, out)
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
        spec = quant.out_spec(m.name, m.dtype, m.shape, header.metadata, compute_dtype)
        if spec is None:
            continue
        out_dtype, out_shape = spec
        total += out_dtype.itemsize * torch.Size(out_shape).numel()
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


#: Conservative output-size estimate per scheme, as a fraction of the
#: COMPUTE-DTYPE checkpoint size (AirLLM's analogous 4-bit/8-bit estimates).
#: Measured: fp8 ~0.55x (skip-list tensors stay bf16), nf4 ~0.30x.
_PREFLIGHT_FACTOR: dict[str | None, float] = {None: 1.0, "fp8": 0.60, "nf4": 0.35}


def _preflight(
    cache_dir: Path,
    src_dir: Path,
    weight_map: dict[str, str],
    compression: Compression,
    dtype: torch.dtype | None,
) -> None:
    """Refuse before writing anything if the output cannot fit.

    Sized from the CAST size, not the file size: Wan's diffusers repos ship
    fp32, so "0.6x the source" would demand double the real fp8 output and
    refuse splits that fit fine. Header parses only — no tensor data is read.
    """
    cast_bytes = 0
    for fname in set(weight_map.values()):
        header = ShardHeader.parse(src_dir / fname)
        for m in header.tensors:
            n = torch.Size(m.shape).numel()
            itemsize = m.dtype.itemsize
            if dtype is not None and m.dtype.is_floating_point and itemsize >= 2:
                itemsize = dtype.itemsize  # _maybe_cast will store it this size
            cast_bytes += n * itemsize
    _check_space(cache_dir, cast_bytes, compression)


def _check_space(cache_dir: Path, cast_bytes: int, compression: Compression) -> None:
    needed = int(cast_bytes * _PREFLIGHT_FACTOR[compression])
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
    gguf_file: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> Manifest:
    """Find-or-create the shard cache for `source`; return its manifest.

    With `gguf_file` (a local .gguf path or 'repo_id:filename'), tensors come
    from the quantized GGUF — dequantized once here, resharded into the chosen
    codec — and only `source`'s config.json is fetched, never its weights.

    `progress(done, total, block_name)` is called once per block — after the
    shard lands, or immediately for a block a previous run already finished, so
    a resumed split reports its true starting point rather than replaying from
    zero. Raising from it aborts the split at a block boundary, which is the
    supported way to stop one: every shard already written keeps its `.done`
    marker and a later call resumes from there.
    """
    dtype = _torch_dtype(compute_dtype)
    # `_torch_dtype` maps None/"none"/"source" to None, so a real dtype here
    # implies `compute_dtype` was a real name.
    dtype_tag: str = "source" if dtype is None else str(compute_dtype)
    if compression is not None:
        if compression not in ("fp8", "nf4"):
            raise ValueError(f"Unknown compression {compression!r} (expected 'fp8', 'nf4', None)")
        if dtype is None:
            raise ValueError(
                f"compression={compression!r} needs an explicit compute_dtype (e.g. "
                "'bfloat16'): the dequantization target must be recorded in the manifest."
            )
        if compression == "nf4" and not torch.cuda.is_available():
            raise quant.QuantError(
                "compression='nf4' requires CUDA at split time (bitsandbytes quantizes on GPU)"
            )

    cache_dir = cache_dir or shard_cache_dir(source, subfolder, compression, dtype_tag, gguf_file)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # FAST PATH before touching the source: a complete cache must be reusable
    # even after the original checkpoint was deleted to reclaim disk — the
    # split-once-delete-original workflow is the whole point of a persistent
    # shard cache. (The cache dir path already encodes source/subfolder/tag.)
    existing: Manifest | None = None
    if (cache_dir / MANIFEST_NAME).is_file():
        existing = Manifest.load(cache_dir)
        if (
            not existing.compatible_with(
                compression=compression, compute_dtype=dtype_tag, model_class=existing.model_class
            )
            or existing.gguf_file != gguf_file
        ):
            raise ValueError(
                f"Shard cache {cache_dir} was built with "
                f"(compression={existing.compression}, dtype={existing.compute_dtype}, "
                f"class={existing.model_class}, gguf={existing.gguf_file}) — pass a "
                f"different cache_dir or delete it."
            )
        if existing.is_complete(cache_dir):
            logger.info("Reusing complete shard cache at %s (source not touched)", cache_dir)
            if progress is not None:
                n = len(existing.blocks)
                progress(n, n, "cached")
            return existing

    if gguf_file is not None:
        from aircanvas.sharding import gguf_source

        gguf_path = gguf_source.resolve_gguf_path(gguf_file, revision=revision, hf_token=hf_token)
        model_class = _model_class_from_config(source, subfolder, revision, hf_token)
    else:
        src_dir = _resolve_source(source, subfolder, revision, hf_token)
        config = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
        # diffusers configs carry _class_name; transformers configs carry
        # architectures — text encoders are streamable transformers too (a 16 GB
        # box cannot even LOAD Wan's 11.4 GB UMT5, but it can stream it).
        found = config.get("_class_name") or next(iter(config.get("architectures", [])), None)
        if not found:
            raise ValueError(f"{src_dir / 'config.json'} names no model class")
        model_class = str(found)

    if existing is not None:
        if existing.model_class != model_class:
            raise ValueError(
                f"Shard cache {cache_dir} was built for {existing.model_class}, but the "
                f"source now has {model_class} — delete the cache or pin a revision."
            )
        logger.warning("Shard cache at %s is incomplete — repairing", cache_dir)

    adapter: ModelAdapter = resolve(model_class)
    if gguf_file is not None:
        ckpt: gguf_source.TensorSource = gguf_source.GGUFCheckpoint(gguf_path)
        names = ckpt.tensor_names()
        plan: BlockPlan = adapter.block_plan(names)
        if not gguf_source.plan_covers(plan, names):
            # Foreign key layout (e.g. BFL-style FLUX ggufs): let diffusers'
            # single-file loader do the renames, then re-plan.
            ckpt = gguf_source.DiffusersGGUFCheckpoint(
                gguf_path, model_class, source, subfolder, revision, hf_token, dtype
            )
            names = ckpt.tensor_names()
            plan = adapter.block_plan(names)
            if not gguf_source.plan_covers(plan, names):
                raise gguf_source.GGUFError(
                    f"{gguf_path.name}: tensor names match neither {model_class}'s "
                    "state dict nor a layout diffusers can convert."
                )
        _check_space(cache_dir, ckpt.materialized_bytes(dtype), compression)

        def gather(ns: tuple[str, ...]) -> dict[str, torch.Tensor]:
            return ckpt.gather(ns, dtype)

    else:
        weight_map = _load_weight_map(src_dir)
        plan = adapter.block_plan(list(weight_map))
        _preflight(cache_dir, src_dir, weight_map, compression, dtype)

        def gather(ns: tuple[str, ...]) -> dict[str, torch.Tensor]:
            return _gather(src_dir, weight_map, ns, dtype)

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
    total_blocks = len(plan.block_names)
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
            if progress is not None:
                progress(i + 1, total_blocks, block)
            continue
        tensors = gather(plan.tensors_by_block[block])
        load_bytes = _materialized_bytes(tensors)
        stored, quant_meta = quant.compress_state_dict(tensors, compression)
        n_bytes, digest = _write_shard(
            cache_dir,
            fname,
            stored,
            {**meta_common, **quant_meta, "block": block},
            hash_shards,
            verify_alignment=compression is not None,
        )
        blocks.append(
            BlockShard(
                name=block, file=fname, n_bytes=n_bytes, sha256=digest, load_bytes=load_bytes
            )
        )
        if progress is not None:
            progress(i + 1, total_blocks, block)

    # The resident shard is never compressed (see module docstring).
    resident_fname = "resident.safetensors"
    if _shard_ready(cache_dir, resident_fname):
        resident_bytes = (cache_dir / resident_fname).stat().st_size
        resident_load_bytes = None
    else:
        tensors = gather(plan.resident_tensors)
        resident_load_bytes = _materialized_bytes(tensors)
        resident_bytes, _ = _write_shard(
            cache_dir, resident_fname, tensors, {**meta_common, "block": "resident"}, hash_shards
        )

    manifest = Manifest(
        # A local source is recorded ABSOLUTE. The cache outlives the shell it
        # was made in — `aircanvas split ./my-model` then `aircanvas studio`
        # from anywhere else must not turn a stale relative path into a Hub
        # repo id and a surprise download.
        source=str(Path(source).resolve()) if Path(source).is_dir() else str(source),
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
        gguf_file=gguf_file,
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
