"""CLI: aircanvas split | run | doctor."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from aircanvas.config import Compression

# --------------------------------------------------------------------------
# Known-model table for `aircanvas doctor`.
#
# Sizes come from docs/RESEARCH.md §3/§4 (measured or derived per-block sizes,
# not marketing numbers). `fp8_ratio` is the share of a block that survives
# compression: norms, modulation/AdaLN projections and embeddings stay bf16
# (sharding/quant.py's skip list), and on FLUX-style blocks the modulation
# projection alone is a third of the bytes — so the real cut is ~0.6x, not the
# 0.5x a naive "fp8 is half of bf16" would suggest.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownModel:
    name: str
    repo_id: str
    kind: str  # "image" | "video"
    n_blocks: int
    largest_block_bytes: int  # bf16
    dit_bytes: int  # bf16, whole DiT
    resident_bytes: int  # embedders/final proj, stays on GPU
    hidden: int
    tokens: int  # default workload token count
    steps: int
    fp8_ratio: float = 0.6
    gated: bool = False
    note: str = ""


MB = 1_000_000
GB = 1_000_000_000

KNOWN_MODELS: tuple[KnownModel, ...] = (
    KnownModel(
        "FLUX.1-schnell",
        "black-forest-labs/FLUX.1-schnell",
        "image",
        57,
        680 * MB,
        int(23.8 * GB),
        1 * GB,
        3072,
        4608,
        4,
        note="4-step distilled: streaming overhead is visible",
    ),
    KnownModel(
        "FLUX.1-dev",
        "black-forest-labs/FLUX.1-dev",
        "image",
        57,
        680 * MB,
        int(23.8 * GB),
        1 * GB,
        3072,
        4608,
        28,
        gated=True,
        note="gated repo: accept the licence on the Hub first",
    ),
    KnownModel(
        "SD3.5-Large",
        "stabilityai/stable-diffusion-3.5-large",
        "image",
        38,
        420 * MB,
        int(16.5 * GB),
        1 * GB,
        2432,
        4608,
        28,
        gated=True,
    ),
    KnownModel(
        "Qwen-Image",
        "Qwen/Qwen-Image",
        "image",
        60,
        680 * MB,
        int(41.0 * GB),
        1 * GB,
        3072,
        4608,
        50,
        note="M5 target; TE is a 16.6 GB Qwen2.5-VL",
    ),
    KnownModel(
        "Wan 2.1 T2V 14B (480p)",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "video",
        40,
        700 * MB,
        int(29.1 * GB),
        1 * GB,
        5120,
        24_000,
        30,
        note="M6 target; needs tiled Wan-VAE decode",
    ),
    KnownModel(
        "HunyuanVideo",
        "hunyuanvideo-community/HunyuanVideo",
        "video",
        60,
        620 * MB,
        int(25.7 * GB),
        1 * GB,
        3072,
        33_000,
        30,
        note="M7 target; 15 GB Llava TE must be CPU/offloaded",
    ),
)


def _synthetic_manifest(model: KnownModel, compression: Compression):
    """A manifest shaped like `model` so doctor uses the REAL budget solver."""
    from aircanvas.sharding.manifest import BlockShard, Manifest

    ratio = model.fp8_ratio if compression == "fp8" else 1.0
    avg_load = model.dit_bytes // model.n_blocks
    blocks = tuple(
        BlockShard(
            name=f"blocks.{i}",
            file=f"block_{i:04d}.safetensors",
            # Largest block first: matches FLUX/Hunyuan two-species layouts and
            # is what the solver's front-loaded residency assumes.
            n_bytes=int((model.largest_block_bytes if i == 0 else avg_load) * ratio),
            load_bytes=model.largest_block_bytes if i == 0 else avg_load,
        )
        for i in range(model.n_blocks)
    )
    return Manifest(
        source=model.repo_id,
        revision=None,
        subfolder="transformer",
        model_class="SyntheticTransformer2DModel",
        adapter="generic",
        compression=compression,
        compute_dtype="bfloat16",
        blocks=blocks,
        resident_bytes=model.resident_bytes,
        resident_load_bytes=model.resident_bytes,
    )


#: Slack above the minimum viable plan below which a model is only "tight".
COMFORT_MARGIN = 1 * GB


def _verdict(model: KnownModel, hardware, compression: Compression) -> str:
    """yes / tight / no, plus the disk time a full generation implies.

    Slack is measured against the MINIMUM plan (nothing resident), not the
    chosen one: the waterfall deliberately spends every spare byte on resident
    blocks, so the chosen plan always looks full no matter how big the card is.
    """
    from aircanvas.streaming.residency import InsufficientVRAMError, Workload, solve

    manifest = _synthetic_manifest(model, compression)
    workload = Workload(steps=model.steps, tokens=model.tokens, hidden=model.hidden)
    try:
        minimal = solve(manifest, hardware, workload, max_resident_blocks=0)
        plan = solve(manifest, hardware, workload)
    except InsufficientVRAMError:
        return "no"
    slack = minimal.vram_budget_bytes - minimal.vram_planned_bytes
    tag = "yes" if slack > COMFORT_MARGIN else "tight"
    seconds = plan.step_read_seconds * max(1, model.steps)
    return f"{tag} (~{seconds / 60:.0f} min IO)" if seconds > 90 else f"{tag} (~{seconds:.0f}s IO)"


def _doctor(probe_disk: bool) -> int:
    import torch

    from aircanvas import __version__
    from aircanvas.config import cache_root
    from aircanvas.utils.hw import DEFAULT_DISK_BW, HardwareProfile, free_disk_bytes

    root = cache_root()
    hardware = HardwareProfile.probe()
    if probe_disk:
        hardware = _probe_real_disk(hardware, root)

    print(f"AirCanvas {__version__} doctor")
    print(f"  torch          {torch.__version__} (cuda {torch.version.cuda or 'n/a'})")
    print(hardware.describe())
    if not probe_disk:
        print("                (assumed — pass --probe-disk to measure)")
    print(f"  shard cache    {root}")
    print(f"  free disk      {free_disk_bytes(root) / GB:.1f} GB")
    if hardware.device != "cuda":
        print("\nNo CUDA device found: AirCanvas will run on CPU (correct, but not fast).")

    print("\nWhat this box can run (per-step disk time is the honest bound):")
    header = (
        f"{'model':<26} {'blocks':>6} {'DiT bf16':>9}  {'fp8 shards':<20} {'no compression':<20}"
    )
    print(header)
    print("-" * len(header))
    for model in KNOWN_MODELS:
        fp8 = _verdict(model, hardware, "fp8")
        raw = _verdict(model, hardware, None)
        print(
            f"{model.name:<26} {model.n_blocks:>6} {model.dit_bytes / GB:>8.1f}G  "
            f"{fp8:<20} {raw:<20}"
        )
    print("\nNotes:")
    for model in KNOWN_MODELS:
        if model.note or model.gated:
            print(f"  {model.name}: {model.note or 'gated repo'}")
    print(
        f"\nDisk bandwidth assumed {hardware.disk_bw_bytes_s / GB:.2f} GB/s"
        f"{' (measured)' if probe_disk else f' (default {DEFAULT_DISK_BW / GB:.1f} GB/s)'}. "
        f"IO estimates are steps x streamed bytes / bandwidth and ignore compute overlap, "
        f"so a real run on video models is usually faster than the number above."
    )
    return 0


def _probe_real_disk(hardware, root: Path):
    """Measure the disk with the largest shard we already have, if any."""
    import dataclasses

    from aircanvas.utils.hw import probe_disk_bandwidth

    candidates = sorted(root.rglob("block_*.safetensors"), key=lambda p: p.stat().st_size)
    if not candidates:
        print("  (no shard cache to probe yet — split a model first for a real measurement)")
        return hardware
    sample = candidates[-1]
    bw = probe_disk_bandwidth(sample, cache_dir=sample.parent)
    return dataclasses.replace(hardware, disk_bw_bytes_s=bw)


def _run(args: argparse.Namespace) -> int:
    from aircanvas.api import AirPipeline

    compression: Compression = None if args.compression == "none" else args.compression
    pipe = AirPipeline.from_pretrained(
        args.source,
        vram_budget=args.vram_budget,
        ram_budget=args.ram_budget,
        compression=compression,
        shard_cache=args.cache_dir,
        compute_dtype=args.compute_dtype,
    )
    result = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
    )
    images = getattr(result, "images", None)
    if images:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(out)
        print(f"Wrote {out}")
    print(pipe.report())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aircanvas",
        description="Run 20B+ image & video diffusion models on 4-8 GB GPUs.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("split", help="Split a model into a per-block shard cache")
    sp.add_argument("source", help="HF repo id (e.g. black-forest-labs/FLUX.1-dev) or local path")
    sp.add_argument("--subfolder", default="transformer")
    sp.add_argument("--cache-dir", type=Path, default=None)
    sp.add_argument("--revision", default=None)
    sp.add_argument(
        "--compute-dtype",
        default="bfloat16",
        help="Cast float weights at split time: bfloat16 | float16 | source (no cast)",
    )
    sp.add_argument("--compression", default="none", choices=["none", "fp8", "nf4"])
    sp.add_argument(
        "--hash", action="store_true", dest="hash_shards", help="Record sha256 per shard (slower)"
    )

    rp = sub.add_parser("run", help="Generate with a streamed pipeline")
    rp.add_argument("source", help="HF repo id or local path")
    rp.add_argument("-p", "--prompt", required=True)
    rp.add_argument("-o", "--out", default="aircanvas.png")
    rp.add_argument("--steps", type=int, default=28, dest="steps")
    rp.add_argument("--height", type=int, default=1024)
    rp.add_argument("--width", type=int, default=1024)
    rp.add_argument("--cache-dir", type=Path, default=None)
    rp.add_argument("--compression", default="fp8", choices=["none", "fp8", "nf4"])
    rp.add_argument("--compute-dtype", default="bfloat16")
    rp.add_argument("--vram-budget", default="auto")
    rp.add_argument("--ram-budget", default="auto")

    dp = sub.add_parser("doctor", help="Probe hardware and report runnable models")
    dp.add_argument(
        "--probe-disk",
        action="store_true",
        help="Measure real read bandwidth from the largest cached shard (a few seconds)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "split":
        from aircanvas.sharding.splitter import shard_cache_dir, split_model

        compression: Compression = None if args.compression == "none" else args.compression
        manifest = split_model(
            args.source,
            cache_dir=args.cache_dir,
            compression=compression,
            subfolder=args.subfolder,
            revision=args.revision,
            compute_dtype=None if args.compute_dtype == "source" else args.compute_dtype,
            hash_shards=args.hash_shards,
        )
        cache = args.cache_dir or shard_cache_dir(
            args.source, args.subfolder, compression, manifest.compute_dtype
        )
        total = sum(b.n_bytes for b in manifest.blocks) + manifest.resident_bytes
        largest = max(b.n_bytes for b in manifest.blocks)
        materialized = sum(b.materialized_bytes for b in manifest.blocks)
        print(
            f"Split complete: {manifest.model_class} ({manifest.adapter} adapter)\n"
            f"  {len(manifest.blocks)} blocks, {total / 1e9:.2f} GB total, "
            f"largest block {largest / 1e6:.0f} MB\n"
            f"  compression: {manifest.compression or 'none'} "
            f"({materialized / 1e9:.2f} GB materialised in {manifest.compute_dtype})\n"
            f"  cache: {cache}"
        )
        return 0

    if args.command == "doctor":
        return _doctor(args.probe_disk)

    if args.command == "run":
        return _run(args)

    raise NotImplementedError(f"'{args.command}' lands in a later milestone; see docs/ROADMAP.md")


if __name__ == "__main__":
    raise SystemExit(main())
