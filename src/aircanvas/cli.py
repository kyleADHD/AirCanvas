"""CLI: aircanvas split | run | doctor."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


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

    sub.add_parser("run", help="Generate with a streamed pipeline (M4)")
    sub.add_parser("doctor", help="Probe hardware and report runnable models (M4)")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "split":
        from aircanvas.sharding.splitter import shard_cache_dir, split_model

        compression = None if args.compression == "none" else args.compression
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
        print(
            f"Split complete: {manifest.model_class} ({manifest.adapter} adapter)\n"
            f"  {len(manifest.blocks)} blocks, {total / 1e9:.2f} GB total, "
            f"largest block {largest / 1e6:.0f} MB\n"
            f"  cache: {cache}"
        )
        return 0

    raise NotImplementedError(f"'{args.command}' lands in a later milestone; see docs/ROADMAP.md")


if __name__ == "__main__":
    raise SystemExit(main())
