"""FLUX.1-schnell headline benchmark on real hardware (ROADMAP M4/M5 acceptance).

One process, pipeline built once, three generations:
  1. 4-step 1024^2  — schnell's native setting, COLD (TE encode + schedule recording)
  2. 4-step again   — WARM (embedding cache hit + recorded schedule + OS page cache)
  3. 28-step 1024^2 — FLUX.1-dev-equivalent workload (identical architecture &
                      per-step cost; dev itself is gated behind a license token)

Usage: python benchmarks/bench_flux.py [--outdir outputs] [--compression fp8]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from aircanvas.api import AirPipeline

PROMPT = "a watercolor fox reading a newspaper in autumn light, cozy study"
MODEL = "black-forest-labs/FLUX.1-schnell"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--compression", default="fp8", choices=["none", "fp8", "nf4"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pipe = AirPipeline.from_pretrained(
        MODEL,
        compression=None if args.compression == "none" else args.compression,
    )
    print(f"pipeline ready in {time.perf_counter() - t0:.1f}s")

    runs = (("cold", 4), ("warm", 4), ("dev-equivalent", 28))
    for label, steps in runs:
        t = time.perf_counter()
        result = pipe(
            PROMPT,
            num_inference_steps=steps,
            height=1024,
            width=1024,
            guidance_scale=0.0,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        )
        dt = time.perf_counter() - t
        out = args.outdir / f"flux_schnell_{steps}step_{label}.png"
        result.images[0].save(out)
        print(f"\n=== {label} ({steps} steps): {dt:.1f}s total -> {out} ===")
        report = pipe.report()
        if isinstance(report, str):
            print(report)


if __name__ == "__main__":
    main()
