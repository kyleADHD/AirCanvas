"""Qwen-Image (20B) headline benchmark — ROADMAP M5 acceptance:
<= 4 min/image on 6-8 GB VRAM, a model that cannot run on this hardware any
other way.

One process, pipeline built once, three generations at 1024^2:
  1. 20-step COLD  — includes the one-time Qwen2.5-VL encode (16.6 GB TE runs
                     on CPU via the auto policy; expect minutes, cached after)
  2. 20-step WARM  — embedding cache hit; the honest per-image number
  3. 50-step WARM  — full-quality reference setting

True CFG (negative prompt) doubles per-step compute, which also helps hide
the per-block disk traffic behind the GPU.

Usage: python benchmarks/bench_qwen.py [--outdir outputs] [--steps-list 20,20,50]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from aircanvas.api import AirPipeline

PROMPT = "a cozy corner bookshop at dusk, warm lamplight, a cat asleep on the counter"
NEGATIVE = " "
MODEL = "Qwen/Qwen-Image"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pipe = AirPipeline.from_pretrained(MODEL, compression="nf4")
    print(f"pipeline ready in {time.perf_counter() - t0:.1f}s")

    for label, steps in (("cold", 20), ("warm", 20), ("warm50", 50)):
        t = time.perf_counter()
        result = pipe(
            PROMPT,
            negative_prompt=NEGATIVE,
            true_cfg_scale=4.0,
            num_inference_steps=steps,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        )
        dt = time.perf_counter() - t
        out = args.outdir / f"qwen_image_{steps}step_{label}.png"
        result.images[0].save(out)
        print(f"\n=== {label} ({steps} steps): {dt:.1f}s total -> {out} ===")
        report = pipe.report()
        if isinstance(report, str):
            print(report)


if __name__ == "__main__":
    main()
