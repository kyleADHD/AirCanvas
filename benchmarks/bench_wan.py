"""Wan 2.1 T2V video benchmark — the M6 acceptance gate, measured directly.

The 1.3B model fits entirely in 6 GB VRAM, so the SAME engine can run it two
ways on the same box:

  streamed   max_resident_blocks=0   -> every block read from disk every step
  resident   max_resident_blocks=all -> zero streaming (full-VRAM reference)

M6 acceptance: streamed per-step time <= 110% of the resident reference —
video compute is supposed to hide the transfer almost entirely
(RESEARCH.md §4). Both runs share the embedding cache, so the UMT5 CPU
encode is paid once.

Usage: python benchmarks/bench_wan.py [--outdir outputs] [--steps 20]
       [--frames 81] [--height 480] [--width 832]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from aircanvas.api import AirPipeline

PROMPT = "a red fox trotting through fresh snow at golden hour, cinematic, shallow depth of field"
NEGATIVE = "blurry, low quality, distorted, watermark"
MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def run(pipe: AirPipeline, args: argparse.Namespace, label: str) -> float:
    t = time.perf_counter()
    result = pipe(
        PROMPT,
        encode_kwargs={"negative_prompt": NEGATIVE},
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        guidance_scale=5.0,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    )
    dt = time.perf_counter() - t
    # Re-create right before writing: a 20-minute render is long enough for
    # Windows temp cleanup to delete the output dir out from under us
    # (it happened; the frames died with the process).
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"wan13b_{args.steps}step_{label}.mp4"
    from diffusers.utils import export_to_video

    export_to_video(result.frames[0], str(out), fps=16)
    print(f"\n=== {label}: {dt:.1f}s total -> {out} ===")
    report = pipe.report()
    if isinstance(report, str):
        print(report)
    phase_stats = getattr(getattr(pipe, "_orchestrator", None), "stats", None)
    per_step = phase_stats.denoise_s / args.steps if phase_stats and args.steps else 0.0
    print(f"per-step (denoise/steps): {per_step:.2f}s")
    return per_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL, help="any Wan-family diffusers repo")
    ap.add_argument(
        "--no-resident",
        action="store_true",
        help="skip the all-resident reference (models that cannot fit in VRAM, e.g. 14B)",
    )
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # text_encoder/tokenizer=None: embeddings come from the disk cache (see
    # scratchpad pre-encode), so the 11.4 GB UMT5 is never loaded — a plain
    # CPU load of it segfaults this 16 GB box. A cache MISS with these None
    # will fail loudly in encode_prompt; that is the correct failure.
    te_less = {"text_encoder": None, "tokenizer": None}
    t0 = time.perf_counter()
    streamed_pipe = AirPipeline.from_pretrained(
        args.model, compression="fp8", max_resident_blocks=0, **te_less
    )
    print(f"pipeline ready in {time.perf_counter() - t0:.1f}s")
    streamed = run(streamed_pipe, args, "streamed")
    del streamed_pipe

    if args.no_resident:
        print(f"\nstreamed-only run: {streamed:.2f}s/step (no in-VRAM reference possible)")
        return

    resident_pipe = AirPipeline.from_pretrained(
        args.model, compression="fp8", max_resident_blocks=10_000, **te_less
    )
    resident = run(resident_pipe, args, "resident")

    if resident > 0:
        ratio = streamed / resident
        verdict = "PASS" if ratio <= 1.10 else "MISS"
        print(
            f"\nM6 gate: streamed {streamed:.2f}s/step vs resident {resident:.2f}s/step "
            f"-> {ratio:.2f}x ({verdict}, target <= 1.10x)"
        )


if __name__ == "__main__":
    main()
