"""Streaming overhead benchmark: sync vs prefetch vs full-VRAM reference.

Synthetic homogeneous DiT-shaped model, sized via CLI (default ~0.8 GB fp32).
Usage: python benchmarks/bench_stream.py [--dim 2048] [--blocks 24] [--steps 4]
Writes its checkpoint + shard cache under --workdir (default: system temp —
NEVER the repo; see CLAUDE.md).
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from aircanvas.sharding.splitter import split_model
from aircanvas.streaming.engine import StreamingEngine


class Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim, bias=False)
        self.lin2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(torch.tanh(self.lin1(x)))


class BigToy(nn.Module):
    def __init__(self, dim: int, n_blocks: int) -> None:
        super().__init__()
        self.x_embedder = nn.Linear(dim, dim, bias=False)
        self.blocks = nn.ModuleList(Block(dim) for _ in range(n_blocks))
        self.proj_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.x_embedder(x)
        for block in self.blocks:
            x = block(x)
        return self.proj_out(x)


def time_steps(fn, steps: int, device: torch.device) -> list[float]:
    times = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--workdir", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="aircanvas_bench_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  workdir={workdir}")

    torch.manual_seed(0)
    model = BigToy(args.dim, args.blocks).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.blocks} blocks, dim {args.dim} -> {n_params * 4 / 1e9:.2f} GB fp32")

    ckpt = workdir / "model" / "transformer"
    if not (ckpt / "diffusion_pytorch_model.safetensors").exists():
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "config.json").write_text('{"_class_name": "BenchToy"}')
        from safetensors.torch import save_file

        save_file(model.state_dict(), str(ckpt / "diffusion_pytorch_model.safetensors"))
    manifest = split_model(str(workdir / "model"), cache_dir=workdir / "cache", compute_dtype=None)

    x = torch.randn(1, args.tokens, args.dim, device=device)

    # Full-VRAM reference
    ref = model.to(device)
    with torch.no_grad():
        ref_times = time_steps(lambda: ref(x), args.steps, device)
    ref.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"full-VRAM reference : {min(ref_times) * 1e3:8.1f} ms/step")

    for label, prefetch in [("sync streaming", False), ("prefetch streaming", True)]:
        streamed = BigToy(args.dim, args.blocks).to("meta").eval()
        with (
            StreamingEngine(
                streamed, manifest, workdir / "cache", device, prefetch=prefetch
            ) as engine,
            torch.no_grad(),
        ):
            times = time_steps(lambda: streamed(x), args.steps, device)
        steady = times[2:] if len(times) > 2 else times[-1:]
        print(
            f"{label:20s}: {statistics.mean(steady) * 1e3:8.1f} ms/step steady "
            f"(first {times[0] * 1e3:.1f} ms)"
        )
        print("  " + engine.report().replace("\n", "\n  "))


if __name__ == "__main__":
    main()
