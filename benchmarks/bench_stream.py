"""Streaming overhead benchmark: sync vs prefetch vs full-VRAM reference.

Synthetic homogeneous DiT-shaped model, sized via CLI (default ~0.8 GB fp32).
Usage: python benchmarks/bench_stream.py [--dim 2048] [--blocks 24] [--steps 4]
                                         [--dtype bfloat16] [--compression fp8]
                                         [--resident 4]
Writes its checkpoint + shard cache under --workdir (default: system temp —
NEVER the repo; see CONTRIBUTING.md).

M4 additions: `--compression fp8` measures the fp8 shard path (half the disk
traffic, plus a per-block upcast and a second slot pool — ADR #8), and
`--resident N` measures the budget solver's resident-block tier by pinning the
first N blocks on the device.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from aircanvas.sharding.manifest import Manifest
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


def bench_engine(
    label: str,
    args: argparse.Namespace,
    manifest: Manifest,
    cache: Path,
    x: torch.Tensor,
    device: torch.device,
    *,
    prefetch: bool,
    resident: int = 0,
) -> None:
    streamed = BigToy(args.dim, args.blocks).to("meta").eval()
    with (
        StreamingEngine(
            streamed, manifest, cache, device, prefetch=prefetch, resident_blocks=resident
        ) as engine,
        torch.no_grad(),
    ):
        times = time_steps(lambda m=streamed: m(x), args.steps, device)
    steady = times[2:] if len(times) > 2 else times[-1:]
    print(
        f"{label:34s}: {statistics.mean(steady) * 1e3:8.1f} ms/step steady "
        f"(first {times[0] * 1e3:.1f} ms)"
    )
    print("  " + engine.report().replace("\n", "\n  "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument(
        "--dtype",
        default="source",
        help="compute dtype for the shard cache: source (no cast) | bfloat16 | float16",
    )
    ap.add_argument(
        "--compression",
        default="none",
        choices=["none", "fp8", "both"],
        help="'both' splits twice and reports the fp8 tradeoff side by side",
    )
    ap.add_argument(
        "--resident", type=int, default=0, help="also measure N permanently resident blocks"
    )
    ap.add_argument("--workdir", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="aircanvas_bench_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  workdir={workdir}  dtype={args.dtype}")

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

    compute_dtype = None if args.dtype == "source" else args.dtype
    schemes = ["none", "fp8"] if args.compression == "both" else [args.compression]
    if "fp8" in schemes and compute_dtype is None:
        compute_dtype = "bfloat16"  # fp8 needs an explicit upcast target
        print("note: fp8 requires an explicit compute dtype; using bfloat16")

    caches: dict[str, tuple[Manifest, Path]] = {}
    for scheme in schemes:
        cache = workdir / f"cache-{scheme}"
        manifest = split_model(
            str(workdir / "model"),
            cache_dir=cache,
            compression=None if scheme == "none" else scheme,
            compute_dtype=compute_dtype,
        )
        caches[scheme] = (manifest, cache)
        print(
            f"shards[{scheme:4s}]: {manifest.disk_bytes() / 1e9:.2f} GB on disk, "
            f"{sum(b.materialized_bytes for b in manifest.blocks) / 1e9:.2f} GB materialised"
        )

    torch_dtype = getattr(torch, compute_dtype) if compute_dtype else torch.float32
    x = torch.randn(1, args.tokens, args.dim, device=device, dtype=torch_dtype)

    # Full-VRAM reference
    ref = model.to(device=device, dtype=torch_dtype)
    with torch.no_grad():
        ref_times = time_steps(lambda: ref(x), args.steps, device)
    ref.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"\n{'full-VRAM reference':34s}: {min(ref_times) * 1e3:8.1f} ms/step")

    for scheme in schemes:
        manifest, cache = caches[scheme]
        for label, prefetch in (("sync streaming", False), ("prefetch streaming", True)):
            bench_engine(f"{label} [{scheme}]", args, manifest, cache, x, device, prefetch=prefetch)
        if args.resident:
            bench_engine(
                f"prefetch +{args.resident} resident [{scheme}]",
                args,
                manifest,
                cache,
                x,
                device,
                prefetch=True,
                resident=args.resident,
            )


if __name__ == "__main__":
    main()
