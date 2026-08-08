# AirCanvas

**Run 20B+ image & video generation models on 4–8 GB GPUs.** Measured, not
promised: a 20.4B Qwen-Image generating 1024² images on a 6 GB laptop RTX
4050 — a model that needs a 40 GB+ card by any other means.

AirCanvas is [AirLLM](https://github.com/lyogavin/airllm) for diffusion
models: it splits a model's transformer into per-block shards on disk
(fp8/NF4-quantized at split time), then streams each block to the GPU
just-in-time during the denoise loop. VRAM scales with the *largest block*,
not the model; system RAM holds only a small pinned prefetch ring.

Why the economics work where AirLLM's don't: an LLM re-reads the whole model
*per token*; a diffusion model runs 20–50 forward passes *total* — and video
steps are so compute-heavy the transfer vanishes entirely. Measured on the
6 GB dev machine ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

| Model | Params | Needs normally | AirCanvas on 6 GB |
|---|---|---|---|
| FLUX.1-schnell | 12B | ~34 GB | 129 s/image warm (4-step 1024²) |
| **Qwen-Image** | **20.4B** | **~41 GB+** | **~7 min/image** (20-step 1024²) |
| Wan 2.1 T2V video | 1.3B–14B | 8–80 GB | **1.02× full-VRAM step time** — streaming is free for video, byte-identical output |

```python
from aircanvas import AirPipeline

pipe = AirPipeline.from_pretrained(
    "Qwen/Qwen-Image",  # 20B — split to nf4 shards on first use
    compression="nf4",
)
image = pipe("a watercolor fox reading a newspaper", num_inference_steps=20).images[0]
print(pipe.report())  # phase times, IO breakdown, residency plan
```

```bash
aircanvas split Qwen/Qwen-Image --compression nf4   # one-time reshard
aircanvas run Qwen/Qwen-Image -p "a watercolor fox" # generate from the CLI
aircanvas doctor                                    # what can THIS box run?
```

## How it works

1. **Split once** — the DiT is resharded into per-block safetensors (fp8 or
   NF4 payloads, quality-sensitive tensors kept high-precision), verified
   against the reader's own parser, resumable, and reusable forever — the
   original checkpoint can be deleted.
2. **Stream at runtime** — the model is built on the meta device (zero
   memory); a prefetch worker walks the recorded block schedule: disk →
   pinned ring → GPU slot pool over a dedicated CUDA stream. No allocations
   in the hot loop. Streamed output is bitwise-equal to full-VRAM execution
   (uncompressed) — enforced by tests and demonstrated on real video.
3. **Phase-aware placement** — text encoders run once and are evicted
   (embedding cache makes repeats free; encoders too big to LOAD on a small
   machine are themselves split and streamed), the DiT streams every step,
   the VAE decodes tiled. A budget solver turns free VRAM/RAM + measured
   disk bandwidth into the plan, printed with every OOM and report.
4. **Cache hierarchy** — leftover VRAM pins blocks resident; leftover RAM
   promotes shards after first read (disk touched once per run); NVMe serves
   the rest. Wan 2.2's dual experts hand over engines at the boundary
   timestep so pool memory is never doubled.

Supported today: FLUX.1, Qwen-Image, SD3/3.5, Wan 2.1/2.2 (incl. MoE
handover), HunyuanVideo, CogVideoX — plus a generic adapter that introspects
any diffusers DiT with a uniform block list.

## Install

```bash
pip install -e .[dev]     # from source (PyPI release pending)
pip install -e .[nf4]     # + bitsandbytes for NF4 shards (CUDA required)
```

Windows and Linux; Python ≥ 3.10; torch ≥ 2.4; diffusers ≥ 0.35. Developed
*on* a 6 GB / 16 GB Windows laptop — the low-resource path is the tested
path, including the failure modes (commit-pressure mmap fallbacks, resumable
everything).

## Honest expectations

On a 6 GB card AirCanvas is slower than a 24 GB card running natively —
minutes per image, near-parity for video. Its promise is different: the
model **runs at all**, correctly, with the bottleneck measured and printed.
See [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/RESEARCH.md](docs/RESEARCH.md).

## License

Apache-2.0.
