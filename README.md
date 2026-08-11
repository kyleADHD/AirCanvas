<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/aircanvas_logo_dark.svg">
    <img alt="AirCanvas" src="assets/aircanvas_logo.svg" width="450">
  </picture>
</p>

<p align="center">
  <b>Run 20B+ image & video generation models on 4–8 GB GPUs.</b>
</p>

<p align="center">
  <a href="#quickstart"><b>Quickstart</b></a> |
  <a href="#configurations"><b>Configurations</b></a> |
  <a href="#supported-models"><b>Supported Models</b></a> |
  <a href="#benchmarks"><b>Benchmarks</b></a> |
  <a href="#faq"><b>FAQ</b></a>
</p>

<p align="center">
  <a href="https://github.com/kyleADHD/AirCanvas/stargazers"><img src="https://img.shields.io/github/stars/kyleADHD/AirCanvas?style=social" alt="GitHub Repo stars"></a>
  <a href="https://github.com/kyleADHD/AirCanvas/actions/workflows/ci.yml"><img src="https://github.com/kyleADHD/AirCanvas/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg" alt="Code License"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <!-- Uncomment after the first PyPI release:
  <a href="https://pypi.org/project/aircanvas/"><img src="https://img.shields.io/pypi/v/aircanvas" alt="PyPI"></a>
  <a href="https://pepy.tech/project/aircanvas"><img src="https://static.pepy.tech/badge/aircanvas" alt="Downloads"></a>
  -->
</p>

Measured, not promised: a 20.4B **Qwen-Image** generating 1024² images on a
6 GB laptop RTX 4050 — a model that needs a 40 GB+ card by any other means.

AirCanvas is [AirLLM](https://github.com/lyogavin/airllm) for diffusion
models: it splits a model's transformer into per-block shards on disk
(fp8/NF4-quantized at split time), then streams each block to the GPU
just-in-time during the denoise loop. VRAM scales with the *largest block*,
not the model; system RAM holds only a small pinned prefetch ring.

Why the economics work where AirLLM's don't: an LLM re-reads the whole model
*per token*; a diffusion model runs 20–50 forward passes *total* — and video
steps are so compute-heavy the transfer vanishes entirely.

## Updates

- **[2026/08] v0.1.0 — first working release.** FLUX.1, Qwen-Image, SD3/3.5,
  Wan 2.1/2.2 (dual-expert handover), HunyuanVideo, CogVideoX adapters; fp8
  and NF4 shard formats; automatic VRAM/RAM/disk budget solver;
  `aircanvas doctor`.
- **[2026/08] Wan 2.1 14B verified end-to-end on a 6 GB GPU** — 562 GB of
  weights streamed through a single overnight run
  ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
- **[2026/08] Video streaming measured effectively free**: Wan 2.1 at
  480×832×81 frames runs at **1.02× the full-VRAM step time** with
  **byte-identical output**.

## Star History

<a href="https://star-history.com/#kyleADHD/AirCanvas&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=kyleADHD/AirCanvas&type=Date&theme=dark">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=kyleADHD/AirCanvas&type=Date">
  </picture>
</a>

## Table of Contents

- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Model compression: fp8 & NF4 shards](#model-compression-fp8--nf4-shards)
- [Configurations](#configurations)
- [Supported Models](#supported-models)
- [Benchmarks](#benchmarks)
- [FAQ](#faq)
- [Acknowledgement](#acknowledgement)
- [Citing AirCanvas](#citing-aircanvas)
- [Contribution](#contribution)

## Quickstart

### 1. Install

```bash
pip install -e .            # from source (PyPI release pending)
pip install -e .[nf4]       # + bitsandbytes for NF4 shards (CUDA required)
pip install -e .[dev]       # + test/lint tooling for contributors
```

Windows and Linux; Python ≥ 3.10; torch ≥ 2.4; diffusers ≥ 0.35. Developed
*on* a 6 GB / 16 GB Windows laptop — the low-resource path is the tested path,
including the failure modes (commit-pressure mmap fallbacks, resumable
everything).

### 2. Generate

```python
from aircanvas import AirPipeline

pipe = AirPipeline.from_pretrained(
    "Qwen/Qwen-Image",  # 20B — split to nf4 shards on first use
    compression="nf4",
)
image = pipe("a watercolor fox reading a newspaper", num_inference_steps=20).images[0]
print(pipe.report())  # phase times, IO breakdown, residency plan
```

The first run downloads the transformer, splits it into a persistent
per-block shard cache (outside the repo, under your HF cache), and streams
from there forever after — the original checkpoint can be deleted.

### 3. Or use the CLI

```bash
aircanvas doctor                                    # what can THIS box run?
aircanvas split Qwen/Qwen-Image --compression nf4   # one-time reshard
aircanvas run Qwen/Qwen-Image -p "a watercolor fox" # generate
```

## How it works

1. **Split once** — the DiT is resharded into per-block safetensors (fp8 or
   NF4 payloads, quality-sensitive tensors kept high-precision), verified
   against the reader's own parser, resumable, and reusable forever.
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

Full design docs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
[docs/RESEARCH.md](docs/RESEARCH.md).

## Model compression: fp8 & NF4 shards

Compression is applied **once, at split time** — runtime just streams smaller
files, so disk traffic (usually the bottleneck for images) drops with it:

| Format | Payload | What stays high-precision | Best for |
|---|---|---|---|
| `None` | source dtype | everything | bitwise-exact verification |
| `"fp8"` (default) | `float8_e4m3fn` + per-tensor scale | norms, modulation, embeddings, all 1-D params | ~1.5–2× less disk traffic, near-lossless |
| `"nf4"` | packed 4-bit (bitsandbytes) + absmax | same | 20B+ models on tiny disks/RAM (CUDA only) |

On the 6 GB dev machine, fp8 shards cut streamed step time from ~300 ms to
150–172 ms on the benchmark model — from IO-bound to compute-bound. NF4
dequantization runs through bitsandbytes' kernel straight into a
pre-allocated pool and is tested bitwise-identical to the reference path.

## Configurations

Everything routes through `AirPipeline.from_pretrained(model_id, ...)`:

| Parameter | Default | Description |
|---|---|---|
| `vram_budget` | `"auto"` | VRAM cap, e.g. `"6GB"`; `"auto"` probes free VRAM |
| `ram_budget` | `"auto"` | system-RAM cap for the pinned ring + shard cache |
| `compression` | `"fp8"` | `None`, `"fp8"`, or `"nf4"` (NF4 needs `[nf4]` extra + CUDA) |
| `shard_cache` | HF cache | where shards live — never inside the repo |
| `device` | auto | CUDA if available |
| `compute_dtype` | `"bfloat16"` | dtype blocks are decompressed to |
| `max_resident_blocks` | solver | pin the first N blocks permanently in VRAM |
| `prefetch` | `True` | background pipelined streaming (off = synchronous loads) |
| `cache_embeddings` | `True` | disk-cache text embeddings; repeat prompts skip the encoders entirely |
| `subfolder` / `revision` / `hf_token` | — | standard Hub options |
| `text_encoder_device` | auto | where the load-run-evict encoders execute |
| `**pipeline_kwargs` | — | forwarded to the underlying diffusers pipeline |

`pipe.report()` prints per-phase wall time, the IO/prefetch breakdown, and
the residency plan the solver chose. `aircanvas doctor` runs the same solver
against your hardware and prints a table of what your box can run.

## Supported Models

| Model | Kind | Params | Status |
|---|---|---|---|
| FLUX.1 (schnell / dev) | image | 12B | ✅ verified on a 6 GB GPU |
| Qwen-Image | image | 20.4B | ✅ verified on a 6 GB GPU (NF4) |
| Wan 2.1 T2V | video | 1.3B / 14B | ✅ verified on a 6 GB GPU, byte-identical streaming |
| Wan 2.2 (A14B MoE) | video | 2×14B | 🧪 adapter complete incl. expert handover; full-scale run pending |
| SD3 / SD3.5 | image | 2–8B | 🧪 validated against the live model index |
| HunyuanVideo | video | 13B | 🧪 validated against the live model index |
| CogVideoX | video | 2–5B | 🧪 validated against the live model index |
| Any diffusers DiT | — | — | generic adapter introspects uniform block lists |

## Benchmarks

All numbers measured on the low-end dev machine (RTX 4050 Laptop 6 GB, 16 GB
RAM, NVMe) — details and methodology in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md):

| Model | Params | Needs normally | AirCanvas on 6 GB |
|---|---|---|---|
| FLUX.1-schnell | 12B | ~34 GB | 129 s/image warm (4-step 1024²) |
| **Qwen-Image** | **20.4B** | **~41 GB+** | **~7 min/image** (20-step 1024²) |
| Wan 2.1 T2V | 1.3B–14B | 8–80 GB | **1.02× full-VRAM step time** — streaming is free for video |

## FAQ

### 1. Is it slower than running on a big GPU?

Yes — that's the honest trade. On a 6 GB card AirCanvas is slower than a
24 GB card running natively: minutes per image, near-parity for video (video
steps are so compute-heavy the streaming hides completely). Its promise is
different: the model **runs at all**, correctly, with the bottleneck measured
and printed by `pipe.report()`.

### 2. Does streaming change the output?

No. Streamed output is **bitwise-equal** to full-VRAM execution with
`compression=None` — enforced by the test suite and demonstrated
byte-identical on real video. fp8/NF4 introduce only the usual quantization
tolerance, applied once at split time.

### 3. Where do the shards go, and how big are they?

In a persistent cache under your HF cache home (override with
`shard_cache=`), never inside the repo. Size ≈ the transformer at the chosen
precision (roughly ½ for fp8, ¼ for NF4). After splitting you can delete the
original checkpoint.

### 4. I got an out-of-memory error

`InsufficientVRAMError` prints the exact plan the solver attempted. The usual
ladder: enable `compression="fp8"` (or `"nf4"`), lower resolution/frames, or
set an explicit `vram_budget` below what other apps are using.

### 5. NF4 says it requires CUDA

Correct — NF4 shards use bitsandbytes' dequantization kernels, which are
CUDA-only. This is enforced with a clear error at split *and* load. Use
`"fp8"` on non-CUDA setups.

### 6. Do I need an NVMe drive?

Strongly recommended. The budget solver measures your disk's real bandwidth
and warns when a SATA drive (or a distilled few-step model, which hides IO
less) will make streaming the bottleneck.

### 7. Does it work on Windows?

Windows is the *primary* development machine — OneDrive-safe paths,
no-`mmap`-under-commit-pressure fallbacks, and cross-platform memory probing
are all part of the tested path.

## Acknowledgement

- [AirLLM](https://github.com/lyogavin/airllm) — the layered-inference idea
  this project adapts to diffusion, and the namesake.
- [diffusers](https://github.com/huggingface/diffusers) — AirCanvas wraps
  stock pipelines and owns only device/memory placement.
- [safetensors](https://github.com/huggingface/safetensors),
  [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes),
  [accelerate](https://github.com/huggingface/accelerate).

## Citing AirCanvas

```bibtex
@software{aircanvas,
  author = {Hulley, Kyle},
  title  = {AirCanvas: Run 20B+ image and video diffusion models on 4--8 GB GPUs},
  url    = {https://github.com/kyleADHD/AirCanvas},
  year   = {2026}
}
```

## Contribution

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
dev setup, the test tiers, and the hard rules (bitwise-equivalence gate, no
allocations in the hot loop) every PR must keep. Security reports go through
[SECURITY.md](SECURITY.md), never public issues.

## License

Apache-2.0.
