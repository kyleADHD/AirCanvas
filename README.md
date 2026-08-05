# AirCanvas

**Run 20B+ image & video generation models on 4–8 GB GPUs.**

AirCanvas is [AirLLM](https://github.com/lyogavin/airllm) for diffusion models: it splits a model's transformer into per-block shards on disk (optionally fp8/NF4-quantized), then streams each block to the GPU just-in-time during the denoise loop — so VRAM scales with the *largest block*, not the model, and system RAM only holds a small prefetch ring.

Where AirLLM is impractically slow for LLM chat (the whole model re-read per *token*), the economics flip for diffusion: an image is 20–50 forward passes *total*, and video DiT steps are so compute-heavy that streaming from NVMe hides almost entirely behind compute.

```python
from aircanvas import AirPipeline

pipe = AirPipeline.from_pretrained(
    "Qwen/Qwen-Image",  # 20B — normally needs a 40 GB+ card
    vram_budget="auto",
    ram_budget="auto",
    compression="fp8",
)
image = pipe("a watercolor fox reading a newspaper", num_inference_steps=30).images[0]
print(pipe.report())  # per-phase time, block IO breakdown, and the memory plan used
```

Not sure what your machine can handle?

```
$ aircanvas doctor
device        cuda  NVIDIA GeForce RTX 4050 Laptop GPU
VRAM          5.32 GB free / 6.44 GB total
RAM           2.63 GB free / 16.87 GB total

model                      blocks  DiT bf16  fp8 shards           no compression
FLUX.1-dev                     57     23.8G  yes (~3 min IO)      yes (~5 min IO)
Qwen-Image                     60     41.0G  yes (~10 min IO)     yes (~16 min IO)
...
```

## Planned model support

| Model | Size | Normally needs | AirCanvas target |
|---|---|---|---|
| FLUX.1 dev/schnell | 12B | ~34 GB | 6 GB VRAM, ≤90 s/image |
| Qwen-Image | 20B | ~42 GB | 6–8 GB VRAM, ≤4 min/image |
| SD3.5 Large | 8B | ~24 GB | 6 GB VRAM |
| Wan 2.1/2.2 T2V/I2V 14B | 14–27B | 80 GB class | 6 GB (480p) / 8 GB (720p), ~free streaming |
| HunyuanVideo | 13B | 45–60 GB | 6–8 GB VRAM |
| CogVideoX-5B | 5B | 26 GB | correctness/CI target |

## Status

🚧 **Pre-alpha.** The full path works end to end — splitter, fp8 shard cache, prefetched streaming engine, budget solver, and the phase-aware `AirPipeline` — and is exercised against a real (tiny) diffusers pipeline in CI. What is *not* yet done is running it on a flagship model: the numbers in the table above are targets derived from [docs/RESEARCH.md](docs/RESEARCH.md), not measurements. Adapters beyond FLUX land in M5–M8.

Measured so far on an RTX 4050 (6 GB), synthetic 24×2048 bf16 DiT, IO-bound like a real image model:

| | ms/step | vs full-VRAM |
|---|---|---|
| full-VRAM reference | 104–114 | 1.0× |
| bf16 shards, streamed | 296–310 | ~2.8× |
| **fp8 shards, streamed** | **150–172** | **~1.5×** |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ROADMAP.md](docs/ROADMAP.md), and the research behind the design in [docs/RESEARCH.md](docs/RESEARCH.md).

## License

Apache-2.0.
