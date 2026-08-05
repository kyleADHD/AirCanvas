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

🚧 **Pre-alpha — design phase complete, implementation starting.** See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ROADMAP.md](docs/ROADMAP.md), and the research behind the design in [docs/RESEARCH.md](docs/RESEARCH.md).

## License

Apache-2.0.
