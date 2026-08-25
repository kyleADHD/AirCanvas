# AirCanvas Studio

A local web UI for [AirCanvas](https://github.com/kyleADHD/AirCanvas). Separate
package from the library: the library owns streaming; this owns the desk.

```
studio/                  ← this project
  aircanvas_studio/      FastAPI → AirPipeline
  README.md
```

First launch is a **welcome setup**: pick image and/or video models, then
AirCanvas splits each DiT into streamed shards. After that the desk is
Imagine-style — prompt dock, a layout that follows the model (text,
reference stills, edit, video, start-frame), plus Gallery and Settings.

The Grok preview is demo mode (no GPU). On your box this server is live.

## Features wired to the library

| Control | AirCanvas API |
|---|---|
| Model picker (FLUX, Qwen-Image, SD3.5, Wan 2.1/2.2, Hunyuan, CogVideoX) | `AirPipeline.from_pretrained(repo)` |
| fp8 / NF4 / bf16 shards | `compression=` |
| VRAM / RAM caps | `vram_budget=` / `ram_budget=` |
| GGUF source | `gguf_file="repo:file.gguf"` |
| LoRA stack + scale | `pipe.load_lora(source, scale=)` — fuse after dequant |
| Prefetch / embedding cache | `prefetch=` / `cache_embeddings=` |
| Steps, size, frames, seed, guidance | forwarded to the wrapped diffusers call |
| Report | `pipe.report(as_dict=True)` |

## Run (GPU machine)

From the AirCanvas checkout:

```bash
pip install -e .[nf4]
pip install -e ./studio
aircanvas-studio --host 0.0.0.0 --port 7860
```

Open the Studio UI (this preview, or any client pointing at `/api`). CORS is
open for local use.

```
POST /api/setup      { repos, compression, hf_token, gguf_file }
POST /api/load       { repo, compression, vram_budget, gguf_file, hf_token }
POST /api/lora       { source, scale, weight_name }
POST /api/generate   { prompt, steps, width, height, num_frames, seed }
GET  /api/models
GET  /api/health
```

Gated models (FLUX.1-dev, SD3.5) need `hf_token` after you accept the Hub
licence. NF4 needs CUDA + `aircanvas[nf4]`.
