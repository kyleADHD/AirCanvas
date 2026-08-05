# AirCanvas Roadmap

Each milestone has a demoable outcome and acceptance criteria. Numbers reference [ARCHITECTURE.md](ARCHITECTURE.md) §6.

## M0 — Scaffold (done when: CI green on empty package)
- `src/` layout, pyproject, ruff, pytest, GitHub Actions (Windows + Linux, CPU).
- Package skeleton with module stubs matching ARCHITECTURE.md §3.

## M1 — Splitter + manifest (done when: any diffusers DiT splits to a valid shard cache)
- Generic adapter (ModuleList introspection) + `flux` adapter.
- `aircanvas split` CLI; `.done` markers; disk preflight; manifest round-trip tests.
- Toy-model unit tests (4-block synthetic FLUX config, CPU CI).

## M2 — Streaming engine, bf16, single-buffer (done when: toy model streamed == full-VRAM reference, bitwise)
- Meta-device init, hooks, GPU slot pool (start with 1 slot, synchronous loads).
- Schedule recording. Correctness before speed.

## M3 — Prefetch pipeline (done when: FLUX.1-schnell generates on the dev box within 2× of RAM-offload baseline)
- Pinned ring, IO thread pool, CUDA copy stream + events, lookahead K.
- `pipe.report()` timing breakdown (io / h2d / dequant / compute).

## M4 — Phase-aware AirPipeline + fp8 shards (done when: **FLUX.1-dev ≤ 90 s/image on 6 GB VRAM + 8 GB RAM** — the headline image demo)
- TE load-run-evict, embedding cache, VAE tiling passthrough.
- fp8 shard storage + upcast-on-load (skip norms/modulation); quant-tolerance tests.
- Budget solver v1 (waterfall policy) + `aircanvas doctor`.

## M5 — Qwen-Image 20B (done when: **Qwen-Image ≤ 4 min/image on 6–8 GB VRAM** — a model that cannot run on this hardware any other way)
- `qwen_image` adapter (60 uniform blocks), Qwen2.5-VL TE strategy (group-offload or CPU).
- NF4 shard option (bitsandbytes).

## M6 — Video: Wan 2.1 (done when: **Wan 14B 480p×81f on 6 GB VRAM, step time ≤ 110% of full-VRAM**)
- `wan` adapter; UMT5 evict strategy; **tiled Wan-VAE decode** (missing upstream — our contribution).
- SDPA/Flash attention verification at video token counts; CFG batch handling.
- Optional **token-chunked FFN execution** (adapter knob): split the ~75k-token sequence through per-block MLPs so the 2–3 GB FFN transient drops to ~0.5 GB at a few % speed cost — makes 720p comfortable on 6 GB.

## M7 — Wan 2.2 MoE + HunyuanVideo
- Per-timestep expert scheduling (only active expert's shards stream).
- Hunyuan dual/single blocks; Llava-8B TE evict; guidance-distilled path (no CFG).

## M8 — Polish & release
- SD3.5-L, CogVideoX adapters; RAM-cache auto-promotion tier; benchmark suite results published; README with reproducible numbers; PyPI release.

## Later / explicitly deferred
- LoRA loading onto quantized shards; HiDream (4-TE stack + MoE FFN); GGUF Q4 shard format; io_uring/GDS fast path; upstreaming pieces to diffusers; distilled-model RAM-tier heuristics; multi-GPU.

## Risks
| Risk | Mitigation |
|---|---|
| Streaming overhead visible on image (unlike video) | fp8/NF4 traffic cut is the plan; RAM-cache tier when available; set expectations in README (minutes, not seconds) |
| diffusers internals shift under us | Pin lower bounds, adapters isolate per-model contact points, toy-model CI catches breakage |
| Windows IO/pinned-memory perf surprises | Dev machine IS Windows; benchmark from M3 onward |
| Distilled few-step models erode the niche | Budget solver detects & prefers RAM tier; still serves the "can't run at all" segment |
