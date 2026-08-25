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

## M4 — Phase-aware AirPipeline + fp8 shards (**mechanism complete**; headline demo pending a FLUX download)
- TE load-run-evict, embedding cache, VAE tiling passthrough. ✅
- fp8 shard storage + upcast-on-load (skip norms/modulation); quant-tolerance tests. ✅
- Budget solver v1 (waterfall policy) + `aircanvas doctor` + `pipe.report()`. ✅
- Verified end-to-end on `hf-internal-testing/tiny-flux-pipe` (`pytest -m network`): split-on-first-use → TE encode+evict → streamed DiT denoise → VAE decode → image.
- **Not yet verified: the acceptance number itself** (FLUX.1-dev ≤ 90 s/image on 6 GB). FLUX.1-dev is a gated ~34 GB download and FLUX.1-schnell is ~33 GB against 57 GB free on the dev box (checkpoint + shard cache would not fit comfortably). Carried into M5 as the first verification task — see below.

Synthetic benchmark on the dev box (RTX 4050 6 GB, 24×2048 blocks, bf16, `benchmarks/bench_stream.py --compression both --resident 6`), which is IO-bound like a real image DiT:

| configuration | ms/step | vs full-VRAM |
|---|---|---|
| full-VRAM reference | 104–114 | 1.0× |
| bf16 shards, prefetched | 296–310 | ~2.8× |
| bf16 shards, prefetched + 6 resident | 228–231 | ~2.1× |
| **fp8 shards, prefetched** | **150–172** | **~1.5×** |
| fp8 shards, prefetched + 6 resident | 155–207 | ~1.7× |

fp8 halves disk traffic and roughly halves step time on this workload. Resident blocks buy ~25% while IO-bound (bf16) and nothing once fp8 has pushed the workload toward compute-bound — which is the budget solver working as designed, not a regression.

## M5 — Qwen-Image 20B (**code complete**; done when: **Qwen-Image ≤ 4 min/image on 6–8 GB VRAM** — a model that cannot run on this hardware any other way)
- `qwen_image` adapter (60 uniform blocks, validated against the live checkpoint index; `encode_prompt_outputs=(prompt_embeds, prompt_embeds_mask)`). ✅
- NF4 shard option (bitsandbytes 0.50): packed 4-bit payload + absmax sibling + original shape in shard `__metadata__`; QuantState rebuilt at load (code table reproduced, not stored); dequant via bnb kernel straight into the pre-allocated second pool (`out=`, ADR #8 holds). CUDA-only at split AND load, enforced with clear errors. Streamed output verified **bitwise-identical** to allocating dequantization across sync and prefetch paths. ✅
- **MEASURED (2026-08-07, RTX 4050 Laptop 6 GB, 16 GB RAM, fp8 shards)** — FLUX.1-schnell 1024², first real-model runs: **cold 4-step 385 s** (encode 144 s on CPU — T5 doesn't fit 6 GB; decode 30 s), **warm 4-step 129 s** (embeddings cached, encode 0 s), **28-step dev-equivalent ~14.1 s/step steady ≈ 7 min/image**. The ≤ 90 s target is missed ~4.5× on THIS box for measured reasons, not design reasons: effective disk read was ~1.6 GB/s under heavy memory pressure (target math assumed 5–7 GB/s NVMe), the 4050 Laptop is ~4× a 4090's step compute, and a per-call engine rebuild costs one all-sync step (dominant at 4 steps, amortised at 28). A machine with a healthy NVMe + 32 GB RAM should land near target; re-verify there. Resilience note: the 28-step run survived a mid-denoise laptop sleep (~20 h gap) and completed correctly.
- **Qwen-Image 20B MEASURED (2026-08-07, same box, nf4 shards)**: 1024², true CFG 4.0 — **20-step 427 s (~7.1 min)**, 50-step 1034 s; encode 0 s warm (16.6 GB Qwen2.5-VL auto-routed to CPU, cached after first prompt); 4 resident blocks + 56 streamed (10.7 GB/step), 97.5 % prefetch rate, 5.21 GB VRAM planned of 5.32 — **a model that needs a 40 GB+ card by any other means**. The ≤4 min target misses ~1.8× here for the same disk/CFG reasons as FLUX. Field bugs fixed en route: inter-call VRAM measurement (clean before measuring) and resident-block leak across engines (close() now releases all weights).

## M6 — Video: Wan 2.1 — **GATE PASSED at 1.02× on 1.3B; 14B COMPLETED on-box** (10.6 h overnight, 562 GB streamed, details in BENCHMARKS.md)
- **MEASURED (2026-08-08, RTX 4050 6 GB, fp8, 480×832×81f, 20 steps, true CFG)**: streamed 49.36 s/step vs all-resident 48.26 s/step = **1.02× (target ≤ 1.10×)** — and the two videos are **byte-identical (same MD5)**. Streaming is effectively free for video, as RESEARCH.md §4 predicted; the 14B ratio should be even better (compute grows faster than transfer).
- En route, the 11.4 GB UMT5 (unloadable on this 16 GB box — OSError 1455 in every stock loader) was **split and streamed through our own engine** for the encode: streamable text encoders are now real infrastructure, and the designed answer for M7's 15 GB Llava.
- `wan` adapter; UMT5 evict strategy; **tiled Wan-VAE decode** (missing upstream — our contribution).
- SDPA/Flash attention verification at video token counts; CFG batch handling.
- Optional **token-chunked FFN execution** (adapter knob): split the ~75k-token sequence through per-block MLPs so the 2–3 GB FFN transient drops to ~0.5 GB at a few % speed cost — makes 720p comfortable on 6 GB.

## M7 — Wan 2.2 MoE + HunyuanVideo — **code complete** (real-model runs pending 26-57 GB downloads)
- Per-timestep expert handover: the low-noise expert's engine is built lazily inside its first forward pre-hook after all prior engines are released — pool VRAM never doubles; transformer_2 auto-detected from model_index.json and meta-built so 28 GB never touches RAM. ✅ (toy-verified)
- HunyuanVideo adapter (20 dual + 40 single, validated against the live index; guidance-distilled, one forward/step); its 15 GB Llava TE uses the split-and-stream recipe proven on UMT5 in M6. ✅

## M8 — Polish & release — **done** (PyPI upload itself needs an account token)
- SD3.5 + CogVideoX adapters ✅ (CogVideoX validated against the live index; SD3.5 gated, pinned from card). RAM-cache auto-promotion tier live ✅ (blobs promoted to pageable RAM after first read; hits in report()). GitHub Actions CI (Windows+Linux, lint+mypy+CPU tests) ✅. docs/BENCHMARKS.md + README rewritten around measured numbers ✅. v0.1.0 + CHANGELOG tagged ✅.

## M9 — GGUF sources — **done** (native K-quant streaming still deferred)
- Split directly from a quantized `.gguf` (`--gguf-file repo:file.gguf`):
  dequant at split time into the verified none/fp8/nf4 codecs. ✅

## M10 — LoRA loading onto quantized shards — **done**
- Fuse PEFT / kohya `.safetensors` adapters **after dequant**, in-place, into
  the compute-dtype views the engine binds. Works with fp8/NF4 because the
  adapter never touches the quantized payload. ✅
- `pipe.load_lora(source, scale=…)` / `unload_lora` / `set_lora_scale`;
  `lora=` on `from_pretrained`; `aircanvas run --lora`. Multiple adapters
  stack. Text-encoder keys skipped (TEs are load-run-evict). ✅
- Hot-loop contract: `down`/`up` materialized onto the engine device once at
  construction; per-block path is `weight.addmm_(up, down, alpha=scale)` into
  already-owned slot storage. Bitwise-equal to a fused full-VRAM reference
  at `compression=None`. ✅

## Later / explicitly deferred
- HiDream (4-TE stack + MoE FFN); GGUF Q4 shard format (native K-quant
  streaming); io_uring/GDS fast path; upstreaming pieces to diffusers;
  distilled-model RAM-tier heuristics; multi-GPU; text-encoder LoRAs;
  LoRA training / merging into a new shard cache.

## Risks
| Risk | Mitigation |
|---|---|
| Streaming overhead visible on image (unlike video) | fp8/NF4 traffic cut is the plan; RAM-cache tier when available; set expectations in README (minutes, not seconds) |
| diffusers internals shift under us | Pin lower bounds, adapters isolate per-model contact points, toy-model CI catches breakage |
| Windows IO/pinned-memory perf surprises | Dev machine IS Windows; benchmark from M3 onward |
| Distilled few-step models erode the niche | Budget solver detects & prefers RAM tier; still serves the "can't run at all" segment |
