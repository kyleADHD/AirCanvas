# AirCanvas Architecture

> Run 20B+ image and video diffusion models on 4–8 GB GPUs (and modest system RAM) by streaming quantized model blocks from disk, AirLLM-style — but with economics that actually work, because diffusion models run 20–50 forward passes per output instead of one per token.

Factual grounding for every number here: [RESEARCH.md](RESEARCH.md).

## 1. Goals and non-goals

**Goals**
- G1. One-line API: `AirPipeline.from_pretrained(model_id, vram_budget="6GB", ram_budget="8GB")` works for any supported diffusers pipeline.
- G2. VRAM floor set by *largest block + activations*, not model size; RAM floor set by *prefetch ring*, not model size.
- G3. Quantized persistent shard cache (fp8/NF4) — split once, reuse forever, shareable.
- G4. Phase-aware execution: text encoders load-run-evict, DiT pipelined-streamed, VAE tiled.
- G5. Automatic budgeting: fill spare VRAM with resident blocks; promote hot shards to pinned RAM; degrade gracefully to NVMe.
- G6. Windows and Linux first-class (primary dev machine is Windows 11).

**Non-goals (v1)**
- Training/fine-tuning/LoRA merging (LoRA *loading* is a later milestone).
- UNet-era models (SDXL and older — heterogeneous conv blocks, and they don't need us).
- Custom CUDA kernels (we compose existing ones: SDPA/FlashAttention, bnb dequant, torch fp8 casts).
- Multi-GPU.
- Serving/batching infrastructure.

## 2. System overview

```
                         ┌─────────────────────────────────────────────┐
 one-time                │  SPLIT (aircanvas split / implicit on load)  │
 ─────────►  HF Hub ───► │  checkpoint index → block plan (adapter)     │
                         │  → per-block shards (safetensors, fp8/NF4)   │
                         │  → manifest.json + .done markers             │
                         └───────────────────┬─────────────────────────┘
                                             │  shard cache dir
 every run                                   ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ AirPipeline (wraps DiffusionPipeline)                                 │
 │                                                                       │
 │  Phase 1: TEXT ENCODE     Phase 2: DENOISE LOOP        Phase 3: DECODE│
 │  load TE → run → evict    for step in steps:           tiled VAE      │
 │  (or CPU inference)         for block in schedule:     (incl. Wan-VAE │
 │                               [streamed via engine]     tiling we    │
 │                                                          ship)        │
 │                                                                       │
 │  ┌─────────────── StreamingEngine (DiT only) ─────────────────────┐  │
 │  │ meta-device model + pre/post forward hooks per block            │  │
 │  │                                                                 │  │
 │  │  NVMe ──io threads──► pinned CPU ring ──copy stream──► GPU slot │  │
 │  │        (lookahead K from deterministic schedule; CUDA events    │  │
 │  │         sync copy-stream → compute; dequant-on-load fp8→bf16    │  │
 │  │         or bnb NF4 dequant on GPU)                              │  │
 │  │                                                                 │  │
 │  │  ResidencyPlanner: first R blocks permanently GPU-resident;     │  │
 │  │  hot shards cached in pinned RAM when ram_budget allows         │  │
 │  └─────────────────────────────────────────────────────────────────┘  │
 └───────────────────────────────────────────────────────────────────────┘
```

## 3. Components

### 3.1 Sharding (`aircanvas.sharding`)

**`splitter.py`** — one-time transform, AirLLM Phase A adapted to diffusers:
- Downloads checkpoint metadata first (`snapshot_download` with weight files excluded), then streams weight shards on demand; `delete_original` option keeps peak disk ≈ 1× model size.
- Uses the adapter's **BlockPlan** (§3.4) to group tensor names by block; writes one safetensors file per block + `.done` marker (write-then-rename for atomicity; incomplete split → re-split).
- Non-block tensors (embedders, final norm/proj, modulation tables) go into a single `resident.safetensors` — they stay on GPU permanently (~0.5 GB).
- Only the **DiT** is sharded. Text encoders and VAE keep their stock single-file form (they use cheaper strategies).

**`quant.py`** — compression applied at split time, recorded in the manifest:
- `fp8` (default): `float8_e4m3fn` payload + a per-tensor fp32 scale (`<name>.__ac_scale`), upcast to compute dtype on GPU at load. No extra deps. Skip list: norms, modulation/AdaLN, embeddings (tiny + sensitive), plus every 1-D parameter. **The real traffic cut is ~1.5–2×, not a clean 2×**: on FLUX-style blocks the skipped `norm1.linear` modulation projection alone is a third of the bytes. Mixed-dtype shards are written largest-element-size-first so every offset stays aligned to its element size — otherwise `prefetch.ShardHeader` refuses the shard (the splitter verifies this with the reader's own parser before writing `.done`).
- `nf4`: bitsandbytes `quantize_nf4` at split, `dequantize_nf4` on GPU at load (AirLLM's exact recipe — quant state stored as sibling tensors). 4× traffic cut.
- `none`: bf16 passthrough.

**`manifest.py`** — `manifest.json` schema (versioned):

```json
{
  "aircanvas_manifest_version": 1,
  "source": {"repo_id": "Qwen/Qwen-Image", "revision": "…", "subfolder": "transformer"},
  "model_class": "QwenImageTransformer2DModel",
  "adapter": "qwen_image",
  "compression": "fp8",
  "compute_dtype": "bfloat16",
  "blocks": [
    {"name": "transformer_blocks.0", "file": "block_0000.safetensors", "bytes": 356515840, "sha256": "…"}
  ],
  "resident": {"file": "resident.safetensors", "bytes": 512000000},
  "expert_groups": null
}
```

`expert_groups` covers Wan 2.2 (two per-timestep experts → two block lists with a scheduling rule).

### 3.2 Streaming engine (`aircanvas.streaming`)

**`engine.py`** — the AirLLM mechanism, upgraded:
- Instantiate the DiT via `from_config` under `init_empty_weights` (meta device, zero memory).
- Register pre/post forward hooks on each block from the BlockPlan. Pre-hook: wait on the block's GPU-ready CUDA event, bind weights (`set_module_tensor_to_device`). Post-hook: release the GPU slot back to the pool (params → meta), notify the prefetcher.
- **GPU slot pool**: 2–3 reusable weight buffers sized to the largest block (double/triple buffering) instead of alloc/free per block — avoids allocator churn and `empty_cache()` calls in the hot loop (AirLLM's `gc.collect()`-per-layer is a known cost we design out).
- First forward records the **block execution schedule**; subsequent steps prefetch against it with lookahead K (video schedules are static; recorded once).

**`prefetch.py`** — 3-stage pipeline:
1. **Disk → pinned ring** — small thread pool reads shard files into a fixed ring of pinned CPU buffers (ring depth = budget-solver output, default 2–4 blocks). On Windows, plain buffered reads first; `O_DIRECT`-style optimizations are a later optimization milestone.
2. **Pinned ring → GPU** — dedicated CUDA copy stream, `copy_(non_blocking=True)`, records a per-block "ready" event; fp8→bf16 upcast or NF4 dequant happens on GPU immediately after the copy.
3. **Compute** — default stream waits on the ready event in the pre-hook.

**`residency.py`** — the budget solver:
- Inputs: free VRAM, free RAM, measured disk read bandwidth (probed once, cached), manifest (block sizes), workload (steps, resolution → activation estimate).
- Outputs: number of permanently GPU-resident blocks R (fill spare VRAM), pinned-RAM shard-cache size (promote most-recently-used shards; with enough RAM the whole model lives in RAM and disk is only touched on first step), ring depth, warnings ("SATA SSD detected: expect ~40% overhead", "distilled 4-step model: streaming overhead will be visible; prefer ram cache").
- Policy is a simple waterfall, not an ILP: activations + slot pool reserved first; leftover VRAM → resident blocks; leftover RAM → shard cache.
- Resident blocks are taken from the **front** of the manifest. Any subset would do (every block runs once per step), but the front is deterministic and, on two-species models, happens to be the big blocks — so pinning them also shrinks the pool the streamed tail needs.
- **Status:** R, ring depth and the slot pool are live as of M4. `ram_cache_bytes` is *sized and reported* but the pinned-RAM tier that would consume it is M8; `ResidencyPlan.describe()` labels it as planned-not-active so a report never overstates what ran.

### 3.3 Runtime orchestration (`aircanvas.runtime`)

**`orchestrator.py`** — `AirPipeline`: wraps a stock `DiffusionPipeline` (we keep diffusers' schedulers, samplers, prompt handling — we only own device/memory placement):
- Phase 1 *encode*: load each text encoder to GPU (whole, or group-offloaded if it alone exceeds budget — HunyuanVideo's 15 GB Llava, HiDream's 4-TE stack), run, cache embeddings, evict. Optional `text_encoder_device="cpu"` mode.
- Phase 2 *denoise*: hand the DiT to the StreamingEngine; run the stock pipeline loop.
- Phase 3 *decode*: tiled VAE decode. **`vae.py`** ships spatial tiling for `AutoencoderKLWan` (missing upstream — diffusers only frame-chunks it) plus passthrough to native `enable_tiling()` elsewhere.
- Embeddings for common negative/empty prompts cached on disk (skips TE load entirely for repeat runs).

### 3.4 Model adapters (`aircanvas.adapters`)

Per-family modules that answer four questions; everything else stays generic:
1. **BlockPlan**: which `ModuleList`(s) are the streamable blocks (`transformer_blocks` + `single_transformer_blocks` for FLUX/Hunyuan; `blocks` ×40 for Wan; ×60 for Qwen-Image), what's resident, what's an expert group (Wan 2.2: per-timestep expert schedule).
- 2. **Component strategy**: TE list + recommended strategy (CPU-able? droppable like SD3.5's T5?), VAE tiling entry point.
3. **Quirks**: T5 dtype-casting forward (breaks naive fp8 storage — keep T5 in bf16 or special-case), attention backend preferences, guidance-distilled (no CFG doubling).
4. **Workload model**: token count as f(resolution, frames) → activation-memory estimate for the budget solver.

`base.py` has a **generic adapter** that introspects any `*Transformer*Model` for `ModuleList`s of identical blocks — unknown DiTs get best-effort support; named adapters (`flux.py`, `qwen_image.py`, `sd3.py`, `wan.py`, `hunyuan_video.py`, `cogvideox.py`) pin down the flagships.

### 3.5 Public API (`aircanvas.api`, `aircanvas.cli`)

```python
from aircanvas import AirPipeline

pipe = AirPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    vram_budget="auto",  # "6GB" | "auto" (probe free VRAM)
    ram_budget="auto",
    compression="fp8",  # None | "fp8" | "nf4"
    shard_cache=None,  # default: <HF cache>/aircanvas/<model>/<compression>/
)
image = pipe(prompt="…", num_inference_steps=30).images[0]
pipe.report()  # where time went: io / h2d / compute / dequant, hit rates
```

CLI: `aircanvas split <repo_id> [--compression fp8] [--delete-original]`, `aircanvas run <repo_id> -p "…"`, `aircanvas doctor` (probe VRAM/RAM/disk bw, print what's runnable).

First call without a shard cache triggers an implicit split (with progress bar + disk-space preflight check, AirLLM's `NotEnoughSpaceException` equivalent).

## 4. Failure/atomicity model

- Shard writes: temp file → fsync → rename; `.done` marker per shard; manifest written last. Any missing marker → that shard re-split.
- Manifest carries source revision — stale cache detected on model update.
- Every OOM path reports the *plan* ("resident=6 blocks, ring=3, activations est. 2.4 GB") so failures are debuggable; `pipe.report()` after success.

## 5. Cross-platform notes (Windows is primary dev)

- No `malloc_trim` (AirLLM calls libc directly — Linux-only); `utils/memory.py` provides platform-appropriate cleanup, and the slot-pool design avoids needing per-block cache purges at all.
- Pinned memory allocation is slower on Windows → allocate the ring once at startup, never in the loop.
- Paths: `pathlib` everywhere; shard cache defaults **outside OneDrive** (`%LOCALAPPDATA%` / HF cache) — never sync 40 GB of shards to the cloud.
- Disk-bandwidth probe uses a real read of the first shard, not platform APIs.

## 6. Performance targets (acceptance criteria)

| Workload | Hardware envelope | Target |
|---|---|---|
| FLUX.1-dev 1024², 28 steps | 6 GB VRAM, 8 GB RAM, NVMe | ≤ 90 s/image (fp8), ≤ 60 s (nf4) |
| FLUX.1-schnell 4 steps | same | ≤ 15 s/image |
| Qwen-Image 20B, 50 steps | 6–8 GB VRAM, 8 GB RAM | ≤ 4 min/image |
| Wan 2.1 14B, 480×832×81f, 30 steps | 6 GB VRAM, 8 GB RAM | ≤ 110% of full-VRAM step time (streaming ~free) |
| Wan 2.1 14B, 720p×81f | 8 GB VRAM | runs; ≤ 110% step time |
| HunyuanVideo 544×960×129f | 6–8 GB VRAM | runs |
| Any of the above with 32 GB RAM | — | shards auto-promoted to RAM cache; ≥ 2× faster than NVMe tier |

Baselines to beat (from research): diffusers disk group-offload = 11.7 GB VRAM / 55.5 s FLUX; kijai/Wan2GP = 10–16 GB VRAM + 24–48 GB RAM floors.

## 7. Testing strategy

- **Unit** (CI, CPU-only): manifest round-trip; splitter on a tiny synthetic DiT (build a 4-block toy `FluxTransformer2DModel` config); quant round-trip error bounds; budget solver given synthetic hardware profiles; adapter BlockPlans against pinned diffusers configs.
- **Integration** (GPU runner / manual): toy-model end-to-end equivalence — streamed output must be bitwise-equal (`none`) or within quant tolerance (fp8/nf4) vs full-VRAM reference; schedule recording; ring under stress (slot pool = 1).
- **Model smoke tests** (manual, gated by env var): FLUX-schnell (smallest real target) on the dev box; golden-image SSIM check.
- **Benchmark harness** (`benchmarks/`): reproduces §6 table; tracked in a results file per release.

## 8. Dependencies & packaging

- Runtime: `torch>=2.4`, `diffusers>=0.35`, `transformers`, `safetensors`, `accelerate`, `huggingface-hub`, `tqdm`. Optional extras: `[nf4]` → bitsandbytes; `[video]` → imageio/av for export; `[attn]` → sage-attention.
- `src/` layout, hatchling build, `aircanvas` console script. Ruff (lint+format), mypy on `sharding`/`streaming` (the correctness-critical core), pytest. GitHub Actions: lint + unit tests on CPU (Windows + Linux matrices). Apache-2.0 (matches AirLLM/diffusers ecosystem).

## 9. Key design decisions (ADR summary)

| # | Decision | Rationale | Alternative rejected |
|---|---|---|---|
| 1 | Custom hook engine, diffusers as substrate | Need quantized-shard load, slot pool, lookahead>1, budget solver — group offloading's hooks don't compose with quant today | Forking/upstreaming `apply_group_offloading` first (revisit later; upstreaming pieces stays on the roadmap) |
| 2 | Quantize at split time into shards | Disk bw is the bottleneck (AirLLM's core lesson); 2–4× traffic cut | Runtime quant (mmgp-style) — pays cost every load |
| 3 | fp8 before NF4 | Zero deps, trivially correct upcast, halves traffic; NF4 adds bnb dep + GPU dequant step | NF4-first (better ratio but more moving parts) |
| 4 | Only DiT is sharded/streamed | TEs run once (evict), VAE runs once (tile) — phase asymmetry is the whole point | Uniform AirLLM-style treatment of everything |
| 5 | GPU slot pool, no per-block alloc | Windows + allocator churn; AirLLM's gc-per-layer is measured overhead | AirLLM-style to('meta') + empty_cache each block |
| 6 | Wrap stock pipelines, own only placement | Schedulers/samplers/prompting are solved problems; smaller surface | Custom pipeline reimplementations |
| 7 | Deterministic recorded schedule for prefetch | Denoise loop is identical every step — perfect prediction | Reactive prefetch (next-module guessing) |
| 8 | fp8 upcast writes into a **second pre-allocated pool**, not the caching allocator | The no-allocation-in-the-hot-loop rule stays absolute and the VRAM cost stays a number we can put in the plan. Compressed staging slots are released as soon as the upcast is *enqueued* (CUDA stream ordering makes that safe), so 2 suffice and the extra cost is `2 x compressed`, not `gpu_slots x compressed` | `payload.to(compute_dtype)` per block, relying on the allocator's steady-state block reuse — same VRAM in practice, but non-deterministic, fragmentation-prone, and a documented exception to a rule that is more valuable un-excepted |
| 9 | Per-tensor fp8 **scale**, not a bare cast | `float8_e4m3fn` denormalises below 2^-6 (0.0156) and DiT weight matrices routinely have amax under that, so an unscaled cast throws away most of the mantissa exactly where it matters. Scaling amax to 448 keeps every value in the normal range, where error is a flat 2^-4 relative half-ulp. Costs one fp32 scalar per tensor and one `mul_` | Scale-free cast (diffusers layerwise-casting semantics) — simpler, but the error becomes a function of a tensor's absolute magnitude |
| 10 | Resident (non-block) shard is never compressed | It is read once and then lives on the GPU forever, so compressing it cuts zero per-step disk traffic (ADR #2's whole rationale) while putting embedders and final projections — the most quality-sensitive tensors in the model — through a lossy codec | Uniform compression of every shard |
| 11 | Pin `_execution_device` via a throwaway pipeline subclass | diffusers derives its device from the first component that has one; our DiT is on `meta` and TEs/VAE are parked on the CPU between phases, so the derivation yields "cpu" and the run silently falls off the GPU. The property is read-only, so an instance attribute cannot shadow it. AirLLM patches `.device` for the same reason (RESEARCH.md §1) | Keeping a dummy module resident on the GPU to bias the derivation (fragile, wastes VRAM); forking diffusers |
| 12 | Budget re-solved per call from height/width/steps | Resolution and step count are solver inputs (§3.2) and are only known at call time — a plan sized for 512² will OOM at 1024². Free VRAM is re-read too, since whatever else is on the card is not ours to spend | One plan fixed at `from_pretrained` (what `workload=` opts into when a caller wants determinism) |

Two solver details worth recording because they look like bugs otherwise:

- **The resident-block scan is exhaustive, not early-exit.** Total VRAM is *not* monotonic in the resident count R: for two-species models (FLUX's 19 big blocks then 38 small ones) pinning the front blocks *shrinks* the slot pool, so a scan that stopped at the first miss would wrongly report R=0.
- **On a CPU run the "device" budget is system RAM**, and the slot pool is subtracted from the RAM waterfall rather than counted twice. CPU is the correctness path, not the fast path.
