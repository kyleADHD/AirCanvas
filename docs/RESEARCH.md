# AirCanvas Research Notes

Synthesized 2026-08-05 from three deep-research passes: (1) AirLLM source-level analysis, (2) open-weight image-model architectures + low-VRAM ecosystem, (3) open-weight video-model architectures + low-VRAM practice. This document is the factual foundation for [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. What AirLLM actually does (and what we borrow)

[AirLLM](https://github.com/lyogavin/airllm) (Apache-2.0, ~29k stars) runs 70B+ LLMs on ~4 GB VRAM. VRAM scales with the **largest single layer**, not total model size, because a transformer executes strictly sequentially.

**Two-phase design (v3.x, current main):**

- **Phase A — one-time split.** `split_and_save_layers()` reads the checkpoint's `weight_map` index, groups tensors by module prefix (`embed`, `layers.0..N`, `norm`, `lm_head`), and writes **one safetensors file per module** plus an empty `.done` marker for atomicity. Options: bitsandbytes 4-bit NF4 / 8-bit block-wise quantization **at split time** (quantized shards mainly to beat disk I/O — the true bottleneck — claimed "up to 3×" end-to-end speedup), `delete_original` to keep peak disk at ~1× model size, hardlink passthrough when a checkpoint shard already maps 1:1 to a module.
- **Phase B — runtime streaming.** Builds the real HF model on the **meta device** (`init_empty_weights`) — zero memory — then registers `forward_pre_hook`/`forward_hook` pairs on every big module. Pre-hook: materialize the module's weights on GPU (`set_module_tensor_to_device`) from a prefetched CPU dict. Post-hook: evict back to `meta`, `gc.collect()` + `torch.cuda.empty_cache()`. transformers owns the whole generation loop; a patched `.device` property keeps KV cache/inputs on CUDA.
- **Prefetch:** one background thread (`ThreadPoolExecutor(max_workers=1)`) loads layer *i+1* from disk while layer *i* computes; tensors are `.pin_memory()`-ed (capped at 2 GiB/layer) for fast H2D. No custom CUDA streams — the H2D copy itself is synchronous.
- **MoE expert streaming** (Kimi K3 2.8T): hooks each expert individually; `safe_open().get_tensor()` seek-reads only the routed experts' few MB from a ~16 GB layer shard.

**Why it's slow for LLMs:** every generated token re-reads essentially the whole model from disk (70B bf16 = 140 GB/forward → ~40 s/token floor at 3.5 GB/s NVMe; real-world reports well under 1 tok/s). This is the key weakness our project sidesteps (§2).

**What we borrow:** persistent per-block quantized shard cache + manifest; meta-device + hook streaming; pinned-memory prefetch; `.done`-marker atomicity; one-line `from_pretrained` ergonomics; treat **disk bandwidth, not VRAM, as the resource to optimize**.

**What we discard:** per-token streaming economics, LLM KV-cache handling, the assumption that all modules deserve the same offload strategy.

---

## 2. Why diffusion models are the *right* target for disk streaming

The inference pattern is fundamentally different from LLMs:

| | LLM (chat) | Image diffusion | Video diffusion |
|---|---|---|---|
| Full model passes per output | ~1 per token (1000s) | 20–50 total (4 for distilled) | 20–50 total |
| Tokens per forward | 1 (decode, GEMV, memory-bound) | ~4,096 (1024², big GEMMs) | 33,000–119,000 |
| Compute per block | sub-ms | ~7–10 ms (4090, FLUX) | 1.7 s (4090) – 11 s (4060), Wan 14B @720p |
| Can transfer hide behind compute? | Never | Partially (Q4 + NVMe ≈ 3–5× compute) | **Almost entirely (transfer = 1–6% of compute on NVMe)** |

Three structural advantages:

1. **Bounded weight traffic.** Total DiT reads = `steps × model_size`, independent of output size. FLUX-dev @28 steps bf16 ≈ 666 GB; Q4 ≈ 182 GB (~36 s I/O floor at 5 GB/s) → **~1 min/image on a 6 GB card**. Qwen-Image 20B Q4 @50 steps ≈ 550 GB → **~3 min/image on 6 GB VRAM + 8 GB RAM**. Video: ~15 GB fp8 × 30 steps ≈ 450 GB/video, and transfer is nearly free behind video compute.
2. **Phase asymmetry.** Text encoders run **once** per prompt (load-run-evict or CPU is fine); the DiT runs **every step** (gets the fast pipelined path); VAE decodes **once** (tiling bounds it). LLM-style uniform treatment is wrong here.
3. **Deterministic schedule.** The block execution order is identical every step — perfect prefetch prediction, unlike MoE routing or speculative branches.

Caveat: distilled few-step models (FLUX-schnell, LTX-distilled, CausVid 4-step Wan) slash compute per step, shrinking the transfer-hiding budget — streaming overhead becomes visible exactly where the speed community is heading. The budget solver must detect this and prefer RAM-cached shards.

---

## 3. Image model targets

| Model | DiT params | Blocks | Hidden | Text encoders | bf16 DiT | Naive VRAM |
|---|---|---|---|---|---|---|
| **Qwen-Image** | 20.4B MMDiT | **60 identical double-stream** (~340M / ~680 MB bf16 each) | 3072 | Qwen2.5-VL-7B (~16.6 GB) | ~41 GB | ~42+ GB |
| **FLUX.1 dev/schnell** | 12B RF-DiT | 19 double (~340M) + 38 single (~140M) | 3072 | T5-XXL 4.7B + CLIP-L | ~23.8 GB | 33.9 GB measured |
| **SD3.5 Large** | 8B MMDiT | 38 | 2432 | CLIP-L + bigG + T5-XXL (droppable) | ~16.5 GB | ~24 GB |
| **HiDream-I1** | 17B sparse DiT | 16 double + 32 single, **MoE FFN (4 routed / 2 active)** | 2560 | CLIP-L + CLIP-G + T5-XXL + **Llama-3.1-8B (all 32 layers' hidden states)** | ~34 GB | 60+ GB |
| SDXL (contrast) | 2.6B UNet | heterogeneous conv+attn | — | CLIP-L + bigG | ~5.2 GB | 8–10 GB |

- **Qwen-Image is the ideal flagship**: 60 *identical* blocks → trivially uniform shard plan; doesn't fit a 4090 at bf16 today; VAE reuses Wan-2.1 VAE.
- FLUX per-block sizes (derived): double ≈ 680 MB bf16 / ~170 MB Q4; single ≈ 280 MB bf16.
- HiDream: text-encoder stack (~13.5B, ~27 GB) is ~80% the size of its DiT — phase-aware TE handling matters more than DiT streaming there. MoE experts are per-token routed → realistically stream all 4.
- SDXL-era UNets are heterogeneous (conv2d, quantizes poorly, awkward block boundaries) — **explicit non-target** for v1; they don't need us anyway.

## 4. Video model targets

| Model | DiT params | Blocks | Hidden | Text encoder | bf16 | Q4_K_M | Official VRAM |
|---|---|---|---|---|---|---|---|
| **Wan 2.1 T2V/I2V 14B** | 14B | **40 identical** (~350M, ~700 MB bf16 / 350 MB fp8) | 5120 | UMT5-XXL 5.7B (11.4 GB) | 29.05 GB | 10.12 GB | ~80 GB naive |
| **Wan 2.2 A14B (MoE)** | 27B total / 14B active | 2 experts × 40 blocks; **per-timestep expert switch** (high-noise → low-noise at t_moe) | 5120 | UMT5-XXL | ~28.6 GB ×2 | 9.65 GB ×2 | — |
| **HunyuanVideo** | 12.8B | 20 dual + 40 single | 3072 | Llava-Llama-3-8B (15 GB!) + CLIP-L | 25.65 GB | 7.88 GB | 45–60 GB |
| LTX-Video 13B | 13B | 48 | 4096 | T5-XXL | 28.58 GB | — (fp8 15.7 GB) | 80 GB full |
| CogVideoX-5B | 5B | 42 | 3072 | T5-XXL | 11.14 GB | — | 5 GB w/ seq-offload |
| Mochi-1 | 10B AsymmDiT | 48 | 3072 | T5-XXL | 20.06 GB | — | ~60 GB |

- **Wan 14B is the highest-demand target** and its backbone is shared by SkyReels-V2, CausVid, VACE, Phantom → one shard plan covers ~10 model families.
- **Wan 2.2's MoE is a gift**: expert switch is per-*timestep*, not per-token — schedule high-noise expert shards for steps < t_moe, low-noise after; 27B params but never more than one block resident.
- **Video compute-vs-transfer (Wan 14B @720p×81f = 75,600 tokens):** ~170 TFLOP per block per step. fp8 block transfer: 14 ms (PCIe 4.0) / 100 ms (NVMe 3.5 GB/s) vs 1.7 s (4090) – 11 s (4060) compute → **NVMe streaming of every block every step is essentially free**. kijai's empirical finding agrees: `prefetch_blocks=1` "offsets most of the transfer overhead."
- **Activations, not weights, set the VRAM floor** at 720p: FFN transient ≈ 2.1 GB (Wan) / 2.9 GB (Hunyuan); residual copies ~0.8 GB. Flash-style attention is mandatory (naive attn matrix ≈ 11 TB). SageAttention2 ≈ 3× vs FA2 is standard in low-VRAM stacks.
- **Wan VAE has no tiling in diffusers** (`AutoencoderKLWan` only frame-chunks via `feat_cache`) — we must ship tiled/spatial-chunked Wan-VAE decode.
- FramePack (constant-memory video length) is orthogonal & complementary: it fixes *sequence* memory; we fix *weight* memory.

**Realistic floors with block streaming + fp8 + VAE tiling (derived):** Wan 14B ≈ **4–5 GB @ 480p, 6–8 GB @ 720p**; HunyuanVideo ≈ **5–6 GB @ 544×960**. Current practice floors for comparison: kijai block-swap ~10-16 GB (needs 48+ GB RAM), Wan2GP ~12 GB class (needs 24–64 GB RAM), diffusers group-offload ~13 GB.

---

## 5. Existing landscape and the gap

| Tool | Mechanism | Floor | Missing |
|---|---|---|---|
| diffusers `enable_model_cpu_offload` | per-component swap | largest component (24 GB FLUX) | useless for 20B on small cards |
| diffusers `enable_sequential_cpu_offload` | per-leaf accelerate hooks | ~GBs | "extremely slow", no prefetch, full model in RAM |
| **diffusers `apply_group_offloading` + `offload_to_disk_path`** ([PR #11682](https://github.com/huggingface/diffusers/pull/11682)) | block groups, CUDA-stream prefetch, disk paging | **11.7 GB VRAM / 2.8 GB RAM, 55.5 s FLUX** (vs 6.6 s baseline) | **bf16-only shards** (24–41 GB disk reads *per step*), per-session serialization, no budget solver, no quant, hand-assembled per-component config |
| kijai `blocks_to_swap` (ComfyUI) | N blocks pinned in CPU RAM, hook swap, prefetch=1 | ~10–16 GB VRAM | needs 48+ GB RAM; ComfyUI-only |
| mmgp / Wan2GP | budgeted RAM offload, int8 on-the-fly, pinned RAM | 6–12 GB VRAM | **24–64 GB RAM floor**; no disk tier |
| ComfyUI core / city96 GGUF | smart partial load; per-forward GGUF dequant | varies | assumes model fits in system RAM |
| Nunchaku SVDQuant | fused W4A4 kernels | ~6.7 GB FLUX | per-model prequantized ckpts, custom engine, not generic |
| stable-diffusion.cpp | ggml, quantize hard | varies | no disk streaming; RAM-resident |
| AirLLM | per-layer disk streaming | 4 GB (70B LLM) | LLMs only; per-token economics |

**The unserved niche — AirCanvas's thesis:**

> **Quantized shards on disk × pipelined block streaming × automatic budgeting × phase-aware pipeline policy**, behind a one-line API, for any diffusers-supported DiT.

Nobody combines these. Specifically:
1. **Quantized shards** (fp8/NF4/Q4) cut per-step disk traffic 2–4× — the single highest-value missing composition (diffusers GGUF proves per-forward dequant works; group offloading proves disk paging works; they don't compose today).
2. **Tiny both-budgets target**: 4–6 GB VRAM **and** ≤8–16 GB RAM (every existing tool assumes one of the two is big).
3. **Persistent shard cache** (AirLLM-style manifest, reused across runs, Hub-shareable) vs diffusers' per-session serialization.
4. **Budget solver**: fill spare VRAM with resident blocks, promote shards to a pinned-RAM cache when RAM exists, degrade to NVMe when it doesn't — a cache **hierarchy**, not a binary mode.
5. **Deterministic lookahead prefetch** exploiting the fixed denoise-loop schedule (depth >1, ring buffer).
6. **Model-coverage quirks handled centrally**: Wan-VAE tiling, T5's dtype-casting forward (breaks layerwise casting), Wan 2.2 per-timestep expert scheduling, HunyuanVideo's 15 GB Llava TE (run-and-evict or CPU), HiDream's 4-TE stack.

## 6. Bandwidth reference (design constants)

- NVMe seq read: ~3.5 GB/s (PCIe 3.0) / ~7 GB/s (4.0) / 12–14 GB/s (5.0); SATA SSD ~0.5 GB/s (fallback tier, warn).
- PCIe H2D: ~13 GB/s real (3.0 ×16), ~25 GB/s real pinned (4.0 ×16). VRAM: 300–1000 GB/s.
- Disk is ~4–7× slower than PCIe, ~100× slower than VRAM → cache hierarchy: **VRAM-resident > pinned-RAM shard cache > NVMe shards**.
- Reads only (no SSD endurance concern); ~450–670 GB read per bf16 image/video generation, 4× less at Q4.

## 7. Design implications (carried into ARCHITECTURE.md)

1. Only the DiT gets the pipelined streaming path; TEs are load-run-evict (or CPU); VAE is tiled — **phase-aware orchestration is the core product insight**.
2. Shard at natural block boundaries (uniform for Qwen-Image/Wan; two-species for FLUX/Hunyuan); generic adapter introspects `ModuleList`s, per-family adapters pin down edge cases.
3. Quantize at split time into the shard cache (fp8 first — simple upcast-on-load, halves traffic; NF4/bnb second; GGUF-style later). Skip norms/modulation (tiny, quality-sensitive).
4. Prefetch = 3-stage pipeline: NVMe→pinned-CPU ring (thread pool) → GPU (dedicated CUDA copy stream + events) → compute (default stream). Lookahead from the recorded deterministic schedule.
5. Budget solver inputs: free VRAM/RAM, measured disk bw, model manifest, steps, resolution → outputs: resident-block count, ring depth, cache tier, quant recommendation, warnings (SATA, distilled models).
6. Windows first-class (dev machine is Windows): no `malloc_trim`, OneDrive-safe paths, pinned-memory behavior differs — cross-platform `clean_memory()` from day one.
7. Activation memory is the real floor for video: integrate SDPA/FlashAttention/Sage, CFG batch splitting option, and ship Wan-VAE tiled decode.
