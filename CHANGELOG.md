# Changelog

## 0.2.0 — Unreleased

- **LoRA loading onto quantized shards** (M10): `pipe.load_lora(source, scale=1.0)`
  — and `lora=` on `AirPipeline.from_pretrained` / `aircanvas run --lora` —
  fuses PEFT and kohya `.safetensors` adapters **after dequant**, in-place,
  into the compute-dtype views the engine is about to bind. fp8/NF4 shard
  caches keep working: the adapter never touches the quantized payload.
  Multiple adapters stack; `unload_lora` / `set_lora_scale` take effect on
  the next generate. Text-encoder keys are skipped (TEs are load-run-evict).
  Fusion is `addmm_` into already-owned slot storage, so the no-allocation
  hot-loop rule holds. Streamed output with a LoRA is bitwise-equal to a
  fully-materialized reference with the same `W += scale * up @ down`.
- **GGUF sources** (`aircanvas[gguf]` extra): `aircanvas split <repo>
  --gguf-file <local.gguf | repo_id:filename>` — and `gguf_file=` on
  `AirPipeline.from_pretrained` — split directly from a quantized GGUF
  checkpoint. Only the source repo's config.json is fetched, so a FLUX-class
  model costs a ~7 GB Q4_K download instead of ~24 GB. Tensors are
  dequantized once at split time into the existing verified codecs
  (none/fp8/nf4); model-native layouts stream through a lazy mmap reader
  (peak RAM: one tensor), BFL-style FLUX layouts convert via diffusers'
  single-file loader. GGUF caches get their own shard-cache tag and manifest
  provenance, and the streaming hot path is untouched.
- CLI: `--hf-token` on `split`/`run` for gated models (ambient
  `hf auth login` / `HF_TOKEN` still works).
- Packaging: `py.typed` marker; PyPI classifiers/keywords; mypy runs under
  the ambient interpreter with optional deps overridden.

## 0.1.0 — 2026-08-08

First working release. Everything below was verified on a 6 GB RTX 4050
Laptop with 16 GB RAM (see docs/BENCHMARKS.md).

- **Core**: one-time per-block resharding (fp8/NF4 at split time, alignment
  verified with the reader's parser, resumable, split-once-delete-original);
  meta-device streaming engine with pinned ring → CUDA copy stream → GPU slot
  pools, deterministic schedule prefetch, bitwise-equivalence gate.
- **Pipeline**: phase-aware `AirPipeline` over stock diffusers pipelines —
  text encoders load-run-evict with a disk embedding cache (or split and
  streamed themselves when too big to load), streamed DiT, tiled VAE decode,
  per-call budget re-solve, `report()` with full attribution.
- **Budgeting**: waterfall solver over probed VRAM/RAM/disk-bandwidth;
  resident-block tier, RAM shard-cache tier (promoted after first read),
  `aircanvas doctor`.
- **Models**: FLUX.1, Qwen-Image, SD3/3.5, Wan 2.1, Wan 2.2 dual-expert
  handover, HunyuanVideo, CogVideoX, generic DiT introspection.
- **Field-hardening**: Windows commit-pressure mmap fallback (ranged reads),
  computed-buffer adoption, tied-embedding re-tie, honest exit codes and
  resumability throughout.
- **Measured**: Qwen-Image 20B at ~7.1 min/image on 6 GB; Wan video streaming
  at 1.02× full-VRAM step time with byte-identical output.
