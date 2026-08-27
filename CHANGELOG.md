# Changelog

## 0.2.0 — Unreleased

- **FLUX.2 klein support** (`adapters/flux2.py`): `Flux2Transformer2DModel`
  gets a named adapter — block plan (`num_layers` double-stream then
  `num_single_layers` single-stream), `encode_prompt` returning
  `(prompt_embeds, text_ids)`, a 512-token text budget and the base 16-px
  latent stride. Without it the model fell to GenericAdapter, which finds the
  right blocks but guesses the encode contract that load-run-evict depends on.
  The Studio catalog gains both klein sizes with figures read from the models'
  own files: 4B is 25 blocks / 7.75 GB DiT (Apache-2.0, ungated), 9B is 32
  blocks / 18.16 GB. Verified end to end on a real `Flux2KleinPipeline`.
- **The Studio can generate from any shard cache on disk**, not just catalog
  models: anything from `aircanvas split <path-or-repo>` now appears on the
  Desk with a verdict solved against its REAL manifest. Previously such a
  cache showed up in Settings but could not be selected.
- Fixed: `pipe.report(as_dict=True)` dropped five plan fields that
  `ResidencyPlan.describe()` prints — including the VRAM budget — so a saved
  report could not redraw the residency plan it described. The plan now
  serialises itself (`ResidencyPlan.as_dict()`), enumerated from the
  dataclass so the two cannot drift again.
- Fixed: a shard cache split from a relative local path recorded that path
  verbatim, so opening the Studio from another directory turned it into a Hub
  repo id and a surprise download. Local sources are recorded absolute.
- **AirCanvas Studio, the local desktop UI** (`aircanvas[studio]` extra):
  `aircanvas studio` serves a single-page app from the same process that owns
  the pipeline — twelve screens across a Simple mode that shows only times,
  sizes and step counts, and a Pro mode where telemetry is the hero. Setup
  verdicts, residency plans and per-step disk times are produced by the real
  budget solver against the live probe, not by a stored table; every size and
  timing in the model catalog is measured (docs/BENCHMARKS.md), published, or
  derived from one of those, and says which. `--demo` shows every screen with
  no GPU, no models and no downloads. The frontend is dependency-free ES
  modules with self-hosted fonts, so it works offline and ships as source.
  See [docs/STUDIO.md](docs/STUDIO.md).
- **Live run telemetry** (`runtime/progress.py`): `AirPipeline.__call__` takes
  an optional `observer` receiving phase and step events, and
  `AirPipeline.live_stats()` exposes the in-flight engine counters (blocks,
  bytes, prefetch hits, stalls) while the denoise loop is running — until now
  those were only readable from `report()` after the fact. Observers are
  advisory: anything they raise is logged and swallowed. Step events come from
  diffusers' own `callback_on_step_end`, chained rather than replacing a
  caller's.
- **Split progress callback**: `split_model(progress=…)` fires once per block,
  including blocks a previous run already finished, so a resumed split reports
  its true starting point. Raising from it stops the split at a block boundary
  — every shard already written keeps its `.done` marker.
- `sharding.manifest.synthetic_manifest()`: build a manifest shaped like a real
  model from published sizes alone, so "can this box run X?" is answered by the
  real solver for a model that is not installed yet. `aircanvas doctor` now
  uses it instead of its own private copy.
- `pipe.report()['hardware']` gains `vram_total_bytes` / `ram_total_bytes`, so
  a persisted report can draw its own memory axes.
- Fixed: `__version__` said `0.1.0` on a 0.2.0 build. It is stamped into every
  shard's metadata and printed by `doctor`, so the drift was visible on disk.
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
