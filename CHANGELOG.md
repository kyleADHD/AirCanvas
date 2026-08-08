# Changelog

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
