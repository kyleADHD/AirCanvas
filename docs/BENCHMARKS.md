# AirCanvas measured results

All numbers from one machine — deliberately the *worst reasonable* machine for
this workload, because that is who AirCanvas is for:

> Windows 11 laptop · RTX 4050 Laptop (6.4 GB VRAM) · 16.9 GB RAM (chronically
> ~70% committed) · NVMe delivering ~1.6-2 GB/s effective under memory
> pressure · torch 2.6 + diffusers 0.39. August 2026.

## The headline: models that cannot otherwise run here

| Model | Params | Needs normally | AirCanvas result |
|---|---|---|---|
| FLUX.1-schnell (fp8 shards) | 12B | ~34 GB VRAM | 1024² 4-step: **129 s warm** (385 s cold) |
| FLUX dev-equivalent 28-step | 12B | ~34 GB | ~14.1 s/step ≈ 7 min/image |
| Qwen-Image (nf4 shards) | 20.4B | ~41 GB+ | 1024² 20-step: **427 s (~7.1 min)**; 50-step: 1034 s |
| Wan 2.1 T2V (fp8 shards) | 1.3B | ~8 GB | 480×832×81f 20-step: 1046 s incl. decode |
| **Wan 2.1 T2V (fp8 shards)** | **14B** | **~80 GB** | **480×832×81f 20-step: 10.6 h overnight — 562 GB streamed, 97.4% prefetched, survived a battery death mid-run** |

Qwen-Image at 50 steps streams **1.07 TB of weights** through the 6 GB card
for a single image, 97.5% of loads served by the prefetcher ahead of demand.

## The thesis measurement: video streaming is free

Same engine, same seed, same box; the only variable is whether the 30 blocks
live in VRAM or stream from disk every step:

```
streamed  49.36 s/step   (55.8 GB read over the run)
resident  48.26 s/step   (zero disk reads)
ratio     1.02x          (M6 gate: <= 1.10x — PASS)
```

The two output videos are **byte-identical (same MD5)** — correctness and
performance in one measurement. Compute grows faster than transfer with model
size, so the 14B ratio is expected to be at least as good.

## Synthetic component benchmark (benchmarks/bench_stream.py)

24×2048 toy DiT, bf16, deliberately IO-bound like an image model:

| configuration | ms/step | vs full-VRAM |
|---|---|---|
| full-VRAM reference | 104-114 | 1.0× |
| bf16 shards, prefetched | 296-310 | ~2.8× |
| bf16 + 6 resident blocks | 228-231 | ~2.1× |
| fp8 shards, prefetched | 150-172 | ~1.5× |

fp8 halves disk traffic and roughly halves step time when IO-bound; resident
blocks help while IO-bound and stop mattering once compression makes the
workload compute-bound — the budget solver's whole job.

## The 14B footnote

The 14B run (Aug 2026) completed unattended overnight at ~31.5 min/step.
On THIS machine it is IO-bound: 28 GB of reads per step against a disk that
delivers only tens of MB/s effective when the OS has ~0.5 GB of RAM free for
cache — the box's chronic commit pressure, not the streaming design, sets
the pace (the 1.3B measurement proves streaming itself costs 2%). With
normal RAM and NVMe throughput the same run computes in the low hours. It
also survived: a battery drain to 0%, a critical-battery sleep mid-step, and
resume — the run completed correctly anyway.

## Honest misses

The aspirational targets (FLUX ≤ 90 s, Qwen ≤ 4 min) missed by ~4.5× and
~1.8× **on this machine**, fully attributed by `pipe.report()`: disk at
~1.6 GB/s effective (targets assumed 5-7 GB/s NVMe), 4050-Laptop compute
(~¼ of a 4090), and true-CFG doubling on Qwen. A desktop with a healthy NVMe
and 32 GB RAM — where the M8 RAM-cache tier holds the whole streamed set —
should close most of the gap. Reproduce with `benchmarks/bench_flux.py`,
`bench_qwen.py`, `bench_wan.py`.
