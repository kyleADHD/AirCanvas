# Contributing to AirCanvas

Thanks for your interest! AirCanvas is a correctness-critical memory system —
a subtle bug doesn't crash, it silently produces wrong images. Contributions
are welcome, but they go through gates designed to keep the product working on
the machines it exists for (4–8 GB GPUs, small RAM, Windows included).

## Development setup

```bash
git clone https://github.com/kyleADHD/AirCanvas
cd AirCanvas
pip install -e .[dev]
```

Verify your environment before changing anything:

```bash
pytest                    # offline CPU suite — green on any machine, no GPU, no network
ruff check . && ruff format --check .
mypy src/aircanvas/sharding src/aircanvas/streaming
```

Two opt-in test tiers exist for changes that touch them:

```bash
pytest -m gpu             # needs a CUDA GPU
pytest -m network         # downloads ~1.5 MB tiny public HF test models
```

## Hard rules (engineering invariants)

These are non-negotiable. PRs that violate them will not be merged, however
good the speedup. Code comments across the repo cite this list.

1. **Correctness before speed.** Any change to sharding or streaming must keep
   the equivalence gate passing: streamed output **bitwise-equal** to
   full-VRAM execution at `compression=None`, within tolerance for fp8/NF4
   (`tests/test_engine.py`).
2. **No allocations in the per-block hot loop.** Never allocate GPU or pinned
   memory while blocks are streaming — use the slot pools and pinned ring
   allocated once at startup (`streaming/prefetch.py`, ADR #8).
3. **Windows is a first-class target** (it is the primary dev machine). No
   libc calls (`malloc_trim`), no hardcoded `/tmp`, `pathlib` everywhere,
   assume paths may be OneDrive-synced.
4. **Nothing is ever written inside the repo.** Shard caches, embedding
   caches, and probe sidecars live under `aircanvas.config.cache_root()`
   (the HF cache home) — never the checkout.
5. **We wrap diffusers, we don't reimplement it.** AirCanvas owns device and
   memory placement only. No forked schedulers, samplers, or prompt handling.
6. **Optional deps stay optional.** bitsandbytes only under the `[nf4]`
   extra; guarded imports with clear error messages.
7. **Crash-safe cache writes.** Shards are written temp-file → rename +
   `.done` marker, manifest last. A missing marker means "re-split this
   shard" — never assume a partial write is valid.
8. **Type hints everywhere.** `mypy` must stay clean in `sharding/` and
   `streaming/` (the correctness-critical core).

## Pull request process

1. Fork, branch from `main`, keep the PR focused — one change per PR.
2. CI must be fully green (lint, format, mypy core, offline test suite on
   Linux **and** Windows). CI is a required check; there is no merge
   without it.
3. New behavior needs tests. Prefer the offline CPU tier (toy models); mark
   GPU- or network-dependent tests with the existing pytest markers.
4. Performance claims need numbers — run `benchmarks/bench_stream.py` and
   include before/after results in the PR description.
5. A maintainer review is required on every PR (enforced via CODEOWNERS).
   Direct pushes to `main` are disabled by branch protection.

## Reporting bugs

Open an issue with the bug template and include:

- `aircanvas doctor` output (it captures VRAM/RAM/disk and OS),
- the exact command or Python snippet,
- the full traceback — and if generation *succeeded but looks wrong*, say so
  explicitly; silent-corruption reports get top priority.

## Security issues

Please **do not** open a public issue for vulnerabilities — see
[SECURITY.md](SECURITY.md) for private reporting.
