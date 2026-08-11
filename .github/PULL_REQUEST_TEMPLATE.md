## What does this PR do?

<!-- One paragraph. Link the issue it closes, if any. -->

## Checklist

- [ ] `pytest` (offline CPU suite) is green
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy src/aircanvas/sharding src/aircanvas/streaming` is clean
- [ ] Touches sharding/streaming → the bitwise equivalence gate
      (`tests/test_engine.py`) still passes
- [ ] No GPU/pinned allocations added inside the per-block hot loop
- [ ] Nothing is written inside the repo at runtime
- [ ] Performance claim → `benchmarks/bench_stream.py` before/after numbers
      included below
- [ ] New behavior is covered by tests (offline tier where possible)

## Benchmarks (if applicable)

<!-- bench_stream.py output, hardware, settings -->
