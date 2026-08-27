# AirCanvas Studio

The local desktop UI: a single-page app served by the same Python process that
owns the pipeline.

```bash
pip install "aircanvas[studio]"
aircanvas studio            # opens http://127.0.0.1:8760/
aircanvas studio --demo     # every screen, no GPU and no downloads
```

Two modes, one app. **Simple** hides every decision the engine can make itself
— only times, sizes in GB and step counts are ever shown. **Pro** does the
opposite: telemetry is the hero. Every other AI-image UI hides the machine;
this one shows it.

---

## Why this shape

**One process.** The Studio is not a client for a server; it is the pipeline
with a window. `AirPipeline` runs in a worker thread of the same interpreter,
so telemetry is a dict read, not a network hop, and cancelling a run is a flag
the denoise loop already checks. There is no daemon to keep alive and no state
that can disagree with the machine.

**No build step.** The frontend is hand-written ES modules and one stylesheet,
served from `aircanvas/studio/static/`. That is a deliberate trade: a React +
Vite bundle would buy component ergonomics and cost a Node toolchain on a
machine whose job is to run a diffusion model, plus a build artifact in the
wheel that nobody can audit by reading. The whole app is ~2,500 lines of plain
JavaScript with no dependencies, and it ships and runs offline exactly as it
sits in the repo.

**Fonts are self-hosted.** Space Grotesk and JetBrains Mono (both SIL Open Font
License) live in `static/fonts/` as latin-subset variable woff2 files, 54 KB
together. A local-first app has to draw correctly with the network unplugged.

**FastAPI is optional.** `pip install aircanvas` does not pull in FastAPI or
uvicorn; `aircanvas doctor` and `aircanvas run` are unaffected. The `[studio]`
extra adds them.

---

## The honesty rules

The product's personality is "measured, not promised", and that constrains the
implementation more than it constrains the copy.

**Verdicts are computed, never tabulated.** "Can this machine run Qwen-Image at
nf4?" is answered by `streaming.residency.solve` against a manifest shaped like
the real model (`sharding.manifest.synthetic_manifest`) — the same solver that
runs at generation time. Change the free VRAM, the shard format or the GGUF
toggle and every chip moves, because nothing is a stored answer. Every chip
also names its reason: `✓ runs · ~6 s IO/step`, `tight · 0.3 GB headroom`,
`✗ needs 41 GB`.

**Numbers keep their provenance.** Every size and timing in
`studio/catalog.py` is measured on the dev box (docs/BENCHMARKS.md), published
by the model's own repo, or derived arithmetically from one of those — and the
comment says which. Where the repo has not verified a quantized source for a
model, that model has no GGUF option rather than an invented download size.

**Estimates say how confident they are.** The Desk shows `7 min` when it has
measured this model at this exact shape on this machine, and `~7 min` when it
is a bound. The rule (`studio/catalog.estimate_seconds`, mirrored in
`static/lib/estimate.js`): a measurement of this shape wins; otherwise the
solver's disk time per step is the floor and a published measurement raises it
to include compute. The Desk's estimate therefore improves as the machine is
actually used, because every finished run is a measurement in the gallery.

**No placeholder artwork.** An image slot with no output shows the recessed
frame and says what is missing. The design handoff's sample imagery was
explicitly a stand-in that must not ship, and a stand-in picture is the one lie
this UI cannot tell.

**Light mode is not shipped from guesswork.** The Appearance control persists
all three choices and the root element carries `data-theme`, so a light pass is
a stylesheet change. Until that pass exists, only dark is drawn and the control
says so.

---

## Runtime seam

The UI needed live telemetry, which did not exist: `pipe.report()` answers
"where did the time go" only once a run is over. Three small additions cover it.

| Where | What |
|---|---|
| `runtime/progress.py` | `RunObserver` — `phase_started` / `phase_finished` / `step`. Advisory: anything an observer raises is logged and swallowed, so a disconnected UI cannot take a generation with it. |
| `AirPipeline.__call__(observer=…)` | Phase boundaries from the orchestrator; steps from diffusers' own `callback_on_step_end`, chained rather than replacing a caller's. |
| `AirPipeline.live_stats()` | The in-flight engine counters — blocks, bytes, prefetch hits, stalls — polled at 5 Hz rather than pushed, because block loads happen thousands of times per run. |
| `split_model(progress=…)` | One call per block, including blocks a previous run already finished, so a resumed split reports its true starting point. Raising from it stops the split at a block boundary, which is how Pause is implemented. |

`AirPipeline` and the CLI are otherwise untouched.

---

## Screens

Twelve, matching the design handoff's artboards.

| Route | Simple | Pro |
|---|---|---|
| `setup` | S1 Welcome — three things this machine can comfortably run | P1 Welcome setup — every model, a verdict per shard format, the machine in the rail |
| `split` | plain progress and a time | P2 Split progress — the per-block grid, blocks/s, eta, resumable queue |
| `desk` | S2 — prompt, shape, quality, a time | P3 / P4 — every solver input, the LoRA stack, budget caps; the video variant swaps in a filmstrip and frame controls |
| `run` | S3 — "Painting step 12 of 20", with the full Pro strip one chevron away | P5 — phase timeline, streaming ticker, VRAM/disk/stall readouts |
| `report` | — | P6 — residency plan, disk bandwidth, per-phase wall time, bottleneck |
| `gallery` | same masonry, simpler rail | P7 — every output keeps its report snapshot; Reproduce restores it exactly |
| `settings` | S4 — storage, account, appearance, and a door to Pro | P8 — cache by model, disk probe with its age, pipeline toggles, token |

The mode switch is global and persistent; S2⇄P3, S3⇄P5 and S4⇄P8 preserve the
prompt, the model and any in-flight run. A live run or split takes the window
over, and the nav highlights where the app *is*, not where the user last
clicked.

---

## API

One GET fills every panel on first paint; one SSE stream carries the deltas.
No screen polls.

| Route | Purpose |
|---|---|
| `GET /api/state` | Everything: mode, theme, desk, settings, machine probe, catalog with verdicts, installed caches, splits, current run, outputs |
| `POST /api/state` | Patch `mode`, `theme`, `seenWelcome`, or the `desk` / `settings` / `setup` sections |
| `GET /api/machine?probe_disk=` | Re-probe; `probe_disk=true` re-measures the disk for real (a few seconds) |
| `GET /api/models` | The catalog with verdicts recomputed for the current probe |
| `POST /api/estimate` | Seconds and seconds-per-step for a shape, plus the source of the estimate |
| `POST /api/splits` · `/{id}/{pause,resume,cancel}` · `/pause-all` | The split queue |
| `POST /api/runs` · `/api/runs/cancel` · `GET /api/runs/current` | One generation at a time |
| `GET /api/outputs` · `DELETE /api/outputs/{id}` · `GET /api/outputs/{id}/{file,report}` · `POST /api/outputs/{id}/reproduced` | The gallery |
| `POST /api/cache/remove` | Delete one shard cache (refuses any path outside the AirCanvas root) |
| `GET /api/events` | Server-sent events: `state`, `machine`, `split`, `run`, `outputs`, `cache`, `installed` |

**Binding and trust.** 127.0.0.1 by default and unauthenticated, the same
posture as a Jupyter server started by hand: everything it exposes is already
readable by whoever is at the keyboard. `--host` widens that, and the CLI help
says what it means.

**Persistence.** `$HF_HOME/aircanvas/studio/studio.json` holds mode, theme,
desk parameters, budget caps, the LoRA stack and the gallery index; outputs
land in `studio/outputs/`. Nothing is ever written inside a project — the same
hard rule the shard cache follows. Telemetry is never persisted: it belongs to
a run, and a run does not survive the process.

---

## Demo mode

`aircanvas studio --demo` shows all twelve screens on a machine with no GPU, no
models and no downloads — which is also how the UI is reviewed.

- The machine is the box docs/BENCHMARKS.md was measured on (RTX 4050 Laptop,
  6.4 GB VRAM, 16.9 GB RAM at ~70% committed, 1.62 GB/s NVMe), declared as data
  rather than probed.
- Every verdict, residency plan and per-step disk time is still produced by the
  real solver against that profile. Nothing is a stored answer.
- Run timings replay measured values (21.1 s/step for Qwen-Image at 1024², 427.4 s
  total, 97.5% prefetched) on a 40× clock, so a run takes seconds.
- The two split-pacing constants are the only figures in the whole app that are
  neither measured nor derived; `studio/demo.py` says so where they are defined.
  The real split screen is driven by the real splitter's callback.
- No artwork is produced, and every screen carries a `demo` marker in the status
  cluster.

---

## Layout

```
src/aircanvas/studio/
  catalog.py    model table, verdicts, estimates      (the honesty rules live here)
  machine.py    probe, installed caches, HF account, download bytes
  store.py      persisted state, atomic + thread-safe
  events.py     SSE fan-out
  jobs.py       SplitManager + RunManager (threads, telemetry, cancellation)
  server.py     FastAPI app, routes, `serve()`
  demo.py       declared machine + replaying managers
  static/
    index.html  styles.css  app.js
    lib/        dom, format, icons, api, store, estimate
    components/ chrome, widgets
    screens/    setup, split, desk, run, report, gallery, settings
    fonts/      Space Grotesk + JetBrains Mono (SIL OFL)
```

## Keyboard

`d` desk · `g` gallery · `s` settings, when focus is not in a field.
Everything else follows platform convention.
