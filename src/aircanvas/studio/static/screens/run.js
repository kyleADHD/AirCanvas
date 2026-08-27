/* P5 Generating (Pro) and S3 Generating (Simple).
 *
 * The signature screen. Every other AI-image UI hides the machine; this one
 * shows it, and the telemetry strip is the hero: a phase timeline that freezes
 * each phase's measured seconds as it completes, and a streaming ticker whose
 * sparkline is the model's own per-block shard sizes filling as they stream.
 *
 * Simple mode runs the same generation behind plain language — a step count, a
 * time, and one sentence about what the machine is doing — with the identical
 * Pro strip one chevron away, because "show me" should never mean "switch
 * modes and lose your place".
 */

import { h } from '../lib/dom.js';
import * as fmt from '../lib/format.js';
import { api } from '../lib/api.js';
import { fail, setUi, store } from '../lib/store.js';
import { topbar } from '../components/chrome.js';
import { chevronDown, chevronRight } from '../lib/icons.js';
import { bar, eyebrow, frame, metric, statusDot } from '../components/widgets.js';

const SPARK_BARS = 48;
const PHASES = ['encode', 'denoise', 'decode'];

/**
 * Phase card state.
 *
 * Decode runs *inside* the pipeline call, so the runtime reports it started
 * before denoise reports finishing — for that window denoise has no measured
 * seconds yet. Ordering settles it: a phase a later phase has overtaken is
 * done, whatever its number says, so the timeline never shows a completed
 * phase as pending.
 */
function phaseState(name, run) {
  const measured = (run.phaseSeconds || {})[name];
  const terminal = ['done', 'cancelled', 'failed'].includes(run.status);
  const here = PHASES.indexOf(name);
  const now = PHASES.indexOf(run.phase);
  if (measured !== undefined && measured !== null) return { klass: '', measured };
  if (terminal) return { klass: '', measured: null };
  if (name === run.phase) return { klass: '.active', measured: null };
  if (now > here) return { klass: '', measured: null };
  return { klass: '.pending', measured: null };
}

export function runScreen() {
  return store.state.mode === 'pro' ? proRun() : simpleRun();
}

function currentRun() {
  return store.state.run || null;
}

function cancel() {
  api.cancelRun().catch(fail);
}

// -- Pro ------------------------------------------------------------------

function proRun() {
  const run = currentRun();
  if (!run) return finishedFallback();

  return h('div.app',
    topbar(),
    h('div.body',
      h('div.stage', { style: { padding: '26px' } },
        frame(previewOutput(run), {
          width: 450,
          height: 450,
          accent: true,
          scrim: [
            h('span.mono.accent', { style: { fontSize: '11px' } }, `preview · step ${run.step}`),
            h('span.push.t3', { style: { fontSize: '11px' } },
              run.previewUrl ? 'latents decoded every 4 steps' : 'the image appears at decode'),
          ],
        }),
      ),
      runRail(run),
    ),
    telemetryStrip(run),
  );
}

function previewOutput(run) {
  if (run.previewUrl) return { id: run.id, file: run.previewUrl, kind: 'image', prompt: run.prompt };
  return null;
}

function runRail(run) {
  const machine = store.state.machine;
  const vramTotal = run.vramTotalBytes || machine.vramTotalBytes || 0;
  const vramUsed = run.vramInUseBytes;

  return h('aside.rail', { style: { padding: '18px', gap: '15px' } },
    h('div.eyebrow', 'Run'),
    h('div.card', { style: { padding: '12px', display: 'flex', flexDirection: 'column', gap: '6px' } },
      h('div.t2', { style: { fontSize: '13px', lineHeight: '1.5' } }, run.prompt),
      h('div.mono.t5', { style: { fontSize: '10px' } },
        [
          run.modelName,
          run.format,
          run.seed === null || run.seed === undefined ? 'seed random' : `seed ${run.seed}`,
          run.loraCount ? `${run.loraCount} LoRA${run.loraCount === 1 ? '' : 's'}` : null,
        ].filter(Boolean).join(' · ')),
    ),
    h('div.col', { style: { gap: '9px' } },
      // VRAM in use is read from the CUDA allocator. Without a CUDA device
      // there is nothing to read, so the row says why instead of showing a
      // dash over an empty track that looks like a stuck meter.
      metric('VRAM in use', vramUsed === null || vramUsed === undefined
        ? (store.state.machine.device === 'cuda' ? fmt.DASH : 'no CUDA device')
        : `${fmt.gbNum(vramUsed, 2)} / ${fmt.gbNum(vramTotal)} GB`),
      vramUsed === null || vramUsed === undefined
        ? null
        : bar(vramTotal ? (vramUsed / vramTotal) * 100 : 0, { height: 4 }),
      metric('Pinned ring', run.pinnedBytes
        ? `${fmt.gbNum(run.pinnedBytes, 2)} GB · ${run.ringBlocks} blocks`
        : fmt.DASH),
      metric('Disk read', fmt.gbps(run.diskReadBytesPerSecond)),
      metric('Stalls', String(run.stalls || 0), { tone: run.stalls ? 'warn' : 'ok' }),
    ),
    h('div.col', { style: { marginTop: 'auto', gap: '8px' } },
      run.status === 'failed'
        ? h('div.banner.fail', h('span.dot.dot-warn'), h('span', run.error || 'The run failed'))
        : null,
      ['done', 'cancelled', 'failed'].includes(run.status)
        ? h('button.btn.btn-lg', { type: 'button', onClick: () => setUi({ route: 'desk' }) }, 'Back to the desk')
        : h('button.btn.btn-danger.btn-lg', { type: 'button', onClick: cancel }, 'Cancel run'),
      h('p.t5', { style: { margin: 0, fontSize: '11px', textAlign: 'center' } },
        'Cancelling keeps the shard cache. Nothing to redownload.'),
    ),
  );
}

/** The persistent strip. Shared verbatim between P5 and S3's disclosure. */
export function telemetryStrip(run) {
  return h('div.telemetry',
    h('div.phases',
      phaseCard('encode', run, { width: 200, note: 'text encoders evicted' }),
      denoiseCard(run),
      phaseCard('decode', run, { width: 200, note: decodeNote(run) }),
    ),
    ticker(run),
  );
}

function decodeNote(run) {
  const done = run.phaseSeconds && run.phaseSeconds.decode;
  if (done) return run.kind === 'video' ? 'tiled VAE · frames written' : 'tiled VAE';
  return 'tiled VAE · after the last step';
}

function phaseCard(name, run, { width, note }) {
  const { klass, measured } = phaseState(name, run);
  const active = klass === '.active';
  const complete = measured !== null;

  return h(`div.phase.side${klass}`, { style: { width: `${width}px` } },
    h('div.row', { style: { gap: '7px' } },
      statusDot(active ? 'run' : complete ? 'ok' : 'idle'),
      h('span.name', name),
      h(`span.push.mono.${complete ? 'ok' : 't5'}`, { style: { fontSize: '12px' } },
        complete ? fmt.seconds(measured) : fmt.DASH),
    ),
    h('div.note', note),
  );
}

function denoiseCard(run) {
  const { klass, measured } = phaseState('denoise', run);
  const active = klass === '.active';
  const complete = measured !== null;
  const pct = run.blocks ? (run.block / run.blocks) * 100 : 0;

  return h(`div.phase.main${klass}`,
    h('div.row', { style: { gap: '7px' } },
      statusDot(active ? 'run' : complete ? 'ok' : 'idle'),
      h('span.name', 'denoise'),
      h('span.mono.t3', { style: { fontSize: '11px' } }, `step ${run.step} of ${run.steps}`),
      h(`span.push.mono.${complete ? 'ok' : 'accent'}`, { style: { fontSize: '12px' } },
        complete ? fmt.seconds(measured) : `${run.elapsed.toFixed(1)} s`),
    ),
    h('div.track.h5', h('i', { style: { width: `${Math.max(0, Math.min(100, pct))}%` } })),
  );
}

/**
 * The streaming ticker.
 *
 * The sparkline is not decoration: each bar is one sampled block of the model,
 * its height the relative size of that block's shard on disk, filled up to the
 * block currently in flight. On a two-species model like FLUX the tall bar at
 * the left IS the 680 MB block the slot pool has to be sized for.
 */
function ticker(run) {
  const profile = run.blockBytes && run.blockBytes.length ? run.blockBytes : null;
  const peak = profile ? Math.max(...profile) : 1;
  const filled = run.blocks ? Math.round((run.block / run.blocks) * SPARK_BARS) : 0;

  const bars = Array.from({ length: SPARK_BARS }, (_, i) => {
    const source = profile
      ? profile[Math.min(profile.length - 1, Math.floor((i / SPARK_BARS) * profile.length))]
      : null;
    const height = source ? Math.max(14, (source / peak) * 100) : 30;
    return h(`i${i < filled ? '.on' : ''}`, { style: { height: `${height}%` } });
  });

  // Nothing streams when the whole model fits: that is the good case, and the
  // strip has to read as one rather than as a stalled counter.
  if (!run.blocks) {
    return h('div.ticker',
      h('span.mono.accent', { style: { fontSize: '12px' } },
        `all ${run.residentBlocks} blocks resident · nothing streamed`),
      h('span.push.t5', { style: { fontSize: '11px' } },
        'this card holds the model — AirCanvas is out of the way'),
    );
  }

  return h('div.ticker',
    h('span.mono.accent', { style: { fontSize: '12px' } },
      `block ${run.block}/${run.blocks} · ${fmt.gbNum(run.bytesStreamed)} GB streamed`
      + (run.prefetchPct === null || run.prefetchPct === undefined ? '' : ` · ${run.prefetchPct}% prefetched`)),
    h('div.spark', bars),
    h('span.mono.t5.nowrap', { style: { fontSize: '11px' } }, `queue ${run.queueDepth}/${run.ringBlocks || 0}`),
  );
}

// -- Simple ---------------------------------------------------------------

function simpleRun() {
  const run = currentRun();
  if (!run) return finishedFallback();
  const expanded = store.ui.showSimpleTelemetry;
  const pct = run.steps ? (run.step / run.steps) * 100 : 0;
  const perStep = run.step > 1 && run.elapsed ? run.elapsed / (run.step - 1) : null;
  const left = perStep ? perStep * (run.steps - run.step) : null;

  return h('div.app',
    topbar(),
    h('div.grow.col', { style: { alignItems: 'center', justifyContent: 'center', background: 'var(--stage)', gap: '26px', padding: '36px' } },
      frame(previewOutput(run), { width: 430, height: 430, accent: true }),
      h('div.col', { style: { width: '430px', gap: '12px' } },
        h('div.row', { style: { alignItems: 'baseline', gap: '10px' } },
          h('span', { style: { fontSize: '16px' } },
            'Painting step ', h('span.mono', String(run.step)), ' of ', h('span.mono', String(run.steps))),
          h('span.push.t3', { style: { fontSize: '14px' } },
            left ? h('span.mono.t2', fmt.remaining(left)) : 'estimating'),
        ),
        bar(pct, { height: 6 }),
        h('p.t4', { style: { margin: 0, fontSize: '13px' } }, reassurance(run)),
      ),
    ),
    expanded ? telemetryStrip(run) : null,
    h('div.row', { style: { flex: 'none', borderTop: '1px solid var(--line-soft)', background: 'var(--rail)', padding: '16px 22px', gap: '14px' } },
      h('button.link.link-quiet.row', {
        type: 'button',
        'aria-expanded': expanded,
        style: { fontSize: '13px' },
        onClick: () => setUi({ showSimpleTelemetry: !expanded }),
      }, 'Show what the machine is doing', expanded ? chevronDown() : chevronRight()),
      h('div.row.push', { style: { gap: '16px' } },
        h('button.link.link-quiet', { type: 'button', style: { fontSize: '13px' }, onClick: cancel }, 'Stop'),
        h('span.t4', { style: { fontSize: '13px' } }, 'Nothing is lost — the model stays on disk.'),
      ),
    ),
  );
}

/** One sentence about what is happening, in words, with no jargon. */
function reassurance(run) {
  const model = store.state.models.find((m) => m.id === run.modelId);
  const params = model ? model.params.replace('B', ' billion') : 'large';
  const vram = fmt.gbNum(store.state.machine.vramTotalBytes, 0);
  return `Streaming a ${params}-parameter model through your ${vram} GB graphics card.`;
}

function finishedFallback() {
  return h('div.app',
    topbar(),
    h('div.stage',
      h('div.col', { style: { alignItems: 'center', gap: '12px' } },
        h('span.t3', 'Nothing is running.'),
        h('button.btn.btn-accent', { type: 'button', onClick: () => setUi({ route: 'desk' }) }, 'Back to the desk'),
      ),
    ),
  );
}
