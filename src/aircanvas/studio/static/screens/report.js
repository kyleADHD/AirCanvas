/* P6 Report — where the time went, and the plan that produced it.
 *
 * Every figure comes from one payload: `pipe.report(as_dict=True)`, persisted
 * with the output. The Gallery's report snapshot reuses the same reader
 * (`residencySegments`), so the miniature bar in the rail and the hero bar
 * here can never disagree.
 *
 * The hero graphic is the residency plan, because it is the answer to the
 * question the whole project exists to answer: what was actually in the 6 GB.
 */

import { h } from '../lib/dom.js';
import * as fmt from '../lib/format.js';
import { api } from '../lib/api.js';
import { fail, setUi, store, toast } from '../lib/store.js';
import { logo } from '../lib/icons.js';
import { applyModel, patchDesk, startRun } from './desk.js';
import { bar, eyebrow, frame, metric, stacked, statusDot } from '../components/widgets.js';

/** VRAM segments of a report's plan, in the order the bar draws them. */
export function residencySegments(report) {
  const plan = (report || {}).plan || {};
  const budget = plan.vram_budget_bytes || 0;
  const planned = plan.vram_planned_bytes || 0;
  return [
    {
      label: 'resident',
      klass: 'seg-resident',
      bytes: (plan.resident_block_bytes || 0) + (plan.resident_shard_bytes || 0),
    },
    { label: 'slots', klass: 'seg-slots', bytes: plan.slot_bytes || 0, light: true },
    { label: 'activations', klass: 'seg-activations', bytes: plan.activation_bytes || 0, light: true },
    { label: 'reserve', klass: 'seg-reserve', bytes: Math.max(0, budget - planned) },
  ];
}

function ramSegments(report) {
  const plan = (report || {}).plan || {};
  const hardware = (report || {}).hardware || {};
  const total = hardware.ram_total_bytes || store.state.machine.ramTotalBytes || 0;
  const pinned = plan.pinned_bytes || 0;
  const cache = plan.ram_cache_bytes || 0;
  return [
    { label: 'pinned ring', klass: 'seg-resident', bytes: pinned },
    { label: 'shard cache', klass: 'seg-slots', bytes: cache, light: true },
    { label: 'other apps + free', klass: 'seg-other', bytes: Math.max(0, total - pinned - cache) },
  ];
}

export function reportScreen() {
  const output = store.state.outputs.find((o) => o.id === store.ui.reportOutputId)
    || store.state.outputs.find((o) => o.report);
  if (!output || !output.report) return missing();

  const report = output.report;
  const phases = report.phases || {};
  const engine = phases.engine || {};
  const plan = report.plan || {};

  return h('div.app',
    h('header.topbar',
      logo(18),
      h('button.link.link-quiet', { type: 'button', onClick: () => setUi({ route: 'desk' }) }, 'Desk'),
      h('span.mono.t5', { style: { fontSize: '11px' } }, `run ${output.id.slice(0, 8)}`),
      h('div.status.push', statusDot('ok'), h('span', fmt.seconds(output.durationSeconds))),
    ),
    h('div.report-body',
      h('div.report-image', frame(output, { width: '100%', height: 'auto' })),
      h('div.report-main',
        header(output),
        h('div.col', { style: { gap: '12px' } },
          eyebrow('Residency plan', 'chosen by the budget solver before step 1'),
          // The axis is the budget the solver was given (free VRAM at plan
          // time), not the card's capacity: labelling it "VRAM 6.40 GB" while
          // the segments add up to 5.10 would leave 1.3 GB of phantom space
          // that was never ours to spend.
          memoryBar('VRAM budget', (plan.vram_budget_bytes || 0), residencySegments(report), [
            `${plan.resident_blocks || 0} blocks pinned`,
            `${plan.gpu_slots || 0} slots · pool ${fmt.gb(plan.slot_bytes, 2)}`,
            `activations ${fmt.gb(plan.activation_bytes, 2)}`,
            `reserve ${fmt.gb(Math.max(0, (plan.vram_budget_bytes || 0) - (plan.vram_planned_bytes || 0)), 2)} for the driver`,
            vramTotal(report) ? `of ${fmt.gb(vramTotal(report), 2)} on the card` : null,
          ]),
          memoryBar('System RAM', ramTotal(report), ramSegments(report), [
            `pinned ring ${plan.ring_depth || 0} blocks`,
            plan.ram_cache_bytes ? 'disk touched once per run' : 'shards re-read from disk each step',
          ]),
        ),
        h('div.grid-2',
          diskCard(report, engine, plan),
          phaseCard(phases),
        ),
        footer(report, plan, phases),
      ),
    ),
  );
}

function ramTotal(report) {
  const hardware = report.hardware || {};
  return hardware.ram_total_bytes || store.state.machine.ramTotalBytes || 0;
}

function vramTotal(report) {
  const hardware = report.hardware || {};
  return hardware.vram_total_bytes || store.state.machine.vramTotalBytes || 0;
}

function header(output) {
  return h('div.row', { style: { gap: '12px' } },
    h('h1.title-panel', 'Report'),
    h('button.btn.btn-sm', { type: 'button', onClick: () => setUi({ route: 'desk' }) }, 'Collapse'),
    h('div.row.push', { style: { gap: '9px' } },
      h('a.btn', { href: api.outputReport(output.id), download: `run-${output.id}.json` }, 'Export JSON'),
      h('button.btn.btn-accent', { type: 'button', onClick: () => reproduce(output) }, 'Reproduce'),
    ),
  );
}

/** Restore prompt, seed, steps, guidance, LoRA stack and caps, then run. */
export function reproduce(output) {
  const model = store.state.models.find((m) => m.id === output.modelId);
  if (!model) {
    toast(`${output.modelName || output.modelId} is not in the catalog any more`, 'fail');
    return;
  }
  applyModel(model);
  patchDesk({
    prompt: output.prompt,
    seed: output.seed,
    steps: output.steps,
    guidance: output.guidance,
    width: output.width,
    height: output.height,
    frames: output.frames,
    loras: output.loras || [],
  }, { immediate: true });
  api.markReproduced(output.id).catch(fail);
  startRun();
}

function memoryBar(label, total, segments, footnotes) {
  return h('div.col', { style: { gap: '7px' } },
    h('div.row', { style: { fontSize: '12px' } },
      h('span.t3', label),
      h('span.push.mono.t2', fmt.gb(total, 2)),
    ),
    stacked(segments),
    h('div.row', { style: { gap: '18px', fontSize: '11px' } },
      footnotes.filter(Boolean).map((note) => h('span.t5', note)),
    ),
  );
}

function diskCard(report, engine, plan) {
  const loads = engine.block_loads || 0;
  const hits = engine.prefetch_hits || 0;
  const bandwidth = (report.hardware || {}).disk_bw_bytes_s;

  return h('div.card.pad', { style: { display: 'flex', flexDirection: 'column', gap: '10px' } },
    h('div.eyebrow', 'Disk bandwidth'),
    h('div.hero-metric',
      h('b', fmt.gbNum(bandwidth, 2)),
      h('span', 'GB/s'),
    ),
    h('div.col', { style: { gap: '6px' } },
      metric('Read this run', fmt.gb(engine.bytes_loaded)),
      metric('Served by prefetch', loads ? fmt.pct((hits / loads) * 100) : fmt.DASH, {
        tone: loads && hits / loads > 0.9 ? 'ok' : '',
      }),
      metric('Demand stalls', `${engine.sync_loads || 0} · ${fmt.seconds(engine.sync_load_s)}`),
      plan.step_read_bytes ? metric('Per step', fmt.gb(plan.step_read_bytes)) : null,
    ),
  );
}

function phaseCard(phases) {
  const total = Math.max(1e-6, phases.total_s || 0);
  const rows = [
    { name: 'encode', value: phases.encode_s || 0, tone: 'green' },
    { name: 'denoise', value: phases.denoise_s || 0, tone: '' },
    { name: 'decode', value: phases.decode_s || 0, tone: 'light' },
  ];
  return h('div.card.pad', { style: { display: 'flex', flexDirection: 'column', gap: '10px' } },
    h('div.eyebrow', 'Per-phase wall time'),
    h('div.col', { style: { gap: '8px' } },
      rows.map((row) => h('div.row', { style: { gap: '9px' } },
        h('span.t3', { style: { fontSize: '12px', width: '60px' } }, row.name),
        bar((row.value / total) * 100, { tone: row.tone, height: 7, fill: true }),
        h('span.mono', { style: { fontSize: '11px', width: '56px', textAlign: 'right' } },
          fmt.seconds(row.value)),
      )),
    ),
    h('div.row', { style: { borderTop: '1px solid var(--line)', paddingTop: '9px', fontSize: '12px' } },
      h('span.t3', `${phases.steps || 0} steps`),
      h('span.push.mono', phases.steps ? `${((phases.denoise_s || 0) / phases.steps).toFixed(1)} s/step` : fmt.DASH),
    ),
  );
}

/**
 * Bottleneck and next lever, derived rather than asserted: the plan knows how
 * many seconds of disk each step implies, and the phases know how long each
 * step actually took. If most of the step was disk, the disk is the answer.
 */
function footer(report, plan, phases) {
  const steps = phases.steps || 0;
  const ioSeconds = (plan.step_read_seconds || 0) * steps;
  const denoise = phases.denoise_s || 0;
  const share = denoise > 0 ? ioSeconds / denoise : 0;
  const diskBound = share > 0.5;
  const compression = report.compression;

  const lever = diskBound
    ? (compression === 'nf4'
      ? 'fp8 instead of nf4 trades disk traffic for VRAM, or more RAM lets the shard cache hold the set'
      : compression === 'fp8'
        ? 'more RAM, so the shard cache holds the streamed set and disk is touched once'
        : 'fp8 shards roughly halve the bytes read per step')
    : 'more VRAM, so more blocks stay resident between steps';

  return h('div.row', {
    style: {
      marginTop: 'auto',
      gap: '22px',
      fontSize: '12px',
      borderTop: '1px solid var(--line-soft)',
      paddingTop: '14px',
    },
  },
    h('span.t5', `Bottleneck: ${diskBound ? 'disk' : 'compute'}`),
    h('span.t5', `Next lever: ${lever}`),
    h('span.push.t5', { title: warningsText(plan) },
      `${Math.round(share * 100)}% of the denoise loop was disk time`),
  );
}

function warningsText(plan) {
  return (plan.warnings || []).join('\n') || 'The solver raised no warnings for this run.';
}

function missing() {
  return h('div.app',
    h('header.topbar', logo(18), h('span.t3', 'Report')),
    h('div.stage',
      h('div.col', { style: { alignItems: 'center', gap: '12px' } },
        h('span.t3', 'No run has produced a report yet.'),
        h('button.btn.btn-accent', { type: 'button', onClick: () => setUi({ route: 'desk' }) }, 'Back to the desk'),
      ),
    ),
  );
}
