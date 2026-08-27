/* P7 Gallery — evidence of use, not a showroom.
 *
 * Tiles keep their real aspect ratio (a 480×832 Wan clip is tall, a 1024²
 * image is square), so the masonry is uneven for the same reason a real
 * folder of outputs is. A queued prompt gets a tile with no artwork, because
 * it is part of the record too.
 *
 * The detail rail's report snapshot reuses the Report screen's own reader, so
 * the miniature residency bar and the full one cannot drift apart.
 */

import { h } from '../lib/dom.js';
import * as fmt from '../lib/format.js';
import { api } from '../lib/api.js';
import { fail, setUi, store } from '../lib/store.js';
import { topbar } from '../components/chrome.js';
import { eyebrow, frame, metric, stacked } from '../components/widgets.js';
import { reproduce, residencySegments } from './report.js';

export function galleryScreen() {
  const outputs = sorted(filtered());
  const selected = outputs.find((o) => o.id === store.ui.selectedOutputId) || outputs[0] || null;

  return h('div.app',
    galleryTopbar(outputs.length),
    h('div.body',
      outputs.length
        ? h('div.masonry', outputs.map((output) => tile(output, selected && output.id === selected.id)))
        : emptyGallery(),
      selected ? detailRail(selected) : null,
    ),
  );
}

function filtered() {
  const { outputs } = store.state;
  const filter = store.ui.galleryFilter;
  if (filter === 'all') return outputs;
  return outputs.filter((output) => output.modelId === filter);
}

function sorted(outputs) {
  if (store.ui.gallerySort === 'slowest') {
    return [...outputs].sort((a, b) => (b.durationSeconds || 0) - (a.durationSeconds || 0));
  }
  return outputs; // the store already keeps newest first
}

/**
 * Two chips and a count, as the design has it. The model filter is a select
 * dressed as a chip rather than one chip per model: a gallery of a hundred
 * outputs across six models would otherwise push the count off the bar.
 */
function galleryTopbar(count) {
  const bar = topbar();
  const models = [...new Set(store.state.outputs.map((o) => o.modelId))]
    .map((id) => store.state.models.find((m) => m.id === id))
    .filter(Boolean);

  const status = bar.querySelector('.status');
  status.replaceChildren(
    h('select.btn.btn-sm', {
      'aria-label': 'Filter by model',
      value: store.ui.galleryFilter,
      onChange: (event) => setUi({ galleryFilter: event.target.value }),
    },
      h('option', { value: 'all', selected: store.ui.galleryFilter === 'all' }, 'All models'),
      models.map((m) => h('option', { value: m.id, selected: store.ui.galleryFilter === m.id }, m.name)),
    ),
    h('select.btn.btn-sm', {
      'aria-label': 'Sort',
      value: store.ui.gallerySort || 'newest',
      onChange: (event) => setUi({ gallerySort: event.target.value }),
    },
      h('option', { value: 'newest', selected: (store.ui.gallerySort || 'newest') === 'newest' }, 'Newest'),
      h('option', { value: 'slowest', selected: store.ui.gallerySort === 'slowest' }, 'Slowest first'),
    ),
    h('span.mono.t5', { style: { fontSize: '11px' } }, `${count} output${count === 1 ? '' : 's'}`),
  );
  return bar;
}

function tile(output, selected) {
  const queued = output.state === 'queued';
  const isVideo = output.kind === 'video';
  const ratio = output.width && output.height ? `${output.width} / ${output.height}` : '1 / 1';

  return h(`button.tile${selected ? '.selected' : ''}${isVideo ? '.video' : ''}${queued ? '.queued' : ''}`, {
    type: 'button',
    onClick: () => setUi({ selectedOutputId: output.id }),
  },
    h('div.art', { style: { aspectRatio: queued ? '5 / 2' : ratio } },
      frame(output, { width: '100%', height: '100%' }),
      isVideo && output.frames
        ? h('div.badge', { style: { right: '7px', bottom: '7px' } }, fmt.clip(output.frames))
        : null,
    ),
    h('div.caption',
      h('div.p', output.prompt),
      queued
        ? null
        : h('div.row', { style: { gap: '7px' } },
          h('span.meta.mono', metaLine(output)),
          output.reproducedCount
            ? h('span.chip.plain', `reproduced ×${output.reproducedCount}`)
            : null,
        ),
    ),
  );
}

function metaLine(output) {
  return [
    output.modelName,
    output.seed === null || output.seed === undefined ? 'random seed' : String(output.seed),
    fmt.duration(output.durationSeconds),
  ].filter(Boolean).join(' · ');
}

function detailRail(output) {
  const report = output.report;
  return h('aside.rail.gallery-rail',
    h('section', { style: { display: 'flex', flexDirection: 'column', gap: '12px' } },
      frame(output, { width: '100%', height: '212px' }),
      h('div', { style: { fontSize: '13px', lineHeight: '1.5' } }, output.prompt),
    ),
    h('section', { style: { display: 'flex', flexDirection: 'column', gap: '9px' } },
      metric('Model', `${output.modelName} · ${output.format}`),
      metric('Seed', output.seed === null || output.seed === undefined ? 'random' : String(output.seed)),
      metric('Steps · guidance', `${output.steps} · ${Number(output.guidance || 0).toFixed(1)}`),
      output.loras && output.loras.length
        ? h('div.metric-row',
          h('span', 'LoRA'),
          h('span.mono', { style: { textAlign: 'right' } },
            output.loras.map((l) => h('div', `${l.id} ${Number(l.scale).toFixed(2)}`))),
        )
        : null,
    ),
    report ? snapshotSection(output, report) : queuedSection(output),
  );
}

function snapshotSection(output, report) {
  const phases = report.phases || {};
  const engine = phases.engine || {};
  const loads = engine.block_loads || 0;
  const hits = engine.prefetch_hits || 0;

  return h('section', { style: { display: 'flex', flexDirection: 'column', gap: '11px' } },
    eyebrow('Report snapshot'),
    stacked(residencySegments(report), { mini: true }),
    h('div.col', { style: { gap: '6px' } },
      metric('Total', `${fmt.seconds(output.durationSeconds)} · ${output.secondsPerStep} s/step`),
      metric('Streamed', `${fmt.gb(engine.bytes_loaded)} · ${loads ? fmt.pct((hits / loads) * 100) : fmt.DASH}`),
      metric('Disk then', fmt.gbps((report.hardware || {}).disk_bw_bytes_s)),
    ),
    h('div.row', { style: { gap: '9px' } },
      h('button.link', { type: 'button', onClick: () => setUi({ route: 'report', reportOutputId: output.id }) }, 'Open the full report'),
      h('button.link.link-quiet.push', {
        type: 'button',
        onClick: () => api.deleteOutput(output.id).catch(fail),
      }, 'Delete'),
    ),
    h('button.btn-primary.wide', { type: 'button', onClick: () => reproduce(output) }, 'Reproduce'),
    h('p.t5', { style: { margin: 0, fontSize: '11px', textAlign: 'center' } },
      'Restores prompt, seed, LoRA stack and budget caps exactly as they were.'),
  );
}

function queuedSection(output) {
  return h('section', { style: { display: 'flex', flexDirection: 'column', gap: '11px' } },
    eyebrow('Queued'),
    h('p.t4', { style: { margin: 0, fontSize: '12px' } },
      'This prompt starts when the current run finishes. Nothing has been measured for it yet.'),
    h('button.btn', { type: 'button', onClick: () => api.deleteOutput(output.id).catch(fail) }, 'Remove from the queue'),
  );
}

function emptyGallery() {
  return h('div.grow.stage',
    h('div.col', { style: { alignItems: 'center', gap: '10px' } },
      h('span.t3', 'Nothing generated yet.'),
      h('span.t5', { style: { fontSize: '12px' } }, 'Every output keeps its report, so you can reproduce it exactly.'),
      h('button.btn.btn-accent', { type: 'button', onClick: () => setUi({ route: 'desk' }) }, 'Go to the desk'),
    ),
  );
}
