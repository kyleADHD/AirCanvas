/* P2 Split progress, and its Simple counterpart.
 *
 * The signature element is the block grid: one cell per transformer block,
 * filled as each becomes its own verified safetensors file. It is not
 * decoration — it is the honest shape of the operation, and it is why
 * interrupting is safe: every filled cell is a `.done` marker on disk that a
 * resumed split will skip.
 *
 * Simple mode gets the same job with none of the vocabulary: a bar, a time,
 * and the promise that stopping loses nothing.
 */

import { h } from '../lib/dom.js';
import * as fmt from '../lib/format.js';
import { api } from '../lib/api.js';
import { fail, modelById, store } from '../lib/store.js';
import { topbar } from '../components/chrome.js';
import { bar } from '../components/widgets.js';

export function splitScreen() {
  const jobs = store.state.splits;
  const active = jobs.find((j) => ['downloading', 'splitting'].includes(j.state));
  const others = jobs.filter((j) => j !== active);
  const pro = store.state.mode === 'pro';

  return h('div.app',
    topbar(),
    h('div.split-screen',
      h('div.col', { style: { width: '100%', gap: pro ? '28px' : '22px' } },
        pro ? proHeading(active) : simpleHeading(active),
        active ? (pro ? activeCard(active) : simpleCard(active)) : emptyCard(),
        others.map((job) => (pro ? queuedCard(job) : simpleCard(job))),
        pro ? proFooter() : simpleFooter(),
      ),
    ),
  );
}

function proHeading(active) {
  const name = active ? active.name : 'your models';
  return h('div.col', { style: { gap: '8px' } },
    h('div', { style: { fontSize: '11px', letterSpacing: '0.05em', color: 'var(--violet-500)' } },
      'Step 2 of 2 · one time'),
    h('h1.title-split', `Splitting ${name} into streamable shards`),
    h('p.t3', { style: { margin: 0, fontSize: '13px' } },
      'Each transformer block becomes its own verified safetensors file. ' +
      'Interrupt safely — the split resumes from the last completed block.'),
  );
}

function simpleHeading(active) {
  return h('div.col', { style: { gap: '8px' } },
    h('h1.title', active && active.state === 'downloading' ? 'Downloading your model' : 'Getting your model ready'),
    h('p.t3', { style: { margin: 0, fontSize: '14px' } },
      'This happens once. You can leave it running — it picks up where it left off.'),
  );
}

function activeCard(job) {
  const model = modelById(job.modelId);
  const downloading = job.state === 'downloading';
  const total = job.blocksTotal || (model ? model.blocks : 0) || 1;

  return h('div.card.lg', { style: { padding: '22px', display: 'flex', flexDirection: 'column', gap: '14px' } },
    h('div.row', { style: { alignItems: 'baseline', gap: '12px' } },
      h('span.card-title', job.name),
      h('span.mono.t3', { style: { fontSize: '11px' } },
        `${model ? model.params : ''} · ${job.format}${job.fromGguf ? ' · from GGUF' : ''}`),
      h('span.push.mono.accent', { style: { fontSize: '13px' } },
        downloading
          ? `${fmt.gbNum(job.bytesDownloaded)} of ${fmt.gbNum(job.bytesExpected)} GB`
          : `block ${job.blocksDone}/${total}`),
    ),
    downloading
      ? bar(job.bytesExpected ? (job.bytesDownloaded / job.bytesExpected) * 100 : 0, { height: 6 })
      : blockGrid(job.blocksDone, total),
    h('div.row', { style: { gap: '26px', fontSize: '11px' } },
      h('span.mono.t3', `${fmt.gbNum(job.bytesWritten)} GB written`),
      h('span.mono.t3', job.blocksPerSecond ? `${job.blocksPerSecond.toFixed(2)} blocks/s` : 'measuring rate'),
      h('span.mono.t3', `verified ${job.blocksDone} of ${total}`),
      h('span.push.mono.t5', job.etaSeconds ? `eta ${fmt.duration(job.etaSeconds)}` : 'eta —'),
    ),
  );
}

/**
 * One cell per block. `grid-template-columns: repeat(N, 1fr)` with N from the
 * model, so a 30-block Wan and a 60-block Qwen both fill the same width.
 */
function blockGrid(done, total) {
  const cells = [];
  for (let i = 0; i < total; i += 1) {
    const klass = i < done - 1 ? 'i.done' : i === done - 1 ? 'i.active' : 'i';
    cells.push(h(klass));
  }
  return h('div.split-cells', { style: { gridTemplateColumns: `repeat(${total}, 1fr)` } }, cells);
}

function queuedCard(job) {
  const paused = job.state === 'paused';
  const failed = job.state === 'failed';
  const done = job.state === 'done';
  const total = job.blocksTotal || 1;
  const pct = done ? 100 : (job.blocksDone / total) * 100;

  return h('div.card.lg.quiet', { style: { padding: '22px', display: 'flex', flexDirection: 'column', gap: '13px' } },
    h('div.row', { style: { alignItems: 'baseline', gap: '12px' } },
      h('span.card-title.t2', job.name),
      h('span.mono.t3', { style: { fontSize: '11px' } }, job.format),
      h(`span.push.mono.${failed ? 'fail' : paused ? 'warn' : done ? 'ok' : 't3'}`, { style: { fontSize: '12px' } },
        failed ? job.error
          : paused ? `paused · ${job.blocksDone} of ${total} blocks`
          : done ? 'ready'
          : 'queued'),
    ),
    done ? null : h('div.row', { style: { gap: '14px' } },
      bar(pct, { tone: paused ? 'amber' : '', height: 6, fill: true }),
      paused || failed
        ? h('button.btn.btn-accent.btn-sm', { type: 'button', onClick: () => api.splitAction(job.id, 'resume').catch(fail) }, 'Resume')
        : h('button.btn.btn-sm', { type: 'button', onClick: () => api.splitAction(job.id, 'pause').catch(fail) }, 'Pause'),
    ),
  );
}

function simpleCard(job) {
  const total = job.blocksTotal || 1;
  const downloading = job.state === 'downloading';
  const pct = downloading
    ? (job.bytesExpected ? (job.bytesDownloaded / job.bytesExpected) * 100 : 0)
    : (job.blocksDone / total) * 100;
  const paused = job.state === 'paused';

  return h('div.card.lg', { style: { padding: '22px', display: 'flex', flexDirection: 'column', gap: '12px' } },
    h('div.row',
      h('span', { style: { fontSize: '15px' } }, job.name),
      h('span.push.t3', { style: { fontSize: '13px' } },
        job.state === 'done' ? 'Ready'
          : paused ? 'Paused'
          : downloading ? 'Downloading' : 'Getting ready'),
    ),
    bar(pct, { tone: paused ? 'amber' : '', height: 6 }),
    h('span.t4', { style: { fontSize: '12px' } },
      job.etaSeconds ? fmt.remaining(job.etaSeconds) : 'starting'),
  );
}

function emptyCard() {
  return h('div.card.lg', { style: { padding: '22px' } },
    h('span.t3', 'Nothing to prepare — every selected model is already on disk.'));
}

function proFooter() {
  return h('div.row', { style: { gap: '16px' } },
    h('p.t4', { style: { margin: 0, fontSize: '12px', maxWidth: '560px' } },
      'Shards land in the persistent cache under your Hugging Face home, never inside a project. ' +
      'Once split, generation runs offline.'),
    h('button.btn.push', { type: 'button', onClick: () => api.pauseAllSplits().catch(fail) }, 'Pause all'),
    h('button.btn.btn-danger', {
      type: 'button',
      onClick: () => Promise.all(store.state.splits.map((j) => api.splitAction(j.id, 'cancel'))).catch(fail),
    }, 'Cancel'),
  );
}

function simpleFooter() {
  return h('div.row', { style: { gap: '16px' } },
    h('button.btn', { type: 'button', onClick: () => api.pauseAllSplits().catch(fail) }, 'Stop'),
    h('span.t4', { style: { fontSize: '12px' } }, 'Nothing is lost — what has finished stays finished.'),
  );
}
