/* S1 Welcome (Simple) and P1 Welcome setup (Pro).
 *
 * The same question in two registers. Pro asks "what should this machine
 * run?" and answers with a verdict per model per shard format, each one
 * solver output for the live probe. Simple asks the user to pick one thing to
 * start with and shows only times, sizes in GB and a plain-words quality
 * note — no format, no shard count, no bandwidth, and the smaller quantized
 * build chosen silently on their behalf.
 */

import { h } from '../lib/dom.js';
import * as fmt from '../lib/format.js';
import { api } from '../lib/api.js';
import { fail, setUi, store, toast } from '../lib/store.js';
import { estimateSeconds, preferredFormat } from '../lib/estimate.js';
import { topbar } from '../components/chrome.js';
import { bar, checkbox, cpuBanner, divider, metric, radio, segmented, toggle, verdictChip } from '../components/widgets.js';

export function setupScreen() {
  return store.state.mode === 'pro' ? proSetup() : simpleSetup();
}

// -- shared ---------------------------------------------------------------

function selection() {
  return (store.state.setup || {}).selection || {};
}

function selectedFor(model) {
  const chosen = selection()[model.id];
  return chosen || null;
}

function patchSelection(modelId, values) {
  const next = { ...selection() };
  if (values === null) delete next[modelId];
  else next[modelId] = { ...(next[modelId] || {}), ...values };
  api.patch({ setup: { selection: next } }).catch(fail);
}

function startSelected() {
  const chosen = Object.entries(selection());
  if (!chosen.length) {
    toast('Pick at least one model first');
    return;
  }
  Promise.all(chosen.map(([modelId, opts]) => api.startSplit({
    modelId,
    format: opts.format,
    fromGguf: Boolean(opts.fromGguf),
  })))
    .then(() => {
      setUi({ route: 'split' });
      api.patch({ seenWelcome: true }).catch(fail);
    })
    .catch(fail);
}

// -- P1 -------------------------------------------------------------------

function proSetup() {
  const models = store.state.models;
  const images = models.filter((m) => m.kind === 'image');
  const videos = models.filter((m) => m.kind === 'video');

  return h('div.app',
    topbar({ setup: true }),
    h('div.setup-body',
      h('div.setup-main',
        h('div.col', { style: { gap: '7px' } },
          h('h1.title', 'What should this machine run?'),
          h('p.t3', { style: { margin: 0, fontSize: '13px', maxWidth: '620px' } },
            'Each model is split into per-block shards on disk, then streamed to the GPU ' +
            'just-in-time. VRAM scales with the largest block, not the model.'),
        ),
        cpuBanner(store.state.machine),
        divider('Image'),
        h('div.grid-2', images.map(imageCard)),
        divider('Video'),
        h('div.grid-3', videos.map(videoCard)),
      ),
      machineRail(),
    ),
  );
}

function imageCard(model) {
  const chosen = selectedFor(model);
  const format = chosen ? chosen.format : preferredFormat(model);
  const fromGguf = chosen ? Boolean(chosen.fromGguf) : Boolean(model.gguf);
  const job = store.state.splits.find((j) => j.modelId === model.id && j.state !== 'done');

  return h(`div.model-card${chosen ? '.selected' : ''}`,
    h('div.row', { style: { alignItems: 'flex-start', gap: '10px' } },
      h('button', {
        type: 'button',
        style: { display: 'contents' },
        'aria-label': `Select ${model.name}`,
        onClick: () => patchSelection(model.id, chosen ? null : { format, fromGguf }),
      }, checkbox(Boolean(chosen))),
      h('div.grow',
        h('div.card-title', model.name),
        h('div.sub', `${model.params} · ${model.blocks} blocks · needs ~${fmt.gbNum(model.nativeBytes, 0)} GB natively`),
      ),
      verdictChip(model.verdicts[format]),
    ),
    h('div.row',
      h('span', { style: { fontSize: '11px', color: 'var(--text-4)', width: '74px', flex: 'none' } }, 'Shard format'),
      segmented(
        model.formats.map((value) => ({ value, label: value })),
        format,
        (value) => patchSelection(model.id, { format: value, fromGguf }),
        { klass: 'fmt' },
      ),
      h('span.push.mono.t5', { style: { fontSize: '11px' } }, `${fmt.gbNum(model.onDisk[format])} GB on disk`),
    ),
    footerRow(model, { format, fromGguf, job }),
  );
}

function footerRow(model, { format, fromGguf, job }) {
  const children = [];

  if (model.gguf) {
    children.push(h('div.row', { style: { gap: '10px' } },
      toggle(fromGguf, (on) => patchSelection(model.id, { format, fromGguf: on }), {
        label: `Split ${model.name} from a quantized GGUF`,
      }),
      h('span', { style: { fontSize: '12px', color: fromGguf ? 'var(--text-1)' : 'var(--text-sub)' } },
        'From GGUF — ',
        h('span.mono.accent', `${fmt.gbNum(model.gguf.downloadBytes)} GB`),
        ` download instead of ${fmt.gbNum(model.nativeBytes)} GB`,
        model.gguf.verified ? h('span.t5', { style: { fontSize: '11px' } }, ' · verified bitwise') : null,
      ),
    ));
  } else if (model.gated) {
    const signedIn = (store.state.huggingFace || {}).authenticated;
    children.push(h('div.row', { style: { gap: '8px' } },
      h('span.t3', { style: { fontSize: '12px' } },
        signedIn
          ? 'Gated repo — accept the licence on the Hub once, then this downloads.'
          : 'Accept the licence on the Hub, then sign in — once.'),
      signedIn ? null : h('button.link.push', {
        type: 'button',
        onClick: () => setUi({ route: 'settings', settingsSection: 'hf' }),
      }, 'Add a token'),
    ));
  } else {
    children.push(h('span.t5', { style: { fontSize: '12px' } }, model.note || 'Splits from the original checkpoint.'));
  }

  if (job) children.push(downloadRow(job));

  return h('div.col', {
    style: { borderTop: '1px solid var(--line)', paddingTop: '11px', gap: '9px' },
  }, children);
}

function downloadRow(job) {
  const paused = job.state === 'paused';
  const total = job.bytesExpected || 0;
  const done = job.bytesDownloaded || 0;
  const pct = total ? (done / total) * 100 : 0;
  return h('div.row', { style: { gap: '11px' } },
    h('div.grow.col', { style: { gap: '5px' } },
      bar(pct, { tone: paused ? 'amber' : '', height: 4 }),
      h('span.mono', { style: { fontSize: '10px', color: paused ? 'var(--warn)' : 'var(--text-5)' } },
        `${paused ? 'paused · ' : ''}${fmt.gbNum(done)} of ${fmt.gbNum(total)} GB`),
    ),
    paused
      ? h('button.btn.btn-accent.btn-sm', { type: 'button', onClick: () => api.splitAction(job.id, 'resume').catch(fail) }, 'Resume')
      : h('button.btn.btn-sm', { type: 'button', onClick: () => api.splitAction(job.id, 'pause').catch(fail) }, 'Pause'),
  );
}

function videoCard(model) {
  const chosen = selectedFor(model);
  const format = chosen ? chosen.format : preferredFormat(model);
  const verdict = model.verdicts[format];
  const impossible = verdict && verdict.tone === 'fail';

  return h(`div.model-card${chosen ? '.selected' : ''}${impossible ? '.faded' : ''}`,
    { style: { padding: '14px', gap: '10px' } },
    h('div.row', { style: { alignItems: 'flex-start', gap: '9px' } },
      h('button', {
        type: 'button',
        style: { display: 'contents' },
        'aria-label': `Select ${model.name} ${model.params}`,
        onClick: () => patchSelection(model.id, chosen ? null : { format, fromGguf: Boolean(model.gguf) }),
      }, checkbox(Boolean(chosen))),
      h('div.grow',
        h('div', { style: { fontSize: '14px', fontWeight: '600' } }, model.name),
        h('div.sub', `${model.params} · ${model.blocks} blocks`),
      ),
    ),
    h('div', { style: { alignSelf: 'flex-start' } }, verdictChip(verdict)),
    h('div.mono.t5.push', { style: { fontSize: '11px', marginTop: 'auto' } },
      impossible ? 'lower resolution or frames to fit' : `${format} · ${fmt.gbNum(model.onDisk[format])} GB on disk`),
  );
}

function machineRail() {
  const machine = store.state.machine;
  const chosen = Object.entries(selection());
  const toDownload = chosen.reduce((sum, [id, opts]) => {
    const model = store.state.models.find((m) => m.id === id);
    if (!model) return sum;
    return sum + (opts.fromGguf && model.gguf ? model.gguf.downloadBytes : model.nativeBytes);
  }, 0);
  const onDisk = chosen.reduce((sum, [id, opts]) => {
    const model = store.state.models.find((m) => m.id === id);
    return model ? sum + (model.onDisk[opts.format] || 0) : sum;
  }, 0);
  const freeAfter = Math.max(0, (machine.cacheFreeBytes || 0) - onDisk);

  return h('aside.setup-rail',
    h('div.col', { style: { gap: '11px' } },
      h('div.eyebrow', 'This machine'),
      h('div.col', { style: { gap: '8px' } },
        metric('GPU', machine.gpu || 'none'),
        metric('VRAM free', `${fmt.gbNum(machine.vramFreeBytes)} / ${fmt.gbNum(machine.vramTotalBytes)} GB`),
        metric('RAM', `${fmt.gbNum(machine.ramTotalBytes)} GB · ${machine.ramCommittedPct}% committed`),
        metric(machine.diskProbed ? 'Disk (probed)' : 'Disk (assumed)', fmt.gbps(machine.diskBytesPerSecond)),
      ),
      h('p.t4', { style: { margin: 0, fontSize: '12px', lineHeight: '1.5', borderTop: '1px solid var(--line-soft)', paddingTop: '11px' } },
        'Verdicts come from the same budget solver that runs at generation time. ' +
        'Re-probe if another app is holding VRAM.'),
      h('button.btn.btn-accent', {
        type: 'button',
        style: { alignSelf: 'flex-start' },
        onClick: () => api.reprobe(true).then(() => toast('Machine re-probed')).catch(fail),
      }, 'Re-probe'),
    ),
    h('div.col', { style: { marginTop: 'auto', gap: '13px' } },
      h('div.col', { style: { borderTop: '1px solid var(--line-soft)', paddingTop: '15px', gap: '8px' } },
        metric('Selected', `${chosen.length} model${chosen.length === 1 ? '' : 's'}`),
        metric('Left to download', fmt.gb(toDownload)),
        metric('On disk after split', fmt.gb(onDisk)),
        metric('Free after', fmt.gb(freeAfter), { tone: 'ok' }),
      ),
      h('button.btn-primary.wide', { type: 'button', disabled: !chosen.length, onClick: startSelected }, 'Split and continue'),
      h('p.t5', { style: { margin: 0, fontSize: '11px', textAlign: 'center' } },
        'Splitting is one-time and resumable. The original checkpoints can be deleted afterwards.'),
    ),
  );
}

// -- S1 -------------------------------------------------------------------

/**
 * The three things this machine can comfortably run, in plain words.
 *
 * Only `good` verdicts: a "tight · 0.3 GB headroom" model is a real option in
 * Pro, where the reason is on the chip, but offering it here would be a
 * promise Simple mode has no way to qualify. Ties break towards a model this
 * project has actually measured, so the time on the card is a measurement
 * rather than a lower bound wherever possible.
 */
function simpleChoices() {
  // On a machine with no CUDA device every verdict is "runs on CPU · not
  // fast", so demanding a `good` tone would offer nothing at all. There, the
  // bar is simply "the memory plan solves".
  const cpuOnly = (store.state.machine || {}).device !== 'cuda';
  const runnable = store.state.models
    .map((model) => {
      const format = preferredFormat(model);
      return { model, format, verdict: model.verdicts[format] };
    })
    .filter((row) => row.verdict && row.model.simple
      && (cpuOnly ? row.verdict.tone !== 'fail' : row.verdict.tone === 'good'));

  const images = runnable.filter((r) => r.model.kind === 'image');
  const videos = runnable.filter((r) => r.model.kind === 'video');
  const byTime = (a, b) => {
    if (Boolean(a.model.measured) !== Boolean(b.model.measured)) return a.model.measured ? -1 : 1;
    return estimateSeconds(a.model, a.format).totalSeconds - estimateSeconds(b.model, b.format).totalSeconds;
  };

  const best = [...images].sort((a, b) => parseFloat(b.model.params) - parseFloat(a.model.params))[0];
  const fastest = [...images].sort(byTime).find((r) => r !== best);
  const video = [...videos].sort(byTime)[0];
  return [best, fastest, video].filter(Boolean);
}

function simpleSetup() {
  const choices = simpleChoices();
  const chosenId = Object.keys(selection())[0] || (choices[0] && choices[0].model.id);

  return h('div.app',
    topbar({ setup: true }),
    h('div.welcome',
      h('div.col', { style: { gap: '10px', maxWidth: '660px' } },
        h('h1.title-lg', "We checked your machine — here's what it can run."),
        h('p.t3', { style: { margin: 0, fontSize: '14px' } },
          'Pick one to start with. You can add the others later, and nothing you download needs downloading twice.'),
      ),
      cpuBanner(store.state.machine),
      choices.length
        ? h('div.grid-cards', choices.map((choice) => simpleCard(choice, chosenId === choice.model.id)))
        : h('div.card.pad.lg.t3', 'No models yet — pick one and AirCanvas will split it into streamable shards.'),
      h('div.row', { style: { gap: '20px' } },
        h('button.btn-primary.tall', {
          type: 'button',
          disabled: !chosenId,
          onClick: () => {
            const choice = choices.find((c) => c.model.id === chosenId);
            if (!choice) return;
            patchSelection(choice.model.id, { format: choice.format, fromGguf: Boolean(choice.model.gguf) });
            setTimeout(startSelected, 60);
          },
        }, 'Set up'),
        h('span.t3', { style: { fontSize: '13px' } },
          'You can leave it running — it picks up where it left off.'),
      ),
    ),
  );
}

function simpleCard({ model, format }, selected) {
  const { totalSeconds, source } = estimateSeconds(model, format);
  const simple = model.simple;
  const download = model.gguf ? model.gguf.downloadBytes : model.nativeBytes;
  const tone = { good: 'ok', warn: 'warn', plain: 't3' }[simple.tone] || 't3';

  return h(`button.simple-card${selected ? '.selected' : ''}`, {
    type: 'button',
    'aria-pressed': selected,
    onClick: () => patchSelection(model.id, selected ? null : { format, fromGguf: Boolean(model.gguf) }),
  },
    h('div.row', { style: { alignItems: 'flex-start', gap: '11px' } },
      radio(selected),
      h('div.grow.col', { style: { gap: '4px' } },
        h('span', { style: { fontSize: '16px', fontWeight: '600' } }, model.name),
        h('span', { style: { fontSize: '13px', color: 'var(--text-sub)' } }, simple.headline),
      ),
    ),
    h('div.col', { style: { gap: '5px' } },
      source === 'cpu'
        ? h('span.t4', { style: { fontSize: '13px' } }, 'Will run without a graphics card, but slowly')
        : h('span.t3', { style: { fontSize: '13px' } },
          source === 'measured' ? 'Takes ' : 'About ',
          h('span.mono.t1', fmt.duration(totalSeconds)),
          ` ${simple.unit} on this PC`),
      h('span.t3', { style: { fontSize: '13px' } },
        'Download ',
        h('span.mono.t1', fmt.gb(download)),
        model.gguf ? ' ' : null,
        model.gguf ? h('span.strike', `instead of ${fmt.gb(model.nativeBytes)}`) : null),
      h(`span.${tone}`, { style: { fontSize: '12px' } }, simple.quality),
    ),
  );
}
