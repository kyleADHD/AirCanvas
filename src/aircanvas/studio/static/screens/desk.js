/* S2 Desk (Simple) and P3/P4 Desk (Pro, image and video).
 *
 * One screen, three faces. The Pro desk shows every parameter the solver
 * takes and the plan it produced; the video variant swaps the square stage for
 * a clip and a filmstrip and adds the controls that only make sense for
 * frames. The Simple desk shows a prompt, a shape, a quality level and a
 * time — no seed, no steps, no guidance, no LoRA, no VRAM, deliberately.
 *
 * Parameter edits are optimistic locally and debounced to the server, so
 * typing never waits on a round trip and a reload never loses the prompt.
 */

import { autosize, h } from '../lib/dom.js';
import * as fmt from '../lib/format.js';
import { api } from '../lib/api.js';
import { currentModel, fail, installedModels, setUi, store, toast } from '../lib/store.js';
import { estimateSeconds, preferredFormat } from '../lib/estimate.js';
import { iconRail, topbar } from '../components/chrome.js';
import { grid, image, video } from '../lib/icons.js';
import { cpuBanner, eyebrow, frame, metric, segmented, slider } from '../components/widgets.js';

let patchTimer = null;
let estimateTimer = null;

/** Merge desk values locally now, persist them shortly. */
export function patchDesk(values, { immediate = false } = {}) {
  Object.assign(store.state.desk, values);
  setUi({});
  clearTimeout(patchTimer);
  const send = () => api.patch({ desk: store.state.desk }).catch(fail);
  if (immediate) send();
  else patchTimer = setTimeout(send, 400);
  scheduleEstimate();
}

function scheduleEstimate() {
  clearTimeout(estimateTimer);
  estimateTimer = setTimeout(() => {
    const model = currentModel();
    if (!model) return;
    const desk = store.state.desk;
    api.estimate({
      modelId: model.id,
      format: desk.format || preferredFormat(model),
      steps: desk.steps,
      width: desk.width,
      height: desk.height,
      frames: desk.frames,
    }).then((estimate) => setUi({ estimate })).catch(() => { /* the local bound still shows */ });
  }, 250);
}

function estimate(model) {
  const cached = store.ui.estimate;
  if (cached) return cached;
  const local = estimateSeconds(model, store.state.desk.format || preferredFormat(model), {
    steps: store.state.desk.steps,
  });
  return { totalSeconds: local.totalSeconds, secondsPerStep: local.secondsPerStep, source: local.source };
}

export function startRun() {
  const model = currentModel();
  if (!model) {
    toast('Pick a model first');
    return;
  }
  const desk = store.state.desk;
  if (!String(desk.prompt || '').trim()) {
    toast('Write a prompt first');
    return;
  }
  api.startRun({
    modelId: model.id,
    format: desk.format || preferredFormat(model),
    prompt: desk.prompt,
    negativePrompt: desk.negativePrompt,
    steps: desk.steps,
    width: desk.width,
    height: desk.height,
    frames: model.kind === 'video' ? desk.frames : 1,
    seed: desk.seed,
    guidance: desk.guidance,
    loras: desk.loras,
    caps: desk.caps,
    fromGguf: Boolean(model.installed && model.installed.fromGguf),
  }).then(() => setUi({ route: 'run' })).catch(fail);
}

export function deskScreen() {
  return store.state.mode === 'pro' ? proDesk() : simpleDesk();
}

// -- Pro ------------------------------------------------------------------

function proDesk() {
  const model = currentModel();
  const isVideo = Boolean(model && model.kind === 'video');
  const last = store.state.outputs.find((o) => o.state === 'done' && (!model || o.modelId === model.id));

  return h('div.app',
    topbar(),
    h('div.body',
      iconRail([
        { title: 'Images', icon: image, active: !isVideo, onClick: () => selectKind('image') },
        { title: 'Video', icon: video, active: isVideo, onClick: () => selectKind('video') },
        { title: 'Gallery', icon: grid, active: false, onClick: () => setUi({ route: 'gallery' }) },
      ]),
      h('div.grow.col',
        isVideo ? videoStage(model, last) : imageStage(model, last),
        dock(model, isVideo),
      ),
      h('aside.rail',
        modelSection(model),
        isVideo ? startFrameSection(model) : null,
        loraSection(),
        budgetSection(model, isVideo),
      ),
    ),
  );
}

function selectKind(kind) {
  const installed = installedModels();
  const next = installed.find((m) => m.kind === kind) || store.state.models.find((m) => m.kind === kind);
  if (!next) {
    toast(`No ${kind} model installed yet`);
    return;
  }
  applyModel(next);
}

export function applyModel(model) {
  patchDesk({
    modelId: model.id,
    format: preferredFormat(model),
    width: model.shape.width,
    height: model.shape.height,
    frames: model.shape.frames,
    steps: model.shape.steps,
    guidance: model.shape.guidance,
  }, { immediate: true });
}

function imageStage(model, last) {
  return h('div.stage',
    h('div.col', { style: { width: '500px', gap: '11px' } },
      frame(last, { width: 500, height: 500 }),
      h('div.row', { style: { gap: '14px', fontSize: '11px' } },
        last
          ? [
            h('span.mono.t5', `${last.width} × ${last.height}`),
            h('span.mono.t5', last.seed === null || last.seed === undefined ? 'seed random' : `seed ${last.seed}`),
            h('span.mono.t5', `${last.steps} steps`),
            h('span.mono.t5', fmt.seconds(last.durationSeconds)),
            h('button.link.push', {
              type: 'button',
              onClick: () => setUi({ route: 'report', reportOutputId: last.id }),
            }, 'Report'),
          ]
          : h('span.mono.t5', model ? 'nothing generated yet' : 'no model installed'),
      ),
    ),
  );
}

function videoStage(model, last) {
  const frames = store.state.desk.frames || (model ? model.shape.frames : 81);
  const fps = model ? model.shape.fps : 16;
  const current = store.ui.videoFrame || 0;
  const tiles = 20;

  return h('div.stage', { style: { flexDirection: 'column', gap: '20px', padding: '30px' } },
    frame(last, {
      width: 660,
      height: 380,
      badge: `${fmt.clip(frames, fps)} · ${fps} fps`,
    }),
    h('div.col', { style: { width: '800px', gap: '8px' } },
      h('div.filmstrip', Array.from({ length: tiles }, (_, i) => h('button', {
        type: 'button',
        'aria-current': Math.round((current / Math.max(1, frames - 1)) * (tiles - 1)) === i,
        'aria-label': `Frame ${Math.round((i / (tiles - 1)) * (frames - 1)) + 1}`,
        onClick: () => setUi({ videoFrame: Math.round((i / (tiles - 1)) * (frames - 1)) }),
      }))),
      h('div.row', { style: { justifyContent: 'space-between', fontSize: '10px' } },
        h('span.mono.t5', '1'),
        h('span.mono.t5', `frame ${current + 1} / ${frames}`),
        h('span.mono.t5', String(frames)),
      ),
    ),
  );
}

function dock(model, isVideo) {
  const desk = store.state.desk;
  const est = estimate(model);

  return h('div.dock',
    promptBox({ klass: 'prompt', min: 46, placeholder: 'Describe what to generate', value: desk.prompt }),
    h('div.row',
      numberField('steps', desk.steps, (value) => patchDesk({ steps: clampInt(value, 1, 200) })),
      isVideo ? numberField('frames', desk.frames, (value) => patchDesk({ frames: clampInt(value, 1, 241) })) : null,
      sizeField(desk),
      seedField(desk),
      isVideo ? null : numberField('guidance', desk.guidance, (value) => patchDesk({ guidance: Number(value) || 0 }), { step: 0.1 }),
      h('div.row.push', { style: { gap: '14px' } },
        h('span.mono.t3', { style: { fontSize: '11px' } },
          est.source === 'cpu'
            ? 'no estimate without a GPU'
            : est.totalSeconds
              ? `${est.source === 'measured' ? '' : '~'}${fmt.duration(est.totalSeconds)} · ${est.secondsPerStep.toFixed(1)} s/step`
              : 'no estimate yet'),
        h('button.btn-primary', { type: 'button', disabled: !model, onClick: startRun }, 'Generate'),
      ),
    ),
  );
}

/** The prompt box: one line tall until the prompt is longer than one line. */
function promptBox({ klass, min, placeholder, value }) {
  const el = h(`textarea.${klass}`, {
    'data-focus-key': 'prompt',
    rows: 1,
    placeholder,
    value: value || '',
    onInput: (event) => {
      autosize(event.target, min);
      patchDesk({ prompt: event.target.value });
    },
  });
  requestAnimationFrame(() => autosize(el, min));
  return el;
}

function clampInt(value, min, max) {
  const n = parseInt(value, 10);
  if (!isFinite(n)) return min;
  return Math.max(min, Math.min(max, n));
}

function numberField(label, value, onChange, { step = 1 } = {}) {
  return h('div.field',
    h('label', { for: `field-${label}` }, label),
    h('input', {
      id: `field-${label}`,
      'data-focus-key': `field-${label}`,
      type: 'number',
      step,
      value: value ?? '',
      onChange: (event) => onChange(event.target.value),
    }),
  );
}

function sizeField(desk) {
  return h('div.field',
    h('label', { for: 'field-size' }, 'size'),
    h('input.wide', {
      id: 'field-size',
      'data-focus-key': 'field-size',
      value: `${desk.width} × ${desk.height}`,
      onChange: (event) => {
        const [w, hh] = String(event.target.value).split(/[^\d]+/).filter(Boolean).map(Number);
        if (w && hh) patchDesk({ width: w, height: hh });
        else setUi({});
      },
    }),
  );
}

function seedField(desk) {
  return h('div.field',
    h('label', { for: 'field-seed' }, 'seed'),
    h('input', {
      id: 'field-seed',
      'data-focus-key': 'field-seed',
      placeholder: 'random',
      value: desk.seed ?? '',
      onChange: (event) => {
        const raw = event.target.value.trim();
        patchDesk({ seed: raw === '' ? null : clampInt(raw, 0, 2 ** 31 - 1) });
      },
    }),
  );
}

function modelSection(model) {
  if (!model) {
    return h('section',
      eyebrow('Model'),
      h('div.dashed', { style: { marginTop: '10px' } },
        h('span.accent', 'No models yet'),
        h('span.t5', { style: { fontSize: '11px' } }, 'pick one and AirCanvas will split it into streamable shards'),
        h('button.link', { type: 'button', onClick: () => setUi({ route: 'setup' }) }, 'Choose a model'),
      ),
    );
  }
  const installed = model.installed;
  const format = store.state.desk.format || preferredFormat(model);
  return h('section', { style: { display: 'flex', flexDirection: 'column', gap: '10px' } },
    eyebrow('Model'),
    h('div.card.selected', { style: { padding: '12px', display: 'flex', flexDirection: 'column', gap: '6px' } },
      h('div.row',
        h('span', { style: { fontSize: '14px', fontWeight: '600' } }, model.name),
        h('span.chip.plain', format),
        h('button.link.push', { type: 'button', onClick: () => setUi({ route: 'setup' }) }, 'Change'),
      ),
      h('span.mono.t3', { style: { fontSize: '11px' } },
        installed
          ? `${model.params} · ${installed.blocks} shards · ${fmt.gb(installed.diskBytes)} cached`
          : `${model.params} · ${model.blocks} blocks · not split yet`),
      model.measured && model.measured.residentRatio
        ? h('span.ok', { style: { fontSize: '11px' } }, 'Byte-identical to full-VRAM execution')
        : null,
    ),
  );
}

function startFrameSection(model) {
  const desk = store.state.desk;
  const frames = desk.frames || model.shape.frames;
  const fps = model.shape.fps || 16;
  return h('section', { style: { display: 'flex', flexDirection: 'column', gap: '12px' } },
    eyebrow('Start frame'),
    h('div.dashed.tall', {
      onDragOver: (event) => { event.preventDefault(); event.currentTarget.classList.add('over'); },
      onDragLeave: (event) => event.currentTarget.classList.remove('over'),
      onDrop: (event) => {
        event.preventDefault();
        event.currentTarget.classList.remove('over');
        const file = event.dataTransfer.files[0];
        if (file) patchDesk({ startFrame: file.name });
      },
    },
      h('span.accent', desk.startFrame || 'Drop an image'),
      h('span.t5', { style: { fontSize: '11px' } }, 'or leave empty for text-to-video'),
    ),
    metric('Frame count', `${frames} · ${(frames / fps).toFixed(1)} s`),
    slider({
      value: frames,
      min: 9,
      max: 241,
      step: 4,
      ariaLabel: 'Frame count',
      onInput: (value) => patchDesk({ frames: value }),
    }),
    h('p.t5', { style: { margin: 0, fontSize: '11px', lineHeight: '1.5' } },
      'More frames cost activations, not weights — VRAM still only holds the largest block.'),
  );
}

function loraSection() {
  const loras = store.state.desk.loras || [];
  return h('section', { style: { display: 'flex', flexDirection: 'column', gap: '12px' } },
    eyebrow('LoRA stack', 'fused after dequant'),
    loras.map((lora, index) => h('div.card', { style: { padding: '11px', display: 'flex', flexDirection: 'column', gap: '8px' } },
      h('div.row',
        h('span.truncate', { style: { fontSize: '13px' }, title: lora.id }, lora.id),
        h('span.push.mono.accent', { style: { fontSize: '12px' } }, Number(lora.scale).toFixed(2)),
        h('button.link.link-quiet', {
          type: 'button',
          'aria-label': `Remove ${lora.id}`,
          onClick: () => patchDesk({ loras: loras.filter((_, i) => i !== index) }, { immediate: true }),
        }, 'Remove'),
      ),
      slider({
        value: Number(lora.scale),
        min: 0,
        max: 1.5,
        step: 0.05,
        ariaLabel: `${lora.id} scale`,
        onInput: (value) => {
          const next = loras.map((l, i) => (i === index ? { ...l, scale: value } : l));
          patchDesk({ loras: next });
        },
      }),
      lora.kind || lora.sizeBytes
        ? h('span.mono.t5', { style: { fontSize: '10px' } },
          [lora.kind, lora.sizeBytes ? `${fmt.mb(lora.sizeBytes)} resident` : null].filter(Boolean).join(' · '))
        : null,
    )),
    h('div.dashed', {
      onDragOver: (event) => { event.preventDefault(); event.currentTarget.classList.add('over'); },
      onDragLeave: (event) => event.currentTarget.classList.remove('over'),
      onDrop: (event) => {
        event.preventDefault();
        event.currentTarget.classList.remove('over');
        const file = event.dataTransfer.files[0];
        if (file) addLora(file.name, file.size);
      },
      onClick: () => {
        const source = window.prompt('LoRA file, folder or Hub repo id');
        if (source) addLora(source.trim(), null);
      },
    },
      h('span.accent', 'Add LoRA'),
      h('span.t5', { style: { fontSize: '11px' } }, 'drop a .safetensors, a folder, or paste a repo id'),
    ),
  );
}

function addLora(source, sizeBytes) {
  const loras = store.state.desk.loras || [];
  if (loras.some((l) => l.id === source)) {
    toast('That adapter is already in the stack');
    return;
  }
  patchDesk({
    loras: [...loras, { id: source, source, scale: 0.8, sizeBytes: sizeBytes || null, kind: null }],
  }, { immediate: true });
}

function budgetSection(model, isVideo) {
  const machine = store.state.machine;
  const desk = store.state.desk;
  const caps = desk.caps || {};
  const vramCap = caps.vramBytes || machine.vramTotalBytes || 0;
  const ramCap = caps.ramBytes || machine.ramTotalBytes || 0;
  const format = desk.format || (model ? preferredFormat(model) : 'fp8');
  const verdict = model ? model.verdicts[format] : null;

  return h('section', { style: { display: 'flex', flexDirection: 'column', gap: '13px' } },
    eyebrow('Budget caps'),
    h('div.col', { style: { gap: '7px' } },
      metric('VRAM', `${fmt.gbNum(vramCap)} / ${fmt.gbNum(machine.vramTotalBytes)} GB`),
      slider({
        value: vramCap,
        min: 1e9,
        max: Math.max(machine.vramTotalBytes || 1e9, vramCap),
        step: 1e8,
        ariaLabel: 'VRAM cap',
        onInput: (value) => patchDesk({ caps: { ...caps, vramBytes: value } }),
      }),
    ),
    isVideo ? null : h('div.col', { style: { gap: '7px' } },
      metric('RAM', `${fmt.gbNum(ramCap)} / ${fmt.gbNum(machine.ramTotalBytes)} GB`),
      slider({
        value: ramCap,
        min: 2e9,
        max: Math.max(machine.ramTotalBytes || 2e9, ramCap),
        step: 1e8,
        ariaLabel: 'RAM cap',
        onInput: (value) => patchDesk({ caps: { ...caps, ramBytes: value } }),
      }),
    ),
    h('div.col', { style: { gap: '10px', borderTop: '1px solid var(--line-soft)', paddingTop: '12px' } },
      metric('Resident blocks', verdict ? `${verdict.residentBlocks} pinned` : fmt.DASH),
      isVideo
        ? metric('Per-step read', verdict ? fmt.gb(verdict.stepReadBytes) : fmt.DASH)
        : metric('Prefetch ring', verdict ? `${verdict.ringDepth} blocks` : fmt.DASH),
    ),
    h('p.t5', { style: { margin: 0, fontSize: '11px', lineHeight: '1.5' } },
      'Lower the caps if another app needs the card. The solver reprints its plan with every run.'),
  );
}

// -- Simple ---------------------------------------------------------------

function simpleDesk() {
  const model = currentModel();
  const desk = store.state.desk;
  const est = estimate(model);
  const shapes = store.state.shapes || {};
  const quality = store.state.quality || { fast: 0.4, balanced: 0.7, best: 1.0 };
  // The stage shows this model's last result, same rule as the Pro desk: a
  // picture from a different model would silently misrepresent the settings
  // shown below it.
  const last = store.state.outputs.find(
    (o) => o.state === 'done' && (!model || o.modelId === model.id),
  );

  return h('div.app',
    topbar(),
    h('div.grow.col',
      h('div.stage', { style: { padding: '36px' } }, frame(last, { width: 470, height: 470 })),
      h('div.dock.simple',
        cpuBanner(store.state.machine),
        promptBox({ klass: 'prompt.simple', min: 62, placeholder: 'Describe the picture you want', value: desk.prompt }),
        h('div.row', { style: { alignItems: 'flex-end', gap: '34px' } },
          h('div.col', { style: { gap: '8px' } },
            h('span.t3', { style: { fontSize: '12px' } }, 'Shape'),
            segmented(
              Object.keys(shapes).map((value) => ({ value, label: fmt.label(value) })),
              desk.shape,
              (value) => patchDesk({ shape: value, ...shapes[value] }),
            ),
          ),
          h('div.col', { style: { gap: '8px' } },
            h('span.t3', { style: { fontSize: '12px' } }, 'Quality'),
            segmented(
              Object.keys(quality).map((value) => ({ value, label: fmt.label(value) })),
              desk.quality,
              (value) => {
                const base = model ? model.shape.steps : 20;
                patchDesk({ quality: value, steps: Math.max(1, Math.round(base * quality[value])) });
              },
            ),
            est.source === 'cpu'
              ? h('span.t4', { style: { fontSize: '12px' } }, 'no time estimate without a graphics card')
              : h('span.t3', { style: { fontSize: '12px' } },
                `${fmt.label(desk.quality)} ≈ `,
                h('span.mono.t2', fmt.duration(est.totalSeconds))),
          ),
          h('div.row.push', { style: { gap: '20px' } },
            h('button.link.link-quiet', {
              type: 'button',
              onClick: () => api.patch({ mode: 'pro' }).catch(fail),
            }, 'Pro controls'),
            h('button.btn-primary.tall', { type: 'button', disabled: !model, onClick: startRun }, 'Generate'),
          ),
        ),
      ),
    ),
  );
}
