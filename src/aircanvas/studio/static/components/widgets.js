/* The parts more than one screen is built from. */

import { h } from '../lib/dom.js';
import * as fmt from '../lib/format.js';
import { check } from '../lib/icons.js';

/** Segmented control. `options` is [{value, label, mono}]. */
export function segmented(options, value, onSelect, { compact = false, klass = '' } = {}) {
  return h(`div.seg${compact ? '.compact' : ''}${klass ? `.${klass}` : ''}`, { role: 'group' },
    options.map((option) => h('button', {
      type: 'button',
      'aria-pressed': option.value === value,
      onClick: () => option.value !== value && onSelect(option.value),
    }, option.label)),
  );
}

export function toggle(on, onChange, { large = false, label = '' } = {}) {
  return h(`button.toggle${large ? '.lg' : ''}`, {
    type: 'button',
    role: 'switch',
    'aria-pressed': Boolean(on),
    'aria-label': label || undefined,
    onClick: () => onChange(!on),
  }, h('span'));
}

/**
 * Labelled slider. The visible track is ours (the design's 4px rail with a
 * 12px thumb); a transparent range input sits over it so keyboard and pointer
 * behaviour stay the platform's.
 */
export function slider({ value, min, max, step = 0.01, onInput, ariaLabel }) {
  const span = max - min || 1;
  const pct = Math.max(0, Math.min(100, ((value - min) / span) * 100));
  return h('div.slider',
    h('i', { style: { width: `${pct}%` } }),
    h('b', { style: { left: `${pct}%` } }),
    h('input', {
      type: 'range',
      min, max, step,
      value,
      'aria-label': ariaLabel,
      onInput: (event) => onInput(Number(event.target.value)),
    }),
  );
}

/** A label/value row, value in mono. */
export function metric(label, value, { tone = '', mono = true } = {}) {
  return h('div.metric-row',
    h('span', label),
    h(`span${mono ? '.mono' : ''}${tone ? `.${tone}` : ''}`, value),
  );
}

/** A verdict chip: the call, and the reason for it. */
export function verdictChip(verdict) {
  if (!verdict) return null;
  return h(`span.chip.${verdict.tone}`, { title: verdictTitle(verdict) }, verdict.label);
}

function verdictTitle(verdict) {
  const parts = [];
  if (verdict.residentBlocks) parts.push(`${verdict.residentBlocks} blocks resident`);
  if (verdict.streamedBlocks) parts.push(`${verdict.streamedBlocks} streamed`);
  if (verdict.secondsPerStep) parts.push(`${fmt.duration(verdict.secondsPerStep)} of disk per step`);
  if (verdict.headroomBytes) parts.push(`${fmt.gb(verdict.headroomBytes)} spare`);
  return parts.length ? `Solved for this machine: ${parts.join(' · ')}` : '';
}

export function checkbox(checked) {
  return h('span.check', { role: 'checkbox', 'aria-checked': Boolean(checked) }, checked ? check() : null);
}

export function radio(checked) {
  return h('span.radio', { role: 'radio', 'aria-checked': Boolean(checked) });
}

/**
 * The one banner every screen that promises a time has to show when there is
 * no CUDA device: `aircanvas doctor` says the same thing, in the same words.
 */
export function cpuBanner(machine) {
  if (!machine || machine.device === 'cuda') return null;
  return h('div.banner',
    h('span.dot.dot-warn'),
    h('span', 'No CUDA device found — AirCanvas will run on this machine, correctly, but not fast. '
      + 'Every timing in this app was measured on a GPU, so none is shown here.'),
  );
}

export function eyebrow(text, trailing) {
  return h('div.row', h('span.eyebrow', text), trailing ? h('span.push.eyebrow', trailing) : null);
}

export function divider(text) {
  return h('div.divider', h('span.eyebrow', text), h('div.rule'));
}

/**
 * An image slot.
 *
 * There is no placeholder artwork anywhere in this app. A slot with no output
 * shows the recessed frame and says what is missing — the design's sample
 * imagery was explicitly a stand-in that must not ship, and inventing a
 * picture is the one lie a "measured, not promised" UI cannot tell.
 */
export function frame(output, { width, height, accent = false, scrim = null, badge = null } = {}) {
  const style = {};
  if (width) style.width = typeof width === 'number' ? `${width}px` : width;
  if (height) style.height = typeof height === 'number' ? `${height}px` : height;

  let content;
  if (output && output.file && output.kind === 'video') {
    content = h('video', { src: `/api/outputs/${output.id}/file`, muted: true, loop: true, autoplay: true, playsinline: true });
  } else if (output && output.file) {
    content = h('img', { src: `/api/outputs/${output.id}/file`, alt: output.prompt || 'output' });
  } else {
    content = h('div.empty',
      h('span', output ? emptyReason(output) : 'No output yet'),
      h('span.mono', output && output.state === 'queued' ? 'starts when the current run finishes' : 'generate to fill this frame'),
    );
  }

  return h(`div.frame${accent ? '.accent' : ''}`, { style },
    content,
    scrim ? h('div.scrim', scrim) : null,
    badge ? h('div.badge', { style: { right: '10px', top: '10px' } }, badge) : null,
  );
}

function emptyReason(output) {
  if (output.state === 'queued') return 'Queued';
  if (output.state === 'failed') return 'This run failed';
  return 'No image on disk';
}

/**
 * Stacked memory bar, as the Report's residency plan draws it.
 *
 * A segment narrower than `LABEL_FLOOR` of the bar gets no inline label: two
 * clipped words overlapping ("activati…" over "reserve 0.3") read as a
 * rendering bug and cost more than the label was worth. The number is still on
 * the segment's tooltip, and the footnotes under the bar carry the detail.
 */
const LABEL_FLOOR = 13;

export function stacked(segments, { mini = false } = {}) {
  const total = segments.reduce((sum, s) => sum + Math.max(0, s.bytes || 0), 0) || 1;
  return h(`div.stacked${mini ? '.mini' : ''}`,
    segments.map((segment) => {
      const share = (Math.max(0, segment.bytes || 0) / total) * 100;
      if (share <= 0) return null;
      const style = { width: `${share}%` };
      if (segment.color) style.background = segment.color;
      const label = `${segment.label} ${fmt.gbNum(segment.bytes, 2)}`;
      return h(`div${segment.klass ? `.${segment.klass}` : ''}`, { style, title: `${segment.label} ${fmt.gb(segment.bytes, 2)}` },
        mini || share < LABEL_FLOOR
          ? null
          : h(`span.${segment.light ? 'on-light' : 'on-dark'}`, label),
      );
    }),
  );
}

/**
 * A flat progress bar. Never gradient — that belongs to the primary CTA.
 * `fill` when it sits in a row and should take the leftover width.
 */
export function bar(pct, { tone = '', height = 6, fill = false } = {}) {
  return h(`div.track.h${height}${fill ? '.fill' : ''}`,
    h(`i${tone ? `.${tone}` : ''}`, { style: { width: `${Math.max(0, Math.min(100, pct || 0))}%` } }));
}

export function statusDot(kind) {
  return h(`span.dot.dot-${kind}`);
}

export function separator() {
  return h('span.sep', '·');
}
