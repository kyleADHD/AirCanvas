/* The top bar: identity, the mode switch, the destinations, and the machine.
 *
 * The right-hand cluster is the app's pulse and changes with state, per the
 * handoff: green dot + `idle` with VRAM and cache size when nothing is
 * running; the whole cluster in violet with the step and elapsed seconds while
 * generating; the machine summary during setup. It is mono, because it is all
 * numbers.
 */

import { h } from '../lib/dom.js';
import * as fmt from '../lib/format.js';
import { logo } from '../lib/icons.js';
import { api } from '../lib/api.js';
import { fail, navigate, runIsLive, setUi, store } from '../lib/store.js';
import { segmented, separator, statusDot } from './widgets.js';

const DESTINATIONS = [
  { route: 'desk', label: 'Desk', proOnly: true },
  { route: 'gallery', label: 'Gallery' },
  { route: 'settings', label: 'Settings' },
];

export function topbar({ setup = false } = {}) {
  const { state, ui } = store;
  const pro = state.mode === 'pro';

  const modeSwitch = segmented(
    [{ value: 'simple', label: 'Simple' }, { value: 'pro', label: 'Pro' }],
    state.mode,
    (mode) => api.patch({ mode }).catch(fail),
    { compact: true, klass: 'on-rail' },
  );

  return h(`header.topbar${setup ? '.setup' : ''}`,
    logo(setup ? 20 : 18),
    setup ? h('span.brand', 'AirCanvas') : null,
    modeSwitch,
    setup ? null : h('nav',
      DESTINATIONS.filter((d) => pro || !d.proOnly).map((d) => h('button', {
        type: 'button',
        'aria-current': (ui.activeRoute || ui.route) === d.route ? 'page' : undefined,
        onClick: () => navigate(d.route),
      }, d.label)),
    ),
    statusCluster({ setup }),
  );
}

export function statusCluster({ setup = false } = {}) {
  const { state } = store;
  const run = state.run;
  const machine = state.machine || {};

  const cluster = h('div.status');
  if (!store.connected) {
    cluster.append(statusDot('warn'), h('span', 'reconnecting'));
    return cluster;
  }

  // Simple mode has no machine language anywhere, and the cluster is nothing
  // but machine language — VRAM, cache size, step counts. The one thing that
  // stays is the demo marker, which is a claim about honesty, not telemetry.
  if (state.mode !== 'pro' && !setup) {
    if (state.demo) cluster.append(demoMark());
    return cluster;
  }

  if (runIsLive(run)) {
    cluster.classList.add('live');
    cluster.append(
      h('span.row', statusDot('run'), h('span', run.status === 'denoise' ? 'denoising' : run.status)),
      separator(),
      h('span', `step ${run.step}/${run.steps}`),
      separator(),
      h('span', `${run.elapsed.toFixed(1)} s`),
    );
    return cluster;
  }

  const cuda = machine.device === 'cuda';

  if (setup) {
    cluster.append(
      h('span', machine.gpu || 'unknown device'),
      separator(),
      h('span', cuda ? `${fmt.gbNum(machine.vramTotalBytes)} GB VRAM` : 'no CUDA device'),
      separator(),
      h('span', `${fmt.gbNum(machine.ramTotalBytes)} GB RAM`),
      separator(),
      h('span', `${machine.diskProbed ? '' : 'assumed '}${fmt.gbps(machine.diskBytesPerSecond)}`),
    );
    if (state.demo) cluster.append(separator(), demoMark());
    return cluster;
  }

  const used = (machine.vramTotalBytes || 0) - (machine.vramFreeBytes || 0);
  cluster.append(
    h('span.row', statusDot('ok'), h('span', 'idle')),
    separator(),
    h('span', cuda ? `VRAM ${fmt.gbNum(used)}/${fmt.gbNum(machine.vramTotalBytes)} GB` : 'cpu'),
    separator(),
    h('span', `cache ${fmt.gbNum((state.cache || {}).totalBytes)} GB`),
  );
  if (state.demo) cluster.append(separator(), demoMark());
  return cluster;
}

/** Demo mode says so on every screen; a replay must never read as a run. */
function demoMark() {
  return h('span.accent', { title: 'Measured runs replayed from docs/BENCHMARKS.md; nothing is loaded' }, 'demo');
}

/** The three destinations of the Pro left rail. */
export function iconRail(items) {
  return h('aside.iconrail',
    items.map((item) => h('button', {
      type: 'button',
      title: item.title,
      'aria-label': item.title,
      'aria-pressed': item.active,
      onClick: item.onClick,
    }, item.icon())),
  );
}

export function toasts() {
  return h('div.toasts',
    store.ui.toasts.map((t) => h(`div.toast${t.kind === 'fail' ? '.fail' : ''}`, t.message)),
  );
}

export function goto(route, extra = {}) {
  setUi({ route, ...extra });
}
