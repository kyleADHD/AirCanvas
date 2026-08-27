/* Bootstrap: connect, route, render.
 *
 * Rendering is deliberately whole-screen. Telemetry arrives at 5 Hz, the DOM
 * for one screen is a few hundred nodes, and rebuilding it is both faster to
 * reason about and cheaper than it sounds — the alternative (fine-grained
 * bindings) is a framework, and a framework is a build step, and a build step
 * is a Node toolchain on a machine that is supposed to be running a diffusion
 * model. Renders coalesce into one animation frame (store.js).
 *
 * The one thing a full rebuild breaks is the caret, so focus and selection are
 * captured and restored around it, keyed on `data-focus-key`.
 */

import { h, fill } from './lib/dom.js';
import { api, connect } from './lib/api.js';
import { mergeState, setConnected, setUi, store, subscribe, fail, runIsLive } from './lib/store.js';
import { toasts } from './components/chrome.js';
import { setupScreen } from './screens/setup.js';
import { splitScreen } from './screens/split.js';
import { deskScreen } from './screens/desk.js';
import { runScreen } from './screens/run.js';
import { reportScreen } from './screens/report.js';
import { galleryScreen } from './screens/gallery.js';
import { settingsScreen } from './screens/settings.js';

const SCREENS = {
  setup: setupScreen,
  split: splitScreen,
  desk: deskScreen,
  run: runScreen,
  report: reportScreen,
  gallery: galleryScreen,
  settings: settingsScreen,
};

const root = document.getElementById('app');

/**
 * Where the app should be, given what the machine is doing.
 *
 * A run or a split takes over the window because it IS the task; everything
 * else follows the user's last navigation. First launch with nothing on disk
 * goes to setup, because there is nothing else to do there.
 */
function route() {
  const { state, ui } = store;
  if (runIsLive(state.run)) return 'run';
  if (state.splits.some((job) => ['downloading', 'splitting'].includes(job.state))) return 'split';
  if (ui.route === 'split' && state.splits.some((job) => job.state === 'paused')) return 'split';
  if (ui.route === 'setup' || ui.route === 'report') return ui.route;
  // First launch with nothing on disk lands on setup, because there is nothing
  // else to do — but only from the default route. An explicit trip to Settings
  // (to paste a token before downloading a gated model, say) has to survive.
  const firstRun = !state.models.some((model) => model.installed) && !state.seenWelcome;
  if (firstRun && !state.demo && ui.route === 'desk') return 'setup';
  return ui.route === 'run' || ui.route === 'split' ? 'desk' : ui.route;
}

let lastRoute = null;

function render() {
  const name = route();
  // The nav highlights where the app IS, not where the user last clicked: a
  // run or a split takes the window over, and the bar has to agree.
  store.ui.activeRoute = name;
  if (name !== lastRoute) {
    lastRoute = name;
    // Leaving a finished run should not strand the Simple disclosure open.
    if (name !== 'run') store.ui.showSimpleTelemetry = false;
  }
  const screen = SCREENS[name] || deskScreen;

  const focus = captureFocus();
  try {
    fill(root, screen(), toasts());
  } catch (error) {
    fill(root, errorScreen(error), toasts());
    console.error(error);
  }
  restoreFocus(focus);
}

function captureFocus() {
  const el = document.activeElement;
  const key = el && el.dataset ? el.dataset.focusKey : null;
  if (!key) return null;
  return { key, start: el.selectionStart, end: el.selectionEnd };
}

function restoreFocus(focus) {
  if (!focus) return;
  const el = root.querySelector(`[data-focus-key="${focus.key}"]`);
  if (!el) return;
  el.focus();
  if (focus.start !== null && focus.start !== undefined && el.setSelectionRange) {
    try {
      el.setSelectionRange(focus.start, focus.end);
    } catch { /* number inputs refuse selection ranges in some browsers */ }
  }
}

function errorScreen(error) {
  return h('div.app',
    h('div.stage',
      h('div.col', { style: { gap: '10px', maxWidth: '560px' } },
        h('h1.title', 'This screen failed to draw'),
        h('p.t3', { style: { margin: 0 } },
          'The rest of the app is still running, and nothing on disk was touched.'),
        h('pre.mono.t5', { style: { fontSize: '11px', whiteSpace: 'pre-wrap', margin: 0 } }, String(error && error.stack || error)),
        h('button.btn.btn-accent', { type: 'button', style: { alignSelf: 'flex-start' }, onClick: () => location.reload() }, 'Reload'),
      ),
    ),
  );
}

/** Apply a server event to the client state. */
function onEvent(type, payload) {
  switch (type) {
    case 'state':
      mergeState(payload.state || payload);
      break;
    case 'machine':
      mergeState({ machine: payload.machine, models: payload.models });
      break;
    case 'models':
      mergeState({ models: payload.models });
      break;
    case 'split':
    case 'splits':
      mergeState({ splits: payload.jobs || store.state.splits });
      break;
    case 'installed':
      mergeState({ installed: payload.models });
      api.state().then(mergeState).catch(() => { /* the stream will catch up */ });
      break;
    case 'run':
      mergeState({ run: payload.run });
      break;
    case 'outputs':
      mergeState({ outputs: payload.outputs });
      break;
    case 'cache':
      mergeState({ cache: payload.cache, models: payload.models || store.state.models });
      break;
    default:
      break;
  }
}

function keyboard(event) {
  if (event.target.matches('input, textarea')) return;
  const map = { d: 'desk', g: 'gallery', s: 'settings' };
  if (map[event.key] && !event.metaKey && !event.ctrlKey) setUi({ route: map[event.key] });
}

subscribe(render);
connect(onEvent, (status) => setConnected(status === 'connected'));
document.addEventListener('keydown', keyboard);
api.state().then(mergeState).catch(fail);
render();

// A debug handle, deliberately: this is a local desktop app, and being able to
// jump to a screen or inspect the last run's payload from the console is worth
// more here than the tidiness of an empty global.
window.aircanvas = { store, setUi, api, render };
