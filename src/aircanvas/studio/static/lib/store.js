/* Client state: what the server said, plus what this window is looking at.
 *
 * Two halves that are deliberately not mixed. `state` is the server's — it
 * arrives whole on connect and in deltas after that, and the client never
 * edits it except by asking the server to. `ui` is this window's: which screen
 * is open, which gallery item is selected, whether the Simple disclosure is
 * expanded. Only `ui` changes on a click without a round trip, which is why a
 * dropped connection degrades to a stale-but-coherent screen instead of a
 * screen that disagrees with the machine.
 *
 * Mode and theme live in `state` (server-persisted) because the handoff calls
 * them global and persistent — switching modes must survive a reload, and a
 * second window must agree.
 */

const listeners = new Set();

export const store = {
  connected: false,
  state: {
    mode: 'simple',
    theme: 'dark',
    seenWelcome: false,
    desk: {},
    settings: {},
    setup: { selection: {} },
    outputs: [],
    machine: {},
    models: [],
    installed: [],
    cache: { models: [] },
    huggingFace: {},
    splits: [],
    run: null,
    shapes: {},
    demo: false,
    versions: {},
  },
  ui: {
    route: 'desk',
    activeRoute: 'desk',
    settingsSection: 'cache',
    selectedOutputId: null,
    reportOutputId: null,
    showSimpleTelemetry: false,
    galleryFilter: 'all',
    gallerySort: 'newest',
    videoFrame: 0,
    estimate: null,
    toasts: [],
  },
};

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

let frame = null;
function emit() {
  // Coalesce: telemetry can land several times between paints.
  if (frame !== null) return;
  frame = requestAnimationFrame(() => {
    frame = null;
    for (const fn of listeners) fn(store);
  });
}

/** Merge a server payload into `state`. */
export function mergeState(patch) {
  Object.assign(store.state, patch);
  applyTheme();
  emit();
}

/** Update this window's own view state. */
export function setUi(patch) {
  Object.assign(store.ui, patch);
  emit();
}

export function navigate(route) {
  if (store.ui.route === route) return;
  setUi({ route });
}

export function setConnected(connected) {
  if (store.connected === connected) return;
  store.connected = connected;
  emit();
}

let toastSeq = 0;
export function toast(message, kind = 'info') {
  const entry = { id: ++toastSeq, message, kind };
  setUi({ toasts: [...store.ui.toasts, entry] });
  setTimeout(() => {
    setUi({ toasts: store.ui.toasts.filter((t) => t.id !== entry.id) });
  }, kind === 'fail' ? 9000 : 4500);
}

/** Report a failed action without swallowing it into the console. */
export function fail(error) {
  console.error(error);
  toast(String(error && error.message ? error.message : error), 'fail');
}

function applyTheme() {
  document.documentElement.dataset.theme = store.state.theme || 'dark';
}

// -- derived --------------------------------------------------------------

export const isPro = () => store.state.mode === 'pro';

export function modelById(id) {
  return store.state.models.find((m) => m.id === id) || null;
}

/** The Desk's current model, falling back to the first installed one. */
export function currentModel() {
  const { desk, models } = store.state;
  return modelById(desk.modelId) || models.find((m) => m.installed) || null;
}

export function installedModels() {
  return store.state.models.filter((m) => m.installed && m.installed.complete);
}

export function activeSplits() {
  return store.state.splits.filter((job) => !['done'].includes(job.state));
}

export function runIsLive(run = store.state.run) {
  return Boolean(run) && ['starting', 'encode', 'denoise', 'decode'].includes(run.status);
}
