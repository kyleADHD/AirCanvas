/* The client half of the API, plus the event stream.
 *
 * Screens never poll. `connect()` opens one EventSource and pushes every
 * server-side change into the store; the fetch helpers below are only for the
 * actions a person takes. The stream replays a short backlog on connect, so a
 * browser reloaded mid-generation redraws the live screen immediately.
 */

const JSON_HEADERS = { 'content-type': 'application/json' };

async function request(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? undefined : JSON_HEADERS,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload && payload.detail) detail = payload.detail;
    } catch { /* a non-JSON error body is not worth a second failure */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

export const api = {
  state: () => request('GET', '/api/state'),
  patch: (payload) => request('POST', '/api/state', payload),
  models: () => request('GET', '/api/models'),
  reprobe: (probeDisk = false) => request('GET', `/api/machine?probe_disk=${probeDisk ? 'true' : 'false'}`),
  estimate: (payload) => request('POST', '/api/estimate', payload),

  startSplit: (payload) => request('POST', '/api/splits', payload),
  splitAction: (id, action) => request('POST', `/api/splits/${id}/${action}`),
  pauseAllSplits: () => request('POST', '/api/splits/pause-all'),

  startRun: (payload) => request('POST', '/api/runs', payload),
  cancelRun: () => request('POST', '/api/runs/cancel'),

  outputs: () => request('GET', '/api/outputs'),
  deleteOutput: (id) => request('DELETE', `/api/outputs/${id}`),
  markReproduced: (id) => request('POST', `/api/outputs/${id}/reproduced`),
  outputFile: (id) => `/api/outputs/${id}/file`,
  outputReport: (id) => `/api/outputs/${id}/report`,

  removeCache: (cacheDir) => request('POST', '/api/cache/remove', { cacheDir }),
};

/**
 * Subscribe to server events. Returns a close function.
 * `onEvent(type, payload)` fires for every event including the initial state.
 */
export function connect(onEvent, onStatus) {
  let source = null;
  let closed = false;
  let retry = 500;

  const open = () => {
    if (closed) return;
    source = new EventSource('/api/events');
    for (const type of ['state', 'machine', 'models', 'split', 'splits', 'run', 'outputs', 'cache', 'installed']) {
      source.addEventListener(type, (event) => {
        retry = 500;
        onStatus?.('connected');
        try {
          onEvent(type, JSON.parse(event.data));
        } catch (error) {
          console.error('Malformed event', type, error);
        }
      });
    }
    source.onopen = () => { retry = 500; onStatus?.('connected'); };
    source.onerror = () => {
      // EventSource reconnects itself, but only on a clean drop. A server
      // restart closes hard, so back off and reopen rather than sit dead.
      onStatus?.('reconnecting');
      source.close();
      if (!closed) setTimeout(open, (retry = Math.min(retry * 2, 8000)));
    };
  };

  open();
  return () => { closed = true; source?.close(); };
}
