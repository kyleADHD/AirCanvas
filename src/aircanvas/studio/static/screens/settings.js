/* P8 Settings (Pro) and S4 Settings (Simple).
 *
 * Pro shows the cache by model, the disk probe with its age, the two
 * pipeline toggles and the token. Simple shows three rows — storage, account,
 * appearance — and a door to the Pro panel, which is also the global mode
 * switch: "Advanced" and "Pro controls" are the same lever.
 *
 * Appearance persists all three choices, but only dark is drawn. The handoff
 * is explicit that light mode was never designed and must not be shipped from
 * guesswork, so the control says so rather than rendering an invented palette.
 */

import { h } from '../lib/dom.js';
import * as fmt from '../lib/format.js';
import { api } from '../lib/api.js';
import { fail, setUi, store, toast } from '../lib/store.js';
import { topbar } from '../components/chrome.js';
import { chevronRight } from '../lib/icons.js';
import { eyebrow, metric, segmented, stacked, toggle } from '../components/widgets.js';

const SECTIONS = [
  { id: 'cache', label: 'Shard cache' },
  { id: 'performance', label: 'Performance' },
  { id: 'hf', label: 'Hugging Face' },
  { id: 'appearance', label: 'Appearance' },
  { id: 'about', label: 'About' },
];

const KEY_COLORS = ['var(--violet-900)', 'var(--violet-500)', 'var(--violet-200)', 'var(--track)'];

export function settingsScreen() {
  return store.state.mode === 'pro' ? proSettings() : simpleSettings();
}

function appearanceControl() {
  return h('div.row', { style: { gap: '10px' } },
    segmented(
      [{ value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }, { value: 'system', label: 'System' }],
      store.state.theme,
      (theme) => {
        api.patch({ theme }).catch(fail);
        if (theme !== 'dark') toast('Light mode has not been designed yet — the choice is saved for when it lands');
      },
    ),
    store.state.theme === 'dark'
      ? null
      : h('span.t5', { style: { fontSize: '11px' } }, 'saved; only dark is drawn so far'),
  );
}

// -- Pro ------------------------------------------------------------------

function proSettings() {
  return h('div.app',
    topbar(),
    h('div.body',
      h('nav.settings-nav',
        SECTIONS.map((section) => h('button', {
          type: 'button',
          'aria-current': store.ui.settingsSection === section.id,
          onClick: () => {
            setUi({ settingsSection: section.id });
            document.getElementById(`settings-${section.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
          },
        }, section.label)),
      ),
      h('div.settings-main',
        cacheSection(),
        performanceSection(),
        huggingFaceSection(),
        aboutSection(),
      ),
    ),
  );
}

function cacheSection() {
  const cache = store.state.cache || { models: [] };
  const models = cache.models || [];
  const free = cache.freeBytes || 0;
  const largest = models.reduce((max, m) => Math.max(max, m.diskBytes || 0), 0);
  const tight = free < largest || free < 20e9;

  return h('div.col', { id: 'settings-cache', style: { gap: '13px' } },
    h('h2.title-section', 'Shard cache'),
    tight ? diskWarning(free, models) : null,
    h('div.card.pad', { style: { display: 'flex', flexDirection: 'column', gap: '13px', padding: '16px' } },
      h('div.row', { style: { gap: '12px' } },
        h('div.grow.col', { style: { gap: '4px' } },
          h('span.t3', { style: { fontSize: '12px' } }, 'Location'),
          h('span.mono.t1', { style: { fontSize: '12px', wordBreak: 'break-all' } }, cache.path),
        ),
        h('button.btn.btn-accent', {
          type: 'button',
          onClick: () => toast('Set HF_HOME before starting the Studio to move the cache'),
        }, 'Change'),
      ),
      h('div.row', {
        style: { alignItems: 'baseline', gap: '10px', borderTop: '1px solid var(--line)', paddingTop: '13px' },
      },
        h('span.mono', { style: { fontSize: '22px' } }, fmt.gb(cache.totalBytes)),
        h('span.t3', { style: { fontSize: '12px' } },
          `across ${models.length} model${models.length === 1 ? '' : 's'} · ${fmt.gb(free)} free on this volume`),
      ),
      models.length
        ? stacked(models.map((model, i) => ({
          label: shortName(model.source),
          bytes: model.diskBytes,
          color: KEY_COLORS[i % KEY_COLORS.length],
        })), { mini: true })
        : null,
      models.length
        ? h('div.col', { style: { gap: '8px' } }, models.map((model, i) => cacheRow(model, i)))
        : h('span.t4', { style: { fontSize: '12px' } },
          'Nothing split yet — models land here the first time you run one.'),
    ),
  );
}

function cacheRow(model, index) {
  return h('div.row', { style: { gap: '10px', fontSize: '12px' } },
    h('span.key', { style: { background: KEY_COLORS[index % KEY_COLORS.length] } }),
    h('span.grow.truncate', { title: model.source }, shortName(model.source)),
    h('span.mono.t3', `${model.format} · ${model.blocks} shards${model.complete ? '' : ' · incomplete'}`),
    h('span.mono', { style: { width: '70px', textAlign: 'right' } }, fmt.gb(model.diskBytes)),
    h('button.link.link-quiet', {
      type: 'button',
      onClick: () => {
        if (!window.confirm(`Delete the ${shortName(model.source)} shard cache? It can be re-split from the source.`)) return;
        api.removeCache(model.cacheDir).then(() => toast('Shard cache removed')).catch(fail);
      },
    }, 'Remove'),
  );
}

function shortName(source) {
  return String(source || '').split('/').pop();
}

function diskWarning(free, models) {
  const biggest = [...models].sort((a, b) => (b.diskBytes || 0) - (a.diskBytes || 0))[0];
  return h('div.banner',
    h('span.dot.dot-warn'),
    h('span',
      'This volume is nearly full — ',
      h('span.mono', { style: { color: '#f0e2c4' } }, fmt.gb(free)),
      biggest ? ` left. ${shortName(biggest.source)} alone needs ${fmt.gb(biggest.diskBytes)} to re-split.` : ' left.'),
    h('button.btn.btn-warn.push', {
      type: 'button',
      onClick: () => toast('Set HF_HOME to a larger volume and restart the Studio'),
    }, 'Move cache'),
  );
}

function performanceSection() {
  const machine = store.state.machine;
  const settings = store.state.settings;

  return h('div.col', { id: 'settings-performance', style: { gap: '13px' } },
    h('h2.title-section', 'Performance'),
    h('div.grid-2',
      h('div.card.pad', { style: { display: 'flex', flexDirection: 'column', gap: '10px', padding: '16px' } },
        h('div.eyebrow', 'Disk bandwidth probe'),
        h('div.hero-metric',
          h('b', fmt.gbNum(machine.diskBytesPerSecond, 2)),
          h('span', machine.diskProbed ? 'GB/s · measured' : 'GB/s · assumed'),
        ),
        h('p.t3', { style: { margin: 0, fontSize: '12px', lineHeight: '1.5' } },
          machine.diskProbed
            ? `Measured ${fmt.ago(machine.diskProbedAt)}. Below 1 GB/s, streaming becomes the bottleneck for image models.`
            : 'No shard cache to read yet, so this is the documented default. Split a model, then probe for a real number.'),
        h('button.btn.btn-accent', {
          type: 'button',
          style: { alignSelf: 'flex-start' },
          onClick: () => api.reprobe(true).then(() => toast('Disk re-probed')).catch(fail),
        }, 'Re-run probe'),
      ),
      h('div.card.pad', { style: { display: 'flex', flexDirection: 'column', gap: '13px', padding: '16px' } },
        settingToggle(
          'Embedding cache',
          'Repeat prompts skip the text encoders entirely.',
          settings.embeddingCache,
          (on) => api.patch({ settings: { embeddingCache: on } }).catch(fail),
        ),
        h('div.hairline'),
        settingToggle(
          'Background prefetch',
          'Pipelined disk to pinned ring to GPU. Off means synchronous loads.',
          settings.backgroundPrefetch,
          (on) => api.patch({ settings: { backgroundPrefetch: on } }).catch(fail),
        ),
      ),
    ),
  );
}

function settingToggle(title, description, value, onChange) {
  return h('div.row', { style: { alignItems: 'flex-start', gap: '12px' } },
    h('div.grow.col', { style: { gap: '3px' } },
      h('span', { style: { fontSize: '13px' } }, title),
      h('span.t3', { style: { fontSize: '12px', lineHeight: '1.5' } }, description),
    ),
    toggle(value, onChange, { large: true, label: title }),
  );
}

function huggingFaceSection() {
  const hf = store.state.huggingFace || {};
  return h('div.col', { id: 'settings-hf', style: { gap: '13px' } },
    h('h2.title-section', 'Hugging Face'),
    h('div.card.pad', { style: { display: 'flex', alignItems: 'center', gap: '14px', padding: '16px' } },
      h('div.grow.col', { style: { gap: '5px' } },
        h('span.t3', { style: { fontSize: '12px' } }, 'Access token'),
        h('div.row', { style: { gap: '10px' } },
          h('span.mono.t1', { style: { fontSize: '12px' } }, hf.masked || 'not set'),
          hf.authenticated
            ? h('span.chip.good', { style: { borderRadius: '4px', padding: '2px 7px' } },
              hf.user ? `Authenticated · ${hf.user}` : 'Authenticated')
            : null,
        ),
        h('span.t5', { style: { fontSize: '11px' } },
          'Only needed for the first download of a gated model. After the split, generation runs offline.'),
      ),
      h('button.btn', {
        type: 'button',
        onClick: () => toast('Run `hf auth login`, or set HF_TOKEN, then re-open the Studio'),
      }, hf.authenticated ? 'Replace' : 'Sign in'),
    ),
  );
}

function aboutSection() {
  const versions = store.state.versions || {};
  return h('div.row', {
    id: 'settings-about',
    style: { gap: '14px', borderTop: '1px solid var(--line-soft)', paddingTop: '16px', marginTop: 'auto' },
  },
    h('span.t3', { style: { fontSize: '13px' } }, 'Appearance'),
    appearanceControl(),
    h('span.push.mono.t5', { style: { fontSize: '11px' } },
      Object.entries(versions).map(([name, value]) => `${name} ${value}`).join(' · ')),
  );
}

// -- Simple ---------------------------------------------------------------

function simpleSettings() {
  const cache = store.state.cache || {};
  const hf = store.state.huggingFace || {};

  return h('div.app',
    topbar(),
    h('div.settings-simple',
      h('h1.title', 'Settings'),
      h('div.grouped',
        h('div.g-row',
          h('div.grow.col', { style: { gap: '4px' } },
            h('span', { style: { fontSize: '15px' } }, 'Storage'),
            h('span.t3', { style: { fontSize: '13px' } },
              h('span.mono.t2', fmt.gb(cache.totalBytes)), ' used by models · ',
              h('span.mono.t2', fmt.gb(cache.freeBytes)), ' free'),
          ),
          h('button.btn.btn-accent.btn-lg', {
            type: 'button',
            onClick: () => { api.patch({ mode: 'pro' }).catch(fail); setUi({ route: 'settings', settingsSection: 'cache' }); },
          }, 'Free up space'),
        ),
        h('div.hairline'),
        h('div.g-row',
          h('div.grow.col', { style: { gap: '4px' } },
            h('span', { style: { fontSize: '15px' } }, 'Hugging Face account'),
            h('span.t3', { style: { fontSize: '13px' } },
              hf.authenticated
                ? `Signed in${hf.user ? ` as ${hf.user}` : ''} — needed once, for models that ask you to accept a licence.`
                : 'Not signed in — only needed for models that ask you to accept a licence.'),
          ),
          h('button.btn.btn-lg', {
            type: 'button',
            onClick: () => toast(hf.authenticated ? 'Run `hf auth logout` to sign out' : 'Run `hf auth login` to sign in'),
          }, hf.authenticated ? 'Sign out' : 'Sign in'),
        ),
        h('div.hairline'),
        h('div.g-row',
          h('div.grow.col', { style: { gap: '4px' } },
            h('span', { style: { fontSize: '15px' } }, 'Appearance'),
            h('span.t3', { style: { fontSize: '13px' } }, 'Follows your system by default.'),
          ),
          appearanceControl(),
        ),
      ),
      h('button.card.lg.quiet', {
        type: 'button',
        style: { display: 'flex', alignItems: 'center', gap: '12px', padding: '18px 22px', width: '100%' },
        onClick: () => { api.patch({ mode: 'pro' }).catch(fail); setUi({ route: 'settings' }); },
      },
        chevronRight(),
        h('div.grow.col', { style: { gap: '4px' } },
          h('span', { style: { fontSize: '15px' } }, 'Advanced'),
          h('span.t4', { style: { fontSize: '13px' } }, 'Shard cache, disk probe, prefetch, budget caps.'),
        ),
      ),
    ),
  );
}
