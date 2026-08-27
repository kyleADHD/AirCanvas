/* Every number the UI prints goes through here.
 *
 * Two conventions, both from the handoff:
 *
 *   - Sizes are decimal GB, matching the rest of the project (a shard cache is
 *     "10.2 GB" because 10.2e9 bytes is what the manifest says, not 9.5 GiB).
 *   - Durations round to the unit a human would use — seconds under 90, then
 *     minutes, then hours — because "about 7 minutes" is the promise the
 *     screen is making, and "427.4 s" is the measurement it reports afterwards.
 *
 * `null` in, empty-ish out: an absent measurement renders as an em dash rather
 * than a zero, so a value that has not been measured never looks measured.
 */

export const DASH = '—';

export function gb(bytes, digits = 1) {
  if (bytes === null || bytes === undefined) return DASH;
  return `${(Number(bytes) / 1e9).toFixed(digits)} GB`;
}

export function gbNum(bytes, digits = 1) {
  if (bytes === null || bytes === undefined) return DASH;
  return (Number(bytes) / 1e9).toFixed(digits);
}

export function mb(bytes, digits = 0) {
  if (bytes === null || bytes === undefined) return DASH;
  return `${(Number(bytes) / 1e6).toFixed(digits)} MB`;
}

export function gbps(bytesPerSecond, digits = 2) {
  if (!bytesPerSecond) return DASH;
  return `${(Number(bytesPerSecond) / 1e9).toFixed(digits)} GB/s`;
}

/** "2.4 s" / "7 min" / "10.6 h" — the unit a person would say out loud. */
export function duration(seconds, { precise = false } = {}) {
  if (seconds === null || seconds === undefined) return DASH;
  const s = Number(seconds);
  if (!isFinite(s)) return DASH;
  if (s >= 3600) return `${(s / 3600).toFixed(1)} h`;
  if (s >= 90) return `${Math.round(s / 60)} min`;
  if (precise || s < 10) return `${s.toFixed(1)} s`;
  return `${Math.round(s)} s`;
}

/**
 * "about 3 min left" — the whole phrase, because the qualifier depends on the
 * magnitude: "about less than a minute left" is not a sentence. Coarse on
 * purpose so it does not jitter on every tick.
 */
export function remaining(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return DASH;
  if (seconds < 60) return 'less than a minute left';
  if (seconds < 5400) return `about ${Math.max(1, Math.round(seconds / 60))} min left`;
  return `about ${(seconds / 3600).toFixed(1)} h left`;
}

export function seconds(value, digits = 1) {
  if (value === null || value === undefined) return DASH;
  return `${Number(value).toFixed(digits)} s`;
}

export function pct(value, digits = 1) {
  if (value === null || value === undefined) return DASH;
  return `${Number(value).toFixed(digits)}%`;
}

export function count(value) {
  if (value === null || value === undefined) return DASH;
  return new Intl.NumberFormat().format(Math.round(Number(value)));
}

/** Clip length from a frame count: "0:05 · 16 fps". */
export function clip(frames, fps = 16) {
  const total = Math.max(0, Number(frames) || 0) / (fps || 16);
  const mins = Math.floor(total / 60);
  const secs = Math.round(total % 60);
  return `${mins}:${String(secs).padStart(2, '0')}`;
}

/** "2 h ago" for a probe timestamp; the screen must not imply it is live. */
export function ago(epochSeconds) {
  if (!epochSeconds) return 'never';
  const delta = Date.now() / 1000 - Number(epochSeconds);
  if (delta < 90) return 'just now';
  if (delta < 5400) return `${Math.round(delta / 60)} min ago`;
  if (delta < 172800) return `${Math.round(delta / 3600)} h ago`;
  return `${Math.round(delta / 86400)} days ago`;
}

/** Sentence-case a machine word without touching an acronym or a format name. */
export function label(value) {
  const text = String(value || '');
  return text.charAt(0).toUpperCase() + text.slice(1);
}
