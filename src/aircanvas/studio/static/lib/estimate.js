/* How long a generation will take, and how confident we are about it.
 *
 * Mirrors `studio/catalog.estimate_seconds` so a card can show a time without
 * a round trip per card. Both ends agree on the rule: a measurement of THIS
 * machine at this exact shape wins; otherwise the solver's disk time per step
 * is the floor and a published measurement raises it to include compute.
 *
 * The `source` is carried through to the screens because the copy differs —
 * "7 min" for something we measured here, "~7 min" for a bound.
 */

import { store } from './store.js';

export function estimateSeconds(model, fmt, { steps } = {}) {
  if (!model) return { totalSeconds: 0, secondsPerStep: 0, source: 'unknown' };
  // Without CUDA there is no honest number to show: every per-step figure the
  // catalog carries was measured on a GPU. The screens say so instead.
  if ((store.state.machine || {}).device !== 'cuda') {
    return { totalSeconds: 0, secondsPerStep: 0, source: 'cpu' };
  }
  const useSteps = steps || model.shape.steps;
  const verdict = (model.verdicts || {})[fmt] || {};
  let perStep = Number(verdict.secondsPerStep) || 0;
  let source = 'solver';
  if (model.measured && model.measured.secondsPerStep > perStep) {
    perStep = model.measured.secondsPerStep;
    source = 'published';
  }
  return { totalSeconds: perStep * useSteps, secondsPerStep: perStep, source };
}

/** "About 7 minutes" vs "7 minutes": the tilde is the honesty marker. */
export function approx(source) {
  return source === 'measured' ? '' : '~';
}

/** The best format to offer for a model: what it is installed as, else its default. */
export function preferredFormat(model) {
  if (!model) return 'fp8';
  if (model.installed && model.installed.format) return model.installed.format;
  return model.defaultFormat;
}
