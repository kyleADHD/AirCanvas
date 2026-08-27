/* The whole icon set: one logo and five glyphs.
 *
 * 16px box, 1.5px stroke, `fill:none`, `stroke:currentColor` — one visual
 * weight, so nothing in the chrome shouts. Icons appear only where a word
 * would be worse: the left rail's three destinations, the two disclosure
 * chevrons, and the check inside a selected model checkbox. No icon decorates
 * a label that already reads clearly.
 *
 * The logo is the repo's own mark (assets/aircanvas_logo_dark.svg): a
 * billowing canvas caught in the wind, four filled paths on the violet
 * gradient. The gradient is defined once per instance so a detached node
 * still paints correctly.
 */

import { svg } from './dom.js';

const LOGO_PATHS = [
  'M114,192 C152,156 198,148 240,150 C270,152 294,140 314,126 C290,150 260,164 232,168 C194,174 152,180 114,192 Z',
  'M318,138 C342,204 332,290 292,348 C306,290 312,206 318,138 Z',
  'M286,352 C232,392 152,392 70,378 C150,374 228,370 286,352 Z',
  'M106,200 C88,248 84,298 94,344 C97,298 101,248 106,200 Z',
];

let gradientSeq = 0;

export function logo(size = 18) {
  const id = `ac-grad-${++gradientSeq}`;
  return svg('svg', { width: size, height: size, viewBox: '60 110 300 260', 'aria-label': 'AirCanvas' },
    svg('defs', {},
      svg('linearGradient', { id, x1: '0', y1: '1', x2: '1', y2: '0' },
        svg('stop', { offset: '0', 'stop-color': '#5b21b6' }),
        svg('stop', { offset: '0.55', 'stop-color': '#8b5cf6' }),
        svg('stop', { offset: '1', 'stop-color': '#c4b5fd' }),
      ),
    ),
    svg('g', { fill: `url(#${id})` }, LOGO_PATHS.map((d) => svg('path', { d }))),
  );
}

function glyph(size, ...children) {
  return svg('svg', {
    width: size,
    height: size,
    viewBox: '0 0 16 16',
    fill: 'none',
    stroke: 'currentColor',
    'stroke-width': '1.5',
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
    'aria-hidden': 'true',
  }, children);
}

export const image = (size = 16) => glyph(size,
  svg('rect', { x: '1.5', y: '2.5', width: '13', height: '11', rx: '2' }),
  svg('circle', { cx: '5.4', cy: '6.4', r: '1.2' }),
  svg('path', { d: 'M2 11.5 6 8l3 2.5 2-1.5 3 2.5' }),
);

export const video = (size = 16) => glyph(size,
  svg('rect', { x: '1.5', y: '3', width: '13', height: '10', rx: '1.5' }),
  svg('path', { d: 'M4.6 3v10M11.4 3v10' }),
);

export const grid = (size = 16) => glyph(size,
  svg('rect', { x: '2', y: '2', width: '5', height: '5', rx: '1' }),
  svg('rect', { x: '9', y: '2', width: '5', height: '5', rx: '1' }),
  svg('rect', { x: '2', y: '9', width: '5', height: '5', rx: '1' }),
  svg('rect', { x: '9', y: '9', width: '5', height: '5', rx: '1' }),
);

export const chevronDown = (size = 14) => glyph(size, svg('path', { d: 'M4 6.5 8 10.5 12 6.5' }));
export const chevronRight = (size = 14) => glyph(size, svg('path', { d: 'M6 4 10 8 6 12' }));

export const check = (size = 10) => svg('svg', {
  width: size,
  height: size,
  viewBox: '0 0 10 10',
  fill: 'none',
  stroke: '#0a0612',
  'stroke-width': '1.8',
  'stroke-linecap': 'round',
  'stroke-linejoin': 'round',
  'aria-hidden': 'true',
}, svg('path', { d: 'M1.5 5.2 4 7.5 8.5 2.6' }));
