/* A 60-line element builder, in place of a framework.
 *
 * The Studio is one window with twelve screens and a 5 Hz telemetry feed. A
 * build step would buy diffing we do not need — screens rebuild on structural
 * change and patch text nodes in place while a run is live (see app.js) — and
 * would cost the thing that matters most here: the app must work offline, from
 * a wheel, with no Node toolchain on the machine.
 *
 *   h('div.card.pad', { onClick }, h('span.mono', '5.44 GB'))
 *
 * Tag syntax is `tag.class.class`. Props are attributes, except: `class` adds
 * to the shorthand classes, `style` takes an object, `on*` takes a listener,
 * and a boolean-valued attribute is omitted entirely when false — which is how
 * `aria-pressed` stays absent instead of reading "false".
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

function build(ns, tag, props, children) {
  const [name, ...classes] = String(tag).split('.');
  const el = ns ? document.createElementNS(ns, name || 'div') : document.createElement(name || 'div');
  if (classes.length) el.setAttribute('class', classes.join(' '));

  if (props && (typeof props !== 'object' || props.nodeType || Array.isArray(props))) {
    children = [props, ...children];
    props = null;
  }

  for (const [key, value] of Object.entries(props || {})) {
    // ARIA states are tri-state strings, not HTML boolean attributes: an
    // element that is not pressed must say aria-pressed="false", and the
    // stylesheet selects on ='true'. Everything else drops on false.
    if (key.startsWith('aria-') && typeof value === 'boolean') {
      el.setAttribute(key, value ? 'true' : 'false');
      continue;
    }
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') {
      el.setAttribute('class', [el.getAttribute('class'), value].filter(Boolean).join(' '));
    } else if (key === 'style' && typeof value === 'object') {
      Object.assign(el.style, value);
    } else if (key === 'text') {
      el.textContent = String(value);
    } else if (key === 'html') {
      el.innerHTML = value;
    } else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === 'value' && 'value' in el) {
      el.value = value;
    } else if (value === true) {
      el.setAttribute(key, '');
    } else {
      el.setAttribute(key, String(value));
    }
  }

  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children.flat(4)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
}

export const h = (tag, props, ...children) => build(null, tag, props, children);
export const svg = (tag, props, ...children) => build(SVG_NS, tag, props, children);

/** Replace an element's children in one shot. */
export function fill(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

/** Set textContent only when it changed — keeps the 5 Hz feed from thrashing. */
export function setText(el, value) {
  const next = value === null || value === undefined ? '' : String(value);
  if (el && el.textContent !== next) el.textContent = next;
  return el;
}

/** Toggle a class without reading the DOM twice. */
export function setClass(el, name, on) {
  if (el) el.classList.toggle(name, Boolean(on));
  return el;
}

/**
 * Grow a textarea to its content.
 *
 * The dock's prompt box is one line tall until the prompt needs two, which a
 * fixed `rows` cannot do. Called after mount and on every input; the reset to
 * `auto` first is what lets it shrink again on delete.
 */
export function autosize(el, min) {
  if (!el) return el;
  el.style.height = 'auto';
  el.style.height = `${Math.max(min, el.scrollHeight)}px`;
  return el;
}

/** A percentage width, clamped, for a progress fill. */
export function setWidth(el, pct) {
  const clamped = Math.max(0, Math.min(100, Number(pct) || 0));
  const next = `${clamped.toFixed(2)}%`;
  if (el && el.style.width !== next) el.style.width = next;
  return el;
}
