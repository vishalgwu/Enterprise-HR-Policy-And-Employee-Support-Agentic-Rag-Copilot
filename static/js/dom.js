/* =============================================================================
   Element building, text formatting and the markdown renderer.

   **Nothing untrusted is ever assigned to innerHTML, anywhere in this console.**
   Answer text comes from a language model, citation titles from uploaded
   documents, audit rows and scratchpad notes from the database, worker findings
   from a model reading an uploaded document. None of it is content this
   application controls, and the admin reading the audit log is exactly the
   session worth stealing.

   So every dynamic string reaches the page through `textContent` or
   `createTextNode`, and the markdown renderer below *builds* `<strong>`,
   `<em>`, `<code>` and `<table>` elements rather than parsing markup into HTML.
   There is therefore no escaping step that can be forgotten: the unsafe
   operation is not available. Keep it that way -- the moment one `innerHTML`
   appears, a model reply containing `<img src=x onerror=...>` runs.

   The one exception is `<use href="#icon">`, which references a symbol already
   in the document and takes no user input. It is built with `setAttributeNS`
   below so even that is not a string of HTML.
   ========================================================================== */

const SVG_NS = "http://www.w3.org/2000/svg";
const XLINK_NS = "http://www.w3.org/1999/xlink";

/** Build an element. `props` sets attributes, `class`, `text` and `on*` handlers. */
export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  apply(node, props);
  append(node, children);
  return node;
}

/** Build an SVG element. Separate because SVG needs its own namespace. */
export function svgEl(tag, props = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "text") {
      node.textContent = String(value);
    } else if (key === "class") {
      node.setAttribute("class", String(value));
    } else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else {
      node.setAttribute(key, String(value));
    }
  }
  append(node, children);
  return node;
}

/** An icon, referencing a <symbol> already in the document. */
export function icon(name, size = 16) {
  const node = svgEl("svg", { width: size, height: size, "aria-hidden": "true" });
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `#${name}`);
  // Old WebKit only honours the namespaced form. Set both; a browser that
  // understands `href` ignores the other.
  use.setAttributeNS(XLINK_NS, "xlink:href", `#${name}`);
  node.appendChild(use);
  return node;
}

function apply(node, props) {
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "text") {
      node.textContent = String(value);
    } else if (key === "class") {
      node.className = String(value);
    } else if (key === "html") {
      // Deliberately not supported. Throwing rather than ignoring, so this
      // fails in development rather than silently producing an empty element
      // that somebody then "fixes" with innerHTML.
      throw new Error("el() has no html option; build nodes, do not parse markup");
    } else if (key === "dataset") {
      Object.assign(node.dataset, value);
    } else if (key === "style" && typeof value === "object") {
      Object.assign(node.style, value);
    } else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) {
      node.setAttribute(key, "");
    } else {
      node.setAttribute(key, String(value));
    }
  }
}

function append(node, children) {
  const list = Array.isArray(children) ? children : [children];
  for (const child of list) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(
      child instanceof Node ? child : document.createTextNode(String(child)),
    );
  }
}

/** Replace a container's children. The only way this console mutates the DOM. */
export function fill(node, children) {
  if (!node) return node;
  node.replaceChildren();
  append(node, children);
  return node;
}

export function show(node, visible = true) {
  if (node) node.hidden = !visible;
}

export function qs(selector, root = document) {
  return root.querySelector(selector);
}

export function qsa(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

/* --- Formatting ----------------------------------------------------------- */

export function num(value, fallback = "—") {
  if (value === null || value === undefined || Number.isNaN(value)) return fallback;
  return Number(value).toLocaleString();
}

export function pct(value, digits = 0) {
  if (value === null || value === undefined) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

/** Milliseconds, in the unit a reader actually wants. */
export function ms(value) {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  if (n < 1) return "<1ms";
  if (n < 1000) return `${Math.round(n)}ms`;
  return `${(n / 1000).toFixed(n < 10000 ? 2 : 1)}s`;
}

/**
 * A projected cost. Sub-cent figures get four decimals rather than rounding to
 * $0.00, because "it cost nothing" and "it cost a twentieth of a cent" are
 * different claims and only one of them is true.
 */
export function usd(value) {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  if (n === 0) return "$0";
  if (n < 0.01) return `$${n.toFixed(4)}`;
  return `$${n.toFixed(2)}`;
}

export function tokens(value) {
  const n = Number(value || 0);
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

export function ago(iso) {
  if (!iso) return "—";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return String(iso);
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function clock(iso) {
  if (!iso) return "—";
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return String(iso);
  return when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function titleCase(text) {
  return String(text || "")
    .replace(/[_.-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** A tag chip whose modifier class is derived from the value itself. */
export function tag(value, label = null) {
  const key = String(value ?? "").toLowerCase().replace(/[^a-z0-9_]+/g, "_");
  return el("span", { class: `tag tag--${key}`, text: label ?? String(value ?? "—") });
}

/* --- Toasts --------------------------------------------------------------- */

const TOAST_MS = 5200;

export function toast(message, kind = "info") {
  const host = qs("#toasts");
  if (!host) return;
  const node = el("div", { class: `toast toast--${kind}`, role: "status" }, [
    el("span", { text: message }),
  ]);
  host.appendChild(node);
  setTimeout(() => {
    node.classList.add("is-leaving");
    node.addEventListener("animationend", () => node.remove(), { once: true });
    // A browser that skips the animation (prefers-reduced-motion collapses it
    // to ~0ms, and animationend may not fire at all) must still remove it.
    setTimeout(() => node.remove(), 400);
  }, TOAST_MS);
}

/* --- Entrance animations -------------------------------------------------- */

/**
 * Add the stagger class only when the container is actually on screen.
 *
 * Load-bearing, and the reason is specific: the trace, timing and source lists
 * are rendered the moment an answer lands, into whichever inspector panels are
 * *not* the open tab -- and a CSS animation never runs on a `display: none`
 * element. With `animation-fill-mode: both` the items were then stuck holding
 * the `from` keyframe, so opening the tab showed an empty panel with the
 * content sitting there at `opacity: 0`.
 */
export function stagger(node) {
  if (!node) return;
  node.classList.remove("stagger-in");
  if (node.offsetParent !== null) node.classList.add("stagger-in");
}

/* --- Markdown ------------------------------------------------------------- */

/*
  Paragraphs, headings, lists, bold, italic, inline code and pipe tables. Every
  one of them *builds* elements.

  Tables and italics were both added after watching real answers: a live web
  answer returned a correct five-line markdown table that fell through to the
  paragraph branch and came out as one run-on line of pipes and dashes, and the
  model routinely cites its source as `*leave-and-time-off.md*`.

  Bold is first in the alternation so `**x**` wins over `*x*`. The order of a
  regex alternation is the whole of its precedence here.
*/
const INLINE = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;

function inlineNodes(text) {
  const out = [];
  for (const part of String(text).split(INLINE)) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      out.push(el("strong", { text: part.slice(2, -2) }));
    } else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      out.push(el("code", { text: part.slice(1, -1) }));
    } else if (
      part.startsWith("*") &&
      part.endsWith("*") &&
      part.length > 2
    ) {
      out.push(el("em", { text: part.slice(1, -1) }));
    } else {
      out.push(document.createTextNode(part));
    }
  }
  return out;
}

const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
const isTableRule = (line) => /^[\s|:-]+$/.test(line) && line.includes("-");

function cells(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function buildTable(rows) {
  const header = cells(rows[0]);
  const body = rows.slice(isTableRule(rows[1] || "") ? 2 : 1).filter(isTableRow);
  const table = el("table", {}, [
    el("thead", {}, [
      el(
        "tr",
        {},
        header.map((cell) => el("th", {}, inlineNodes(cell))),
      ),
    ]),
    el(
      "tbody",
      {},
      body.map((row) =>
        el(
          "tr",
          {},
          cells(row).map((cell) => el("td", {}, inlineNodes(cell))),
        ),
      ),
    ),
  ]);
  // Its own scroll box: a wide table must not make the whole conversation
  // scroll sideways.
  return el("div", { class: "bubble__tablebox" }, [table]);
}

/** Render markdown-ish text as a DocumentFragment of real elements. */
export function markdown(text) {
  const fragment = document.createDocumentFragment();
  const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");

  let index = 0;
  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    if (isTableRow(line)) {
      const rows = [];
      while (index < lines.length && isTableRow(lines[index])) {
        rows.push(lines[index]);
        index += 1;
      }
      fragment.appendChild(buildTable(rows));
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      const level = Math.min(4, heading[1].length + 2);
      fragment.appendChild(el(`h${level}`, {}, inlineNodes(heading[2])));
      index += 1;
      continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || ordered) {
      const tagName = bullet ? "ul" : "ol";
      const items = [];
      while (index < lines.length) {
        const current = lines[index];
        const match = bullet
          ? current.match(/^\s*[-*+]\s+(.*)$/)
          : current.match(/^\s*\d+[.)]\s+(.*)$/);
        if (!match) break;
        items.push(el("li", {}, inlineNodes(match[1])));
        index += 1;
      }
      fragment.appendChild(el(tagName, {}, items));
      continue;
    }

    // A paragraph runs until a blank line or the start of another block.
    const paragraph = [];
    while (index < lines.length) {
      const current = lines[index];
      if (
        !current.trim() ||
        isTableRow(current) ||
        /^(#{1,4})\s+/.test(current) ||
        /^\s*[-*+]\s+/.test(current) ||
        /^\s*\d+[.)]\s+/.test(current)
      ) {
        break;
      }
      paragraph.push(current.trim());
      index += 1;
    }
    if (paragraph.length) {
      fragment.appendChild(el("p", {}, inlineNodes(paragraph.join(" "))));
    }
  }

  return fragment;
}
