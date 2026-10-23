/* =============================================================================
   Charts, hand-built from SVG primitives.

   No charting library, and that is a decision rather than an omission. This
   console has no build step -- `CLAUDE.md` says so and the Dockerfile depends
   on it -- so a library means a CDN script tag, which means a third-party
   request on a page that is otherwise entirely same-origin. Four chart types
   in two hundred lines is cheaper than that.

   Colours come from the stylesheet's tokens, read once per render rather than
   hardcoded, so switching theme recolours the charts without re-rendering them
   from data.
   ========================================================================== */

import { el, svgEl } from "./dom.js";

/**
 * The categorical series colours, read from the CSS tokens.
 *
 * Read from the computed style rather than listed here so the palette has one
 * home — and read *per render* rather than once, because the tokens differ
 * between the light and dark themes and a cached palette would leave a chart
 * holding the other theme's colours until its next data refresh.
 *
 * The tokens are named `--chart-N` rather than after their hues. A token
 * called `--violet` cannot be rebranded without editing this file as well,
 * which is how a palette ends up half-changed.
 *
 * Six is the limit on purpose: a seventh category in a donut is a slice nobody
 * can tell from the sixth, so the renderers below group the tail into "other"
 * instead of inventing a colour for it.
 */
function palette() {
  const style = getComputedStyle(document.documentElement);
  const fallbacks = [
    "#b0512d",
    "#3f6b5c",
    "#6d6195",
    "#9c7730",
    "#46688a",
    "#8a5566",
  ];
  return fallbacks.map(
    (fallback, index) =>
      (style.getPropertyValue(`--chart-${index + 1}`) || fallback).trim(),
  );
}

function tokenColour(name, fallback) {
  return (
    getComputedStyle(document.documentElement).getPropertyValue(name) || fallback
  ).trim();
}

/** Group a {label: value} map into at most `keep` slices plus "other". */
function topSlices(data, keep = 5) {
  const rows = Object.entries(data || {})
    .map(([label, value]) => [label, Number(value) || 0])
    .filter(([, value]) => value > 0)
    .sort((a, b) => b[1] - a[1]);
  if (rows.length <= keep) return rows;
  const head = rows.slice(0, keep);
  const tail = rows.slice(keep).reduce((sum, [, value]) => sum + value, 0);
  return tail > 0 ? [...head, ["other", tail]] : head;
}

/* --- Donut ---------------------------------------------------------------- */

function arc(cx, cy, radius, start, end) {
  const point = (angle) => [
    cx + radius * Math.cos(angle),
    cy + radius * Math.sin(angle),
  ];
  const [x1, y1] = point(start);
  const [x2, y2] = point(end);
  const large = end - start > Math.PI ? 1 : 0;
  return `M ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2}`;
}

/**
 * A donut with a figure in the hole.
 *
 * Drawn as stroked arcs rather than filled wedges: a stroke has a width, so the
 * ring thickness is one number, and there is no path to close through the
 * centre -- which is what produces the hairline seams a filled donut shows at
 * slice boundaries.
 */
export function donut(svg, data, { total = null, label = "" } = {}) {
  if (!svg) return [];
  const rows = topSlices(data);
  const sum = total ?? rows.reduce((acc, [, value]) => acc + value, 0);
  svg.replaceChildren();

  const cx = 110;
  const cy = 86;
  const radius = 58;
  const width = 20;
  const colours = palette();

  if (!sum) {
    svg.appendChild(
      svgEl("circle", {
        cx,
        cy,
        r: radius,
        fill: "none",
        stroke: tokenColour("--surface-3", "#e8e5dc"),
        "stroke-width": width,
      }),
    );
    svg.appendChild(
      svgEl("text", {
        x: cx,
        y: cy + 6,
        "text-anchor": "middle",
        class: "donut__hole",
        text: "0",
      }),
    );
    return [];
  }

  // Start at the top. Without the offset the first slice begins at three
  // o'clock, which reads as the chart being rotated.
  let angle = -Math.PI / 2;
  const legend = [];
  rows.forEach(([name, value], index) => {
    const span = (value / sum) * Math.PI * 2;
    const colour = colours[index % colours.length];
    // A full circle has no arc: start and end coincide, so the path draws
    // nothing at all. One slice is a circle.
    if (rows.length === 1) {
      svg.appendChild(
        svgEl("circle", {
          cx,
          cy,
          r: radius,
          fill: "none",
          stroke: colour,
          "stroke-width": width,
        }),
      );
    } else {
      svg.appendChild(
        svgEl("path", {
          d: arc(cx, cy, radius, angle, angle + span - 0.012),
          fill: "none",
          stroke: colour,
          "stroke-width": width,
          "stroke-linecap": "butt",
        }),
      );
    }
    legend.push({ name, value, colour, share: value / sum });
    angle += span;
  });

  svg.appendChild(
    svgEl("text", {
      x: cx,
      y: cy + 2,
      "text-anchor": "middle",
      class: "donut__hole",
      text: String(sum.toLocaleString()),
    }),
  );
  if (label) {
    svg.appendChild(
      svgEl("text", {
        x: cx,
        y: cy + 18,
        "text-anchor": "middle",
        class: "donut__sub",
        text: label,
      }),
    );
  }
  return legend;
}

/* --- Horizontal bars ------------------------------------------------------ */

export function bars(svg, data, { formatter = (v) => v.toLocaleString() } = {}) {
  if (!svg) return [];
  const rows = topSlices(data, 6);
  svg.replaceChildren();
  if (!rows.length) return [];

  const max = Math.max(...rows.map(([, value]) => value));
  const colours = palette();
  const rowHeight = 24;
  const labelWidth = 118;
  const width = 320;
  const height = rows.length * rowHeight + 8;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  rows.forEach(([name, value], index) => {
    const y = index * rowHeight + 4;
    const span = Math.max(2, ((width - labelWidth - 54) * value) / max);
    svg.appendChild(
      svgEl("text", {
        x: 0,
        y: y + 13,
        text: name.length > 16 ? `${name.slice(0, 15)}…` : name,
      }),
    );
    svg.appendChild(
      svgEl("rect", {
        class: "chart__bar",
        x: labelWidth,
        y,
        width: span,
        height: 14,
        rx: 5,
        fill: colours[index % colours.length],
      }),
    );
    svg.appendChild(
      svgEl("text", {
        x: labelWidth + span + 7,
        y: y + 12,
        text: formatter(value),
      }),
    );
  });
  return rows.map(([name, value], index) => ({
    name,
    value,
    colour: colours[index % colours.length],
  }));
}

/* --- Latency ------------------------------------------------------------- */

/**
 * p50 / p95 / p99 against a shared axis, with the mean marked.
 *
 * Shown as three bars and not a distribution because three bars is what the
 * data is: `/api/metrics` reports percentiles, not buckets. Drawing a smooth
 * curve through three points would imply a shape nobody measured.
 */
export function latency(svg, stats) {
  if (!svg) return;
  svg.replaceChildren();
  const rows = [
    ["p50", stats?.latency_p50_ms],
    ["p95", stats?.latency_p95_ms],
    ["p99", stats?.latency_p99_ms],
  ].filter(([, value]) => value !== null && value !== undefined);

  const width = 640;
  const height = 200;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (!rows.length) {
    svg.appendChild(
      svgEl("text", {
        x: width / 2,
        y: height / 2,
        "text-anchor": "middle",
        text: "No answers timed yet",
      }),
    );
    return;
  }

  const mean = Number(stats?.average_latency_ms || 0);
  const max = Math.max(mean, ...rows.map(([, value]) => Number(value))) * 1.12 || 1;
  const left = 56;
  const bottom = height - 34;
  const span = width - left - 28;
  // p50 / p95 / p99, deliberately a ramp from calm to alarming rather than
  // three arbitrary series colours: these are one measurement at three points
  // of the distribution, and the tail is the one worth noticing.
  const colours = [
    tokenColour("--ok", "#2f7150"),
    tokenColour("--warn", "#8a5a12"),
    tokenColour("--bad", "#a33a35"),
  ];

  for (let i = 0; i <= 4; i += 1) {
    const y = bottom - (bottom - 20) * (i / 4);
    svg.appendChild(
      svgEl("line", { class: "chart__grid", x1: left, y1: y, x2: width - 20, y2: y }),
    );
    svg.appendChild(
      svgEl("text", {
        x: left - 8,
        y: y + 3.5,
        "text-anchor": "end",
        text: `${Math.round((max * i) / 4)}`,
      }),
    );
  }

  const barWidth = Math.min(96, span / (rows.length * 1.7));
  rows.forEach(([name, value], index) => {
    const centre = left + span * ((index + 0.5) / rows.length);
    const tall = ((bottom - 20) * Number(value)) / max;
    svg.appendChild(
      svgEl("rect", {
        class: "chart__bar",
        x: centre - barWidth / 2,
        y: bottom - tall,
        width: barWidth,
        height: Math.max(2, tall),
        rx: 7,
        fill: colours[index % colours.length],
      }),
    );
    svg.appendChild(
      svgEl("text", {
        x: centre,
        y: bottom - tall - 7,
        "text-anchor": "middle",
        text: `${Math.round(Number(value))}ms`,
      }),
    );
    svg.appendChild(
      svgEl("text", {
        x: centre,
        y: bottom + 17,
        "text-anchor": "middle",
        text: name,
      }),
    );
  });

  if (mean > 0) {
    const y = bottom - ((bottom - 20) * mean) / max;
    svg.appendChild(
      svgEl("line", {
        x1: left,
        y1: y,
        x2: width - 20,
        y2: y,
        stroke: tokenColour("--text-faint", "#8c877c"),
        "stroke-width": 1.5,
        "stroke-dasharray": "5 4",
      }),
    );
    svg.appendChild(
      svgEl("text", {
        x: width - 24,
        y: y - 6,
        "text-anchor": "end",
        text: `mean ${Math.round(mean)}ms`,
      }),
    );
  }
}

/* --- Legend -------------------------------------------------------------- */

export function legend(host, rows, { formatter = null } = {}) {
  if (!host) return;
  host.replaceChildren();
  for (const row of rows) {
    host.appendChild(
      el("div", { class: "legend__row" }, [
        el("span", {
          class: "legend__swatch",
          style: { background: row.colour },
        }),
        el("span", { class: "legend__name", text: row.name }),
        el("span", {
          class: "legend__value",
          text: formatter
            ? formatter(row)
            : row.share !== undefined
              ? `${row.value} · ${(row.share * 100).toFixed(0)}%`
              : String(row.value),
        }),
      ]),
    );
  }
}
