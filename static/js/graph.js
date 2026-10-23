/* =============================================================================
   Drawing a compiled graph, laid out from the server's own node and edge lists.

   `CLAUDE.md` records that the console's diagram and the real edge set can
   drift, and that renaming a trace tag silently stops lighting a node. Half of
   that is fixed here: nothing in this file knows the shape of either graph. It
   fetches `/api/graph`, which serves the constants `build_graph` and
   `build_supervisor_graph` actually wire from, and lays the nodes out itself --
   so a node added to the agent appears in the diagram with no change here.

   The other half, mapping a trace tag to a node, cannot be derived that way: a
   tag is prose a node chose to log. `TRACE_NODE` below is that mapping, and it
   is followed by a token-overlap fallback so a renamed tag degrades to "lights
   a slightly less specific node" rather than to "lights nothing".

   The layout is a layered one: BFS depth from the entry node, first visit wins.
   First visit wins is what makes a cycle work -- `rewrite_query -> retrieve_kb`
   is the one edge in this graph worth seeing, and a longest-path layering would
   not terminate on it.
   ========================================================================== */

import { svgEl } from "./dom.js";

const BOX_W = 128;
const BOX_H = 26;
const GAP_X = 14;
const GAP_Y = 30;
const PAD = 14;
// How many boxes may sit side by side before a layer wraps. Three keeps both
// graphs roughly square in a panel a third of the page wide.
const MAX_PER_ROW = 3;

/*
  Trace tag -> node id. The tags come from `note()` calls in
  `app/agent/nodes.py` and `app/agent/supervisor.py`; renaming one there means
  renaming it here. The fallback below means a miss is cosmetic rather than
  silent, but it is still a miss -- change both together.
*/
const TRACE_NODE = {
  contextualiser: "contextualize",
  router: "route_question",
  "kb retriever": "retrieve_kb",
  "kb grader": "grade_kb_evidence",
  tavily: "search_web",
  "web grader": "grade_web_evidence",
  rewriter: "rewrite_query",
  "generate/kb": "generate_from_kb",
  "generate/web": "generate_from_web",
  "generate/direct": "direct_answer",
  fallback: "answer_insufficient",
  retriever: "retrieve_kb",
  extractive: "generate_from_kb",
  planner: "plan",
  supervisor: "supervise",
  composer: "compose",
  escalation: "escalate",
};

/** Resolve a trace tag to a node id, exactly then approximately. */
export function nodeForTag(tag, nodes) {
  const key = String(tag || "").trim().toLowerCase();
  if (!key) return null;
  const exact = TRACE_NODE[key];
  if (exact && nodes.includes(exact)) return exact;
  if (nodes.includes(key)) return key;

  // Token overlap: "KB Grader" still finds `grade_kb_evidence` when the map is
  // stale. Requires two shared tokens so "Grader" alone does not match three
  // nodes and light the wrong one.
  const wanted = new Set(key.split(/[^a-z0-9]+/).filter((t) => t.length > 1));
  let best = null;
  let bestScore = 0;
  for (const node of nodes) {
    const tokens = new Set(node.split(/[^a-z0-9]+/).filter(Boolean));
    let score = 0;
    for (const token of wanted) if (tokens.has(token)) score += 1;
    if (score > bestScore) {
      bestScore = score;
      best = node;
    }
  }
  return bestScore >= 2 ? best : null;
}

/** Every edge, conditional ones included, as {from, to, conditional}. */
function allEdges(definition) {
  const edges = (definition.edges || []).map((edge) => ({
    from: edge.from,
    to: edge.to,
    conditional: false,
  }));
  for (const [source, branch] of Object.entries(definition.branches || {})) {
    for (const target of Object.values(branch.targets || {})) {
      edges.push({ from: source, to: target, conditional: true });
    }
  }
  return edges;
}

/** Assign each node a layer and a slot within it. */
function layout(definition) {
  const nodes = definition.nodes || [];
  const edges = allEdges(definition);
  const out = new Map(nodes.map((node) => [node, []]));
  for (const edge of edges) {
    if (out.has(edge.from)) out.get(edge.from).push(edge.to);
  }

  const layerOf = new Map();
  const entry = definition.entry && nodes.includes(definition.entry)
    ? definition.entry
    : nodes[0];
  const queue = entry ? [entry] : [];
  if (entry) layerOf.set(entry, 0);

  while (queue.length) {
    const current = queue.shift();
    for (const next of out.get(current) || []) {
      // First visit wins, which is what makes the rewrite loop terminate: the
      // back edge finds `retrieve_kb` already placed and leaves it there.
      if (!layerOf.has(next)) {
        layerOf.set(next, layerOf.get(current) + 1);
        queue.push(next);
      }
    }
  }
  // A node no edge reaches still has to be drawn -- it is almost certainly a
  // wiring mistake, and a diagram that hides it is a diagram that hides the
  // mistake.
  let orphanLayer = Math.max(0, ...layerOf.values()) + 1;
  for (const node of nodes) {
    if (!layerOf.has(node)) layerOf.set(node, orphanLayer);
  }

  const layers = new Map();
  for (const node of nodes) {
    const layer = layerOf.get(node);
    if (!layers.has(layer)) layers.set(layer, []);
    layers.get(layer).push(node);
  }

  /*
    Wrap a wide layer onto several visual rows.

    The supervisor has six workers in one layer, which laid out flat is 1150px
    wide and 166px tall -- in a card a third of that width it renders as an
    unreadable strip. Chunking keeps the aspect ratio close to square, which is
    the shape a diagram has to be to sit beside anything else.

    The *logical* layer is kept on each node and the visual row is separate,
    because `edgePath` decides whether an edge goes backwards from the logical
    layer. Conflating them would make the second half of a wrapped layer look
    like a back edge and route every one of those curves out round the side.
  */
  const rows = [];
  for (const [layer, members] of Array.from(layers.entries()).sort(
    (a, b) => a[0] - b[0],
  )) {
    for (let start = 0; start < members.length; start += MAX_PER_ROW) {
      rows.push({ layer, members: members.slice(start, start + MAX_PER_ROW) });
    }
  }

  const widest = Math.max(...rows.map((row) => row.members.length));
  const width = widest * BOX_W + (widest - 1) * GAP_X + PAD * 2;
  const height = rows.length * BOX_H + (rows.length - 1) * GAP_Y + PAD * 2;

  const placed = new Map();
  rows.forEach((row, rowIndex) => {
    const rowWidth = row.members.length * BOX_W + (row.members.length - 1) * GAP_X;
    const left = (width - rowWidth) / 2;
    row.members.forEach((node, index) => {
      placed.set(node, {
        x: left + index * (BOX_W + GAP_X),
        y: PAD + rowIndex * (BOX_H + GAP_Y),
        layer: row.layer,
      });
    });
  });

  return { placed, edges, width, height, nodes };
}

/** A curve between two boxes. Side-routed when it goes backwards. */
function edgePath(from, to) {
  const x1 = from.x + BOX_W / 2;
  const y1 = from.y + BOX_H;
  const x2 = to.x + BOX_W / 2;
  const y2 = to.y;

  if (to.layer <= from.layer) {
    // A back edge. Routed out to the left and round, because a straight line
    // through four boxes is unreadable and this is the one edge in the graph
    // somebody is looking for.
    const bulge = Math.min(from.x, to.x) - 20;
    return `M ${from.x} ${from.y + BOX_H / 2} C ${bulge} ${from.y + BOX_H / 2}, ` +
      `${bulge} ${to.y + BOX_H / 2}, ${to.x} ${to.y + BOX_H / 2}`;
  }
  const mid = (y1 + y2) / 2;
  return `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`;
}

/**
 * Draw `definition` into `svg`.
 *
 * `visited` is the ordered list of node ids the last run went through;
 * consecutive pairs in it are exactly the edges traversed, which is how the
 * loop shows as a loop. `terminal` and `escalation` only change the colour of
 * a visited node.
 */
export function renderGraph(svg, definition, visited = []) {
  if (!svg || !definition) return;
  const { placed, edges, width, height, nodes } = layout(definition);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();

  const visitedSet = new Set(visited);
  const taken = new Set();
  for (let i = 0; i + 1 < visited.length; i += 1) {
    taken.add(`${visited[i]}->${visited[i + 1]}`);
  }

  const terminal = new Set(definition.terminal || []);
  const last = visited[visited.length - 1];

  const defs = svgEl("defs", {}, [
    svgEl(
      "marker",
      {
        id: `arrow-${svg.id || "g"}`,
        viewBox: "0 0 10 10",
        refX: "9",
        refY: "5",
        markerWidth: "5",
        markerHeight: "5",
        orient: "auto-start-reverse",
      },
      [svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "currentColor" })],
    ),
  ]);
  svg.appendChild(defs);

  for (const edge of edges) {
    const from = placed.get(edge.from);
    const to = placed.get(edge.to);
    if (!from || !to) continue;
    const classes = ["graph__edge"];
    if (to.layer <= from.layer) classes.push("is-loop");
    if (taken.has(`${edge.from}->${edge.to}`)) classes.push("is-taken");
    svg.appendChild(
      svgEl("path", {
        class: classes.join(" "),
        d: edgePath(from, to),
        "marker-end": `url(#arrow-${svg.id || "g"})`,
      }),
    );
  }

  for (const node of nodes) {
    const spot = placed.get(node);
    if (!spot) continue;
    const classes = ["graph__node"];
    if (visitedSet.has(node)) classes.push("is-visited");
    if (terminal.has(node)) classes.push("is-terminal");
    if (node === "escalate" || node === "answer_insufficient") {
      classes.push("is-escalation");
    }
    if (node === last) classes.push("is-last");

    // A node id is longer than its box in a few cases. Truncated with an
    // ellipsis and given a <title>, rather than shrunk: a diagram where one
    // label is 7px is a diagram with one unreadable label.
    const label = node.length > 20 ? `${node.slice(0, 19)}…` : node;

    svg.appendChild(
      svgEl("g", { class: classes.join(" ") }, [
        svgEl("title", { text: node }),
        svgEl("rect", {
          x: spot.x,
          y: spot.y,
          width: BOX_W,
          height: BOX_H,
          rx: 7,
        }),
        svgEl("text", {
          x: spot.x + BOX_W / 2,
          y: spot.y + BOX_H / 2 + 3.2,
          "text-anchor": "middle",
          text: label,
        }),
      ]),
    );
  }
}

/**
 * The node path a graph answer took, from its trace.
 *
 * Consecutive visited nodes are exactly the edges traversed, including
 * `rewrite_query -> retrieve_kb`, so duplicates are kept deliberately: a run
 * that retrieved twice should light that edge, and de-duplicating the list
 * would erase the one path worth seeing.
 */
export function pathFromTrace(trace, nodes) {
  const path = [];
  for (const step of trace || []) {
    const tag = String(step).split(":", 1)[0];
    const node = nodeForTag(tag, nodes);
    if (node && path[path.length - 1] !== node) path.push(node);
  }
  return path;
}

/** The supervisor's node path, from its worker path. */
export function supervisorPath(workerPath, escalated, nodes) {
  const path = ["plan"];
  for (const worker of workerPath || []) {
    path.push("supervise");
    if (nodes.includes(worker)) path.push(worker);
  }
  path.push(escalated ? "escalate" : "compose");
  return path.filter((node) => nodes.includes(node));
}
