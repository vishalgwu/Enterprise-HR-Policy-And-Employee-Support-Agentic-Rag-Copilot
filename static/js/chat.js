/* =============================================================================
   The Ask view: the conversation, and the inspector beside it.

   **Conversations live in this browser.** `/api/chat` is open by design, so
   there is no identity a server-side conversation could be scoped to; an
   endpoint that listed them would either need authentication this product does
   not have, or would let anyone read anyone else's HR questions. The client
   owns the thread and posts it back with each turn, which is also why the
   copilot is stateless.

   They are in `localStorage` rather than `sessionStorage`, unlike the keys:
   these are the user's own questions and losing them on a tab close would be a
   worse default than keeping them. A credential is the opposite trade, which is
   why the two live in different stores.

   The streamed path and the posted path produce the same answer; the server
   pins that with a test. Streaming is attempted first and a failure falls back
   to `POST /api/chat`, because a proxy that strips `text/event-stream` should
   cost the live stages, not the answer.
   ========================================================================== */

import { ApiError, api, streamChat } from "./api.js";
import {
  ago,
  clock,
  el,
  fill,
  icon,
  markdown,
  ms,
  num,
  qs,
  show,
  stagger,
  tag,
  titleCase,
  toast,
  usd,
} from "./dom.js";
import { pathFromTrace, renderGraph, supervisorPath } from "./graph.js";

const STORE = "hr-copilot/chats";
const MAX_CONVERSATIONS = 24;
// Eight turns is what `normalise_history` keeps anyway, and sending more would
// be paying to have it discarded server-side.
const HISTORY_TURNS = 8;

const state = {
  conversations: [],
  activeId: null,
  last: null, // the most recent AnswerResult
  graphs: null, // /api/graph, fetched once
  mode: null, // a per-question override, or null for the deployment's own
  busy: false,
};

/* --- Persistence ---------------------------------------------------------- */

function load() {
  try {
    const raw = localStorage.getItem(STORE);
    const parsed = raw ? JSON.parse(raw) : null;
    if (parsed && Array.isArray(parsed.conversations)) {
      state.conversations = parsed.conversations.filter(
        (c) => c && typeof c.id === "string" && Array.isArray(c.turns),
      );
      state.activeId = parsed.activeId || state.conversations[0]?.id || null;
    }
  } catch {
    // A malformed store is dropped rather than repaired. It came from an older
    // shape of this console and guessing at it would show someone a thread
    // that is half another thread.
    state.conversations = [];
  }
  if (!state.conversations.length) newConversation(false);
}

function save() {
  try {
    localStorage.setItem(
      STORE,
      JSON.stringify({
        conversations: state.conversations.slice(0, MAX_CONVERSATIONS),
        activeId: state.activeId,
      }),
    );
  } catch {
    // Quota exceeded, or storage blocked. The conversation still works for this
    // page load; it just will not be there tomorrow. Not worth a toast on every
    // turn, so this is silent by design.
  }
}

function active() {
  return state.conversations.find((c) => c.id === state.activeId) || null;
}

function newConversation(render = true) {
  const id =
    crypto?.randomUUID?.() ||
    `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  // Short, and matching the server's `CONVERSATION_ID_PATTERN`: letters,
  // digits, dot, dash or underscore, 1-64 characters. A uuid satisfies it.
  state.conversations.unshift({
    id,
    title: "New chat",
    turns: [],
    updatedAt: new Date().toISOString(),
  });
  state.conversations = state.conversations.slice(0, MAX_CONVERSATIONS);
  state.activeId = id;
  state.last = null;
  save();
  if (render) {
    renderThread();
    renderConversations();
    renderInspector(null);
  }
  return id;
}

/* --- The thread ----------------------------------------------------------- */

/**
 * The speaker label above a turn.
 *
 * A label, not a coloured disc. The assistant's turn is set as prose on the
 * page rather than in a bubble (see the note at the top of `views.css`), and a
 * monogram in a circle beside a document is a chat affordance on something
 * that is not a chat message. The user's is hidden by CSS — their own turn is
 * already marked by being tinted and right-aligned, so naming it as well is
 * telling someone what they just typed.
 */
function avatar(role) {
  return el("div", {
    class: "turn__avatar",
    text: role === "user" ? "You" : "HR Copilot",
  });
}

function userTurn(text) {
  return el("div", { class: "turn turn--user" }, [
    avatar("user"),
    el("div", { class: "turn__body" }, [
      el("div", { class: "bubble", text }),
    ]),
  ]);
}

function metaRow(result) {
  const bits = [];
  if (result.answer_mode) bits.push(tag(result.answer_mode));
  if (result.source_used) bits.push(tag(result.source_used));
  if (result.cached) bits.push(tag("cached", `cached · ${result.cache_similarity ?? ""}`));
  if (result.escalated) bits.push(tag("escalated"));
  if (result.model) {
    bits.push(el("span", { class: "tag", text: result.model }));
  }
  if (result.latency_ms !== null && result.latency_ms !== undefined) {
    bits.push(el("span", { class: "tag", text: ms(result.latency_ms) }));
  }
  if (result.cost_usd) {
    bits.push(el("span", { class: "tag", text: usd(result.cost_usd) }));
  }
  if (result.pii_redacted?.length) {
    bits.push(tag("pii", `redacted: ${result.pii_redacted.join(", ")}`));
  }
  if (result.stop_reason) bits.push(tag("degraded", "partial"));
  if (result.degraded?.length) bits.push(tag("degraded", "degraded call"));
  return el("div", { class: "turn__meta" }, bits);
}

function agentTurn(result) {
  const classes = ["bubble"];
  if (result.source_used === "error" || result.blocked) classes.push("bubble--error");
  if (result.stop_reason) classes.push("bubble--partial");

  const body = el("div", { class: "turn__body" }, [
    el("div", { class: classes.join(" ") }, [markdown(result.answer)]),
    metaRow(result),
  ]);

  if (result.sources?.length) {
    body.appendChild(
      el(
        "div",
        { class: "turn__cites" },
        result.sources.map((source) =>
          el("span", { class: "cite", title: source.title || source.source }, [
            icon("i-book", 12),
            source.source,
          ]),
        ),
      ),
    );
  }
  if (result.resolved_question) {
    body.appendChild(
      el("div", {
        class: "turn__meta",
        text: `Resolved your follow-up to: "${result.resolved_question}"`,
      }),
    );
  }
  return el("div", { class: "turn turn--agent" }, [avatar("agent"), body]);
}

function renderThread() {
  const inner = qs("#thread-inner");
  const conversation = active();
  if (!inner) return;

  if (!conversation || !conversation.turns.length) {
    // The welcome block is in the template, so it is rebuilt here rather than
    // kept around hidden -- a hidden node would keep its suggestion chips in
    // the tab order.
    fill(inner, [welcomeNode()]);
    wireSuggestions();
    return;
  }

  fill(
    inner,
    conversation.turns.map((turn) =>
      turn.role === "user"
        ? userTurn(turn.content)
        : agentTurn(turn.result || { answer: turn.content }),
    ),
  );
  scrollToEnd();
}

let welcomeTemplate = null;

function welcomeNode() {
  // Captured on first render, before anything replaces it, so the markup lives
  // in the template with the rest of the page's copy rather than in a string
  // here.
  if (!welcomeTemplate) {
    const original = qs("#welcome");
    welcomeTemplate = original ? original.cloneNode(true) : el("div");
  }
  return welcomeTemplate.cloneNode(true);
}

function wireSuggestions() {
  for (const chip of document.querySelectorAll("#thread-inner .suggest .chip")) {
    chip.addEventListener("click", () => {
      const input = qs("#question");
      input.value = chip.textContent.trim();
      input.dispatchEvent(new Event("input"));
      input.focus();
    });
  }
}

function scrollToEnd() {
  const thread = qs("#thread");
  if (thread) thread.scrollTop = thread.scrollHeight;
}

/* --- Progress ------------------------------------------------------------- */

const STAGE_WORDS = {
  guard: "checking the question",
  blocked: "refusing",
  cache: "looking for an answer already given",
  cache_hit: "found one",
  route: "choosing a model",
  mode: "starting",
  output_guard: "checking the answer",
  Contextualiser: "resolving your follow-up",
  Router: "routing",
  "KB Retriever": "searching internal policy",
  "KB Grader": "grading the evidence",
  Tavily: "searching the web",
  "Web Grader": "grading the web results",
  Rewriter: "rephrasing and trying again",
  "Generate/KB": "writing from policy",
  "Generate/Web": "writing from the web",
  "Generate/Direct": "replying",
  Fallback: "admitting it does not know",
  Planner: "deciding whether to split this",
  Supervisor: "choosing a specialist",
  Compose: "merging what the specialists found",
  Escalate: "handing this to a person",
};

/*
  The supervisor's worker tags are `Worker/<name>`, which is a shape rather
  than a fixed string, so it cannot sit in the table above. Named here because
  this is the one path where the wait is long enough that the stage text is the
  only thing on screen for twenty seconds, and "starting" for all of it is what
  this exists to stop.
*/
function stageWords(name) {
  if (!name) return "thinking";
  const worker = String(name).match(/^Worker\/(.+)$/);
  if (worker) return `asking the ${worker[1]} specialist`;
  return STAGE_WORDS[name] || name;
}

function thinkingNode() {
  return el("div", { class: "turn turn--agent", id: "thinking" }, [
    avatar("agent"),
    el("div", { class: "turn__body" }, [
      el("div", { class: "thinking" }, [
        el("span", { class: "thinking__dots" }, [
          el("i"),
          el("i"),
          el("i"),
        ]),
        el("span", { class: "thinking__stage", id: "thinking-stage", text: "thinking" }),
      ]),
    ]),
  ]);
}

/* --- The inspector -------------------------------------------------------- */

function kv(rows) {
  return rows
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([label, value]) =>
      el("div", { class: "verdict__row" }, [
        el("span", { text: label }),
        el("b", { text: String(value) }),
      ]),
    );
}

function renderVerdict(result) {
  const host = qs("#verdict");
  if (!host) return;
  if (!result) {
    show(host, false);
    return;
  }
  show(host, true);
  fill(
    host,
    kv([
      ["Mode", result.answer_mode],
      ["Route", result.route],
      ["Source", result.source_used],
      ["Retrieval", result.retrieval_mode],
      ["KB grade", result.kb_grade],
      ["Web grade", result.web_grade],
      ["Chunks", result.kb_chunks],
      ["Rewrites", result.retry_count],
      ["Complexity", result.complexity],
      ["Model", result.model],
      ["Latency", ms(result.latency_ms)],
      ["Cached", result.cached ? `yes (${result.cache_similarity})` : "no"],
    ]),
  );
}

function renderTrace(result) {
  const host = qs("#trace");
  const empty = qs("#trace-empty");
  const steps = result?.trace || [];
  qs("#n-trace").textContent = steps.length ? String(steps.length) : "";
  show(empty, steps.length === 0);
  if (!host) return;

  fill(
    host,
    steps.map((step) => {
      const [rawTag, ...rest] = String(step).split(":");
      const detail = rest.join(":").trim();
      const classes = [];
      if (/continuing with fallback|rate limited/i.test(step)) classes.push("is-degraded");
      if (/error|failed/i.test(step) && !classes.length) classes.push("is-error");
      return el("li", { class: classes.join(" ") }, [
        el("span", { class: "trace__tag", text: rawTag.trim() }),
        detail ? el("span", { class: "trace__text", text: ` ${detail}` }) : null,
      ]);
    }),
  );
  stagger(host);
}

function renderTiming(result) {
  const host = qs("#timing");
  const empty = qs("#timing-empty");
  const spans = (result?.spans || []).filter((span) => span.duration_ms > 0);
  qs("#n-timing").textContent = spans.length ? String(spans.length) : "";
  show(empty, spans.length === 0);
  if (!host) return;

  const widest = Math.max(1, ...spans.map((span) => span.duration_ms));
  fill(
    host,
    el(
      "div",
      { class: "spans" },
      spans.map((span) =>
        el(
          "div",
          {
            class: "span-row",
            dataset: { kind: span.kind, status: span.status },
            title: `${span.name} · ${span.kind} · ${span.status}`,
          },
          [
            el("div", { class: "span-row__name" }, [
              el("span", {
                class: "span-row__label",
                // Indented by depth, so the span tree reads as a tree without
                // drawing one.
                text: `${"  ".repeat(Math.min(3, span.depth))}${span.name}`,
              }),
              el("span", {
                class: "span-row__bar",
                style: { width: `${Math.max(3, (span.duration_ms / widest) * 110)}px` },
              }),
            ]),
            el("span", { class: "span-row__ms", text: ms(span.duration_ms) }),
          ],
        ),
      ),
    ),
  );
  stagger(host.firstChild);
}

function renderSources(result) {
  const host = qs("#sources");
  const empty = qs("#sources-empty");
  const sources = result?.sources || [];
  const urls = result?.web_urls || [];
  qs("#n-sources").textContent = sources.length + urls.length
    ? String(sources.length + urls.length)
    : "";
  show(empty, sources.length + urls.length === 0);
  if (!host) return;

  const cards = sources.map((source) =>
    el("div", { class: "source" }, [
      el("div", { class: "source__title", text: source.title || source.source }),
      el("div", { class: "source__meta" }, [
        el("span", { class: "tag", text: source.source }),
        source.department ? tag(source.department) : null,
        source.doc_type ? el("span", { class: "tag", text: source.doc_type }) : null,
        source.score !== null && source.score !== undefined
          ? el("span", {
              class: "source__score",
              text: `score ${Number(source.score).toFixed(2)}`,
            })
          : null,
      ]),
    ]),
  );

  for (const url of urls) {
    cards.push(
      el("div", { class: "source" }, [
        el("div", { class: "source__title" }, [
          el("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: url }),
        ]),
        el("div", { class: "source__meta" }, [tag("web_search", "public web")]),
      ]),
    );
  }

  fill(host, cards);
  stagger(host);
}

function renderPad(notes) {
  const host = qs("#pad");
  const empty = qs("#pad-empty");
  const rows = notes || [];
  qs("#n-pad").textContent = rows.length ? String(rows.length) : "";
  show(empty, rows.length === 0);
  if (!host) return;

  fill(
    host,
    rows.map((note) =>
      el("div", { class: "pad-note", dataset: { kind: note.kind } }, [
        el("div", { class: "pad-note__head" }, [
          el("span", { class: "pad-note__author", text: note.author }),
          el("span", { text: note.kind }),
          el("span", { class: "pad-note__time", text: clock(note.created_at) }),
        ]),
        el("div", { text: note.text }),
      ]),
    ),
  );
  stagger(host);
}

function bar(label, used, limit, formatter = num) {
  const share = limit > 0 ? Math.min(1, used / limit) : 0;
  const classes = ["meter__fill"];
  if (share > 0.85) classes.push("meter__fill--warn");
  return el("div", { class: "budget-bar" }, [
    el("div", { class: "budget-bar__head" }, [
      el("span", { text: label }),
      el("b", { text: `${formatter(used)} / ${formatter(limit)}` }),
    ]),
    el("div", { class: "meter" }, [
      el("div", {
        class: classes.join(" "),
        style: { width: `${(share * 100).toFixed(1)}%` },
      }),
    ]),
  ]);
}

function renderCost(result) {
  const host = qs("#budget");
  const detail = qs("#cost-detail");
  if (!host) return;
  const budget = result?.budget;
  if (!budget) {
    fill(host, el("p", { class: "empty", text: "Ask something first." }));
    show(detail, false);
    return;
  }

  fill(host, [
    bar("Reasoning steps", budget.steps, budget.max_steps),
    bar("Tool calls", budget.tool_calls, budget.max_tool_calls),
    bar("Model calls", budget.llm_calls, budget.max_llm_calls),
    bar("Tokens", budget.tokens, budget.max_tokens),
    bar("Wall clock", budget.elapsed_seconds, budget.max_seconds, (v) =>
      `${Number(v).toFixed(1)}s`,
    ),
    bar("Projected cost", budget.cost_usd, budget.max_cost_usd, usd),
  ]);

  show(detail, true);
  fill(
    detail,
    kv([
      ["Prompt tokens", num(result.prompt_tokens)],
      ["Completion tokens", num(result.completion_tokens)],
      ["Projected cost", usd(result.cost_usd)],
      ["Tier", result.model_tier],
      ["Routing reason", result.routing_reason],
      ["Own key", result.byok ? "yes" : "no"],
      ["Stopped early", budget.stop_reason || "no"],
    ]),
  );
}

function renderGraphPanel(result) {
  if (!state.graphs) return;
  const useSupervisor = result?.answer_mode === "supervisor";
  const definition = useSupervisor ? state.graphs.supervisor : state.graphs.pipeline;
  const path = !result
    ? []
    : useSupervisor
      ? supervisorPath(result.worker_path, result.escalated, definition.nodes)
      : pathFromTrace(result.trace, definition.nodes);
  renderGraph(qs("#graph"), definition, path);
}

export function renderInspector(result) {
  renderGraphPanel(result);
  renderVerdict(result);
  renderTrace(result);
  renderTiming(result);
  renderSources(result);
  renderCost(result);
  renderPad(result?.scratchpad);
}

function renderConversations() {
  const host = qs("#convos");
  if (!host) return;
  fill(
    host,
    state.conversations.map((conversation) =>
      el(
        "button",
        {
          class: `convo-item${conversation.id === state.activeId ? " is-active" : ""}`,
          onclick: (event) => {
            if (event.target.closest(".convo-item__del")) return;
            state.activeId = conversation.id;
            state.last = null;
            save();
            renderThread();
            renderConversations();
            renderInspector(null);
            loadNotes();
          },
        },
        [
          icon("i-chat", 15),
          el("span", { class: "convo-item__text" }, [
            el("span", { class: "convo-item__title", text: conversation.title }),
            el("span", {
              class: "convo-item__meta",
              text: `${conversation.turns.filter((t) => t.role === "user").length} question(s) · ${ago(conversation.updatedAt)}`,
            }),
          ]),
          el(
            "span",
            {
              class: "convo-item__del",
              role: "button",
              title: "Delete this conversation",
              onclick: (event) => {
                event.stopPropagation();
                state.conversations = state.conversations.filter(
                  (c) => c.id !== conversation.id,
                );
                if (state.activeId === conversation.id) {
                  state.activeId = state.conversations[0]?.id || null;
                  state.last = null;
                }
                if (!state.conversations.length) newConversation(false);
                save();
                renderThread();
                renderConversations();
                renderInspector(null);
              },
            },
            [icon("i-close", 14)],
          ),
        ],
      ),
    ),
  );
}

/* --- Scratchpad ----------------------------------------------------------- */

async function loadNotes() {
  const conversation = active();
  if (!conversation) return;
  try {
    const body = await api.notes(conversation.id);
    renderPad(body.notes);
  } catch {
    // A scratchpad this console cannot read is an agent with no notes, which is
    // how every conversation starts. Not worth interrupting anyone over.
    renderPad([]);
  }
}

async function addNote() {
  const input = qs("#pad-text");
  const text = input.value.trim();
  if (!text) return;
  const conversation = active();
  if (!conversation) return;
  try {
    await api.addNote({ conversation_id: conversation.id, text, kind: "user" });
    input.value = "";
    await loadNotes();
    toast("Noted. The next turn will see it.", "ok");
  } catch (error) {
    toast(error.detail || "The note could not be saved.", "bad");
  }
}

/* --- Asking --------------------------------------------------------------- */

function historyFor(conversation) {
  return conversation.turns
    .slice(-HISTORY_TURNS)
    .map((turn) => ({
      role: turn.role,
      content: turn.role === "user" ? turn.content : turn.result?.answer || turn.content,
    }))
    .filter((turn) => turn.content);
}

async function ask(question) {
  if (state.busy) return;
  const conversation = active() || { id: newConversation(false) };
  const current = active();
  state.busy = true;

  const send = qs("#send");
  send.disabled = true;

  // The user's turn goes in immediately. Waiting for the server to echo it
  // makes the input feel like it swallowed the question.
  current.turns.push({ role: "user", content: question });
  if (current.turns.length === 1) {
    current.title = question.length > 48 ? `${question.slice(0, 47)}…` : question;
  }
  current.updatedAt = new Date().toISOString();
  save();
  renderThread();
  renderConversations();

  const inner = qs("#thread-inner");
  inner.appendChild(thinkingNode());
  scrollToEnd();

  const payload = {
    question,
    history: historyFor({ turns: current.turns.slice(0, -1) }),
    conversation_id: current.id,
  };
  if (state.mode) payload.mode = state.mode;

  const stageNode = () => qs("#thinking-stage");
  const liveTrace = [];

  let result = null;
  try {
    result = await streamChat(payload, (name, data) => {
      if (name === "stage" && data?.name) {
        const node = stageNode();
        if (node) node.textContent = stageWords(data.name);
      } else if (name === "trace" && data?.text) {
        liveTrace.push(data.text);
        renderTrace({ trace: liveTrace });
      } else if (name === "error" && data?.message) {
        toast(data.message, data.status === 429 ? "warn" : "bad");
      }
    });
  } catch (error) {
    // Streaming failed outright -- a proxy stripping the content type, an
    // aborted connection. Fall back to the plain endpoint: losing the live
    // stages is a cosmetic loss and losing the answer is not.
    if (error instanceof ApiError && error.status === 429) {
      toast(error.detail, "warn");
    } else {
      try {
        result = await api.chat(payload);
      } catch (fallbackError) {
        toast(fallbackError.detail || "The copilot could not answer that.", "bad");
      }
    }
  }

  qs("#thinking")?.remove();
  state.busy = false;
  send.disabled = !qs("#question").value.trim();

  if (!result) {
    renderThread();
    return;
  }

  current.turns.push({ role: "assistant", content: result.answer, result });
  current.updatedAt = new Date().toISOString();
  save();
  state.last = result;
  renderThread();
  renderConversations();
  renderInspector(result);

  if (result.blocked) toast("That question was refused by the input guard.", "warn");
  else if (result.stop_reason) toast("That answer is partial: a budget ran out.", "warn");
}

/* --- Wiring --------------------------------------------------------------- */

function autosize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(180, textarea.scrollHeight)}px`;
}

export function setMode(mode) {
  state.mode = mode || null;
  const text = qs("#mode-text");
  if (text) {
    text.textContent = state.mode || document.body.dataset.agentMode || "auto";
  }
}

export function currentMode() {
  return state.mode;
}

export function lastResult() {
  return state.last;
}

export async function initChat() {
  load();
  renderThread();
  renderConversations();

  try {
    state.graphs = await api.graph();
    renderGraphPanel(null);
  } catch {
    // The diagram is the one panel that genuinely cannot degrade to text, so a
    // failure here leaves it empty rather than filling it with a guess.
  }
  loadNotes();

  const input = qs("#question");
  const send = qs("#send");
  const count = qs("#char-count");

  input.addEventListener("input", () => {
    autosize(input);
    const length = input.value.length;
    count.textContent = `${length} / 3000`;
    count.classList.toggle("is-near", length > 2700);
    send.disabled = !input.value.trim() || state.busy;
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      qs("#composer").requestSubmit();
    }
  });

  qs("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || state.busy) return;
    input.value = "";
    autosize(input);
    count.textContent = "0 / 3000";
    ask(question);
  });

  qs("#chat-new").addEventListener("click", () => {
    newConversation();
    loadNotes();
    qs("#question").focus();
  });

  qs("#chat-clear").addEventListener("click", () => {
    state.conversations = [];
    newConversation(false);
    save();
    renderThread();
    renderConversations();
    renderInspector(null);
    toast("Conversations cleared from this browser.", "ok");
  });

  qs("#pad-add").addEventListener("click", addNote);
  qs("#pad-text").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addNote();
    }
  });

  // Inspector tabs.
  for (const button of document.querySelectorAll("#view-ask .tab")) {
    button.addEventListener("click", () => {
      for (const other of document.querySelectorAll("#view-ask .tab")) {
        other.classList.toggle("is-active", other === button);
      }
      for (const panel of document.querySelectorAll("#panels .panel")) {
        panel.classList.toggle(
          "is-active",
          panel.id === `panel-${button.dataset.panel}`,
        );
      }
      // Re-render the newly visible panel so its entrance animation runs now
      // that it is on screen -- see `stagger` in dom.js for why it cannot have
      // run earlier.
      if (state.last) renderInspector(state.last);
    });
  }

  input.focus();
}

export { renderPad, loadNotes };
