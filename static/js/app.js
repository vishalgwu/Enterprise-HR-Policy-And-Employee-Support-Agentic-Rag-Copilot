/* ============================================================================
   HR Copilot console — front end
   ----------------------------------------------------------------------------
   Talks to the same origin it was served from: /health, /api/chat, /api/upload
   and /api/audit. No framework and no build step, because the target
   architecture serves this directory from Nginx.

   ## Nothing untrusted is ever assigned to innerHTML.

   Answer text comes out of an LLM, citation titles come out of uploaded
   documents, and audit rows come out of the database — all three are content
   this application does not control. `answer.innerHTML = ...` with a model
   reply containing `<img src=x onerror=...>` executes it, and the admin reading
   the audit log is exactly the session worth stealing. Every dynamic string
   here reaches the page through `textContent` or `createTextNode`, and the
   light markdown renderer builds `<strong>`/`<code>` elements rather than
   parsing markup, so there is no escaping step that can be forgotten.

   ## The admin key lives in sessionStorage, not localStorage.

   It is a credential. sessionStorage is scoped to the tab and cleared when the
   tab closes, so a shared machine does not keep it. It is sent as the
   `X-Admin-Key` header and never as a query parameter — a URL ends up in the
   history, in the proxy log and in the Referer header.
   ============================================================================ */

(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const body = document.body;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ── Transport ─────────────────────────────────────────────────────────── */

  class ApiError extends Error {
    constructor(status, detail) {
      super(detail);
      this.status = status;
      this.detail = detail;
    }
  }

  const adminKey = {
    get: () => sessionStorage.getItem("hrcopilot.adminKey") || "",
    set: (value) => sessionStorage.setItem("hrcopilot.adminKey", value),
    clear: () => sessionStorage.removeItem("hrcopilot.adminKey"),
  };

  /** Read `detail` out of a FastAPI error body, whatever shape it took.
   *  A HTTPException gives a string; a 422 from pydantic gives a list of
   *  {loc, msg} objects. Falling back to the status keeps the UI honest when
   *  the response is not JSON at all — a proxy timeout, say. */
  function errorDetail(payload, status) {
    const detail = payload && payload.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length) {
      return detail.map((d) => d.msg || String(d)).join("; ");
    }
    return `Request failed (${status}).`;
  }

  async function api(path, { method = "GET", json, form, admin = false } = {}) {
    const headers = {};
    let bodyInit;

    if (json !== undefined) {
      headers["Content-Type"] = "application/json";
      bodyInit = JSON.stringify(json);
    } else if (form !== undefined) {
      // No Content-Type: the browser must set the multipart boundary itself.
      bodyInit = form;
    }

    if (admin) {
      const key = adminKey.get();
      if (!key) throw new ApiError(0, "No admin key is set for this tab.");
      headers["X-Admin-Key"] = key;
    }

    const response = await fetch(path, { method, headers, body: bodyInit });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new ApiError(response.status, errorDetail(payload, response.status));
    return payload;
  }

  /* ── Toasts ────────────────────────────────────────────────────────────── */

  const ICONS = { ok: "✓", bad: "✕", info: "◆" };

  function toast(kind, message, ttl = 4600) {
    const el = document.createElement("div");
    el.className = `toast toast--${kind}`;

    const icon = document.createElement("span");
    icon.className = "toast__icon";
    icon.textContent = ICONS[kind] || ICONS.info;

    const text = document.createElement("span");
    text.textContent = message;

    el.append(icon, text);
    $("#toasts").appendChild(el);

    setTimeout(() => {
      el.classList.add("is-leaving");
      el.addEventListener("animationend", () => el.remove(), { once: true });
    }, ttl);
  }

  /* ── Safe text rendering ───────────────────────────────────────────────── */

  // Bold before italic: alternation is left-to-right, so `**x**` must get the
  // chance to match before `*x*` claims the first asterisk pair. Italic is here
  // because the model routinely cites its source as *leave-and-time-off.md*,
  // and unhandled asterisks show up verbatim in the answer.
  const INLINE = /\*\*([^*]+)\*\*|\*([^*\n]+)\*|`([^`]+)`/g;
  const INLINE_TAG = ["strong", "em", "code"];

  /** Append `text` to `parent`, turning **bold**, *italic* and `code` into
   *  elements. Everything else becomes a text node, so no markup in the source
   *  string is ever interpreted. */
  function appendInline(parent, text) {
    let last = 0;
    let match;
    INLINE.lastIndex = 0;
    while ((match = INLINE.exec(text)) !== null) {
      if (match.index > last) {
        parent.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      const group = [1, 2, 3].find((n) => match[n] !== undefined);
      const el = document.createElement(INLINE_TAG[group - 1]);
      el.textContent = match[group];
      parent.appendChild(el);
      last = INLINE.lastIndex;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
  }

  const BULLET = /^\s*([-*•]|\d+[.)])\s+/;

  // A pipe table's second line is its separator: |---|:--:|---|
  const TABLE_RULE = /^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$/;

  const isTable = (lines) =>
    lines.length >= 2 && lines[0].trim().startsWith("|") && TABLE_RULE.test(lines[1]);

  const cellsOf = (line) =>
    line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

  /** Build a real <table> from a markdown pipe table.
   *
   *  Without this the block fell through to the paragraph branch, which joins
   *  its lines with a space -- so a table arrived as one run-on line of pipes
   *  and dashes. Observed on a live web-fallback answer. Cells go through
   *  appendInline, so **bold** inside a cell works and nothing is parsed as
   *  markup. */
  function appendTable(container, lines) {
    const wrap = document.createElement("div");
    wrap.className = "table-scroll";
    const table = document.createElement("table");

    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const cell of cellsOf(lines[0])) {
      const th = document.createElement("th");
      appendInline(th, cell);
      headRow.appendChild(th);
    }
    head.appendChild(headRow);
    table.appendChild(head);

    const tbody = document.createElement("tbody");
    for (const line of lines.slice(2)) {
      const tr = document.createElement("tr");
      for (const cell of cellsOf(line)) {
        const td = document.createElement("td");
        appendInline(td, cell);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    container.appendChild(wrap);
  }

  function renderRich(container, text) {
    container.textContent = "";
    for (const block of String(text).split(/\n{2,}/)) {
      const lines = block.split("\n").filter((line) => line.trim() !== "");
      if (!lines.length) continue;

      if (isTable(lines)) {
        appendTable(container, lines);
      } else if (lines.every((line) => BULLET.test(line))) {
        const list = document.createElement("ul");
        for (const line of lines) {
          const item = document.createElement("li");
          appendInline(item, line.replace(BULLET, ""));
          list.appendChild(item);
        }
        container.appendChild(list);
      } else {
        const para = document.createElement("p");
        appendInline(para, lines.join(" "));
        container.appendChild(para);
      }
    }
  }

  /** Reveal an answer word by word, then re-render it with formatting.
   *  Capped at ~1.1s regardless of length: a long policy answer typed at a
   *  fixed rate is a wait, not a flourish. */
  function typeOut(container, text, done) {
    if (reduceMotion) {
      renderRich(container, text);
      done && done();
      return;
    }
    const tokens = String(text).match(/\S+\s*/g) || [text];
    const perFrame = Math.max(1, Math.ceil(tokens.length / 40));
    let index = 0;

    container.classList.add("is-typing");
    container.textContent = "";

    const step = () => {
      container.textContent += tokens.slice(index, index + perFrame).join("");
      index += perFrame;
      if (index < tokens.length) {
        setTimeout(step, 26);
      } else {
        container.classList.remove("is-typing");
        renderRich(container, text);
        done && done();
      }
    };
    step();
  }

  /* ── Small builders ────────────────────────────────────────────────────── */

  /** Bytes at a human scale. A 74-byte policy stub reported as "0k" reads like
   *  the upload was empty. */
  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function badge(className, label, value) {
    const node = el("span", `badge ${className}`);
    node.appendChild(document.createTextNode(label));
    if (value !== undefined && value !== null) {
      node.appendChild(el("b", null, value));
    }
    return node;
  }

  const SOURCE_LABEL = {
    private_kb: "internal policy",
    web_search: "public web",
    direct: "no lookup needed",
    insufficient_evidence: "no reliable evidence",
    error: "failed",
  };

  /* ── View switching ────────────────────────────────────────────────────── */

  function showView(name) {
    $$(".view").forEach((view) => view.classList.toggle("is-active", view.id === `view-${name}`));
    $$(".rail__btn[data-view]").forEach((btn) => {
      const active = btn.dataset.view === name;
      btn.classList.toggle("is-active", active);
      if (active) btn.setAttribute("aria-current", "page");
      else btn.removeAttribute("aria-current");
    });
    if (name === "audit") audit.load();
  }

  $$(".rail__btn[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });

  /** Replay a container's staggered entrance.
   *
   *  The lists in the inspector are rendered the moment an answer lands, into
   *  whichever panels are *not* the open tab — and an animation on a
   *  `display:none` element never runs. With `animation-fill-mode: both` the
   *  items are then stuck holding the `from` keyframe, so opening the tab
   *  showed an empty panel with the content sitting there at opacity 0.
   *  Gating the animation on a class, and adding it only while the container is
   *  on screen, is what makes the reveal happen when it is actually watched. */
  function stagger(container) {
    if (!container || reduceMotion || !container.offsetParent) return;
    container.classList.remove("stagger-in");
    void container.offsetWidth; // reflow, so the animation restarts
    container.classList.add("stagger-in");
  }

  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.toggle("is-active", t === tab));
      $$(".panel").forEach((p) => p.classList.toggle("is-active", p.id === `panel-${tab.dataset.tab}`));
      stagger($(`#panel-${tab.dataset.tab}`));
    });
  });

  /* ── Theme ─────────────────────────────────────────────────────────────── */

  const savedTheme = localStorage.getItem("hrcopilot.theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;

  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("hrcopilot.theme", next);
  });

  /* ── Health ────────────────────────────────────────────────────────────── */

  const health = {
    async poll() {
      const dot = $("#health-dot");
      const text = $("#health-text");
      try {
        const data = await api("/health");
        const missing = data.missing_secrets || [];
        dot.className = `pulse ${missing.length ? "is-warn" : "is-ok"}`;
        text.textContent = missing.length
          ? `degraded · missing ${missing.join(", ")}`
          : `${data.llm_provider} · ${data.llm_model}`;
        $("#health-pill").title =
          `env ${data.env} · index ${data.pinecone_index} · namespace ${data.private_namespace}\n` +
          `embeddings ${data.embedding_model} (${data.embedding_dim}d)\n` +
          `tracing ${data.tracing ? (data.tracing_redacted ? "on, redacted" : "on, UNREDACTED") : "off"}`;
        // /health is authoritative about the admin surface; the server-rendered
        // attribute is only the first paint.
        body.dataset.admin = data.admin_enabled ? "on" : "off";
      } catch {
        dot.className = "pulse is-bad";
        text.textContent = "unreachable";
      }
    },
  };

  /* ── The decision graph ────────────────────────────────────────────────── */

  // Trace entries are written by `note()` in app/agent/nodes.py as
  // "<tag>: <message>". The tag is the stable half, so the mapping keys on it.
  const TRACE_NODE = {
    Contextualiser: "contextualize",
    Router: "route",
    "KB Retriever": "retrieve",
    "KB Grader": "gradekb",
    Tavily: "web",
    "Web Grader": "gradeweb",
    Rewriter: "rewrite",
    "Generate/KB": "genkb",
    "Generate/Web": "genweb",
    "Generate/Direct": "direct",
    Fallback: "insufficient",
  };

  function splitStep(entry) {
    const at = String(entry).indexOf(":");
    if (at === -1) return { tag: "", message: String(entry) };
    return { tag: entry.slice(0, at).trim(), message: entry.slice(at + 1).trim() };
  }

  function stepKind(message) {
    if (/^failed \(|rate limited/i.test(message)) return "fail";
    if (/^good\b/.test(message)) return "good";
    if (/^weak\b/.test(message)) return "weak";
    return "step";
  }

  const graph = {
    reset() {
      $$("#graph .node").forEach((n) => n.classList.remove("is-live", "is-final"));
      $$("#graph .graph__edges path").forEach((p) => p.classList.remove("is-live"));
      $("#verdict").hidden = true;
    },

    /** Light the nodes the run visited, and the edges between them.
     *  The trace *is* the execution order, so consecutive pairs are exactly the
     *  edges traversed — including `rewrite -> retrieve`, the loop. */
    render(trace) {
      this.reset();

      const visited = [];
      for (const entry of trace || []) {
        const node = TRACE_NODE[splitStep(entry).tag];
        if (node && visited[visited.length - 1] !== node) visited.push(node);
      }
      if (!visited.length) return;

      // `contextualize` runs on every question but only *reports* when there
      // was history to resolve against — a first question costs no LLM call
      // there and writes no trace step. Lighting it regardless is the honest
      // picture: the node ran, it just had nothing to do.
      if (visited[0] !== "contextualize") visited.unshift("contextualize");

      const paint = (index) => {
        if (index >= visited.length) return;
        const node = $(`#graph .node[data-node="${visited[index]}"]`);
        if (node) {
          node.classList.add("is-live");
          if (index === visited.length - 1) node.classList.add("is-final");
        }
        const from = index === 0 ? "start" : visited[index - 1];
        const edge = $(`#graph path[data-edge="${from}-${visited[index]}"]`);
        if (edge) edge.classList.add("is-live");
        if (!reduceMotion) setTimeout(() => paint(index + 1), 130);
        else paint(index + 1);
      };
      paint(0);
    },

    verdict(result, latencyMs) {
      $("#v-route").textContent = result.route || "—";
      $("#v-kb").textContent = result.kb_grade || "—";
      $("#v-web").textContent = result.web_grade || "—";
      $("#v-chunks").textContent = result.kb_chunks ?? "—";
      $("#v-retry").textContent = result.retry_count ?? "—";
      $("#v-latency").textContent = latencyMs ? `${latencyMs} ms` : "—";
      $("#verdict").hidden = false;
    },
  };

  /* ── Trace panel ───────────────────────────────────────────────────────── */

  function renderTrace(trace) {
    const list = $("#trace");
    list.textContent = "";
    $("#trace-empty").hidden = Boolean(trace && trace.length);
    (trace || []).forEach((entry, index) => {
      const { tag, message } = splitStep(entry);
      const item = el("li", "trace__step");
      item.dataset.kind = stepKind(message);
      item.style.setProperty("--i", index);
      item.appendChild(el("div", "trace__tag", tag || "step"));
      item.appendChild(el("div", "trace__msg", message));
      list.appendChild(item);
    });
  }

  /* ── Sources panel ─────────────────────────────────────────────────────── */

  function renderSources(result) {
    const wrap = $("#sources");
    wrap.textContent = "";

    const sources = result.sources || [];
    const urls = result.web_urls || [];
    $("#sources-empty").hidden = Boolean(sources.length || urls.length);

    let index = 0;

    for (const source of sources) {
      const card = el("div", "source");
      card.style.setProperty("--i", index++);
      card.appendChild(el("div", "source__title", source.title || source.source));
      card.appendChild(el("div", "source__file", source.source));

      const meta = el("div", "source__meta");
      if (source.department) meta.appendChild(el("span", "tagline", source.department));
      if (source.doc_type) meta.appendChild(el("span", "tagline", source.doc_type));
      meta.appendChild(el("span", "tagline", "internal"));
      card.appendChild(meta);
      wrap.appendChild(card);
    }

    for (const url of urls) {
      const card = el("div", "source source--web");
      card.style.setProperty("--i", index++);
      card.appendChild(el("div", "source__title", "Public web result"));

      const link = el("a", null, url);
      link.href = url;
      link.target = "_blank";
      // noopener: the opened page must not get a handle on this one.
      link.rel = "noopener noreferrer";
      const holder = el("div", "source__file");
      holder.appendChild(link);
      card.appendChild(holder);

      const meta = el("div", "source__meta");
      meta.appendChild(el("span", "tagline", "external"));
      card.appendChild(meta);
      wrap.appendChild(card);
    }
  }

  /* ── Chat ──────────────────────────────────────────────────────────────── */

  const thread = $("#thread");
  const input = $("#question");
  const sendBtn = $("#send");

  const STAGES = [
    ["Routing", "deciding whether this needs a policy lookup"],
    ["Retrieving", "searching the approved knowledge base"],
    ["Grading", "checking the evidence actually answers it"],
    ["Composing", "writing the answer from what was found"],
  ];

  function scrollThread() {
    thread.scrollTop = thread.scrollHeight;
  }

  function addQuestion(text) {
    const wrap = el("div", "msg msg--me");
    wrap.appendChild(el("div", "msg__bubble", text));
    thread.appendChild(wrap);
    scrollThread();
  }

  function addThinking() {
    const node = el("div", "thinking");
    const orbit = el("div", "orbit");
    orbit.append(el("span"), el("span"), el("span"));

    const text = el("div", "thinking__text");
    const stage = el("span", "thinking__stage", STAGES[0][0] + "…");
    const sub = el("span", "thinking__sub", STAGES[0][1]);
    text.append(stage, sub);
    node.append(orbit, text);
    thread.appendChild(node);
    scrollThread();

    let i = 0;
    const timer = setInterval(() => {
      i = (i + 1) % STAGES.length;
      stage.textContent = STAGES[i][0] + "…";
      sub.textContent = STAGES[i][1];
    }, 1900);

    return {
      stop() {
        clearInterval(timer);
        node.remove();
      },
    };
  }

  function addAnswer(result, animate = true) {
    const wrap = el("div", "msg msg--bot");
    const card = el("div", "msg__card");

    const head = el("div", "msg__head");
    head.appendChild(el("div", "msg__avatar", "◆"));
    head.appendChild(el("span", "msg__who", "HR Copilot"));
    card.appendChild(head);

    const bodyEl = el("div", "msg__body");
    card.appendChild(bodyEl);

    const foot = el("div", "msg__foot");
    card.appendChild(foot);

    wrap.appendChild(card);
    thread.appendChild(wrap);

    const decorate = () => {
      const source = result.source_used || "unknown";
      foot.appendChild(badge(`badge--source badge--${source}`, SOURCE_LABEL[source] || source));

      // Shown only when the agent had to put a referent back, because that is
      // the one thing a reader cannot reconstruct from their own message.
      if (result.resolved_question) {
        const chip = el("span", "badge badge--resolved", "read as: ");
        chip.appendChild(el("b", null, result.resolved_question));
        foot.appendChild(chip);
      }

      if (result.kb_grade) foot.appendChild(badge(`badge--${result.kb_grade}`, "kb ", result.kb_grade));
      if (result.web_grade) foot.appendChild(badge(`badge--${result.web_grade}`, "web ", result.web_grade));
      if (result.retry_count) foot.appendChild(badge("", "rewrites ", result.retry_count));

      for (const cite of result.sources || []) {
        const chip = el("button", "cite-chip", cite.source);
        chip.type = "button";
        chip.title = "Show this document in the Evidence panel";
        chip.addEventListener("click", () => {
          $('.tab[data-tab="sources"]').click();
        });
        foot.appendChild(chip);
      }
      if (result.audit_id) {
        foot.appendChild(badge("", "audit #", result.audit_id));
      }
      scrollThread();
    };

    // Rehydrating a saved conversation replays many answers at once; typing
    // each one out would make switching chats take as long as asking did.
    if (animate) {
      typeOut(bodyEl, result.answer || "", decorate);
    } else {
      renderRich(bodyEl, result.answer || "");
      decorate();
    }

    scrollThread();
  }

  function addFailure(message) {
    const wrap = el("div", "msg msg--bot");
    const card = el("div", "msg__card");
    const head = el("div", "msg__head");
    head.appendChild(el("div", "msg__avatar", "!"));
    head.appendChild(el("span", "msg__who", "Could not answer"));
    card.appendChild(head);
    const bodyEl = el("div", "msg__body");
    renderRich(bodyEl, message);
    card.appendChild(bodyEl);
    wrap.appendChild(card);
    thread.appendChild(wrap);
    scrollThread();
  }

  /* ── Conversations ─────────────────────────────────────────────────────── */

  /**
   * Conversations live in this browser, not on the server.
   *
   * `/api/chat` is deliberately open — employees ask without logging in — so
   * there is no identity a server-side conversation could be scoped to. An
   * endpoint that listed conversations would either need auth this product does
   * not have, or would let anyone read anyone else's HR questions. Keeping the
   * thread client-side and posting it back with each turn avoids inventing a
   * half-authenticated store for the most sensitive data here.
   *
   * The server still gets the record it needs: `conversation_id` goes on the
   * audit row, so an admin can group a thread in `/api/audit` without any new
   * unauthenticated read path existing.
   *
   * localStorage rather than sessionStorage, deliberately, and the opposite
   * choice from the admin key: a conversation is the user's own work and should
   * survive closing the tab; the admin key is a credential and should not.
   */
  const KEY = "hrcopilot.conversations";
  const MAX_CONVERSATIONS = 30;
  const HISTORY_TURNS = 8; // matches normalise_history() on the server

  const chats = {
    items: [],
    activeId: null,

    load() {
      try {
        const raw = JSON.parse(localStorage.getItem(KEY) || "{}");
        this.items = Array.isArray(raw.items) ? raw.items : [];
        this.activeId = raw.activeId || null;
      } catch {
        // A corrupt or unreadable store must not leave the page blank.
        this.items = [];
        this.activeId = null;
      }
      if (!this.active()) this.create({ render: false });
    },

    save() {
      try {
        this.items = this.items.slice(0, MAX_CONVERSATIONS);
        localStorage.setItem(
          KEY,
          JSON.stringify({ items: this.items, activeId: this.activeId })
        );
      } catch {
        // Quota, or a browser blocking site data. The conversation still works
        // for this session; only its persistence is lost.
        toast("info", "This conversation will not be remembered — browser storage is full.");
      }
    },

    active() {
      return this.items.find((c) => c.id === this.activeId) || null;
    },

    create({ render = true } = {}) {
      const convo = {
        id: (crypto.randomUUID && crypto.randomUUID()) || String(Date.now()),
        title: "New conversation",
        updated: Date.now(),
        turns: [],
      };
      this.items.unshift(convo);
      this.activeId = convo.id;
      this.save();
      if (render) {
        renderConvos();
        openConversation(convo);
      }
      return convo;
    },

    remove(id) {
      this.items = this.items.filter((c) => c.id !== id);
      if (this.activeId === id) {
        // Never leave the app with no conversation open.
        this.activeId = this.items.length ? this.items[0].id : null;
        if (!this.activeId) this.create({ render: false });
      }
      this.save();
      renderConvos();
      openConversation(this.active());
    },

    select(id) {
      this.activeId = id;
      this.save();
      renderConvos();
      openConversation(this.active());
    },

    append(turn) {
      const convo = this.active() || this.create({ render: false });
      convo.turns.push(turn);
      convo.updated = Date.now();
      // The first thing asked names the thread; a conversation called "New
      // conversation" in a list of ten is no help to anyone.
      if (turn.role === "user" && convo.turns.filter((t) => t.role === "user").length === 1) {
        convo.title = turn.content.slice(0, 60);
      }
      // Most recently used first, which is the order a chat list wants.
      this.items = [convo, ...this.items.filter((c) => c.id !== convo.id)];
      this.save();
      renderConvos();
    },

    /** The turns to send as context: role and content only.
     *  Never the stored metadata — the server has no use for our badges, and
     *  the contextualise prompt is charged by the token. */
    historyFor() {
      const convo = this.active();
      if (!convo) return [];
      return convo.turns
        .slice(-HISTORY_TURNS)
        .map((t) => ({ role: t.role, content: t.content }));
    },
  };

  function relativeTime(ms) {
    const mins = Math.round((Date.now() - ms) / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.round(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.round(hours / 24)}d ago`;
  }

  function renderConvos() {
    const wrap = $("#convos");
    wrap.textContent = "";

    for (const convo of chats.items) {
      const row = el("div", `convo${convo.id === chats.activeId ? " is-active" : ""}`);

      const open = el("button", "convo__open");
      open.type = "button";
      open.appendChild(el("div", "convo__title", convo.title));
      const asked = convo.turns.filter((t) => t.role === "user").length;
      open.appendChild(
        el("div", "convo__meta",
          `${asked} question${asked === 1 ? "" : "s"} · ${relativeTime(convo.updated)}`)
      );
      open.addEventListener("click", () => chats.select(convo.id));

      const del = el("button", "convo__del", "×");
      del.type = "button";
      del.title = "Delete this conversation";
      del.setAttribute("aria-label", `Delete conversation: ${convo.title}`);
      del.addEventListener("click", (event) => {
        event.stopPropagation();
        // Deleting is irreversible and there is no server copy to recover from.
        if (asked && !window.confirm(`Delete "${convo.title}"? This cannot be undone.`)) {
          return;
        }
        chats.remove(convo.id);
        toast("info", "Conversation deleted.");
      });

      row.append(open, del);
      wrap.appendChild(row);
    }
  }

  /** Rebuild the thread for a conversation, without animating anything. */
  function openConversation(convo) {
    thread.textContent = "";
    if (!convo || !convo.turns.length) {
      thread.appendChild(buildWelcome());
      graph.reset();
      renderTrace([]);
      renderSources({});
      input.focus();
      return;
    }
    for (const turn of convo.turns) {
      if (turn.role === "user") addQuestion(turn.content);
      else if (turn.error) addFailure(turn.content);
      else addAnswer({ ...(turn.meta || {}), answer: turn.content }, false);
    }
    // The inspector shows the last answer, which is what "where am I?" means
    // after switching back into a conversation.
    const last = [...convo.turns].reverse().find((t) => t.role === "assistant" && !t.error);
    if (last && last.meta) {
      renderTrace(last.meta.trace);
      renderSources(last.meta);
      graph.render(last.meta.trace);
      graph.verdict(last.meta, last.meta.latency_ms);
    } else {
      graph.reset();
    }
    scrollThread();
  }

  async function ask(question) {
    const welcome = $("#welcome");
    if (welcome) welcome.remove();

    // Captured before the user's turn is appended: the server resolves the
    // follow-up against what came *before* it.
    const history = chats.historyFor();

    addQuestion(question);
    chats.append({ role: "user", content: question });
    input.value = "";
    autosize();
    updateCount();

    sendBtn.disabled = true;
    sendBtn.classList.add("is-busy");
    const spinner = addThinking();
    const started = performance.now();

    try {
      const result = await api("/api/chat", {
        method: "POST",
        json: {
          question,
          history,
          conversation_id: chats.activeId,
        },
      });
      const latency = Math.round(performance.now() - started);

      spinner.stop();
      addAnswer(result);
      chats.append({
        role: "assistant",
        content: result.answer || "",
        meta: { ...result, latency_ms: latency },
      });
      renderTrace(result.trace);
      renderSources(result);
      stagger($(".panel.is-active"));
      graph.render(result.trace);
      graph.verdict(result, latency);

      if (result.source_used === "insufficient_evidence") {
        toast("info", "No reliable evidence found — the agent said so rather than guessing.");
      } else if (result.source_used === "error") {
        toast("bad", "The answer could not be generated.");
      }
    } catch (error) {
      spinner.stop();
      const message = error.detail || "The request did not complete.";
      addFailure(message);
      // Stored too, so reopening the conversation does not silently lose the
      // fact that a question went unanswered.
      chats.append({ role: "assistant", content: message, error: true });
      toast("bad", message);
    } finally {
      sendBtn.classList.remove("is-busy");
      sendBtn.disabled = input.value.trim().length < 2;
      input.focus();
    }
  }

  /* Composer ---------------------------------------------------------------- */

  function autosize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  }

  function updateCount() {
    const length = input.value.length;
    $("#char-count").textContent = `${length} / 3000`;
    sendBtn.disabled = input.value.trim().length < 2;
  }

  input.addEventListener("input", () => {
    autosize();
    updateCount();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!sendBtn.disabled) $("#composer").requestSubmit();
    }
  });

  $("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (question.length >= 2) ask(question);
  });

  // Kept as a detached clone of the server-rendered markup, so an empty
  // conversation can put the welcome back without this file owning a second
  // copy of the copy. Cloned again per use: a node can only be in one place.
  const WELCOME = $("#welcome") ? $("#welcome").cloneNode(true) : null;

  function bindSuggestions(root) {
    $$(".suggest__item", root).forEach((btn) => {
      btn.addEventListener("click", () => ask(btn.dataset.q));
    });
  }

  function buildWelcome() {
    if (!WELCOME) return el("div");
    const node = WELCOME.cloneNode(true);
    bindSuggestions(node); // listeners do not survive cloneNode
    return node;
  }

  bindSuggestions(document);

  /* ── Confetti ──────────────────────────────────────────────────────────── */

  function celebrate(anchor) {
    if (reduceMotion) return;
    const rect = anchor.getBoundingClientRect();
    const originX = rect.left + rect.width / 2;
    const originY = rect.top + rect.height / 2;
    const colours = ["#7c5cff", "#22d3ee", "#f472b6", "#34d399", "#fbbf24"];

    for (let i = 0; i < 26; i += 1) {
      const bit = document.createElement("i");
      bit.className = "confetti";
      bit.style.left = `${originX}px`;
      bit.style.top = `${originY}px`;
      bit.style.background = colours[i % colours.length];
      document.body.appendChild(bit);

      const angle = (Math.PI * 2 * i) / 26 + Math.random() * 0.4;
      const distance = 90 + Math.random() * 150;

      bit.animate(
        [
          { transform: "translate(-50%, -50%) scale(1) rotate(0deg)", opacity: 1 },
          {
            transform:
              `translate(${Math.cos(angle) * distance - 50}%, ` +
              `${Math.sin(angle) * distance + 120}%) scale(0.3) rotate(${Math.random() * 720}deg)`,
            opacity: 0,
          },
        ],
        { duration: 900 + Math.random() * 500, easing: "cubic-bezier(.15,.7,.3,1)" }
      ).onfinish = () => bit.remove();
    }
  }

  /* ── Upload ────────────────────────────────────────────────────────────── */

  const drop = $("#drop");
  const fileInput = $("#file");
  const uploadBtn = $("#do-upload");
  let chosen = null;

  function chooseFile(file) {
    chosen = file || null;
    const label = $("#drop-file");
    if (!chosen) {
      label.hidden = true;
      drop.classList.remove("is-loaded");
      uploadBtn.disabled = true;
      return;
    }
    label.textContent = `${chosen.name} · ${formatBytes(chosen.size)}`;
    label.hidden = false;
    drop.classList.add("is-loaded");
    uploadBtn.disabled = false;
  }

  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((type) =>
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add("is-over");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.remove("is-over");
    })
  );
  drop.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) chooseFile(file);
  });

  function addReceipt(ok, name, detail, stats) {
    const card = el("div", `receipt${ok ? "" : " receipt--bad"}`);
    card.appendChild(el("div", "receipt__seal", ok ? "✓" : "✕"));

    const middle = el("div");
    middle.appendChild(el("div", "receipt__name", name));
    middle.appendChild(el("div", "receipt__meta", detail));
    card.appendChild(middle);

    const box = el("div", "receipt__stats");
    for (const [value, label] of stats || []) {
      const chip = el("div", "stat-chip");
      chip.appendChild(el("b", null, value));
      chip.appendChild(el("span", null, label));
      box.appendChild(chip);
    }
    card.appendChild(box);

    $("#receipts").prepend(card);
    if (ok) celebrate(card);
  }

  uploadBtn.addEventListener("click", async () => {
    if (!chosen) return;
    if (!adminKey.get()) {
      toast("info", "Set the admin key first — uploading writes to the knowledge base.");
      $("#admin-dialog").showModal();
      return;
    }

    const form = new FormData();
    form.append("file", chosen);
    form.append("department", $("#department").value.trim() || "general");

    uploadBtn.disabled = true;
    uploadBtn.querySelector(".btn__label").textContent = "Indexing…";

    try {
      const report = await api("/api/upload", { method: "POST", form, admin: true });
      addReceipt(true, report.file, `${report.department} · ${report.namespace}`, [
        [report.chunks, "chunks"],
        [report.vectors_in_namespace, "vectors"],
        [formatBytes(report.bytes_received), "size"],
      ]);
      toast("ok", `${report.file} indexed — ${report.chunks} chunks are now searchable.`);
      chooseFile(null);
      fileInput.value = "";
    } catch (error) {
      addReceipt(false, chosen.name, error.detail || "Upload failed", []);
      toast("bad", error.detail || "Upload failed.");
    } finally {
      uploadBtn.querySelector(".btn__label").textContent = "Index document";
      uploadBtn.disabled = !chosen;
    }
  });

  /* ── Audit ─────────────────────────────────────────────────────────────── */

  const audit = {
    async load() {
      if (!adminKey.get()) {
        $("#audit-empty").textContent = "Set the admin key to read the audit log.";
        $("#audit-empty").hidden = false;
        $("#audit-body").textContent = "";
        $("#stats").textContent = "";
        return;
      }
      try {
        const [page, stats] = await Promise.all([
          api("/api/audit?limit=50", { admin: true }),
          api("/api/audit/stats", { admin: true }),
        ]);
        this.renderStats(stats);
        this.renderRows(page.entries || []);
      } catch (error) {
        $("#stats").textContent = "";
        $("#audit-body").textContent = "";
        $("#audit-empty").textContent = error.detail || "Could not read the audit log.";
        $("#audit-empty").hidden = false;
      }
    },

    renderStats(stats) {
      const wrap = $("#stats");
      wrap.textContent = "";

      const cards = [
        ["Answered", String(stats.total ?? 0), null],
        ["Grounded", `${Math.round((stats.grounded_rate || 0) * 100)}%`, stats.grounded_rate || 0],
        ["Rewritten", String(stats.rewritten ?? 0), null],
        ["Avg latency", stats.average_latency_ms ? `${stats.average_latency_ms} ms` : "—", null],
      ];

      cards.forEach(([label, value, ratio], index) => {
        const card = el("div", "stat");
        card.style.setProperty("--i", index);
        card.appendChild(el("div", "stat__k", label));
        card.appendChild(el("div", `stat__v${label === "Grounded" ? " grad" : ""}`, value));
        if (ratio !== null) {
          const meter = el("div", "meter");
          const fill = el("i");
          fill.style.width = `${Math.round(ratio * 100)}%`;
          meter.appendChild(fill);
          card.appendChild(meter);
        }
        wrap.appendChild(card);
      });

      const bySource = stats.by_source || {};
      Object.keys(bySource).forEach((key, index) => {
        const card = el("div", "stat");
        card.style.setProperty("--i", index + 4);
        card.appendChild(el("div", "stat__k", SOURCE_LABEL[key] || key));
        card.appendChild(el("div", "stat__v", String(bySource[key])));
        wrap.appendChild(card);
      });

      stagger(wrap);
    },

    renderRows(entries) {
      const tbody = $("#audit-body");
      tbody.textContent = "";
      $("#audit-empty").hidden = entries.length > 0;
      $("#audit-empty").textContent = "Nothing logged yet.";

      for (const entry of entries) {
        const row = document.createElement("tr");

        row.appendChild(el("td", "num", `#${entry.id}`));
        row.appendChild(el("td", "num", (entry.asked_at || "").replace("T", " ").replace("+00:00", "")));
        row.appendChild(el("td", "q", entry.question));

        const source = document.createElement("td");
        const used = entry.source_used || "unknown";
        source.appendChild(badge(`badge--${used}`, SOURCE_LABEL[used] || used));
        row.appendChild(source);

        const grades = document.createElement("td");
        if (entry.kb_grade) grades.appendChild(badge(`badge--${entry.kb_grade}`, "kb ", entry.kb_grade));
        if (entry.web_grade) grades.appendChild(badge(`badge--${entry.web_grade}`, "web ", entry.web_grade));
        row.appendChild(grades);

        row.appendChild(el("td", "num", entry.latency_ms ?? "—"));

        row.title = (entry.trace || []).join("\n");
        tbody.appendChild(row);
      }
    },
  };

  $("#audit-refresh").addEventListener("click", () => audit.load());

  /* ── Admin key dialog ──────────────────────────────────────────────────── */

  const dialog = $("#admin-dialog");

  $("#admin-open").addEventListener("click", () => {
    $("#admin-key").value = adminKey.get();
    dialog.showModal();
  });

  $("#admin-form").addEventListener("submit", (event) => {
    // A dialog form submits with the clicked button's `value`; Cancel must not
    // overwrite a stored key with whatever is in the box.
    if (event.submitter && event.submitter.value !== "save") return;
    const value = $("#admin-key").value.trim();
    if (!value) {
      adminKey.clear();
      toast("info", "Admin key cleared.");
      return;
    }
    adminKey.set(value);
    toast("ok", "Admin key saved for this tab.");
    if ($("#view-audit").classList.contains("is-active")) audit.load();
  });

  $("#admin-clear").addEventListener("click", () => {
    adminKey.clear();
    $("#admin-key").value = "";
    toast("info", "Admin key forgotten.");
  });

  /* ── Boot ──────────────────────────────────────────────────────────────── */

  $("#chat-new").addEventListener("click", () => {
    // An untouched empty conversation is already a new one; making a second
    // would just leave a row of identical empties in the list.
    const current = chats.active();
    if (current && !current.turns.length) {
      toast("info", "This conversation is already empty.");
      input.focus();
      return;
    }
    chats.create();
    toast("ok", "New conversation started.");
  });

  chats.load();
  renderConvos();
  openConversation(chats.active());

  health.poll();
  setInterval(() => health.poll(), 30000);
  updateCount();
  input.focus();
})();
