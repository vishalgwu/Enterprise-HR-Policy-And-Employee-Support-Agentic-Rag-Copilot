"""The graph's nodes, built as closures over shared clients.

`build_nodes` takes the LLM, retriever and search tool and closes over them, so
one run constructs each once and a test can inject fakes. That is how the
rewrite loop and the insufficient-evidence path are exercised deterministically,
without waiting for a real question that happens to fail twice.

**Every node degrades; none may raise.** A node that throws aborts the whole
run, so one flaky Pinecone or Tavily call would cost an answer the other source
could have given. `guard()` wraps each network call: retrieval failure becomes
zero chunks (which routes to the web fallback), search failure becomes empty
evidence, a rewrite failure keeps the original query while still incrementing
`retry_count` so a broken LLM cannot spin the loop, and a generation failure
returns GENERATION_FAILED with `source_used="error"` rather than an empty
answer.

Every node also returns its own steps in `trace`, which the `add` reducer on
`AgentState` appends rather than overwrites. `note()` both logs a step and
returns it, so the console line and the state entry cannot drift apart, and
`guard()` records a degraded call there too -- a run that answered from the web
because Pinecone was down looks identical to one that answered from the web
because the evidence was weak, unless the trace says which it was.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from app.agent.decisions import get_evidence_grader, get_router, grade_evidence
from app.agent.decisions import route_question as classify_route
from app.agent.prompts import (
    CONTEXTUALIZE_PROMPT,
    DIRECT_ANSWER_PROMPT,
    GENERATION_FAILED,
    INSUFFICIENT_ANSWER,
    KB_ANSWER_PROMPT,
    REWRITE_PROMPT,
    WEB_ANSWER_PROMPT,
)
from app.agent.schemas import AgentState
from app.core.config import Settings, get_logger, get_settings
from app.rag.clients import (
    get_embeddings,
    get_llm,
    get_web_search,
    is_rate_limit,
    message_text,
    web_result_urls,
    web_results_to_text,
)
from app.rag.loaders import cite_sources, format_documents
from app.rag.retrieval import get_kb_retriever

_log = get_logger("graph")

# Node names, in one place. `build_graph` wires from this tuple and the
# conditional path maps are checked against it, so a typo fails at build time
# rather than on whichever question first reaches the mistyped edge.
NODE_NAMES = (
    "contextualize",
    "route_question",
    "retrieve_kb",
    "grade_kb_evidence",
    "search_web",
    "grade_web_evidence",
    "rewrite_query",
    "generate_from_kb",
    "generate_from_web",
    "direct_answer",
    "answer_insufficient",
)

TERMINAL_NODES = (
    "generate_from_kb",
    "generate_from_web",
    "direct_answer",
    "answer_insufficient",
)


def build_nodes(
    llm: Any = None,
    retriever: Any = None,
    web_search: Any = None,
    embedding: Any = None,
    verbose: bool = True,
    settings: Settings | None = None,
) -> dict[str, Callable]:
    """Construct every node once, closed over shared clients.

    Each dependency is injectable and each is built only if not supplied, so a
    test can pass fakes without any network access at all.
    """
    settings = settings or get_settings()
    llm = llm or get_llm(settings=settings)
    if retriever is None:
        embedding = embedding or get_embeddings()
        retriever = get_kb_retriever(embedding=embedding, settings=settings)
    if web_search is None:
        web_search = get_web_search(settings=settings)

    router = get_router(llm)
    grader = get_evidence_grader(llm)
    max_retries = settings.max_retries

    def note(tag: str, message: Any) -> list[str]:
        """Log one step and return it for the node to put in `trace`.

        `verbose` silences the console line only. The trace is state the API
        returns and the UI renders, so it is recorded either way.
        """
        if verbose:
            _log.info("[%s] %s", tag, message)
        return [f"{tag}: {message}"]

    def guard(
        tag: str,
        action: Callable[[], Any],
        fallback: Any,
        steps: list[str] | None = None,
    ) -> Any:
        """Run a network call; on failure log it and degrade to `fallback`.

        Pass `steps` to have the degradation recorded in the trace as well: a
        caller reading the answer cannot otherwise tell an outage from a real
        absence of evidence.
        """
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - a node must never abort the graph
            if is_rate_limit(exc):
                _log.error(
                    "[%s] RATE LIMITED by the provider - results below are "
                    "degraded, not a knowledge gap (%s)",
                    tag,
                    str(exc)[:160],
                )
                detail = "rate limited by the provider; results are degraded"
            else:
                _log.warning(
                    "[%s] failed (%s: %s); continuing with fallback",
                    tag,
                    type(exc).__name__,
                    str(exc)[:200],
                )
                detail = f"failed ({type(exc).__name__}); continuing with fallback"
            if steps is not None:
                steps.append(f"{tag}: {detail}")
            return fallback

    def asked(state: AgentState) -> str:
        """The question every node downstream of `contextualize` should use.

        `question` stays the employee's verbatim wording for the audit log and
        the UI; this is that question resolved against the conversation. They
        are the same string whenever there is no history, which is why a
        single-turn run is unchanged by any of this.
        """
        return state.get("standalone_question") or state["question"]

    # --- 0. Resolve a follow-up against the conversation ----------------------
    def contextualize(state: AgentState) -> dict[str, Any]:
        """Turn "what about contractors?" into a question that stands alone.

        **No history means no LLM call.** The first question of every
        conversation takes this node, and paying a round trip to be told a
        standalone question is already standalone would tax the common case to
        serve the rarer one.

        Degrades to the original wording, like every other node: a failure here
        costs the follow-up its context, not the answer.
        """
        question = state["question"]
        history = state.get("history") or []
        if not history:
            return {"standalone_question": question, "current_query": question}

        transcript = "\n".join(
            f"{'Employee' if turn['role'] == 'user' else 'Assistant'}: "
            f"{turn['content']}"
            for turn in history
        )
        steps: list[str] = []
        resolved = guard(
            "Contextualiser",
            lambda: message_text(
                (CONTEXTUALIZE_PROMPT | llm).invoke(
                    {"history": transcript, "question": question}
                )
            ),
            question,
            steps,
        ).strip()
        resolved = resolved or question

        if resolved != question:
            steps += note("Contextualiser", f"resolved to {resolved!r}")
        else:
            steps += note("Contextualiser", "already standalone")
        return {
            "standalone_question": resolved,
            "current_query": resolved,
            "trace": steps,
        }

    # --- 1. Route ------------------------------------------------------------
    def route_question(state: AgentState) -> dict[str, Any]:
        question = asked(state)
        # classify_route survives a Groq tool_use_failed 400, which the router
        # hits often enough to matter: it is on the path of every question.
        route = classify_route(question, router)
        return {
            "route": route,
            "current_query": question,
            "trace": note("Router", route),
        }

    def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
        return "retrieve_kb" if state.get("route") == "kb" else "direct_answer"

    # --- 2. Retrieve from the private KB -------------------------------------
    def retrieve_kb(state: AgentState) -> dict[str, Any]:
        query = state["current_query"]
        # A Pinecone outage degrades to "no evidence", which routes to the web
        # fallback -- the same path a genuinely empty retrieval takes.
        steps: list[str] = []
        docs = guard("KB Retriever", lambda: retriever.invoke(query), [], steps)
        steps += note("KB Retriever", f"{len(docs)} chunk(s) for {query!r}")
        # Citations are written beside the chunks they describe, so a rewrite
        # that replaces `kb_docs` replaces them in the same update.
        return {"kb_docs": docs, "citations": cite_sources(docs), "trace": steps}

    # --- 3. Grade private KB evidence ----------------------------------------
    def grade_kb_evidence(state: AgentState) -> dict[str, Any]:
        docs = state.get("kb_docs") or []
        # The similarity gate already returns nothing for out-of-domain
        # questions; grading an empty context would just spend a call to be told
        # "weak".
        if not docs:
            return {
                "kb_grade": "weak",
                "trace": note("KB Grader", "weak (nothing passed the similarity gate)"),
            }

        grade = grade_evidence(asked(state), format_documents(docs), grader)
        return {"kb_grade": grade, "trace": note("KB Grader", grade)}

    def decide_after_kb_grade(
        state: AgentState,
    ) -> Literal["generate_from_kb", "search_web"]:
        return "generate_from_kb" if state.get("kb_grade") == "good" else "search_web"

    # --- 4. Tavily web search fallback ---------------------------------------
    def search_web(state: AgentState) -> dict[str, Any]:
        query = state["current_query"]
        steps = note("Tavily", f"searching {query!r}")
        # This node is itself the fallback, so it needs one of its own: a Tavily
        # outage must degrade to empty evidence and let the graph rewrite or
        # admit defeat, not abort the run.
        raw = guard("Tavily", lambda: web_search.invoke({"query": query}), None, steps)
        web_text = web_results_to_text(raw)
        steps += note("Tavily", f"{len(web_text)} characters")
        return {
            "web_results": web_text,
            "web_urls": web_result_urls(raw),
            "trace": steps,
        }

    # --- 5. Grade web evidence -----------------------------------------------
    def grade_web_evidence(state: AgentState) -> dict[str, Any]:
        web_results = state.get("web_results") or ""
        if not web_results.strip():
            return {
                "web_grade": "weak",
                "trace": note("Web Grader", "weak (search returned nothing)"),
            }

        grade = grade_evidence(asked(state), web_results, grader)
        return {"web_grade": grade, "trace": note("Web Grader", grade)}

    def decide_after_web_grade(
        state: AgentState,
    ) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
        if state.get("web_grade") == "good":
            return "generate_from_web"
        if state.get("retry_count", 0) < max_retries:
            return "rewrite_query"
        return "answer_insufficient"

    # --- 6. Rewrite the query ------------------------------------------------
    def rewrite_query(state: AgentState) -> dict[str, Any]:
        # Rewrite from the original wording, not from an earlier rewrite, so
        # successive attempts cannot drift away from what was asked.
        question = asked(state)
        steps: list[str] = []
        rewritten = guard(
            "Rewriter",
            lambda: message_text((REWRITE_PROMPT | llm).invoke({"question": question})),
            question,
            steps,
        ).strip()
        # An empty rewrite would retrieve nothing at all; keep the original.
        rewritten = rewritten or question
        steps += note("Rewriter", rewritten)
        # retry_count increments even when the rewrite failed, so a broken LLM
        # cannot spin this loop.
        return {
            "current_query": rewritten,
            "retry_count": state.get("retry_count", 0) + 1,
            "trace": steps,
        }

    # --- 7/8/9. Generation ---------------------------------------------------
    def _generate(
        tag: str, prompt: Any, variables: dict[str, Any], source_used: str
    ) -> dict[str, Any]:
        """Render one answer, reporting an honest failure rather than raising."""
        steps: list[str] = []
        answer = guard(
            tag, lambda: message_text((prompt | llm).invoke(variables)), "", steps
        )
        if not answer.strip():
            return {
                "answer": GENERATION_FAILED,
                "source_used": "error",
                "trace": steps + note(tag, "produced no answer; reporting an error"),
            }
        return {
            "answer": answer,
            "source_used": source_used,
            "trace": steps + note(tag, f"answered from {source_used}"),
        }

    def generate_from_kb(state: AgentState) -> dict[str, Any]:
        return _generate(
            "Generate/KB",
            KB_ANSWER_PROMPT,
            {
                "question": asked(state),
                "context": format_documents(state.get("kb_docs") or []),
            },
            "private_kb",
        )

    def generate_from_web(state: AgentState) -> dict[str, Any]:
        return _generate(
            "Generate/Web",
            WEB_ANSWER_PROMPT,
            {"question": asked(state), "context": state.get("web_results") or ""},
            "web_search",
        )

    def direct_answer(state: AgentState) -> dict[str, Any]:
        return _generate(
            "Generate/Direct",
            DIRECT_ANSWER_PROMPT,
            {"question": asked(state)},
            "direct",
        )

    # --- 10. Insufficient evidence -------------------------------------------
    def answer_insufficient(state: AgentState) -> dict[str, Any]:
        return {
            "answer": INSUFFICIENT_ANSWER,
            "source_used": "insufficient_evidence",
            "trace": note("Fallback", "no sufficient evidence in KB or web"),
        }

    return {
        "contextualize": contextualize,
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb_evidence": grade_kb_evidence,
        "search_web": search_web,
        "grade_web_evidence": grade_web_evidence,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "answer_insufficient": answer_insufficient,
        "route_after_router": route_after_router,
        "decide_after_kb_grade": decide_after_kb_grade,
        "decide_after_web_grade": decide_after_web_grade,
    }
