"""The agentic RAG graph: retrieve, grade, rewrite, fall back, answer.

    START -> route
      route == direct                       -> direct_answer            -> END
      route == kb                           -> retrieve_kb
    retrieve_kb                             -> grade_kb
      kb_grade == good                      -> generate_from_kb         -> END
      kb_grade == weak                      -> search_web
    search_web                              -> grade_web
      web_grade == good                     -> generate_from_web        -> END
      web_grade == weak, retries left       -> rewrite_query -> retrieve_kb
      web_grade == weak, no retries         -> answer_insufficient      -> END

The rewrite loops back to the private KB rather than straight to the web,
because a vocabulary mismatch is the most common cause of a weak retrieval and
internal policy is the preferred source. `retry_count` bounds the loop.

Nodes are closures over the LLM, retriever and search tool so a run constructs
each once, and tests can inject fakes.
"""

import sys

if __package__ in (None, ""):  # allow `python Rag/graph.py`
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from Rag import config
from Rag.clients import (
    get_embeddings,
    get_llm,
    get_web_search,
    is_rate_limit,
    web_results_to_text,
)
from Rag.config import get_logger
from Rag.loaders import format_documents
from Rag.private_kb import get_kb_retriever
from Rag.schemas import (
    MAX_RETRIES,
    AgentState,
    get_evidence_grader,
    get_router,
    grade_evidence,
    initial_state,
)
from Rag.schemas import route_question as classify_route

_log = get_logger("graph")

INSUFFICIENT_ANSWER = (
    "I could not find enough reliable evidence in the private knowledge base or "
    "in web search results to answer this confidently. Please rephrase the "
    "question, or point me at the specific policy document that covers it."
)

KB_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer employee questions from internal HR policy.\n"
            "Use ONLY the supplied context. Never invent a number, date or "
            "entitlement that is not in it.\n"
            "If the context covers only part of the question, answer that part "
            "and say plainly which part it does not cover.\n"
            "Write for a colleague, not a lawyer: lead with the answer, then the "
            "detail. Cite the source document names you used.\n"
            "End with: Source: Private KB.",
        ),
        ("human", "Question:\n{question}\n\nPrivate KB context:\n{context}"),
    ]
)

WEB_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Internal HR policy did not cover this question, so the answer comes "
            "from a public web search.\n"
            "Use ONLY the supplied search results. Never invent details.\n"
            "Open by saying this is general public information and not this "
            "company's policy, so the reader does not mistake it for an "
            "entitlement. Include the most useful URLs.\n"
            "End with: Source: Web Search.",
        ),
        ("human", "Question:\n{question}\n\nWeb search context:\n{context}"),
    ]
)

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an internal HR assistant. This message needs no policy "
            "lookup. Reply in one or two warm, natural sentences. Do not invent "
            "policy. If the person seems to want something, invite the question.",
        ),
        ("human", "{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite an employee's question so it retrieves better from an HR "
            "policy knowledge base and from web search.\n"
            "Preserve the original intent exactly. Prefer the formal vocabulary "
            "a policy document would use over casual phrasing. Keep it one "
            "sentence.\n"
            "Do not answer the question. Return only the rewritten query.",
        ),
        ("human", "{question}"),
    ]
)


GENERATION_FAILED = (
    "I could not produce an answer just now because the language model was "
    "unreachable. The question was understood; please try again shortly."
)


def build_nodes(llm=None, retriever=None, web_search=None, embedding=None, verbose=True):
    """Construct every node once, closed over shared clients.

    Every node that touches the network degrades instead of raising. A node that
    throws aborts the whole graph, which would mean one flaky Pinecone or Tavily
    call costs the employee an answer that the other source could have given.
    """
    llm = llm or get_llm()
    embedding = embedding or get_embeddings()
    retriever = retriever if retriever is not None else get_kb_retriever(embedding=embedding)
    web_search = web_search if web_search is not None else get_web_search()

    router = get_router(llm)
    grader = get_evidence_grader(llm)

    def log(tag, message):
        if verbose:
            _log.info("[%s] %s", tag, message)

    def guard(tag, action, fallback):
        """Run a network call; on failure log it and degrade to `fallback`."""
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - a node must never abort the graph
            if is_rate_limit(exc):
                # Called out at error level: every downstream grade will read
                # "weak" and the run will look like a knowledge gap rather than
                # an exhausted quota.
                _log.error("[%s] RATE LIMITED by the provider - results below are "
                           "degraded, not a knowledge gap (%s)", tag, str(exc)[:160])
            else:
                _log.warning("[%s] failed (%s: %s); continuing with fallback",
                             tag, type(exc).__name__, str(exc)[:200])
            return fallback

    # --- Node 1: route -------------------------------------------------------
    def route_question(state: AgentState):
        question = state["question"]
        # classify_route survives a Groq tool_use_failed 400, which the router
        # hits often enough to matter: it is on the path of every question.
        route = classify_route(question, router)
        log("Router", route)
        return {"route": route, "current_query": question}

    def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
        return "retrieve_kb" if state.get("route") == "kb" else "direct_answer"

    # --- Node 2: retrieve from the private KB --------------------------------
    def retrieve_kb(state: AgentState):
        query = state["current_query"]
        # A Pinecone outage degrades to "no evidence", which routes to the web
        # fallback -- the same path an genuinely empty retrieval takes.
        docs = guard("KB Retriever", lambda: retriever.invoke(query), [])
        log("KB Retriever", f"{len(docs)} chunk(s) for {query!r}")
        return {"kb_docs": docs}

    # --- Node 3: grade private KB evidence -----------------------------------
    def grade_kb_evidence(state: AgentState):
        docs = state.get("kb_docs") or []
        # The similarity gate already returns nothing for out-of-domain
        # questions; grading an empty context would just spend a call to be
        # told "weak".
        if not docs:
            log("KB Grader", "weak (no chunks passed the similarity gate)")
            return {"kb_grade": "weak"}

        grade = grade_evidence(state["question"], format_documents(docs), grader)
        log("KB Grader", grade)
        return {"kb_grade": grade}

    def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
        return "generate_from_kb" if state.get("kb_grade") == "good" else "search_web"

    # --- Node 4: Tavily web search fallback ----------------------------------
    def search_web(state: AgentState):
        query = state["current_query"]
        log("Tavily", f"searching {query!r}")
        # This node is itself the fallback, so it needs one of its own: a Tavily
        # outage must degrade to empty evidence and let the graph rewrite or
        # admit defeat, not abort the run.
        raw = guard("Tavily", lambda: web_search.invoke({"query": query}), None)
        web_text = web_results_to_text(raw)
        log("Tavily", f"{len(web_text)} characters")
        return {"web_results": web_text}

    # --- Node 5: grade web evidence ------------------------------------------
    def grade_web_evidence(state: AgentState):
        web_results = state.get("web_results") or ""
        if not web_results.strip():
            log("Web Grader", "weak (search returned nothing)")
            return {"web_grade": "weak"}

        grade = grade_evidence(state["question"], web_results, grader)
        log("Web Grader", grade)
        return {"web_grade": grade}

    def decide_after_web_grade(
        state: AgentState,
    ) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
        if state.get("web_grade") == "good":
            return "generate_from_web"
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "rewrite_query"
        return "answer_insufficient"

    # --- Node 6: rewrite the query -------------------------------------------
    def rewrite_query(state: AgentState):
        # Rewrite from the original wording, not from an earlier rewrite, so
        # successive attempts cannot drift away from what was asked.
        question = state["question"]
        rewritten = guard(
            "Rewriter",
            lambda: (REWRITE_PROMPT | llm).invoke({"question": question}).content.strip(),
            question,
        )
        # An empty rewrite would retrieve nothing at all; keep the original.
        rewritten = rewritten or question
        log("Rewriter", rewritten)
        # retry_count increments even when the rewrite failed, so a broken LLM
        # cannot spin this loop.
        return {"current_query": rewritten, "retry_count": state.get("retry_count", 0) + 1}

    # --- Node 7: generate from the private KB --------------------------------
    def _generate(tag, prompt, variables, source_used):
        """Render one answer, reporting an honest failure rather than raising."""
        answer = guard(tag, lambda: (prompt | llm).invoke(variables).content, None)
        if not answer or not answer.strip():
            return {"answer": GENERATION_FAILED, "source_used": "error"}
        return {"answer": answer, "source_used": source_used}

    def generate_from_kb(state: AgentState):
        return _generate(
            "Generate/KB",
            KB_ANSWER_PROMPT,
            {
                "question": state["question"],
                "context": format_documents(state.get("kb_docs") or []),
            },
            "private_kb",
        )

    # --- Node 8: generate from web search ------------------------------------
    def generate_from_web(state: AgentState):
        return _generate(
            "Generate/Web",
            WEB_ANSWER_PROMPT,
            {"question": state["question"], "context": state.get("web_results") or ""},
            "web_search",
        )

    # --- Node 9: direct answer -----------------------------------------------
    def direct_answer(state: AgentState):
        return _generate(
            "Generate/Direct",
            DIRECT_ANSWER_PROMPT,
            {"question": state["question"]},
            "direct",
        )

    # --- Node 10: insufficient evidence --------------------------------------
    def answer_insufficient(state: AgentState):
        log("Fallback", "no sufficient evidence in KB or web")
        return {"answer": INSUFFICIENT_ANSWER, "source_used": "insufficient_evidence"}

    return {
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


def build_graph(llm=None, retriever=None, web_search=None, embedding=None, verbose=True):
    """Wire and compile the agent graph."""
    n = build_nodes(llm, retriever, web_search, embedding, verbose)

    builder = StateGraph(AgentState)
    for name in (
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
    ):
        builder.add_node(name, n[name])

    builder.add_edge(START, "route_question")
    # The path maps are explicit rather than inferred from the condition's
    # return annotation: they are what the rendered diagram draws, and they make
    # a branch that returns an unmapped name fail at compile time instead of at
    # run time on whichever question happens to reach it first.
    builder.add_conditional_edges(
        "route_question",
        n["route_after_router"],
        {"retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"},
    )
    builder.add_edge("retrieve_kb", "grade_kb_evidence")
    builder.add_conditional_edges(
        "grade_kb_evidence",
        n["decide_after_kb_grade"],
        {"generate_from_kb": "generate_from_kb", "search_web": "search_web"},
    )
    builder.add_edge("search_web", "grade_web_evidence")
    builder.add_conditional_edges(
        "grade_web_evidence",
        n["decide_after_web_grade"],
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "answer_insufficient": "answer_insufficient",
        },
    )
    # A rewrite retries the private KB first: internal policy is the preferred
    # source, and a vocabulary mismatch is the likeliest cause of a weak hit.
    builder.add_edge("rewrite_query", "retrieve_kb")
    for terminal in (
        "generate_from_kb",
        "generate_from_web",
        "direct_answer",
        "answer_insufficient",
    ):
        builder.add_edge(terminal, END)

    return builder.compile()


def ask(question: str, graph=None, verbose: bool = True) -> AgentState:
    """Run one question through the graph and return the final state.

    Raises ValueError on an empty question rather than spending a routing call
    to discover there was nothing to answer.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    graph = graph or build_graph(verbose=verbose)
    return graph.invoke(initial_state(question.strip()))


RULE = "=" * 86
THIN = "-" * 86


def print_result(result: AgentState, show_sources: bool = True) -> None:
    """Print a question, how it was answered, and the evidence behind it."""
    kb_docs = result.get("kb_docs") or []
    web_results = result.get("web_results") or ""

    print(RULE)
    print(f"QUESTION      {result.get('question', '')}")
    if result.get("current_query") and result["current_query"] != result.get("question"):
        print(f"REWRITTEN AS  {result['current_query']}")
    print(THIN)
    print(f"ROUTE         {result.get('route', '-')}")
    print(f"SOURCE USED   {result.get('source_used', '-')}")
    print(f"RETRIES       {result.get('retry_count', 0)}")
    print(f"KB EVIDENCE   {len(kb_docs)} chunk(s), graded "
          f"{result.get('kb_grade', '-')}")
    print(f"WEB EVIDENCE  {len(web_results)} chars, graded "
          f"{result.get('web_grade', '-')}")

    if show_sources and kb_docs:
        seen = []
        for doc in kb_docs:
            src = doc.metadata.get("source")
            if src and src not in seen:
                seen.append(src)
        print(f"KB SOURCES    {', '.join(seen)}")

    print(THIN)
    print("ANSWER")
    print((result.get("answer") or "").strip())
    print(RULE)


def ask_agent(question: str, graph=None, verbose: bool = True,
              show_sources: bool = True) -> AgentState:
    """Ask one question and print the answer with its provenance.

    Pass an existing `graph` when asking several questions: building one
    constructs the router, grader, retriever and search tool, and there is no
    reason to repeat that per question.
    """
    result = ask(question, graph=graph, verbose=verbose)
    print_result(result, show_sources=show_sources)
    return result


# --- Visualisation -----------------------------------------------------------
# Rendered from the compiled graph, so a diagram can never drift from the wiring.

DIAGRAM_PATH = config.PROJECT_ROOT / "docs" / "agent-graph.md"


def graph_mermaid(graph=None) -> str:
    """Mermaid source for the compiled graph."""
    return (graph or build_graph(verbose=False)).get_graph().draw_mermaid()


def save_graph_diagram(path: Path | None = None, graph=None) -> Path:
    """Write the diagram to a markdown file. GitHub renders mermaid inline."""
    path = Path(path) if path else DIAGRAM_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "# Agent graph\n\n"
        "Generated from the compiled graph by "
        "`python Rag/graph.py --diagram`. Do not edit by hand.\n\n"
        "Solid arrows are unconditional edges; dotted arrows are branches.\n\n"
        "```mermaid\n" + graph_mermaid(graph).strip() + "\n```\n"
    )
    path.write_bytes(body.encode("utf-8"))
    return path


def save_graph_png(path: Path | None = None, graph=None) -> Path:
    """Write a PNG of the graph.

    Not called anywhere by default: `draw_mermaid_png` posts the graph to the
    public mermaid.ink service. Harmless here (node names only) but it is a
    network call to a third party, so it stays opt-in.
    """
    path = Path(path) if path else config.PROJECT_ROOT / "docs" / "agent-graph.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    png = (graph or build_graph(verbose=False)).get_graph().draw_mermaid_png()
    path.write_bytes(png)
    return path


def main():
    config.configure_stdout()
    config.configure_logging()

    if "--diagram" in sys.argv:
        written = save_graph_diagram()
        print(graph_mermaid())
        print(f"\nWrote {written}")
        return

    # Sample questions live in demo.py rather than being duplicated here.
    # Imported lazily: demo imports this module.
    from Rag.demo import run

    run()


if __name__ == "__main__":
    main()
