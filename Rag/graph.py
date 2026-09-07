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

if __package__ in (None, ""):  # allow `python Rag/graph.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from Rag import config
from Rag.clients import get_embeddings, get_llm, get_web_search, web_results_to_text
from Rag.loaders import format_documents
from Rag.private_kb import get_kb_retriever
from Rag.schemas import (
    MAX_RETRIES,
    AgentState,
    get_evidence_grader,
    get_router,
    initial_state,
)

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


def build_nodes(llm=None, retriever=None, web_search=None, embedding=None, verbose=True):
    """Construct every node once, closed over shared clients."""
    llm = llm or get_llm()
    embedding = embedding or get_embeddings()
    retriever = retriever if retriever is not None else get_kb_retriever(embedding=embedding)
    web_search = web_search if web_search is not None else get_web_search()

    router = get_router(llm)
    grader = get_evidence_grader(llm)

    def log(tag, message):
        if verbose:
            print(f"[{tag}] {message}")

    # --- Node 1: route -------------------------------------------------------
    def route_question(state: AgentState):
        question = state["question"]
        route = router.invoke({"question": question}).route
        log("Router", route)
        return {"route": route, "current_query": question}

    def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
        return "retrieve_kb" if state.get("route") == "kb" else "direct_answer"

    # --- Node 2: retrieve from the private KB --------------------------------
    def retrieve_kb(state: AgentState):
        query = state["current_query"]
        docs = retriever.invoke(query)
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

        grade = grader.invoke(
            {"question": state["question"], "evidence": format_documents(docs)}
        ).grade
        log("KB Grader", grade)
        return {"kb_grade": grade}

    def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
        return "generate_from_kb" if state.get("kb_grade") == "good" else "search_web"

    # --- Node 4: Tavily web search fallback ----------------------------------
    def search_web(state: AgentState):
        query = state["current_query"]
        log("Tavily", f"searching {query!r}")
        web_text = web_results_to_text(web_search.invoke({"query": query}))
        log("Tavily", f"{len(web_text)} characters")
        return {"web_results": web_text}

    # --- Node 5: grade web evidence ------------------------------------------
    def grade_web_evidence(state: AgentState):
        web_results = state.get("web_results") or ""
        if not web_results.strip():
            log("Web Grader", "weak (search returned nothing)")
            return {"web_grade": "weak"}

        grade = grader.invoke(
            {"question": state["question"], "evidence": web_results}
        ).grade
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
        rewritten = (REWRITE_PROMPT | llm).invoke(
            {"question": state["question"]}
        ).content.strip()
        log("Rewriter", rewritten)
        return {"current_query": rewritten, "retry_count": state.get("retry_count", 0) + 1}

    # --- Node 7: generate from the private KB --------------------------------
    def generate_from_kb(state: AgentState):
        answer = (KB_ANSWER_PROMPT | llm).invoke(
            {
                "question": state["question"],
                "context": format_documents(state.get("kb_docs") or []),
            }
        ).content
        return {"answer": answer, "source_used": "private_kb"}

    # --- Node 8: generate from web search ------------------------------------
    def generate_from_web(state: AgentState):
        answer = (WEB_ANSWER_PROMPT | llm).invoke(
            {"question": state["question"], "context": state.get("web_results") or ""}
        ).content
        return {"answer": answer, "source_used": "web_search"}

    # --- Node 9: direct answer -----------------------------------------------
    def direct_answer(state: AgentState):
        answer = (DIRECT_ANSWER_PROMPT | llm).invoke(
            {"question": state["question"]}
        ).content
        return {"answer": answer, "source_used": "direct"}

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
    builder.add_conditional_edges("route_question", n["route_after_router"])
    builder.add_edge("retrieve_kb", "grade_kb_evidence")
    builder.add_conditional_edges("grade_kb_evidence", n["decide_after_kb_grade"])
    builder.add_edge("search_web", "grade_web_evidence")
    builder.add_conditional_edges("grade_web_evidence", n["decide_after_web_grade"])
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
    """Run one question through the graph and return the final state."""
    graph = graph or build_graph(verbose=verbose)
    return graph.invoke(initial_state(question))


def main():
    config.configure_stdout()
    graph = build_graph()

    questions = [
        "hi there",                                        # -> direct
        "How many PTO days do I get per year?",            # -> private KB
        "Who won the football World Cup in 2022?",         # -> KB miss, web
    ]
    for question in questions:
        print("\n" + "=" * 74)
        print(f"Q: {question}")
        print("=" * 74)
        final = ask(question, graph)
        print(f"\n[source_used] {final.get('source_used')}   "
              f"[retries] {final.get('retry_count')}")
        print(f"[answer]\n{final.get('answer', '').strip()}")


if __name__ == "__main__":
    main()
