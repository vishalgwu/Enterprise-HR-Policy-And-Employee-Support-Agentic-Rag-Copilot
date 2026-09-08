"""Wiring for the agentic RAG graph.

    START -> contextualize                  -> route
      route == direct                       -> direct_answer            -> END
      route == kb                           -> retrieve_kb
    retrieve_kb                             -> grade_kb_evidence
      kb_grade == good                      -> generate_from_kb         -> END
      kb_grade == weak                      -> search_web
    search_web                              -> grade_web_evidence
      web_grade == good                     -> generate_from_web        -> END
      web_grade == weak, retries left       -> rewrite_query -> retrieve_kb
      web_grade == weak, no retries         -> answer_insufficient      -> END

The rewrite loops back to the private KB rather than straight to the web,
because a vocabulary mismatch is the most common cause of a weak retrieval and
internal policy is the preferred source. `retry_count` bounds the loop.

This module only wires; the node bodies live in `app.agent.nodes`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import NODE_NAMES, TERMINAL_NODES, build_nodes
from app.agent.schemas import AgentState, initial_state
from app.core.config import Settings

# Conditional branches, as {source node: (condition, {returned name: target})}.
# The path maps are explicit rather than inferred from each condition's return
# annotation: they are what the rendered diagram draws, and they make a branch
# that returns an unmapped name fail at compile time instead of at run time on
# whichever question happens to reach it first.
BRANCHES = {
    "route_question": (
        "route_after_router",
        {"retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"},
    ),
    "grade_kb_evidence": (
        "decide_after_kb_grade",
        {"generate_from_kb": "generate_from_kb", "search_web": "search_web"},
    ),
    "grade_web_evidence": (
        "decide_after_web_grade",
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "answer_insufficient": "answer_insufficient",
        },
    ),
}

# Unconditional edges.
EDGES = (
    # Every question passes through contextualisation, but only a follow-up
    # costs an LLM call there -- with no history the node resolves to the
    # question it was given and returns.
    ("contextualize", "route_question"),
    ("retrieve_kb", "grade_kb_evidence"),
    ("search_web", "grade_web_evidence"),
    # A rewrite retries the private KB first: internal policy is the preferred
    # source, and a vocabulary mismatch is the likeliest cause of a weak hit.
    ("rewrite_query", "retrieve_kb"),
)


def build_graph(
    llm: Any = None,
    retriever: Any = None,
    web_search: Any = None,
    embedding: Any = None,
    verbose: bool = True,
    settings: Settings | None = None,
):
    """Wire and compile the agent graph.

    Every client is injectable, so a test builds the real graph over fakes.
    """
    nodes = build_nodes(llm, retriever, web_search, embedding, verbose, settings)

    builder = StateGraph(AgentState)
    for name in NODE_NAMES:
        builder.add_node(name, nodes[name])

    builder.add_edge(START, "contextualize")
    for source, target in EDGES:
        builder.add_edge(source, target)
    for source, (condition, path_map) in BRANCHES.items():
        unknown = set(path_map.values()) - set(NODE_NAMES)
        if unknown:
            raise ValueError(f"Branch {source!r} targets unknown node(s): {unknown}")
        builder.add_conditional_edges(source, nodes[condition], path_map)
    for terminal in TERMINAL_NODES:
        builder.add_edge(terminal, END)

    return builder.compile()


def ask(
    question: str,
    graph: Any = None,
    verbose: bool = True,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> AgentState:
    """Run one question through the graph and return the final state.

    Raises ValueError on an empty question rather than spending a routing call
    to discover there was nothing to answer.

    `history` is the conversation so far, oldest first, as
    `{"role": "user"|"assistant", "content": ...}`. It is what lets a follow-up
    resolve; `initial_state` normalises and caps it. Omit it and the run is a
    single-turn one, identical to before this existed.

    Pass an existing `graph` when asking several questions: building one
    constructs the router, grader, retriever and search tool, none of which vary
    per question.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    graph = graph or build_graph(verbose=verbose)
    return graph.invoke(initial_state(question.strip(), history))
