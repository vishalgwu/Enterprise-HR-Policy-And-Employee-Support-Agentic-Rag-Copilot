"""Every path through the graph, driven by fakes.

This is what dependency injection in `build_nodes` buys: the rewrite loop and
the insufficient-evidence path are exercised deterministically instead of
waiting for a real question that happens to fail twice.
"""

from __future__ import annotations

import pytest
from conftest import FakeLLM, FakeRetriever, FakeWebSearch, kb_doc
from langchain_core.messages import AIMessage

from app.agent.graph import BRANCHES, ask, build_graph
from app.agent.nodes import NODE_NAMES, TERMINAL_NODES


def run(llm, retriever, web_search, question="How many PTO days do I get?"):
    graph = build_graph(
        llm=llm, retriever=retriever, web_search=web_search, verbose=False
    )
    return ask(question, graph=graph, verbose=False)


# --- The four paths ----------------------------------------------------------


def test_good_kb_evidence_answers_from_the_private_kb():
    retriever = FakeRetriever([kb_doc()])
    state = run(FakeLLM(route="kb", grades=["good"]), retriever, FakeWebSearch())

    assert state["source_used"] == "private_kb"
    assert state["kb_grade"] == "good"
    assert state["retry_count"] == 0
    assert len(retriever.queries) == 1


def test_weak_kb_evidence_falls_back_to_web_search():
    """Chunks came back, the grader judged them weak, the web answered instead.

    The retriever returns a chunk on purpose: an empty retrieval skips the
    grader entirely, which is a different path (see the next test).
    """
    web = FakeWebSearch(answer="UK statutory minimum is 28 days.",
                        results=[{"url": "https://gov.uk/holiday", "title": "Holiday"}])
    llm = FakeLLM(route="kb", grades=["weak", "good"])
    state = run(llm, FakeRetriever([kb_doc()]), web)

    assert state["source_used"] == "web_search"
    assert state["kb_grade"] == "weak"
    assert state["web_grade"] == "good"
    assert web.queries == ["How many PTO days do I get?"]
    assert state["web_urls"] == ["https://gov.uk/holiday"]
    assert state["retry_count"] == 0


def test_an_empty_retrieval_grades_weak_without_spending_a_call():
    """The similarity gate already said no; grading nothing costs a call to
    be told 'weak'."""
    llm = FakeLLM(route="kb", grades=["good"])
    state = run(llm, FakeRetriever([]), FakeWebSearch(answer="Public guidance."))

    assert state["kb_grade"] == "weak"
    # Exactly one grading call, and it was the web's.
    assert llm.calls.count("EvidenceGrade") == 1


def test_direct_route_skips_retrieval_entirely():
    retriever = FakeRetriever([kb_doc()])
    state = run(FakeLLM(route="direct"), retriever, FakeWebSearch(), question="hi there")

    assert state["source_used"] == "direct"
    assert retriever.queries == []
    assert "kb_grade" not in state


def test_everything_weak_rewrites_once_then_admits_defeat():
    retriever = FakeRetriever([])
    state = run(
        FakeLLM(route="kb", grades=["weak"], rewrite="What is the annual PTO allowance?"),
        retriever,
        FakeWebSearch(),
    )

    assert state["source_used"] == "insufficient_evidence"
    assert state["retry_count"] == 1
    # Retrieved once with the original wording, once with the rewrite.
    assert len(retriever.queries) == 2
    assert retriever.queries[1] == "What is the annual PTO allowance?"


def test_the_original_question_survives_a_rewrite():
    """`question` is what the employee asked; `current_query` is what was run."""
    state = run(
        FakeLLM(route="kb", grades=["weak"], rewrite="formal phrasing"),
        FakeRetriever([]),
        FakeWebSearch(),
    )
    assert state["question"] == "How many PTO days do I get?"
    assert state["current_query"] == "formal phrasing"


# --- The trace and the citations ---------------------------------------------


def test_the_trace_records_the_nodes_that_ran_in_order():
    state = run(FakeLLM(route="direct"), FakeRetriever([]), FakeWebSearch(),
                question="hi there")
    assert state["trace"] == [
        "Router: direct",
        "Generate/Direct: answered from direct",
    ]


def test_the_trace_accumulates_instead_of_being_overwritten():
    """The `add` reducer on `trace` is what makes this pass.

    Without it each node's return *replaces* the key and only the last node's
    entry survives -- so the rewrite loop, the one run whose path is actually
    worth seeing, would erase its own history.
    """
    state = run(
        FakeLLM(route="kb", grades=["weak"], rewrite="annual PTO allowance"),
        FakeRetriever([]),
        FakeWebSearch(),
    )
    trace = state["trace"]

    assert trace[0] == "Router: kb"
    assert trace[-1] == "Fallback: no sufficient evidence in KB or web"
    # Retrieval ran twice: once on the original wording, once on the rewrite.
    assert sum(step.startswith("KB Retriever:") for step in trace) == 2
    assert "Rewriter: annual PTO allowance" in trace


def test_a_degraded_call_is_recorded_in_the_trace():
    """An outage and a genuine absence of evidence produce the same answer;
    only the trace distinguishes them."""
    web = FakeWebSearch(answer="Public guidance.", results=[{"url": "https://x"}])
    state = run(FakeLLM(route="kb", grades=["good"]), FakeRetriever(fail=True), web)

    assert state["source_used"] == "web_search"
    assert any(
        step.startswith("KB Retriever: failed (RuntimeError)")
        for step in state["trace"]
    )


def test_citations_are_written_beside_the_chunks_they_describe():
    state = run(FakeLLM(route="kb", grades=["good"]), FakeRetriever([kb_doc()]),
                FakeWebSearch())
    assert [c["source"] for c in state["citations"]] == ["leave-and-time-off.md"]


def test_citations_are_replaced_by_a_rewrite_not_accumulated():
    """The retry retrieved nothing, so nothing may still be cited.

    A reducer here would leave the first attempt's sources attached to an
    answer that no longer rests on them.
    """
    state = run(
        FakeLLM(route="kb", grades=["weak"], rewrite="annual PTO allowance"),
        FakeRetriever([[kb_doc()], []]),
        FakeWebSearch(),
    )
    assert state["source_used"] == "insufficient_evidence"
    assert state["kb_docs"] == []
    assert state["citations"] == []


# --- Degradation -------------------------------------------------------------


def test_a_retrieval_outage_becomes_zero_chunks_not_a_crash():
    """One flaky Pinecone call must not cost an answer the web could have given."""
    web = FakeWebSearch(answer="Public guidance.", results=[{"url": "https://x"}])
    state = run(FakeLLM(route="kb", grades=["good"]), FakeRetriever(fail=True), web)

    assert state["kb_docs"] == []
    assert state["source_used"] == "web_search"
    assert state["retry_count"] == 0
    assert web.queries == ["How many PTO days do I get?"]


def test_a_search_outage_becomes_empty_evidence_not_a_crash():
    state = run(
        FakeLLM(route="kb", grades=["weak"]),
        FakeRetriever([]),
        FakeWebSearch(fail=True),
    )
    assert state["web_results"] == ""
    assert state["source_used"] == "insufficient_evidence"


def test_a_failed_rewrite_still_increments_the_retry_count():
    """Otherwise a broken LLM spins the loop forever."""
    llm = FakeLLM(route="kb", grades=["weak"])
    llm.rewrite = None
    state = run(llm, FakeRetriever([]), FakeWebSearch())
    assert state["retry_count"] == 1
    assert state["source_used"] == "insufficient_evidence"


def test_a_total_provider_outage_still_returns_an_answer():
    """Router -> kb, every grade -> weak, so it lands on the honest fallback."""
    state = run(FakeLLM(fail=True), FakeRetriever([]), FakeWebSearch())
    assert state["source_used"] == "insufficient_evidence"
    assert state["answer"].strip()


def test_a_reply_of_content_blocks_does_not_abort_the_run():
    """`.content` may be a list, and `.strip()` on a list raises.

    That AttributeError used to escape the node, and a node that raises kills
    the whole graph -- so a provider shape change would have cost every answer
    rather than degrading one.
    """

    class _BlockReply(FakeLLM):
        def invoke(self, input, config=None, **kwargs):
            return AIMessage(
                content=[{"type": "text", "text": "Twenty days of PTO."}]
            )

    state = run(
        _BlockReply(route="kb", grades=["good"]),
        FakeRetriever([kb_doc()]),
        FakeWebSearch(),
    )
    assert state["source_used"] == "private_kb"
    assert state["answer"] == "Twenty days of PTO."


def test_a_generation_failure_is_reported_not_swallowed():
    class _GenerationFails(FakeLLM):
        def invoke(self, input, config=None, **kwargs):
            raise RuntimeError("generation down")

    state = run(_GenerationFails(route="kb", grades=["good"]), FakeRetriever([kb_doc()]),
                FakeWebSearch())
    assert state["source_used"] == "error"
    assert "could not produce an answer" in state["answer"]


# --- Input validation and wiring ---------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", None])
def test_an_empty_question_raises_before_spending_a_call(question):
    with pytest.raises(ValueError):
        ask(question, graph=object(), verbose=False)


def test_every_branch_target_is_a_real_node():
    for source, (_condition, path_map) in BRANCHES.items():
        assert source in NODE_NAMES
        assert set(path_map.values()) <= set(NODE_NAMES)


def test_terminal_nodes_are_nodes():
    assert set(TERMINAL_NODES) <= set(NODE_NAMES)


EXPECTED_EDGES = {
    ("__start__", "route_question"),
    ("route_question", "retrieve_kb"),
    ("route_question", "direct_answer"),
    ("retrieve_kb", "grade_kb_evidence"),
    ("grade_kb_evidence", "generate_from_kb"),
    ("grade_kb_evidence", "search_web"),
    ("search_web", "grade_web_evidence"),
    ("grade_web_evidence", "generate_from_web"),
    ("grade_web_evidence", "rewrite_query"),
    ("grade_web_evidence", "answer_insufficient"),
    ("rewrite_query", "retrieve_kb"),
    ("generate_from_kb", "__end__"),
    ("generate_from_web", "__end__"),
    ("direct_answer", "__end__"),
    ("answer_insufficient", "__end__"),
}


def test_the_compiled_graph_matches_the_documented_design():
    """The README draws this shape; this is what stops the picture drifting.

    An added, removed or misdirected edge fails here rather than showing up as
    a question that quietly takes the wrong path.
    """
    graph = build_graph(
        llm=FakeLLM(), retriever=FakeRetriever([]), web_search=FakeWebSearch(),
        verbose=False,
    )
    actual = {(e.source, e.target) for e in graph.get_graph().edges}
    assert actual == EXPECTED_EDGES


def test_the_compiled_graph_has_exactly_the_declared_nodes():
    graph = build_graph(
        llm=FakeLLM(), retriever=FakeRetriever([]), web_search=FakeWebSearch(),
        verbose=False,
    )
    compiled = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    assert compiled == set(NODE_NAMES)
