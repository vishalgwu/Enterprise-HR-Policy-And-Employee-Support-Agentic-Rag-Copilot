"""The answer pipeline: everything that happens around the agent.

The graph answers a question. This decides whether the question should be
answered at all, whether it already has been, which model should do it, which
orchestration is the right one, and what has to be true of the answer before
anyone sees it. Eleven stages, in a fixed order, and the order is the design:

     1. trace and budget        one tracer and one budget per run, shared
     2. input guard             refuse, or redact and carry on
     3. semantic cache          an answer already given to this question
     4. model routing           complexity -> tier -> model id
     5. orchestration choice    graph | react | supervisor
     6. run                     streaming the stages as they happen
     7. generation fallback     retrieved evidence, no model -> extract
     8. output guard            redact what the model echoed; flag the rest
     9. accounting              tokens, cost, llm calls, budget report
    10. cache store             only an answer worth repeating
    11. assemble                one AnswerResult, which is the whole contract

Each stage is short and every one of them can be switched off by configuration,
because a feature nobody can turn off is a feature nobody can measure. The
whole pipeline with every stage off is the behaviour this project shipped with.

**Why the guard runs before the cache.** A blocked question must not be able to
poison the cache, and a redacted question is the key the cache should be keyed
on -- otherwise two employees asking the same thing with different phone
numbers in it are two cache misses, and the entry that does get stored holds
somebody's phone number.

**Why routing runs after the cache and before retrieval.** After, because a
cache hit should not pay for a classification. Before retrieval, because the
model has to exist to run the graph -- which means the cost projection is made
without knowing how much evidence will be retrieved, and `context_chars` is
therefore an estimate. That is stated here rather than hidden: the projection
is for a ceiling and a dashboard, not an invoice.

**Why generation has a non-model fallback.** If retrieval succeeded and
generation failed, the run *found the policy* and then failed to write prose
about it. Returning an apology throws away the part that worked. The same code
serves the deployment with no provider key at all, which is what makes a public
demo possible -- see `app/services/extractive.py`.

**On compiling a graph per run.** Compiling the `StateGraph` is microseconds;
what is expensive is the retriever, which loads an embedding model, and that is
built once and shared. So this holds the clients and compiles per question,
which is what lets every run have its own tracer -- a tracer bound into a
cached graph would collect every question's spans into one list for the life of
the process.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.agent.budgets import Budget
from app.agent.graph import build_graph
from app.agent.prompts import (
    BLOCKED_BY_GUARD,
    BUDGET_STOPPED_NOTE,
    GENERATION_FAILED,
    INSUFFICIENT_ANSWER,
)
from app.agent.react import ReActAgent
from app.agent.schemas import initial_state, normalise_history
from app.agent.supervisor import SupervisorTeam
from app.core.config import Settings, get_logger, get_settings
from app.core.tracing import Tracer
from app.rag.clients import get_llm, get_web_search
from app.rag.hybrid import build_retriever
from app.rag.loaders import cite_sources
from app.services.authorization import ANONYMOUS, Principal
from app.services.cache import SemanticCache
from app.services.copilot import AnswerResult, SourceRef, to_answer
from app.services.extractive import answer_from_documents
from app.services.guardrails import InputGuard, OutputGuard
from app.services.routing import (
    CHARS_PER_TOKEN,
    ModelRouter,
    estimate_tokens,
    project_cost,
)
from app.services.streaming import (
    EVENT_ANSWER,
    EVENT_CITATIONS,
    EVENT_STAGE,
    EVENT_TRACE,
)
from app.services.toolbox import build_registry

log = get_logger("pipeline")

__all__ = ["AnswerPipeline"]

# Trace tags are "<Tag>: <detail>"; the tag is what the console maps to a graph
# node. Split on the first colon only -- a rewritten query legitimately contains
# one.
def _tag_of(step: str) -> str:
    return step.split(":", 1)[0].strip()


class AnswerPipeline:
    """Holds the shared clients and runs one question through every stage.

    One instance per application (or per caller-supplied provider key, through
    `app/services/byok.py`). Everything it holds is either stateless or a
    deliberate process-wide cache: the retriever's embedding model, the
    semantic cache's LRU, the guards' compiled patterns. Nothing per-request
    lives here, which is what keeps one employee's question out of another's.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: Any = None,
        web_search: Any = None,
        embedding: Any = None,
        llm: Any = None,
        graph: Any = None,
        cache: SemanticCache | None = None,
        router: ModelRouter | None = None,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
        scratchpad: Any = None,
        facts: Any = None,
        approvals: Any = None,
        idempotency: Any = None,
        supervisor_factory: Callable[..., Any] | None = None,
        react_factory: Callable[..., Any] | None = None,
        verbose: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.verbose = verbose

        # Injected clients win; anything else is built on first use. `graph` and
        # `llm` being injectable is what lets the existing offline tests drive
        # this whole pipeline over fakes.
        self._graph = graph
        self._llm = llm
        self._retriever = retriever
        self._web_search = web_search
        self._embedding = embedding

        self.cache = cache if cache is not None else SemanticCache(settings=self.settings)
        self.router = router or ModelRouter(settings=self.settings)
        self.input_guard = input_guard or InputGuard(self.settings)
        self.output_guard = output_guard or OutputGuard(self.settings)
        self.scratchpad = scratchpad
        self.facts = facts
        self.approvals = approvals
        self.idempotency = idempotency
        # Which services the caller supplied, as opposed to which ones defaulted.
        # `adopt_services` needs the difference: see its docstring.
        self._injected = {
            name
            for name, value in (
                ("cache", cache),
                ("scratchpad", scratchpad),
                ("facts", facts),
                ("approvals", approvals),
                ("idempotency", idempotency),
            )
            if value is not None
        }
        # Default to the real ones. They are classes, not instances: both close
        # over a per-request tracer and budget, so there is nothing to build
        # ahead of time and a shared instance would be the bug.
        self._supervisor_factory = supervisor_factory or SupervisorTeam
        self._react_factory = react_factory or ReActAgent

    def adopt_services(self, **services: Any) -> AnswerPipeline:
        """Take the application's stores, where this pipeline defaulted its own.

        `create_app` is the composition root, and a pipeline built outside it --
        by a test injecting a graph of fakes, by a CLI -- defaulted to a private
        semantic cache and no scratchpad at all. Left alone, that produces two
        confusing behaviours in the assembled application: `/api/cache/
        invalidate` drops a cache nothing is reading from, and an answer comes
        back with no scratchpad because the pipeline has a different one from
        the endpoint that wrote to it. Both read as the feature being broken.

        **Only what defaulted.** A service the constructor was handed is kept,
        because that caller said something specific and the root does not know
        better than an explicit argument. That is the whole reason
        `self._injected` exists -- without it this would be a method that
        quietly overwrites what it was told.
        """
        for name, service in services.items():
            if service is None or name in self._injected:
                continue
            if not hasattr(self, name):  # pragma: no cover - a typo, not a state
                raise AttributeError(f"AnswerPipeline has no service named {name!r}")
            setattr(self, name, service)
        return self

    # --- Lazily built shared clients ----------------------------------------
    @property
    def retriever(self) -> Any:
        """The shared retriever. Built once; it may load an embedding model."""
        if self._retriever is None:
            self._retriever = build_retriever(
                settings=self.settings, embedding=self._embedding
            )
        return self._retriever

    @property
    def web_search(self) -> Any:
        """The shared web-search client, or None when there is no key.

        None rather than an exception: the graph's `search_web` node already
        treats an empty result as "no web evidence", which is the honest state
        of a deployment with no Tavily key, and refusing to build a pipeline
        over it would make the private KB unreachable too.
        """
        if self._web_search is None:
            try:
                self._web_search = get_web_search(settings=self.settings)
            except Exception:  # noqa: BLE001 - a missing key is a reduced shape
                log.warning("No web search configured; the web fallback is off")
                self._web_search = _NoWebSearch()
        return self._web_search

    def _chat_model(self, model: str | None) -> Any:
        if self._llm is not None:
            return self._llm
        return get_llm(model=model or None, settings=self.settings)

    def _graph_for(self, model: str | None, tracer: Tracer, budget: Budget) -> Any:
        """The compiled graph for one run.

        An injected graph is returned untouched and **no chat model is built**.
        That ordering is what keeps the offline tests offline: they inject a
        graph closed over a fake LLM, and constructing a real client to then
        not use it would need a provider key.
        """
        if self._graph is not None:
            return self._graph
        return build_graph(
            llm=self._chat_model(model),
            retriever=self.retriever,
            web_search=self.web_search,
            embedding=self._embedding,
            verbose=self.verbose,
            settings=self.settings,
            tracer=tracer,
            budget=budget,
        )

    @property
    def has_model(self) -> bool:
        """Whether a chat model is reachable for this pipeline.

        An injected `llm` or graph counts: that is a test, or a caller-supplied
        key that `app/services/byok.py` has already folded into these settings.
        Without one of those, `settings.has_llm` decides -- and False there is
        what routes a question down the extractive path rather than failing it.
        """
        return self._llm is not None or self._graph is not None or self.settings.has_llm

    # --- The pipeline --------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        conversation_id: str | None = None,
        principal: Principal | None = None,
        mode: str | None = None,
        byok: bool = False,
        stream: Any = None,
        tracer: Tracer | None = None,
        budget: Budget | None = None,
    ) -> AnswerResult:
        """Answer one question, all stages, never raising.

        `stream` is an optional `app/services/streaming.EventStream`; when
        given, each stage is emitted as it happens. Nothing about the pipeline
        changes when it is absent, which is deliberate -- a streamed answer and
        a posted one must not be able to differ.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        started = time.perf_counter()
        principal = principal or ANONYMOUS
        tracer = tracer or Tracer(
            enabled=self.settings.trace_spans,
            redact_payloads=not self.settings.trace_payloads,
        )
        budget = (budget or Budget.from_settings(self.settings)).start()
        history = normalise_history(history)
        question = question.strip()

        def stage(name: str, **data: Any) -> None:
            if stream is not None:
                stream.emit(EVENT_STAGE, name=name, **data)

        with tracer.span("answer", kind="agent", question=question) as root:
            # --- 2. Input guard ---------------------------------------------
            stage("guard")
            verdict = self.input_guard.check(question, tracer)
            if not verdict.allowed:
                stage("blocked", category=verdict.category)
                return self._finish(
                    self._blocked(question, verdict, principal, byok),
                    started,
                    tracer,
                    budget,
                    conversation_id,
                    principal,
                    stream,
                )
            asked = verdict.text or question
            redacted = asked if asked != question else None
            root.set(redacted=bool(redacted))

            # --- 3. Semantic cache ------------------------------------------
            stage("cache")
            # `is not None`, never a truth test. `SemanticCache` defines
            # `__len__`, so an *empty* cache is falsy -- `if self.cache` skipped
            # both the lookup and the store, which meant the cache could never
            # fill and therefore never became truthy. It read as "the cache is
            # off" with every setting saying it was on.
            hit = self.cache.get(asked, tracer) if self.cache is not None else None
            replayed = (
                self._from_cache(question, redacted, verdict, hit, byok)
                if hit is not None
                else None
            )
            if replayed is not None:
                stage("cache_hit", similarity=round(hit.similarity, 4))
                return self._finish(
                    replayed,
                    started,
                    tracer,
                    budget,
                    conversation_id,
                    principal,
                    stream,
                )

            # --- 4. Model routing -------------------------------------------
            decision = self.router.choose(
                asked, history=history, tracer=tracer, budget=budget
            )
            stage("route", model=decision.model, tier=decision.tier)

            # --- 5/6/7. Run -------------------------------------------------
            chosen = mode or self._pick_mode(decision.complexity)
            if not self.has_model:
                # No model anywhere: retrieve and extract. The one path that is
                # structurally incapable of inventing a policy number.
                chosen = "extractive"
            stage("mode", mode=chosen)
            result = self._run(
                chosen,
                question=question,
                asked=asked,
                history=history,
                decision=decision,
                conversation_id=conversation_id,
                principal=principal,
                tracer=tracer,
                budget=budget,
                stream=stream,
            )

            # Carry forward what the stages above established. Done here, once,
            # rather than inside each orchestration, so a new orchestration
            # cannot forget a field the audit row depends on.
            result.redacted_question = redacted
            result.guard_input = verdict.to_dict()
            result.pii_redacted = list(verdict.redacted)
            result.answer_mode = chosen
            result.retrieval_mode = self.settings.effective_retrieval_mode
            result.model = decision.model
            result.provider = decision.provider
            result.model_tier = decision.tier
            result.complexity = decision.complexity
            result.routing_reason = decision.reason
            result.byok = byok

            # --- 8. Output guard --------------------------------------------
            stage("output_guard")
            out = self.output_guard.check(
                result.answer, tracer, grounded=result.grounded
            )
            result.guard_output = out.to_dict()
            if not out.allowed:
                # The only blocking output case is a leaked credential, and an
                # answer containing one must not be shown with a hole in it.
                log.error("The output guard refused an answer (%s)", out.category)
                result.answer = out.text
                result.blocked = True
            elif out.modified:
                result.answer = out.text
                result.pii_redacted = sorted(
                    set(result.pii_redacted) | set(out.redacted)
                )

            # --- 10. Cache store --------------------------------------------
            if self.cache is not None and not result.blocked and not result.partial:
                self.cache.put(asked, result.model_dump(), tracer)

            return self._finish(
                result, started, tracer, budget, conversation_id, principal, stream
            )

    # --- Orchestration choice ------------------------------------------------
    def _pick_mode(self, complexity: str) -> str:
        """Which orchestration answers this question.

        `auto` is the interesting value and the rule is deliberately narrow: the
        supervisor is for a question that spans policy areas, which is exactly
        what `classify_complexity` calls `complex`, and the graph is for
        everything else. That is the honest trade -- the supervisor costs a
        delegation call plus one model call per worker, so spending it on "what
        are core hours?" would make the common question four times slower to
        demonstrate a capability it does not need.

        ReAct is never chosen automatically. It is the most expensive and least
        predictable of the three, and its value here is that a question needing
        a *tool* -- arithmetic, a ticket, an escalation -- is one a human is
        better placed to recognise than a keyword heuristic. `AGENT_MODE=react`
        selects it, and so does the console's mode switch.
        """
        configured = self.settings.agent_mode
        if configured != "auto":
            return configured
        # An injected graph is the caller saying "*this* graph answers
        # questions" -- a CLI holding one, a test holding one over fakes.
        # Routing past it to a different orchestration would quietly ignore the
        # injection, and the thing it would reach for instead needs clients the
        # injector deliberately did not give us.
        if complexity == "complex" and self._graph is None:
            return "supervisor"
        return "graph"

    def _run(
        self,
        mode: str,
        *,
        question: str,
        asked: str,
        history: list[dict[str, str]],
        decision: Any,
        conversation_id: str | None,
        principal: Principal,
        tracer: Tracer,
        budget: Budget,
        stream: Any,
    ) -> AnswerResult:
        """Dispatch to one orchestration, degrading to extraction rather than
        raising -- and also when the orchestration *returns* a failure."""
        try:
            if mode == "extractive":
                return self._run_extractive(question, asked, tracer)
            if mode == "supervisor":
                result = self._run_supervisor(
                    question, asked, history, decision, conversation_id,
                    tracer, budget, stream,
                )
            elif mode == "react":
                result = self._run_react(
                    question, asked, history, decision, conversation_id,
                    principal, tracer, budget, stream,
                )
            else:
                result = self._run_graph(
                    question, asked, history, decision, tracer, budget, stream
                )
            return self._rescue(result, question, asked, tracer)
        except Exception:  # noqa: BLE001 - never leak a provider error upwards
            log.exception("The %s orchestration raised; falling back", mode)
            tracer.degraded("orchestration", reason=f"{mode} raised", kind="agent")
            # Retrieval may still work when generation does not, and an answer
            # built from the policy text beats an apology.
            fallback = self._run_extractive(question, asked, tracer)
            if fallback.answer:
                return fallback
            return AnswerResult(
                question=question,
                answer=GENERATION_FAILED,
                source_used="error",
                trace=[f"Pipeline: the {mode} run failed; reported as an error"],
            )

    def _rescue(
        self, result: AnswerResult, question: str, asked: str, tracer: Tracer
    ) -> AnswerResult:
        """Replace a reported generation failure with the policy text itself.

        **A failed generation does not raise.** `_generate` in
        `app/agent/nodes.py` returns `GENERATION_FAILED` with
        `source_used="error"`, because a node that throws aborts the whole run
        -- which is right for the node and leaves the pipeline holding a
        completed run that failed. Without this, the `except` below never fires
        and the employee gets an apology.

        What makes the apology avoidable is that *retrieval already worked*:
        the run found the policy and then failed to write prose about it.
        Handing back the retrieved text beats handing back "the language model
        was unreachable", and it is the same code the no-key deployment uses.

        The original trace is kept and the substitution is recorded in it, so
        the run still reads as a generation failure rather than as a successful
        extractive answer -- which is a different thing and should not be
        disguised as one.
        """
        if result.source_used != "error":
            return result

        tracer.degraded("generation", reason="recovering from retrieved text")
        fallback = self._run_extractive(question, asked, tracer)
        if not fallback.answer or fallback.source_used != "private_kb":
            return result

        fallback.trace = [
            *result.trace,
            "Pipeline: generation failed; answering from the retrieved text instead",
            *fallback.trace,
        ]
        fallback.degraded = [*result.degraded, "generation failed"]
        return fallback

    # --- Orchestrations ------------------------------------------------------
    def _run_graph(
        self,
        question: str,
        asked: str,
        history: list[dict[str, str]],
        decision: Any,
        tracer: Tracer,
        budget: Budget,
        stream: Any,
    ) -> AnswerResult:
        graph = self._graph_for(decision.model, tracer, budget)
        state = initial_state(asked, history)
        # `question` in state is what the nodes see; the employee's verbatim
        # wording is restored afterwards so the UI and the audit row show what
        # was typed rather than what was redacted.
        final = self._drive(graph, state, stream)
        result = to_answer(final)
        result.question = question
        return result

    def _drive(self, graph: Any, state: Any, stream: Any) -> Any:
        """Run the graph, streaming each node's trace lines as they land.

        `stream_mode="values"` yields the whole state after every step, so the
        reducers have already been applied and the last value is the final
        state -- which is why this does not have to merge updates itself and
        cannot get `trace`'s `add` reducer wrong.

        Falls back to `invoke` when streaming is unavailable: a LangGraph
        version without `.stream`, or a graph a test injected that only
        implements `invoke`. The answer must not depend on which.
        """
        if stream is None or not hasattr(graph, "stream"):
            return graph.invoke(state)

        final = state
        seen = 0
        try:
            for value in graph.stream(state, stream_mode="values"):
                final = value
                steps = list(value.get("trace") or [])
                for step in steps[seen:]:
                    stream.emit(EVENT_TRACE, text=step, tag=_tag_of(step))
                    stream.emit(EVENT_STAGE, name=_tag_of(step))
                seen = len(steps)
                citations = value.get("citations")
                if citations:
                    stream.emit(EVENT_CITATIONS, sources=list(citations))
        except Exception:  # noqa: BLE001 - fall back rather than lose the answer
            log.exception("Streaming the graph failed; running it plainly")
            return graph.invoke(state)
        return final

    def _run_supervisor(
        self,
        question: str,
        asked: str,
        history: list[dict[str, str]],
        decision: Any,
        conversation_id: str | None,
        tracer: Tracer,
        budget: Budget,
        stream: Any,
    ) -> AnswerResult:
        llm = self._chat_model(decision.model)
        team = self._supervisor_factory(
            llm=llm,
            retriever=self.retriever,
            web_search=self.web_search,
            settings=self.settings,
            scratchpad=self.scratchpad,
            # So an escalation lands in the queue an HR advisor actually reads,
            # not only in a scratchpad scoped to one conversation id that
            # nothing lists. Without it the employee is told a person will pick
            # their question up and no person ever learns of it.
            approvals=self.approvals,
            verbose=self.verbose,
        )
        def on_step(steps: list[str], worker_path: list[str]) -> None:
            """Report each step as it lands, rather than all of them at the end.

            A supervisor run is twenty seconds of model calls on a real
            question, and the sequence of workers it consults is the thing
            being demonstrated. Showing it only once the answer arrives shows
            it to somebody who has stopped watching.
            """
            for step in steps:
                stream.emit(EVENT_TRACE, text=step, tag=_tag_of(step))
                stream.emit(
                    EVENT_STAGE, name=_tag_of(step), workers=list(worker_path)
                )

        out = team.ask(
            asked,
            history=history,
            tracer=tracer,
            budget=budget,
            conversation_id=conversation_id,
            on_step=on_step if stream is not None else None,
        )
        findings = list(out.get("findings") or [])
        citations = list(out.get("citations") or [])
        if stream is not None and citations:
            stream.emit(EVENT_CITATIONS, sources=citations)
        return AnswerResult(
            question=question,
            answer=(out.get("answer") or "").strip(),
            source_used=out.get("source_used"),
            route="kb",
            kb_chunks=sum(int(f.get("chunks") or 0) for f in findings),
            sources=[SourceRef(**ref) for ref in citations],
            trace=list(out.get("trace") or []),
            worker_path=list(out.get("worker_path") or []),
            handoffs=list(out.get("handoffs") or []),
            findings=findings,
            escalated=bool(out.get("escalated")),
        )

    def _run_react(
        self,
        question: str,
        asked: str,
        history: list[dict[str, str]],
        decision: Any,
        conversation_id: str | None,
        principal: Principal,
        tracer: Tracer,
        budget: Budget,
        stream: Any,
    ) -> AnswerResult:
        llm = self._chat_model(decision.model)
        # The registry is built per request, not per process: `raise_ticket` and
        # `escalate_to_human` close over *this* question and *this* conversation
        # id, so a shared one would file one employee's escalation against
        # another's question. One `Budget` object is handed to both the registry
        # and the agent, which is what makes `budget.report()["tool_calls"]` a
        # fact rather than two opinions.
        registry = build_registry(
            settings=self.settings,
            retriever=self.retriever,
            web_search=self.web_search,
            scratchpad=self.scratchpad,
            facts=self.facts,
            approvals=self.approvals,
            idempotency=self.idempotency,
            tracer=tracer,
            budget=budget,
            question=asked,
            conversation_id=conversation_id,
            subject=principal.subject,
        )
        agent = self._react_factory(
            llm=llm,
            registry=registry,
            settings=self.settings,
            tracer=tracer,
            budget=budget,
            verbose=self.verbose,
        )
        out = agent.run(
            asked,
            history=history,
            principal=principal,
            conversation_id=conversation_id,
        )
        data = out.to_dict() if hasattr(out, "to_dict") else dict(out)
        if stream is not None:
            for step in data.get("trace") or []:
                stream.emit(EVENT_TRACE, text=step, tag=_tag_of(step))
        citations = list(data.get("citations") or [])
        return AnswerResult(
            question=question,
            answer=(data.get("answer") or "").strip(),
            source_used=data.get("source_used"),
            sources=[SourceRef(**ref) for ref in citations],
            trace=list(data.get("trace") or []),
            steps=len(data.get("steps") or []),
            tool_calls=sum(1 for s in (data.get("steps") or []) if s.get("action")),
            escalated=data.get("source_used") == "escalated",
            stop_reason=(
                (data.get("budget") or {}).get("stop_reason")
                if data.get("stopped_on_budget")
                else None
            ),
        )

    def _run_extractive(
        self, question: str, asked: str, tracer: Tracer
    ) -> AnswerResult:
        """Retrieve, then stitch sentences. No model involved at any point."""
        with tracer.span("extractive", kind="agent"):
            docs: list[Any] = []
            try:
                docs = list(self.retriever.invoke(asked) or [])
            except Exception:  # noqa: BLE001
                log.exception("Retrieval failed on the extractive path")
                tracer.degraded("retriever", reason="raised", kind="retrieval")
            extracted = answer_from_documents(asked, docs, tracer=tracer)

        if extracted.is_empty:
            return AnswerResult(
                question=question,
                answer=INSUFFICIENT_ANSWER,
                source_used="insufficient_evidence",
                kb_chunks=len(docs),
                sources=[SourceRef(**ref) for ref in cite_sources(docs)],
                trace=[
                    f"Retriever: {len(docs)} chunk(s)",
                    "Extractive: nothing covered the question closely enough",
                ],
            )
        return AnswerResult(
            question=question,
            answer=extracted.text,
            source_used="private_kb",
            route="kb",
            kb_grade="good" if extracted.coverage >= 0.4 else "weak",
            kb_chunks=len(docs),
            sources=[SourceRef(**ref) for ref in extracted.citations],
            trace=[
                f"Retriever: {len(docs)} chunk(s)",
                f"Extractive: {extracted.sentences_used} sentence(s), "
                f"coverage {extracted.coverage:.2f}",
            ],
        )

    # --- Non-run results -----------------------------------------------------
    def _blocked(
        self, question: str, verdict: Any, principal: Principal, byok: bool
    ) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer=BLOCKED_BY_GUARD,
            source_used="direct",
            answer_mode="blocked",
            blocked=True,
            guard_input=verdict.to_dict(),
            pii_redacted=list(verdict.redacted),
            byok=byok,
            trace=[f"Input guard: refused ({verdict.category})"],
        )

    def _from_cache(
        self,
        question: str,
        redacted: str | None,
        verdict: Any,
        hit: Any,
        byok: bool,
    ) -> AnswerResult | None:
        """Rebuild an answer from a cache entry, or None to treat it as a miss.

        The stored payload is replayed **whole**, citations included, and only
        the fields that describe *this* request are overwritten. Recomputing the
        citations would attach today's sources to yesterday's text, which is the
        one way a cache can turn a correct answer into a wrong one.

        A payload that no longer validates means the contract changed under a
        warm process -- a redeploy with the old process still serving, or a
        field removed. The whole cache is dropped and the question is answered
        properly: `None` here, not an exception, because the one thing that must
        not happen is a cache implementation detail turning into a 500 on a
        question that would otherwise have been answered.
        """
        try:
            result = AnswerResult.model_validate(hit.entry.payload)
        except Exception:  # noqa: BLE001 - a stale payload shape must not 500
            log.exception("A cached payload no longer validates; treating as a miss")
            self.cache.invalidate()
            return None

        result.question = question
        result.redacted_question = redacted
        result.guard_input = verdict.to_dict()
        result.answer_mode = "cached"
        result.cached = True
        result.cache_similarity = round(hit.similarity, 4)
        result.byok = byok
        # The replayed trace describes the *original* run, which is useful and
        # also misleading on its own -- nothing was retrieved or graded on this
        # request. One line says so.
        result.trace = [
            f"Cache: {hit.kind} hit at similarity {hit.similarity:.3f}; "
            f"nothing was retrieved or graded on this request",
            *result.trace,
        ]
        return result

    # --- Accounting and assembly --------------------------------------------
    def _finish(
        self,
        result: AnswerResult,
        started: float,
        tracer: Tracer,
        budget: Budget,
        conversation_id: str | None,
        principal: Principal,
        stream: Any,
    ) -> AnswerResult:
        """Stage 9 and 11: account for what the run spent, then assemble.

        **Tokens are estimated, not measured.** LangChain surfaces usage
        metadata inconsistently across providers and not at all through a
        compiled graph's final state, so counting the characters that went into
        and came out of the models is the number available without threading a
        callback handler through every node. It is characters/4 -- about +/-20%
        on English prose -- which is the right accuracy for a ceiling and a
        dashboard and the wrong accuracy for reconciling a bill. `/api/metrics`
        says so where the figure is shown.
        """
        spans = tracer.to_list()
        llm_calls = sum(1 for span in spans if span.get("kind") == "llm")

        # Every model call sees the question; the generating call also sees the
        # evidence. Approximated from what is known rather than threaded: a
        # chunk count times the configured chunk size is closer than ignoring
        # it. Divided by `CHARS_PER_TOKEN` directly rather than through
        # `estimate_tokens`, which takes text -- building a four-thousand
        # character placeholder string to measure its length would be absurd.
        prompt_chars = len(result.question) * max(1, llm_calls)
        prompt_chars += result.kb_chunks * self.settings.chunk_size
        result.prompt_tokens = (prompt_chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN
        result.completion_tokens = estimate_tokens(result.answer)
        if result.model:
            result.cost_usd = project_cost(
                result.model, result.prompt_tokens, result.completion_tokens
            )
        if not result.cached:
            budget.charge_llm(
                tokens=result.prompt_tokens + result.completion_tokens,
                cost_usd=result.cost_usd,
                calls=llm_calls,
            )

        report = budget.report()
        result.budget = report
        result.stop_reason = result.stop_reason or report.get("stop_reason")
        if result.stop_reason and not result.answer.endswith(
            BUDGET_STOPPED_NOTE.strip()
        ):
            result.answer += BUDGET_STOPPED_NOTE
        result.steps = result.steps or int(report.get("steps") or 0)
        result.tool_calls = result.tool_calls or int(report.get("tool_calls") or 0)

        result.spans = spans
        result.span_summary = tracer.summary()
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        result.conversation_id = conversation_id
        result.subject = principal.subject if principal else None

        if self.scratchpad is not None and conversation_id:
            try:
                result.scratchpad = [
                    note.to_dict()
                    for note in self.scratchpad.notes(conversation_id, limit=60)
                ]
            except Exception:  # noqa: BLE001 - notes are not worth an error
                log.exception("Could not read the scratchpad for the response")

        if stream is not None:
            stream.emit(EVENT_ANSWER, text=result.answer)
            if result.sources:
                stream.emit(
                    EVENT_CITATIONS,
                    sources=[s.model_dump() for s in result.sources],
                )
        return result


class _NoWebSearch:
    """A web-search client for a deployment with no Tavily key.

    Returns the empty shape the real client returns rather than raising, so
    `search_web` records "0 characters" and the graph takes the same path it
    takes when the web genuinely had nothing. A `None` here would have to be
    special-cased in the node, which is how the no-key deployment would start
    behaving differently from the keyed one in ways nobody tested.
    """

    def invoke(self, payload: Any = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
        query = payload.get("query") if isinstance(payload, Mapping) else ""
        return {"query": query, "answer": "", "results": []}
