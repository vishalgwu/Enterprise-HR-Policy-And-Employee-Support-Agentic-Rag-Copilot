"""Score the router and the evidence grader against a labelled set.

    python scripts/eval_decisions.py
    python scripts/eval_decisions.py --fewshot
    python scripts/eval_decisions.py --router-only
    python scripts/eval_decisions.py --runs 3

This is the measurement behind the README's "21/21 on a labelled set", and the
A/B harness for `FEWSHOT_GRADING`. It existed only as a filename in a comment
in `app/agent/prompts.py` until a documentation pass noticed the script did not
exist -- which meant the number was reproducible by hand and by nobody else.

**Routing and grading are the two decisions that cannot fail quietly.** A
wrong route sends an answerable question to the web, or answers an external one
from memory. A wrong grade either discards good evidence or passes bad evidence
to the generator with a citation attached. Both are prompt-sensitive, so both
need a fixed set to re-measure against after a prompt is touched.

**Why `--runs`.** `temperature=0` is not determinism on Groq. `CLAUDE.md`
records a genuinely borderline grading case that returned `weak` 8/8 in
isolation and flipped to `good` inside a longer suite run. One pass over this
set is a sample, not a score; `--runs 3` reports the spread, and a case that
disagrees with itself across runs is marked rather than averaged away.

Needs live credentials -- it is measuring a model, so there is nothing to fake.
The offline suite covers the recovery paths around these calls.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/eval_decisions.py`
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from collections import Counter
from typing import Any

from app.agent.decisions import get_evidence_grader, get_router
from app.agent.decisions import grade_evidence as run_grade
from app.agent.decisions import route_question as run_route
from app.core.config import get_settings
from app.rag.clients import get_llm
from scripts._cli import parser_for

# --- The labelled sets -------------------------------------------------------
#
# Both are deliberately small and deliberately adversarial. The value is not
# coverage -- it is that every entry is a case that has been got wrong by some
# version of these prompts.

# (message, expected route). The router prompt is a two-sided calibration and
# this set is both sides: external factual questions must reach `kb` (otherwise
# the model answers from memory with no evidence), while greetings AND
# questions about the assistant itself must stay `direct`. An earlier prompt
# pushed everything informational to `kb`, so "who are you?" retrieved nothing,
# fell through to web search, and answered from an unrelated page.
ROUTER_CASES: tuple[tuple[str, str], ...] = (
    # --- direct: no evidence lookup would help -------------------------------
    ("hi", "direct"),
    ("good morning", "direct"),
    ("thanks, that's helpful", "direct"),
    ("who are you?", "direct"),
    ("what can you help me with?", "direct"),
    ("bye", "direct"),
    # --- kb: a fact is being asked for ---------------------------------------
    ("How many days of paid time off do I get?", "kb"),
    ("What is the notice period if I resign?", "kb"),
    ("When are approved expenses paid out?", "kb"),
    ("Can I work fully remotely from another city?", "kb"),
    ("What is the process for raising a grievance?", "kb"),
    ("Does my insurance cover dental?", "kb"),
    ("How do I get a copy of my payslip?", "kb"),
    ("What are the core working hours?", "kb"),
    # External facts route to `kb` too. Downstream stages check whether the
    # knowledge base covers it and fall back to the web when it does not, so
    # routing here is never wasted -- whereas `direct` would answer from
    # memory, which is the one thing this product must not do.
    ("What is the statutory minimum notice period in India?", "kb"),
    ("What is the capital of France?", "kb"),
)

# (question, evidence, expected grade). The hard cases are the `weak` ones:
# evidence from the right general subject that does not contain the specific
# fact. The similarity gate cannot catch those -- a wrong-source chunk has
# scored 0.521 against a weakest-correct of 0.171 -- so they are the grader's
# whole job.
GRADER_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "How many days of paid time off do full-time employees get?",
        "[1] source=leave-and-time-off.md\nFull-time employees accrue 20 days "
        "of paid time off per calendar year, accruing monthly from the start "
        "date.",
        "good",
    ),
    (
        "When are approved expenses paid out?",
        "[1] source=expense-reimbursement.md\nApproved expenses are paid with "
        "the next payroll run after approval.",
        "good",
    ),
    (
        "What are the core hours?",
        "[1] source=remote-and-hybrid-work.md\nCore hours are 10:00 to 16:00 "
        "local time, during which employees are expected to be reachable.",
        "good",
    ),
    (
        "When are expenses paid out?",
        "[1] source=attendance-and-working-hours.md\nCore hours are 10:00 to "
        "16:00 local time.\n\n[2] source=expense-reimbursement.md\nApproved "
        "expenses are paid with the next payroll run after approval.",
        "good",
    ),
    # Right document, wrong fact. The commonest real failure.
    (
        "How much notice do I have to give to resign?",
        "[1] source=onboarding-and-offboarding.md\nOffboarding begins when HR "
        "receives written notice. The manager completes an exit checklist and "
        "IT revokes access on the final working day.",
        "weak",
    ),
    # Adjacent subject, plausible-looking.
    (
        "Can I claim a taxi home after working late?",
        "[1] source=remote-and-hybrid-work.md\nEmployees working from home may "
        "claim a one-off home office allowance within their first month.",
        "weak",
    ),
    # Generic, on-topic, answers nothing.
    (
        "How is my bonus calculated?",
        "[1] source=payroll-and-compensation.md\nPayroll is processed monthly. "
        "Queries should be raised with the HR Service Desk.",
        "weak",
    ),
    # Nothing at all. Handled before the model is called, and asserted here so
    # the shortcut stays correct rather than merely fast.
    ("What is the sabbatical policy?", "", "weak"),
)


# --- Running -----------------------------------------------------------------


def _score(results: list[tuple[str, str, str]]) -> tuple[int, int]:
    """(correct, total) over (subject, expected, actual) rows."""
    return sum(1 for _s, want, got in results if want == got), len(results)


def _report(title: str, results: list[tuple[str, str, str]]) -> int:
    correct, total = _score(results)
    print(f"\n{title}: {correct}/{total}")
    for subject, want, got in results:
        if want == got:
            continue
        print(f"  MISS  expected {want:7s} got {got:7s}  {subject[:70]!r}")
    return total - correct


def run_router(llm: Any, runs: int) -> list[tuple[str, str, str]]:
    router = get_router(llm)
    results: list[tuple[str, str, str]] = []
    for message, expected in ROUTER_CASES:
        seen = Counter(run_route(message, router) for _ in range(runs))
        # The majority answer, with disagreement reported rather than hidden:
        # a case that is 2/3 is not the same evidence as one that is 3/3, and
        # averaging it away is how a flaky decision looks stable.
        actual, votes = seen.most_common(1)[0]
        if votes < runs:
            print(f"  FLAKY {dict(seen)}  {message[:60]!r}")
        results.append((message, expected, actual))
    return results


def run_grader(llm: Any, runs: int, settings: Any) -> list[tuple[str, str, str]]:
    grader = get_evidence_grader(llm, settings=settings)
    results: list[tuple[str, str, str]] = []
    for question, evidence, expected in GRADER_CASES:
        seen = Counter(run_grade(question, evidence, grader) for _ in range(runs))
        actual, votes = seen.most_common(1)[0]
        if votes < runs:
            print(f"  FLAKY {dict(seen)}  {question[:60]!r}")
        results.append((question, expected, actual))
    return results


def main() -> None:
    settings = get_settings()

    parser = parser_for(__doc__)
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help=(
            "grade with GRADER_PROMPT_FEWSHOT instead of GRADER_PROMPT. Run "
            "both and compare before changing the default -- the 0.15 gate and "
            "the published 21/21 were measured against the plain prompt."
        ),
    )
    parser.add_argument("--router-only", action="store_true")
    parser.add_argument("--grader-only", action="store_true")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="repeat each case N times and take the majority. temperature=0 "
        "is not determinism on Groq.",
    )
    parser.add_argument("--model", default=None, help="override the chat model")
    args = parser.parse_args()
    # After parse_args, so --help works in a checkout with no credentials.
    settings.validate_required()

    if args.fewshot:
        settings = settings.model_copy(update={"fewshot_grading": True})

    llm = get_llm(model=args.model, settings=settings)

    print(f"Model         {args.model or settings.active_llm_model}")
    print(f"Provider      {settings.llm_provider}")
    print(f"Grader prompt {'few-shot' if settings.fewshot_grading else 'plain'}")
    print(f"Runs per case {args.runs}")

    misses = 0
    if not args.grader_only:
        misses += _report("Router", run_router(llm, args.runs))
    if not args.router_only:
        misses += _report("Grader", run_grader(llm, args.runs, settings))

    total = (0 if args.grader_only else len(ROUTER_CASES)) + (
        0 if args.router_only else len(GRADER_CASES)
    )
    print(f"\nOverall: {total - misses}/{total}")
    if misses:
        print(
            "\nA miss is a prompt problem or a quota problem, and they look "
            "alike: every safe default here is the conservative one, so an "
            "exhausted token budget grades everything weak. Check the quota "
            "before editing a prompt."
        )


if __name__ == "__main__":
    main()
