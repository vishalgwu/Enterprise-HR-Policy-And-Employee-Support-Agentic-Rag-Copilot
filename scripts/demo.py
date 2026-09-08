"""Four demos, one per path the agent can take.

    python scripts/demo.py        run all four
    python scripts/demo.py 2      run demo 2 only
    python scripts/demo.py -q     run all four without the per-node trace

Demos 2 and 4 both end in web search but for different reasons, and the
distinction is the point: demo 2 is a genuine HR question this company's
knowledge base does not cover, demo 4 is a question no internal policy could
ever answer. Both must be visibly labelled as public information rather than
company policy.

The node trace goes to stderr via logging, so `python scripts/demo.py > out.txt`
captures the answers without it.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/demo.py`
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse

from app.core.config import configure_logging, configure_stdout, get_settings
from app.services.copilot import Copilot, render_answer

DEMOS = [
    (
        "Demo 1 - Answer from the Private KB",
        "How many PTO days do I get per year, and do unused days carry over?",
        "In the KB. Should retrieve, grade good, and answer with citations.",
    ),
    (
        "Demo 2 - Web Search Fallback",
        "What is the statutory minimum paid holiday entitlement in the UK?",
        "A real HR question this KB does not cover. Should fall back to the web "
        "and be marked as public information, not company policy.",
    ),
    (
        "Demo 3 - Direct Answer",
        "hi there, thanks for the help earlier!",
        "No lookup needed. Should route direct and skip retrieval entirely.",
    ),
    (
        "Demo 4 - Current / External Question",
        "What does recent research say about four-day work weeks and productivity?",
        "Current external knowledge, nothing internal could answer it. Should "
        "reach the web with cited URLs.",
    ),
]


def run(selected: int | None = None, verbose: bool = True) -> None:
    configure_stdout()
    configure_logging()
    get_settings().validate_required()

    # One Copilot for every demo: it holds a single compiled graph, and the
    # router, grader, retriever and search client do not vary per question.
    copilot = Copilot(verbose=verbose)

    chosen = DEMOS if selected is None else [DEMOS[selected - 1]]
    results = []
    for title, question, expectation in chosen:
        print("\n" + "#" * 86)
        print(f"# {title}")
        print(f"# {expectation}")
        print("#" * 86)
        result = copilot.ask(question)
        print(render_answer(result))
        results.append((title, result))

    if len(results) > 1:
        print("\n" + "=" * 86)
        print("SUMMARY")
        print("=" * 86)
        print(f"{'demo':<40} {'route':<8} {'source_used':<22} retries")
        for title, result in results:
            print(
                f"{title[:38]:<40} {str(result.route):<8} "
                f"{str(result.source_used):<22} {result.retry_count}"
            )


def main() -> None:
    # argparse rather than reading sys.argv by hand. Hand-rolled parsing here
    # silently ignored --help and ran all four demos instead of printing it --
    # four live Groq, Pinecone and Tavily round trips for someone who asked
    # what the flags were. Parsing first is also what keeps --help working in a
    # checkout with no credentials, since run() calls validate_required().
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "demo",
        nargs="?",
        type=int,
        choices=range(1, len(DEMOS) + 1),
        help="run one demo instead of all four",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the per-node trace on stderr",
    )
    args = parser.parse_args()
    run(args.demo, verbose=not args.quiet)


if __name__ == "__main__":
    main()
