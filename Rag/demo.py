"""Four demos, one per path the agent can take.

    python Rag/demo.py        run all four
    python Rag/demo.py 2      run demo 2 only
    python Rag/demo.py -q     run all four without the per-node trace

Demos 2 and 4 both end in web search but for different reasons, and the
distinction is the point: demo 2 is a genuine HR question this company's
knowledge base does not cover, demo 4 is a question no internal policy could
ever answer. Both must be visibly labelled as public information rather than
company policy.
"""

import sys

if __package__ in (None, ""):  # allow `python Rag/demo.py`
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from Rag.config import configure_logging, configure_stdout
from Rag.graph import ask_agent, build_graph

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
    # Node trace goes to stderr via logging, so stdout carries only the answers.
    configure_logging()
    # One graph for every demo: building it constructs the router, grader,
    # retriever and search tool, none of which vary per question.
    graph = build_graph(verbose=verbose)

    chosen = DEMOS if selected is None else [DEMOS[selected - 1]]
    results = []
    for title, question, expectation in chosen:
        print("\n" + "#" * 86)
        print(f"# {title}")
        print(f"# {expectation}")
        print("#" * 86)
        results.append((title, ask_agent(question, graph=graph, verbose=verbose)))

    if len(results) > 1:
        print("\n" + "=" * 86)
        print("SUMMARY")
        print("=" * 86)
        print(f"{'demo':<40} {'route':<8} {'source_used':<22} retries")
        for title, r in results:
            print(f"{title[:38]:<40} {str(r.get('route')):<8} "
                  f"{str(r.get('source_used')):<22} {r.get('retry_count', 0)}")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    verbose = "-q" not in args
    numbers = [int(a) for a in args if a.isdigit()]
    if numbers and not 1 <= numbers[0] <= len(DEMOS):
        raise SystemExit(f"demo must be 1..{len(DEMOS)}")
    run(numbers[0] if numbers else None, verbose=verbose)


if __name__ == "__main__":
    main()
