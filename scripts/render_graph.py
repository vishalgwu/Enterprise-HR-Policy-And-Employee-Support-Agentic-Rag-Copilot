"""Regenerate docs/agent-graph.md from the compiled graph.

    python scripts/render_graph.py
    python scripts/render_graph.py --png    # posts the graph to mermaid.ink

Rendering from the compiled graph is what stops the diagram drifting from the
wiring. Building the graph constructs no clients here -- fakes are injected --
so this needs no API keys and makes no calls, except with --png.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/render_graph.py`
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from langchain_core.runnables import Runnable

from app.agent.diagram import graph_mermaid, save_graph_diagram, save_graph_png
from app.agent.graph import build_graph
from scripts._cli import parser_for


class _Unused(Runnable):
    """Stands in for a client the diagram never calls.

    A real Runnable subclass, not a duck type: `prompt | llm` goes through
    `coerce_to_runnable`, which rejects anything else outright.
    """

    def invoke(self, input=None, config=None, **kwargs):  # pragma: no cover
        raise AssertionError("rendering must not invoke a client")

    def with_structured_output(self, *args, **kwargs):
        return self


def main() -> None:

    parser = parser_for(__doc__)
    parser.add_argument(
        "--png",
        action="store_true",
        help="render a PNG instead -- posts the graph to the public mermaid.ink",
    )
    args = parser.parse_args()

    graph = build_graph(
        llm=_Unused(), retriever=_Unused(), web_search=_Unused(), verbose=False
    )

    if args.png:
        print(f"Wrote {save_graph_png(graph=graph)}")
        return

    print(graph_mermaid(graph))
    print(f"\nWrote {save_graph_diagram(graph=graph)}")


if __name__ == "__main__":
    main()
