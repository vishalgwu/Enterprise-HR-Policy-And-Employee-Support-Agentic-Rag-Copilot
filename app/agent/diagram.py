"""Render the compiled graph, so a diagram can never drift from the wiring.

Separate from `graph.py` because rendering is a build-time concern and one of
the two renderers makes a network call, which nothing importing the graph should
risk pulling in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agent.graph import build_graph
from app.core.config import PROJECT_ROOT

DIAGRAM_PATH = PROJECT_ROOT / "docs" / "agent-graph.md"
PNG_PATH = PROJECT_ROOT / "docs" / "agent-graph.png"


def graph_mermaid(graph: Any = None) -> str:
    """Mermaid source for the compiled graph."""
    return (graph or build_graph(verbose=False)).get_graph().draw_mermaid()


def save_graph_diagram(path: Path | None = None, graph: Any = None) -> Path:
    """Write the diagram to a markdown file. GitHub renders mermaid inline."""
    path = Path(path) if path else DIAGRAM_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "# Agent graph\n\n"
        "Generated from the compiled graph by "
        "`python scripts/render_graph.py`. Do not edit by hand.\n\n"
        "Solid arrows are unconditional edges; dotted arrows are branches taken "
        "on a structured decision.\n\n"
        "```mermaid\n" + graph_mermaid(graph).strip() + "\n```\n"
    )
    path.write_bytes(body.encode("utf-8"))
    return path


def save_graph_png(path: Path | None = None, graph: Any = None) -> Path:
    """Write a PNG of the graph.

    Opt-in, never called by default: `draw_mermaid_png` posts the graph to the
    public mermaid.ink service. Harmless here -- node names only -- but it is
    still a network call to a third party.
    """
    path = Path(path) if path else PNG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    png = (graph or build_graph(verbose=False)).get_graph().draw_mermaid_png()
    path.write_bytes(png)
    return path
