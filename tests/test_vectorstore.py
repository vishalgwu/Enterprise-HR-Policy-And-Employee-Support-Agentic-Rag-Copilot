"""Pure helpers in the vector-store layer, and the filter builder above it.

Everything here runs without Pinecone. The score-scale conversion gets its own
test because getting it wrong silently disables the gate rather than raising.
"""

from __future__ import annotations

import pytest

from app.rag.retrieval import build_filter
from app.rag.vectorstore import _attr, _index_ready, to_relevance_scale


# --- Score scale -------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0), (0.15, 0.575)],
)
def test_raw_cosine_maps_onto_langchains_relevance_scale(raw, expected):
    assert to_relevance_scale(raw) == pytest.approx(expected)


def test_passing_a_raw_cosine_straight_through_would_filter_nothing():
    """0.15 read as a relevance score means raw cosine -0.70, which gates nothing."""
    assert to_relevance_scale(-0.70) == pytest.approx(0.15)


# --- Pinecone shape tolerance ------------------------------------------------
# The v7 client returns objects or dicts depending on the call path.


class _Status:
    ready = True


class _Description:
    status = _Status()
    dimension = 384


def test_attr_reads_objects_and_dicts_alike():
    assert _attr({"dimension": 384}, "dimension") == 384
    assert _attr(_Description(), "dimension") == 384
    assert _attr({}, "missing") is None
    assert _attr(_Description(), "missing") is None


def test_index_ready_reads_both_shapes():
    assert _index_ready({"status": {"ready": True}})
    assert _index_ready(_Description())
    assert not _index_ready({"status": {"ready": False}})
    assert not _index_ready({})


# --- Metadata filters --------------------------------------------------------


def test_no_constraints_means_no_filter():
    assert build_filter() is None


def test_department_filter():
    assert build_filter(department="payroll") == {"department": {"$eq": "payroll"}}


def test_filters_compose():
    assert build_filter(department="payroll", doc_type="pdf") == {
        "department": {"$eq": "payroll"},
        "doc_type": {"$eq": "pdf"},
    }
