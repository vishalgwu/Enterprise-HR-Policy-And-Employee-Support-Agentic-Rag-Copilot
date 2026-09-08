"""Loading, chunk identity, chunking and evidence rendering."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.rag.loaders import (
    SUPPORTED_SUFFIXES,
    chunk_id,
    cite_sources,
    class_matcher,
    format_documents,
    is_supported,
    load_directory,
    load_file,
    read_document_text,
    separators_for,
    split_documents,
)

# --- Chunk identity ----------------------------------------------------------


def _chunk(origin="hr-docs", source="a.md", start=0):
    return Document(
        page_content="text",
        metadata={"origin": origin, "source": source, "start_index": start},
    )


def test_chunk_id_is_stable_across_calls():
    assert chunk_id(_chunk()) == chunk_id(_chunk())


def test_chunk_id_changes_with_offset():
    assert chunk_id(_chunk(start=0)) != chunk_id(_chunk(start=900))


def test_chunk_id_is_independent_of_content():
    """An edited chunk must update in place, not leave its old version behind."""
    a, b = _chunk(), _chunk()
    b.page_content = "completely different text"
    assert chunk_id(a) == chunk_id(b)


def test_chunk_id_separates_corpora_with_the_same_file_name():
    """Without origin in the basis, two departments' benefits.pdf collide."""
    assert chunk_id(_chunk(origin="hr-docs")) != chunk_id(_chunk(origin="public-web"))


# --- Format support ----------------------------------------------------------


def test_the_four_brief_formats_are_supported():
    assert {".md", ".txt", ".pdf", ".docx"} <= set(SUPPORTED_SUFFIXES)


def test_unsupported_suffix_is_rejected_by_name(tmp_path):
    path = tmp_path / "policy.rtf"
    path.write_text("x", encoding="utf-8")
    assert not is_supported(path)
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_document_text(path)


def test_docx_tables_are_read(tmp_path):
    """Benefit tiers and expense caps live in tables, not in paragraphs."""
    docx = pytest.importorskip("docx")
    path = tmp_path / "benefits.docx"
    document = docx.Document()
    document.add_paragraph("Benefits")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Tier A"
    table.rows[0].cells[1].text = "90% covered"
    document.save(str(path))

    text = read_document_text(path)
    assert "Tier A" in text and "90% covered" in text


# --- Metadata ----------------------------------------------------------------


def test_load_file_records_the_metadata_retrieval_filters_on(tmp_path):
    path = tmp_path / "leave.md"
    path.write_text("# Leave Policy\n\nTwenty days.", encoding="utf-8")

    document = load_file(path, origin="hr-docs", department="people-ops")
    assert document is not None
    assert document.metadata["source"] == "leave.md"
    assert document.metadata["title"] == "Leave Policy"
    assert document.metadata["department"] == "people-ops"
    assert document.metadata["doc_type"] == "markdown"
    assert document.metadata["origin"] == "hr-docs"


def test_empty_file_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "blank.md"
    path.write_text("   \n\n", encoding="utf-8")
    assert load_file(path, origin="hr-docs") is None


def test_subdirectory_becomes_the_department(tmp_path):
    (tmp_path / "payroll").mkdir()
    (tmp_path / "payroll" / "bonus.md").write_text("# Bonus", encoding="utf-8")
    (tmp_path / "handbook.md").write_text("# Handbook", encoding="utf-8")

    loaded = load_directory(tmp_path, origin="hr-docs")
    by_source = {d.metadata["source"]: d for d in loaded}
    assert by_source["bonus.md"].metadata["department"] == "payroll"
    assert by_source["handbook.md"].metadata["department"] == "general"


def test_unsupported_files_are_skipped_not_fatal(tmp_path):
    (tmp_path / "notes.rtf").write_text("ignore me", encoding="utf-8")
    (tmp_path / "policy.md").write_text("# Policy", encoding="utf-8")
    documents = load_directory(tmp_path, origin="hr-docs")
    assert [d.metadata["source"] for d in documents] == ["policy.md"]


def test_directory_with_nothing_usable_raises(tmp_path):
    (tmp_path / "notes.rtf").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="No supported documents"):
        load_directory(tmp_path, origin="hr-docs")


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "nope", origin="hr-docs")


# --- Chunking ----------------------------------------------------------------


def test_markdown_splits_on_headings_and_prose_does_not():
    assert separators_for("markdown")[0] == "\n## "
    assert separators_for("text")[0] == "\n\n"
    assert separators_for(None)[0] == "\n\n"


def test_split_documents_chooses_separators_per_document():
    """A mixed directory is one call, not one call per format."""
    markdown = Document(
        page_content="# T\n\n## A\n" + ("a" * 600) + "\n## B\n" + ("b" * 600),
        metadata={"doc_type": "markdown", "source": "a.md", "origin": "hr-docs"},
    )
    text = Document(
        page_content=("word " * 400),
        metadata={"doc_type": "text", "source": "b.txt", "origin": "hr-docs"},
    )
    chunks = split_documents([markdown, text], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 2
    assert {c.metadata["source"] for c in chunks} == {"a.md", "b.txt"}
    assert all("start_index" in c.metadata for c in chunks)


def test_chunks_get_distinct_ids():
    document = Document(
        page_content=("word " * 800),
        metadata={"doc_type": "text", "source": "b.txt", "origin": "hr-docs"},
    )
    chunks = split_documents([document], chunk_size=400, chunk_overlap=40)
    ids = [chunk_id(c) for c in chunks]
    assert len(ids) == len(set(ids))


# --- Rendering ---------------------------------------------------------------


def test_format_documents_numbers_and_labels_each_chunk():
    docs = [
        Document(page_content="A", metadata={"source": "one.md"}),
        Document(page_content="B", metadata={"source": "two.md"}),
    ]
    rendered = format_documents(docs)
    assert rendered == "[1] source=one.md\nA\n\n[2] source=two.md\nB"


def test_format_documents_of_nothing_is_empty():
    assert format_documents([]) == ""


def test_cite_sources_deduplicates_by_source():
    docs = [
        Document(page_content="A", metadata={"source": "one.md", "title": "One"}),
        Document(page_content="B", metadata={"source": "one.md", "title": "One"}),
        Document(page_content="C", metadata={"source": "two.md", "title": "Two"}),
    ]
    assert [s["source"] for s in cite_sources(docs)] == ["one.md", "two.md"]


# --- bs4 ---------------------------------------------------------------------


def test_class_matcher_matches_one_token_of_a_class_attribute():
    """bs4 compares class_ against the whole attribute, so a string fails."""
    matches = class_matcher("entry-content")
    assert matches("entry-content section-padding relative")
    assert matches("entry-content")
    assert not matches("entry-content-wrapper")
    assert not matches("")
    assert not matches(None)
