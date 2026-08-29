"""Unit tests for the RAG pipeline's pure (non-network) building blocks.

Covers chunking and reranking, which have no external dependency, and
therefore need no API keys or Pinecone/OpenAI access to test. The RAG
*agent's* end-to-end grounded-answer behaviour (Task 3's "ask a question
not in the KB" requirement) is exercised via
``docs/sample_conversations.md`` / manual evaluation instead, since it
inherently requires a live embedding model and vector index.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nectar_agent.rag.ingestion import chunk_text
from nectar_agent.rag.reranker import rerank
from nectar_agent.rag.vector_store import RetrievedChunk


def test_chunk_text_respects_target_size() -> None:
    paragraph = "This is a sentence about AHU troubleshooting. " * 40
    text = "\n\n".join([paragraph] * 3)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    # Allow some slack for the overlap prefix appended to each chunk.
    assert all(len(c) <= 500 + 50 + 5 for c in chunks)


def test_chunk_text_single_short_document_returns_one_chunk() -> None:
    text = "Short safety note: always de-energize equipment before servicing."
    chunks = chunk_text(text, chunk_size=800, overlap=120)
    assert len(chunks) == 1
    assert chunks[0] == text


def _make_chunk(chunk_id: str, text: str, score: float, doc_id: str = "doc1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        doc_id=doc_id,
        title="Test Doc",
        source_type="synthetic_doc",
        source_url=None,
    )


def test_rerank_prefers_higher_lexical_overlap_when_scores_tie() -> None:
    chunks = [
        _make_chunk("a", "General information about building maintenance schedules.", 0.5),
        _make_chunk("b", "AHU airflow troubleshooting: check filters and belts.", 0.5),
    ]
    ranked = rerank("AHU airflow troubleshooting", chunks, top_n=2)
    assert ranked[0].chunk_id == "b"


def test_rerank_deduplicates_near_identical_chunks() -> None:
    chunks = [
        _make_chunk("a", "Chiller supply water temperature should read 6-8C.", 0.8),
        _make_chunk("b", "Chiller supply water temperature should read 6-8C.", 0.79),
    ]
    ranked = rerank("chiller supply water temperature", chunks, top_n=5)
    assert len(ranked) == 1


def test_rerank_respects_top_n() -> None:
    chunks = [_make_chunk(str(i), f"chunk {i} about HVAC", 0.9 - i * 0.01) for i in range(10)]
    ranked = rerank("HVAC", chunks, top_n=4)
    assert len(ranked) == 4
