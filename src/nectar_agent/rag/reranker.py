"""Lightweight reranking/filtering stage for retrieved chunks (Task 3).

Pinecone's cosine similarity is a good first pass but is purely
vector-geometric; this module applies a cheap, dependency-free lexical
overlap boost on top of it to demote chunks that are vector-similar but
topically off (a known weakness of embedding-only retrieval on short
technical queries), and to deduplicate near-identical chunks from the
same document before they reach the LLM's context window. This keeps the
pipeline's "Reranker / Filtering" stage from the brief's diagram cheap
and dependency-light while still meaningfully improving precision; a
cross-encoder model could be dropped in here later without changing the
call signature.
"""

from __future__ import annotations

import re

from nectar_agent.rag.vector_store import RetrievedChunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase and tokenize text into a set of alphanumeric words.

    Args:
        text: Input text.

    Returns:
        Set of lowercase word tokens.

    Raises:
        RuntimeError: If tokenization fails unexpectedly (e.g. ``text``
            is not a string).
    """
    try:
        return set(_WORD_RE.findall(text.lower()))
    except Exception as exc:
        raise RuntimeError(f"Failed to tokenize text: {exc}") from exc


def rerank(query_text: str, chunks: list[RetrievedChunk], top_n: int = 4) -> list[RetrievedChunk]:
    """Rerank retrieved chunks using a similarity-plus-lexical-overlap score.

    Args:
        query_text: The original user query.
        chunks: Chunks returned by ``retriever.retrieve``, already above
            the minimum vector-similarity threshold.
        top_n: Maximum number of chunks to keep after reranking.

    Returns:
        Up to ``top_n`` chunks, reordered by combined score, with
        duplicate/near-duplicate chunks from the same document collapsed.

    Raises:
        RuntimeError: If reranking fails unexpectedly.
    """
    try:
        if not chunks:
            return []

        query_terms = _tokenize(query_text)
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            overlap = len(query_terms & _tokenize(chunk.text))
            lexical_boost = overlap / max(len(query_terms), 1)
            combined = (0.75 * chunk.score) + (0.25 * lexical_boost)
            scored.append((combined, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        seen_doc_chunks: set[tuple[str, str]] = set()
        deduped: list[RetrievedChunk] = []
        for _, chunk in scored:
            key = (chunk.doc_id, chunk.text[:80])
            if key in seen_doc_chunks:
                continue
            seen_doc_chunks.add(key)
            deduped.append(chunk)
            if len(deduped) >= top_n:
                break
        return deduped
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to rerank chunks: {exc}") from exc
