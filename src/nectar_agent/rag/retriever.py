"""Top-level retrieval interface for the RAG pipeline (Task 3).

Combines query embedding, Pinecone similarity search, and the minimum
relevance threshold into one function so agent code has a single call to
make. This is also where the "answer does not exist in the knowledge
base" decision is made: if nothing clears ``rag_min_similarity``, the
caller is told so explicitly instead of being handed weak matches to
hallucinate around.
"""

from __future__ import annotations

from nectar_agent.config import get_settings
from nectar_agent.rag import vector_store
from nectar_agent.rag.embeddings import embed_query
from nectar_agent.rag.vector_store import RetrievedChunk


def retrieve(query_text: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Retrieve the most relevant knowledge-base chunks for a query.

    Args:
        query_text: The natural-language question to search for.
        top_k: Number of chunks to retrieve; defaults to
            ``settings.rag_top_k`` if not given.

    Returns:
        List of chunks that meet ``settings.rag_min_similarity``,
        ordered by descending similarity. Empty if nothing relevant
        enough was found - callers must treat this as "not found in the
        knowledge base," not silently proceed.

    Raises:
        RuntimeError: If embedding the query or querying the vector
            store fails.
    """
    try:
        settings = get_settings()
        k = top_k or settings.rag_top_k
        query_vector = embed_query(query_text)
        matches = vector_store.query(query_vector, top_k=k)
        return [m for m in matches if m.score >= settings.rag_min_similarity]
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to retrieve knowledge-base chunks: {exc}") from exc
