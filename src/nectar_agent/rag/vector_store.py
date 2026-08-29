"""Pinecone vector store wrapper for the RAG pipeline (Task 3).

Wraps the Pinecone client so the rest of the codebase interacts with a
small, typed interface (``upsert_chunks`` / ``query``) instead of the raw
SDK. Also owns index creation, since a serverless Pinecone index must
exist with the right dimensionality before any vectors can be written.
"""

from __future__ import annotations

from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec

from nectar_agent.config import get_settings
from nectar_agent.rag.ingestion import DocumentChunk

_pinecone_client: Pinecone | None = None


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity search, with its score.

    Attributes:
        chunk_id: ID of the matched chunk.
        text: Chunk text content.
        score: Cosine similarity score in [0, 1] (higher = more similar).
        doc_id: Source document identifier.
        title: Source document title.
        source_type: "web_scrape" or "synthetic_doc".
        source_url: Origin URL, if the chunk came from a web-scraped page.
    """

    chunk_id: str
    text: str
    score: float
    doc_id: str
    title: str
    source_type: str
    source_url: str | None


def _get_client() -> Pinecone:
    """Return a lazily-initialized, module-level Pinecone client.

    Returns:
        A configured ``Pinecone`` client instance.

    Raises:
        RuntimeError: If the client cannot be constructed (e.g. an
            invalid API key).
    """
    global _pinecone_client
    try:
        if _pinecone_client is None:
            settings = get_settings()
            _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
        return _pinecone_client
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize the Pinecone client: {exc}") from exc


def ensure_index_exists() -> None:
    """Create the configured Pinecone index if it does not already exist.

    Uses a serverless spec so no capacity planning is required for a
    prototype-scale knowledge base. Safe to call repeatedly - a no-op if
    the index is already present.

    Raises:
        RuntimeError: If listing or creating the index fails (network
            error, invalid key, unsupported region, etc.).
    """
    try:
        settings = get_settings()
        client = _get_client()
        existing = {idx["name"] for idx in client.list_indexes()}
        if settings.pinecone_index_name not in existing:
            client.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.embedding_dimensions,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud, region=settings.pinecone_region
                ),
            )
    except Exception as exc:
        raise RuntimeError(f"Failed to ensure Pinecone index exists: {exc}") from exc


def upsert_chunks(chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
    """Upsert a batch of document chunks and their embeddings into Pinecone.

    Args:
        chunks: Chunks to store (provides IDs and metadata).
        embeddings: Embedding vectors, aligned by index with ``chunks``.

    Returns:
        Number of vectors upserted.

    Raises:
        ValueError: If ``chunks`` and ``embeddings`` have different lengths.
        RuntimeError: If the upsert request(s) to Pinecone fail.
    """
    try:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length.")
        if not chunks:
            return 0

        settings = get_settings()
        index = _get_client().Index(settings.pinecone_index_name)

        vectors = [
            {
                "id": chunk.chunk_id,
                "values": vector,
                "metadata": {
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source_type": chunk.source_type,
                    "source_url": chunk.source_url or "",
                    "chunk_index": chunk.chunk_index,
                },
            }
            for chunk, vector in zip(chunks, embeddings)
        ]

        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[start : start + batch_size])
        return len(vectors)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to upsert chunks to Pinecone: {exc}") from exc


def query(vector: list[float], top_k: int) -> list[RetrievedChunk]:
    """Run a similarity search against the knowledge-base index.

    Args:
        vector: Query embedding vector.
        top_k: Maximum number of results to return.

    Returns:
        List of ``RetrievedChunk`` results, ordered by descending
        similarity score.

    Raises:
        RuntimeError: If the Pinecone query request fails.
    """
    try:
        settings = get_settings()
        index = _get_client().Index(settings.pinecone_index_name)
        response = index.query(vector=vector, top_k=top_k, include_metadata=True)

        results: list[RetrievedChunk] = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            results.append(
                RetrievedChunk(
                    chunk_id=match["id"],
                    text=metadata.get("text", ""),
                    score=match.get("score", 0.0),
                    doc_id=metadata.get("doc_id", ""),
                    title=metadata.get("title", ""),
                    source_type=metadata.get("source_type", ""),
                    source_url=metadata.get("source_url") or None,
                )
            )
        return results
    except Exception as exc:
        raise RuntimeError(f"Failed to query Pinecone: {exc}") from exc
