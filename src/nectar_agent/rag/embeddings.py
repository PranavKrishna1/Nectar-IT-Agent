"""Embedding model wrapper for the RAG pipeline (Task 3).

Isolates the choice of embedding provider behind a small interface
(``embed_texts`` / ``embed_query``) so ``ingestion``/``retriever`` code
never touches a provider SDK directly.

Two backends are supported, selected by ``settings.embedding_provider``:

- ``"pinecone"`` (default): Pinecone's hosted Inference embeddings. Uses
  the same ``PINECONE_API_KEY`` as the vector store, and is included in
  Pinecone's free Starter plan allowance - so the whole RAG pipeline
  needs no additional paid API key.
- ``"openai"``: OpenAI's embedding endpoint, for when higher-quality or
  higher-throughput embeddings are worth a paid key.

Note the two backends produce different vector dimensionalities
(``multilingual-e5-large`` = 1024, ``text-embedding-3-small`` = 1536).
``settings.embedding_dimensions`` must match the active model, and the
Pinecone index must be created with that same dimension - switching
providers therefore requires re-creating the index and re-ingesting.
"""

from __future__ import annotations

from nectar_agent.config import get_settings

_openai_client = None


def _get_openai_client():
    """Return a lazily-initialized, module-level OpenAI client.

    Lazy initialization avoids requiring an OpenAI API key at import
    time, which matters because the default configuration doesn't use
    OpenAI at all.
    """
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        settings = get_settings()
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _embed_texts_openai(texts: list[str]) -> list[list[float]]:
    """Embed texts using OpenAI's embeddings endpoint, aligned by index with ``texts``."""
    settings = get_settings()
    response = _get_openai_client().embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def _embed_texts_pinecone(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed texts using Pinecone's hosted Inference embeddings.

    Pinecone's embedding models distinguish between indexing passages and
    embedding a search query, and produce better retrieval quality when
    told which is which - hence the explicit ``input_type``, either
    ``"passage"`` or ``"query"``.
    """
    from pinecone import Pinecone

    settings = get_settings()
    client = Pinecone(api_key=settings.pinecone_api_key)
    response = client.inference.embed(
        model=settings.embedding_model,
        inputs=texts,
        parameters={"input_type": input_type, "truncate": "END"},
    )
    return [record["values"] for record in response.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document chunks for storage in the vector database.

    Returns an empty list for empty input without calling any API.

    Raises:
        ValueError: If ``settings.embedding_provider`` is not supported.
    """
    if not texts:
        return []
    settings = get_settings()
    if settings.embedding_provider == "pinecone":
        return _embed_texts_pinecone(texts, input_type="passage")
    if settings.embedding_provider == "openai":
        return _embed_texts_openai(texts)
    raise ValueError(f"Unsupported embedding_provider: {settings.embedding_provider}")


def embed_query(query: str) -> list[float]:
    """Embed a single user query for similarity search.

    Raises:
        ValueError: If ``settings.embedding_provider`` is not supported.
    """
    settings = get_settings()
    if settings.embedding_provider == "pinecone":
        return _embed_texts_pinecone([query], input_type="query")[0]
    if settings.embedding_provider == "openai":
        return _embed_texts_openai([query])[0]
    raise ValueError(f"Unsupported embedding_provider: {settings.embedding_provider}")
