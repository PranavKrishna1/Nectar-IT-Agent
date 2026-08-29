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

    Returns:
        A configured ``OpenAI`` client instance.

    Raises:
        RuntimeError: If the client cannot be constructed (e.g. the
            ``openai`` package is missing or the API key is invalid).
    """
    global _openai_client
    try:
        if _openai_client is None:
            from openai import OpenAI

            settings = get_settings()
            _openai_client = OpenAI(api_key=settings.openai_api_key)
        return _openai_client
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize the OpenAI client: {exc}") from exc


def _embed_texts_openai(texts: list[str]) -> list[list[float]]:
    """Embed texts using OpenAI's embeddings endpoint.

    Args:
        texts: Texts to embed, in order.

    Returns:
        Embedding vectors aligned by index with ``texts``.

    Raises:
        RuntimeError: If the OpenAI embeddings request fails (network
            error, invalid key, rate limit, etc.).
    """
    try:
        settings = get_settings()
        response = _get_openai_client().embeddings.create(
            model=settings.embedding_model, input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as exc:
        raise RuntimeError(f"OpenAI embedding request failed: {exc}") from exc


def _embed_texts_pinecone(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed texts using Pinecone's hosted Inference embeddings.

    Pinecone's embedding models distinguish between indexing passages and
    embedding a search query, and produce better retrieval quality when
    told which is which - hence the explicit ``input_type``.

    Args:
        texts: Texts to embed, in order.
        input_type: Either ``"passage"`` (for documents being indexed) or
            ``"query"`` (for a user's search query).

    Returns:
        Embedding vectors aligned by index with ``texts``.

    Raises:
        RuntimeError: If the Pinecone Inference request fails (network
            error, invalid key, rate limit, etc.).
    """
    try:
        from pinecone import Pinecone

        settings = get_settings()
        client = Pinecone(api_key=settings.pinecone_api_key)
        response = client.inference.embed(
            model=settings.embedding_model,
            inputs=texts,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        return [record["values"] for record in response.data]
    except Exception as exc:
        raise RuntimeError(f"Pinecone embedding request failed: {exc}") from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document chunks for storage in the vector database.

    Args:
        texts: Chunk texts to embed, in order.

    Returns:
        List of embedding vectors, one per input text, in the same order.
        Returns an empty list for empty input without calling any API.

    Raises:
        ValueError: If ``settings.embedding_provider`` is not supported.
        RuntimeError: If the underlying embedding request fails for any
            other reason.
    """
    try:
        if not texts:
            return []
        settings = get_settings()
        if settings.embedding_provider == "pinecone":
            return _embed_texts_pinecone(texts, input_type="passage")
        if settings.embedding_provider == "openai":
            return _embed_texts_openai(texts)
        raise ValueError(f"Unsupported embedding_provider: {settings.embedding_provider}")
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to embed texts: {exc}") from exc


def embed_query(query: str) -> list[float]:
    """Embed a single user query for similarity search.

    Args:
        query: The natural-language query text.

    Returns:
        The query's embedding vector.

    Raises:
        ValueError: If ``settings.embedding_provider`` is not supported.
        RuntimeError: If the underlying embedding request fails for any
            other reason.
    """
    try:
        settings = get_settings()
        if settings.embedding_provider == "pinecone":
            return _embed_texts_pinecone([query], input_type="query")[0]
        if settings.embedding_provider == "openai":
            return _embed_texts_openai([query])[0]
        raise ValueError(f"Unsupported embedding_provider: {settings.embedding_provider}")
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to embed query: {exc}") from exc
