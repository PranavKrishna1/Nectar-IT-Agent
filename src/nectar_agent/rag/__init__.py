"""RAG pipeline: ingestion, embeddings, vector storage, retrieval, reranking.

Submodules:
    web_scraper: Scrapes nectarit.com into the web-sourced knowledge base.
    ingestion: Loads and chunks markdown documents from both KB sources.
    embeddings: Wraps the embedding model provider.
    vector_store: Wraps the Pinecone client (index management, upsert, query).
    retriever: Top-level "query -> relevant chunks" interface with a
        minimum-relevance cutoff.
    reranker: Cheap lexical-overlap reranking/deduplication on top of
        vector similarity.
"""
