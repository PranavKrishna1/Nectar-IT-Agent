"""One-off script: scrape nectarit.com, chunk all documents, and upsert to Pinecone.

Usage:
    python scripts/ingest_knowledge_base.py [--skip-scrape]

Run this once before the RAG agent can answer anything (and again
whenever the knowledge-base documents change). Requires
``OPENAI_API_KEY`` and ``PINECONE_API_KEY`` to be set (see .env.example).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nectar_agent.config import get_settings  # noqa: E402
from nectar_agent.rag import vector_store, web_scraper  # noqa: E402
from nectar_agent.rag.embeddings import embed_texts  # noqa: E402
from nectar_agent.rag.ingestion import build_chunks  # noqa: E402

WEB_SOURCED_DIR = PROJECT_ROOT / "data" / "knowledge_base" / "web_sourced"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "knowledge_base" / "synthetic"


def main() -> None:
    """Run the full ingestion pipeline: scrape -> chunk -> embed -> upsert."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip re-scraping nectarit.com and use the existing web_sourced/ files.",
    )
    args = parser.parse_args()

    settings = get_settings()

    if not args.skip_scrape:
        print(f"Scraping {settings.nectar_website_url} ...")
        pages = web_scraper.scrape_site(settings.nectar_website_url)
        written = web_scraper.write_scraped_pages(pages, WEB_SOURCED_DIR)
        print(f"  wrote {len(written)} web-sourced document(s) to {WEB_SOURCED_DIR}")
    else:
        print("Skipping scrape step (--skip-scrape).")

    print("Chunking documents from both knowledge-base sources ...")
    chunks = build_chunks(WEB_SOURCED_DIR, SYNTHETIC_DIR)
    print(f"  produced {len(chunks)} chunk(s) from web_sourced/ and synthetic/")

    if not chunks:
        print("No chunks to ingest - check that the knowledge_base directories are populated.")
        return

    print(f"Ensuring Pinecone index '{settings.pinecone_index_name}' exists ...")
    vector_store.ensure_index_exists()

    print(f"Embedding {len(chunks)} chunk(s) with model '{settings.embedding_model}' ...")
    batch_size = 100
    total_upserted = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_texts([c.text for c in batch])
        total_upserted += vector_store.upsert_chunks(batch, vectors)
        print(f"  upserted {total_upserted}/{len(chunks)}")

    print(f"Done. {total_upserted} vectors are now indexed in Pinecone.")


if __name__ == "__main__":
    main()
