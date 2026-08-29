"""Document loading and chunking for the RAG pipeline (Task 3).

Loads markdown documents from both knowledge-base sources
(``data/knowledge_base/web_sourced/`` and
``data/knowledge_base/synthetic/``), splits them into overlapping chunks
sized for embedding, and attaches metadata (source type, URL, doc id,
section) that is later stored alongside each vector in Pinecone. This
metadata is what lets the RAG agent cite exactly where an answer came
from and distinguish platform-marketing content from facility-technical
content.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from nectar_agent.config import get_settings


@dataclass
class DocumentChunk:
    """A single chunk of a source document, ready to be embedded.

    ``chunk_id`` is a deterministic hash of doc path + chunk index, used
    as the Pinecone vector ID so re-ingestion overwrites rather than
    duplicates. ``doc_id`` is the source document's filename stem.
    ``source_type`` is "web_scrape" or "synthetic_doc"; ``source_url``
    is set only for web-scraped content.
    """

    chunk_id: str
    doc_id: str
    text: str
    source_type: str
    source_url: str | None
    title: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """Split a markdown file's YAML-ish front matter from its body.

    Only supports the simple ``key: value`` front matter written by
    ``web_scraper.write_scraped_pages`` - not a full YAML parser, since
    that is all this project's documents use.

    Returns a ``(front_matter_dict, body_text)`` tuple, falling back to
    ``({}, raw)`` if there's no front matter block or parsing fails.
    """
    try:
        match = re.match(r"^---\n(.*?)\n---\n\n?(.*)$", raw, re.DOTALL)
        if not match:
            return {}, raw
        fm_block, body = match.groups()
        fm: dict = {}
        for line in fm_block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fm[key.strip()] = value.strip()
        return fm, body
    except Exception:
        return {}, raw


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping fixed-size chunks on paragraph bounds.

    Splitting first on blank-line paragraph breaks (rather than a pure
    character slice) keeps procedural steps and list items intact where
    possible, which matters for troubleshooting-guide style content.
    ``overlap`` trailing characters are repeated at the start of the next
    chunk to preserve context across chunk boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            # A single paragraph longer than chunk_size: hard-split it.
            for start in range(0, len(para), chunk_size - overlap):
                chunks.append(para[start : start + chunk_size])
            current = ""
    if current:
        chunks.append(current)

    # Apply overlap between consecutive chunks for better retrieval recall
    # across boundaries.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, curr in zip(chunks, chunks[1:]):
            prefix = prev[-overlap:]
            overlapped.append(f"{prefix}\n{curr}")
        return overlapped
    return chunks


def load_documents(*directories: Path) -> list[tuple[Path, dict, str]]:
    """Load all markdown documents from one or more directories.

    Returns a list of ``(file_path, front_matter, body_text)`` tuples.
    """
    docs: list[tuple[Path, dict, str]] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            front_matter, body = _parse_front_matter(raw)
            docs.append((path, front_matter, body))
    return docs


def build_chunks(web_sourced_dir: Path, synthetic_dir: Path) -> list[DocumentChunk]:
    """Load and chunk every document in both knowledge-base sources.

    Returns a flat list of ``DocumentChunk`` objects across all
    documents, ready to be embedded and upserted into Pinecone.
    """
    settings = get_settings()
    all_chunks: list[DocumentChunk] = []

    for path, front_matter, body in load_documents(web_sourced_dir, synthetic_dir):
        doc_id = path.stem
        source_type = front_matter.get("source_type", "synthetic_doc")
        source_url = front_matter.get("source_url")
        # Title falls back to the first markdown heading, then the filename.
        title = front_matter.get("title") or _first_heading(body) or doc_id

        for index, text in enumerate(
            chunk_text(body, settings.rag_chunk_size, settings.rag_chunk_overlap)
        ):
            chunk_id = hashlib.sha256(f"{doc_id}:{index}".encode()).hexdigest()[:24]
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    source_type=source_type,
                    source_url=source_url,
                    title=title,
                    chunk_index=index,
                )
            )
    return all_chunks


def _first_heading(body: str) -> str | None:
    """Extract the first markdown ``#`` heading from a document body, if any."""
    try:
        match = re.search(r"^#{1,6}\s+(.*)$", body, re.MULTILINE)
        return match.group(1).strip() if match else None
    except Exception:
        return None
