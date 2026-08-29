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

    Attributes:
        chunk_id: Deterministic unique ID (hash of doc path + chunk index),
            used as the Pinecone vector ID so re-ingestion overwrites
            rather than duplicates.
        doc_id: Identifier of the source document (its filename stem).
        text: The chunk's text content.
        source_type: Either "web_scrape" or "synthetic_doc".
        source_url: Origin URL for web-scraped content, else ``None``.
        title: Document title/heading.
        chunk_index: Position of this chunk within its source document.
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

    Args:
        raw: Full text content of the markdown file.

    Returns:
        A ``(front_matter_dict, body_text)`` tuple. If the file has no
        front matter block, returns an empty dict and the original text.
        Falls back to ``({}, raw)`` if parsing fails unexpectedly.

    Raises:
        None: Parsing failures are caught internally and degrade to
            treating the whole input as body text with no front matter.
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

    Args:
        text: Full document body text.
        chunk_size: Target maximum characters per chunk.
        overlap: Number of trailing characters repeated at the start of
            the next chunk, to preserve context across chunk boundaries.

    Returns:
        List of text chunks, in document order.

    Raises:
        RuntimeError: If chunking fails unexpectedly (e.g. ``overlap``
            greater than or equal to ``chunk_size``, causing a
            non-advancing split range).
    """
    try:
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
    except Exception as exc:
        raise RuntimeError(f"Failed to chunk text: {exc}") from exc


def load_documents(*directories: Path) -> list[tuple[Path, dict, str]]:
    """Load all markdown documents from one or more directories.

    Args:
        *directories: Directories to scan for ``*.md`` files.

    Returns:
        List of ``(file_path, front_matter, body_text)`` tuples.

    Raises:
        RuntimeError: If reading a document file fails (e.g. a
            permissions error or invalid encoding).
    """
    try:
        docs: list[tuple[Path, dict, str]] = []
        for directory in directories:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.md")):
                raw = path.read_text(encoding="utf-8")
                front_matter, body = _parse_front_matter(raw)
                docs.append((path, front_matter, body))
        return docs
    except Exception as exc:
        raise RuntimeError(f"Failed to load documents: {exc}") from exc


def build_chunks(
    web_sourced_dir: Path, synthetic_dir: Path
) -> list[DocumentChunk]:
    """Load and chunk every document in both knowledge-base sources.

    Args:
        web_sourced_dir: Directory containing scraped nectarit.com pages.
        synthetic_dir: Directory containing authored technical documents.

    Returns:
        Flat list of ``DocumentChunk`` objects across all documents,
        ready to be embedded and upserted into Pinecone.

    Raises:
        RuntimeError: If loading or chunking any document fails.
    """
    try:
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
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to build chunks: {exc}") from exc


def _first_heading(body: str) -> str | None:
    """Extract the first markdown ``#`` heading from a document body.

    Args:
        body: Document body text.

    Returns:
        The heading text without the leading ``#`` characters, or
        ``None`` if no heading is present or extraction fails
        unexpectedly.

    Raises:
        None: Failures are caught internally and degrade to ``None``.
    """
    try:
        match = re.search(r"^#{1,6}\s+(.*)$", body, re.MULTILINE)
        return match.group(1).strip() if match else None
    except Exception:
        return None
