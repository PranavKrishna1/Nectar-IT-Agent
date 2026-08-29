"""Scraper for the web-sourced portion of the knowledge base (Task 3).

Pulls pages from the Nectar IT website (https://www.nectarit.com/) and
converts them into clean markdown files under
``data/knowledge_base/web_sourced/``, tagged with their source URL so the
RAG agent can cite them accurately. This covers platform/product/company
content (e.g. "what protocols does the platform support"); it is
deliberately kept separate from the authored technical documents in
``data/knowledge_base/synthetic/`` (chiller manuals, AHU troubleshooting,
etc.) that the website itself does not publish - see docs/design_decisions.md
for why both sources are needed.

This module only depends on ``httpx`` and a lightweight HTML-to-text
pass; no headless browser is required since the target pages are static
marketing content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

DEFAULT_PAGES: tuple[str, ...] = (
    "",  # homepage
    "solutions/connected-buildings",
    "solutions/smart-generators",
    "solutions/heavy-machines",
    "solutions/fleet-management",
    "about",
)


@dataclass
class ScrapedPage:
    """A single scraped and cleaned web page: URL, title, and body text."""

    url: str
    title: str
    text: str


def _clean_html(html: str) -> tuple[str, str]:
    """Strip an HTML page down to a ``(title, text)`` tuple.

    Navigation, scripts, styles, and boilerplate elements are removed;
    consecutive blank lines are collapsed so the result reads as clean
    paragraphs. Falls back to ``("Untitled", "")`` if parsing fails, so a
    single malformed page cannot abort the whole scrape run.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"

        text = soup.get_text(separator="\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)
        return title, text
    except Exception:
        return "Untitled", ""


def scrape_page(base_url: str, path: str, client: httpx.Client) -> ScrapedPage | None:
    """Fetch and clean a single page from the target site.

    ``path`` is relative to ``base_url`` (e.g. "solutions/connected-
    buildings"); empty string fetches the homepage. Returns ``None`` if
    the request failed, returned no usable content, or any other
    unexpected error occurred - a single bad page must never abort the
    whole scrape run.
    """
    try:
        url = base_url.rstrip("/") + "/" + path.lstrip("/") if path else base_url
        response = client.get(url, timeout=15.0, follow_redirects=True)
        response.raise_for_status()

        title, text = _clean_html(response.text)
        if not text:
            return None
        return ScrapedPage(url=url, title=title, text=text)
    except Exception:
        return None


def scrape_site(
    base_url: str, pages: tuple[str, ...] = DEFAULT_PAGES
) -> list[ScrapedPage]:
    """Scrape a fixed list of pages (plus the homepage) from the target site.

    Failed fetches are skipped silently, since marketing pages
    occasionally 404/redirect and that should not abort the whole
    ingestion run.
    """
    results: list[ScrapedPage] = []
    with httpx.Client(headers={"User-Agent": "NectarFacilityAgent-KB-Ingest/1.0"}) as client:
        for path in pages:
            page = scrape_page(base_url, path, client)
            if page is not None:
                results.append(page)
    return results


def write_scraped_pages(pages: list[ScrapedPage], output_dir: Path) -> list[Path]:
    """Write scraped pages to markdown files with source-URL front matter.

    ``output_dir`` is typically ``data/knowledge_base/web_sourced/``.
    Returns the list of file paths written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, page in enumerate(pages):
        slug = re.sub(r"[^a-z0-9]+", "-", page.title.lower()).strip("-") or f"page-{i}"
        file_path = output_dir / f"{slug}.md"
        front_matter = (
            f"---\nsource_url: {page.url}\nsource_type: web_scrape\n"
            f"title: {page.title}\n---\n\n"
        )
        file_path.write_text(front_matter + f"# {page.title}\n\n{page.text}\n", encoding="utf-8")
        written.append(file_path)
    return written
