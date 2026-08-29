"""Tests for API-key format validation in Settings.

These validators exist to convert a silent, late-surfacing failure (a
401 Unauthorized thrown halfway through an ingestion run, after the
scrape and chunking have already completed) into an immediate,
actionable error at configuration load time. The malformed-key cases
below are real copy-paste damage patterns, not hypotheticals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nectar_agent.config import Settings


def _settings(**overrides) -> Settings:
    """Build a Settings instance from explicit values, ignoring any .env.

    Args:
        **overrides: Field values to set on the Settings instance.

    Returns:
        The constructed ``Settings`` object.
    """
    return Settings(_env_file=None, **overrides)


def test_valid_pinecone_key_accepted() -> None:
    settings = _settings(pinecone_api_key="pcsk_abc123")
    assert settings.pinecone_api_key == "pcsk_abc123"


def test_pinecone_key_with_stray_leading_character_rejected() -> None:
    """The exact reported failure: a duplicated leading char from pasting."""
    with pytest.raises(ValueError, match="pcsk_"):
        _settings(pinecone_api_key="Ppcsk_abc123")


def test_pinecone_key_surrounding_quotes_and_whitespace_stripped() -> None:
    settings = _settings(pinecone_api_key='  "pcsk_abc123"  ')
    assert settings.pinecone_api_key == "pcsk_abc123"


def test_empty_pinecone_key_allowed() -> None:
    """An unset key must not raise - it's checked later, by check_setup."""
    assert _settings(pinecone_api_key="").pinecone_api_key == ""


def test_valid_gemini_key_accepted() -> None:
    settings = _settings(gemini_api_key="AIzaSyExample")
    assert settings.gemini_api_key == "AIzaSyExample"


def test_malformed_gemini_key_rejected() -> None:
    with pytest.raises(ValueError, match="AIza"):
        _settings(gemini_api_key="sk-wrong-provider-key")


def test_empty_gemini_key_allowed() -> None:
    assert _settings(gemini_api_key="").gemini_api_key == ""


def test_openai_key_quotes_stripped_without_format_check() -> None:
    """OpenAI prefixes vary, so only stripping is applied - no rejection."""
    settings = _settings(openai_api_key='  "sk-proj-anything"  ')
    assert settings.openai_api_key == "sk-proj-anything"
