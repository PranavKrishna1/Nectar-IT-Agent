"""Centralized application configuration.

All environment-dependent values (API keys, model names, index names,
thresholds) live here as a single ``pydantic-settings`` model so every
module imports one ``settings`` object instead of scattering
``os.environ.get(...)`` calls, and a missing/malformed value fails fast
at startup with a clear error. See ``.env.example`` for the full list
of variables with descriptions and example values.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM provider -------------------------------------------------
    # Defaults target the FREE tier: Google Gemini via AI Studio, which
    # issues a key with no credit card required. Pydantic AI reads the
    # key from GEMINI_API_KEY (see providers/google.py), so that variable
    # is exported by get_settings() below rather than passed in code. To
    # use OpenAI instead, set OPENAI_API_KEY and change the two model
    # names to "openai:gpt-4o-mini" / "openai:gpt-4o".
    #
    # Model choice controls your free daily quota: avoid "-latest" aliases
    # - they silently resolve to whatever preview model is newest, and
    # preview models carry far lower free-tier limits (as low as 20
    # requests/day). Pin an explicit, stable, dated model instead - see
    # scripts/check_setup.py if you hit HTTP 429.
    gemini_api_key: str = Field(
        default="",
        description="Google AI Studio (Gemini) API key - free tier, no card required.",
    )
    openai_api_key: str = Field(
        default="", description="OpenAI API key (only needed if using openai:* models)."
    )
    llm_model_router: str = Field(
        default="google:gemini-3.5-flash",
        description="Cheap/fast model used for intent routing (Task 2).",
    )
    llm_model_reasoning: str = Field(
        default="google:gemini-3.5-flash",
        description="Higher-capability model used for multi-step "
        "reasoning in the orchestrator and RAG/action agents.",
    )

    # --- Pinecone (vector database) -----------------------------------
    pinecone_api_key: str = Field(default="", description="Pinecone API key.")
    pinecone_index_name: str = Field(
        default="nectar-facility-kb",
        description="Name of the Pinecone index storing knowledge-base "
        "embeddings.",
    )
    pinecone_cloud: str = Field(default="aws", description="Pinecone serverless cloud.")
    pinecone_region: str = Field(
        default="us-east-1",
        description="Pinecone serverless region. The free Starter tier only "
        "supports us-east-1.",
    )

    # --- Embeddings ------------------------------------------------------
    # Defaults target the FREE tier: Pinecone's own hosted Inference
    # embeddings, included in the Starter plan's allowance and reusing
    # PINECONE_API_KEY - so no third API key is needed. Set
    # embedding_provider="openai" to use OpenAI instead.
    embedding_provider: str = Field(
        default="pinecone",
        description="Embedding backend: 'pinecone' (free, hosted, uses "
        "PINECONE_API_KEY) or 'openai' (requires OPENAI_API_KEY).",
    )
    embedding_model: str = Field(
        default="multilingual-e5-large",
        description="Embedding model name. For embedding_provider='pinecone' "
        "use a Pinecone-hosted model such as 'multilingual-e5-large'; for "
        "'openai' use e.g. 'text-embedding-3-small'.",
    )
    embedding_dimensions: int = Field(
        default=1024,
        description="Vector dimensionality of the chosen embedding model "
        "(multilingual-e5-large = 1024; text-embedding-3-small = 1536). Must "
        "match the model, or Pinecone will reject the upsert.",
    )

    # --- Speech services -------------------------------------------------
    # Defaults target the FREE tier: faster-whisper runs Whisper locally
    # (no key, no network), and edge-tts uses Microsoft Edge's free voice
    # service (no key, no signup).
    stt_provider: str = Field(
        default="faster-whisper",
        description="Speech-to-text backend: 'faster-whisper' (free, local, "
        "no key), 'whisper' (OpenAI API), or 'deepgram'.",
    )
    tts_provider: str = Field(
        default="edge-tts",
        description="Text-to-speech backend: 'edge-tts' (free, no key), "
        "'elevenlabs', or 'aws'.",
    )
    faster_whisper_model: str = Field(
        default="base",
        description="faster-whisper model size: tiny | base | small | medium | "
        "large-v3. Larger is more accurate but slower and a bigger download.",
    )
    edge_tts_voice: str = Field(
        default="en-US-AriaNeural",
        description="Voice name for edge-tts. Run `edge-tts --list-voices` to see options.",
    )
    deepgram_api_key: str = Field(default="", description="Deepgram API key, if used.")
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key, if used.")
    elevenlabs_voice_id: str = Field(
        default="", description="Voice ID to use for ElevenLabs synthesis."
    )

    # --- RAG behaviour --------------------------------------------------
    rag_top_k: int = Field(default=5, description="Chunks retrieved per query.")
    rag_min_similarity: float = Field(
        default=0.30,
        description="Minimum cosine similarity for a retrieved chunk to be "
        "considered relevant; below this the RAG agent reports 'not found' "
        "instead of answering from weak matches.",
    )
    rag_chunk_size: int = Field(default=800, description="Target characters per chunk.")
    rag_chunk_overlap: int = Field(default=120, description="Character overlap between chunks.")

    # --- Routing behaviour ------------------------------------------------
    router_min_confidence: float = Field(
        default=0.55,
        description="Minimum router confidence required to act on a route "
        "without asking a clarifying question first.",
    )

    # --- Orchestration safety limits --------------------------------------
    orchestrator_max_steps: int = Field(
        default=8,
        description="Hard cap on tool/sub-agent calls per user turn, to "
        "guarantee the autonomous loop always terminates.",
    )
    require_confirmation_for_actions: bool = Field(
        default=True,
        description="If true (default), any write/action tool call must be "
        "confirmed by the user before it executes.",
    )

    # --- Web-sourced knowledge ------------------------------------------
    nectar_website_url: str = Field(
        default="https://www.nectarit.com/",
        description="Root URL scraped by rag/web_scraper.py for the "
        "web-sourced portion of the knowledge base.",
    )

    # --- MCP server -------------------------------------------------------
    mcp_server_host: str = Field(default="127.0.0.1")
    mcp_server_port: int = Field(default=8765)

    # -- Validators --------------------------------------------------------
    # These catch malformed keys at load time with an actionable message,
    # rather than letting a stray character surface much later as an
    # opaque "401 Unauthorized" deep inside an ingestion run. Copy-paste
    # damage (a duplicated leading character, wrapping quotes, a trailing
    # newline) is by far the most common cause of a 401 here.

    @field_validator("pinecone_api_key")
    @classmethod
    def _validate_pinecone_key(cls, value: str) -> str:
        """Strip whitespace/quotes and reject keys not shaped like Pinecone's."""
        cleaned = value.strip().strip("\"'")
        if cleaned and not cleaned.startswith("pcsk_"):
            raise ValueError(
                f"PINECONE_API_KEY looks malformed: it starts with "
                f"{cleaned[:6]!r} but Pinecone keys start with 'pcsk_'. "
                "Check for a stray character at the start of the value in "
                "your .env (e.g. 'Ppcsk_...' instead of 'pcsk_...')."
            )
        return cleaned

    @field_validator("gemini_api_key")
    @classmethod
    def _validate_gemini_key(cls, value: str) -> str:
        """Strip whitespace/quotes and reject an OpenAI-style key pasted here.

        Most Google AI Studio keys start with "AIza", but not all valid
        keys do, so that prefix is not enforced as a hard requirement -
        only the unambiguous "sk-..." (OpenAI) case is rejected.
        """
        cleaned = value.strip().strip("\"'")
        if cleaned and cleaned.startswith("sk-"):
            raise ValueError(
                "GEMINI_API_KEY looks like an OpenAI-style key ('sk-...'), not "
                "a Google AI Studio key (these usually start with 'AIza', "
                "though other formats exist). Re-copy it from "
                "https://aistudio.google.com/apikey."
            )
        return cleaned

    @field_validator("openai_api_key")
    @classmethod
    def _validate_openai_key(cls, value: str) -> str:
        """Strip surrounding whitespace/quotes (not format-checked further)."""
        return value.strip().strip("\"'")


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide ``Settings`` instance.

    ``lru_cache`` means the environment is read/validated once per
    process and every caller shares the same settings object.

    As a side effect, this also copies the configured LLM API keys into
    ``os.environ`` under the names Pydantic AI's providers read
    (``GEMINI_API_KEY`` / ``OPENAI_API_KEY``) - Pydantic AI builds its
    model clients from those environment variables directly, so loading
    a key into this Settings object alone isn't enough to make it visible
    to the provider.
    """
    settings = Settings()
    if settings.gemini_api_key:
        os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    return settings
