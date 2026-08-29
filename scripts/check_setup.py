"""Preflight check: verify configuration and credentials before running anything.

Run this first, and after any .env change:

    python scripts/check_setup.py

It verifies, in order, and reports each result with an actionable fix:

  1. .env exists and is being loaded
  2. Both required keys are present and correctly formatted
  3. The Pinecone key actually authenticates (catches 401s here rather
     than halfway through an ingestion run)
  4. The Gemini key authenticates AND has remaining daily quota
     (catches 429 rate-limit exhaustion before a demo, not during it)
  5. The configured embedding dimensions match any existing index

Every check is cheap - the Gemini check spends exactly one request - so
this is safe to run whenever something looks wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def _print(status: str, label: str, detail: str = "") -> None:
    """Print one aligned check result line.

    Args:
        status: One of the OK/FAIL/WARN markers.
        label: Short name of the check.
        detail: Optional extra explanation or remediation hint.
    """
    line = f"{status} {label}"
    if detail:
        line += f"\n       {detail}"
    print(line)


def check_env_file() -> bool:
    """Verify that a .env file exists at the project root.

    Returns:
        True if .env exists, False otherwise.
    """
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        _print(OK, ".env file found", str(env_path))
        return True
    _print(
        FAIL,
        ".env file NOT found",
        f"Expected at {env_path}\n       Fix: copy .env.example to .env, then fill in your two keys.",
    )
    return False


def check_settings():
    """Load application settings, surfacing validation errors clearly.

    Returns:
        The loaded ``Settings`` object, or ``None`` if loading failed
        (e.g. a malformed API key tripped a validator).
    """
    try:
        from nectar_agent.config import get_settings

        settings = get_settings()
        _print(OK, "Configuration loaded and validated")
        return settings
    except Exception as exc:  # noqa: BLE001 - we want to report any failure
        _print(FAIL, "Configuration failed to load", str(exc))
        return None


def check_keys_present(settings) -> bool:
    """Verify the keys required by the active provider configuration are set.

    Args:
        settings: Loaded application settings.

    Returns:
        True if every key the current configuration needs is present.
    """
    ok = True

    if settings.llm_model_router.startswith("google:") or settings.llm_model_reasoning.startswith(
        "google:"
    ):
        if settings.gemini_api_key:
            _print(OK, "GEMINI_API_KEY present", f"{settings.gemini_api_key[:8]}...")
        else:
            _print(
                FAIL,
                "GEMINI_API_KEY missing",
                "Fix: get a free key at https://aistudio.google.com/apikey and put it in .env",
            )
            ok = False

    if settings.llm_model_router.startswith("openai:") or settings.embedding_provider == "openai":
        if settings.openai_api_key:
            _print(OK, "OPENAI_API_KEY present")
        else:
            _print(FAIL, "OPENAI_API_KEY missing but required by current config")
            ok = False

    if settings.pinecone_api_key:
        _print(OK, "PINECONE_API_KEY present", f"{settings.pinecone_api_key[:8]}...")
    else:
        _print(
            FAIL,
            "PINECONE_API_KEY missing",
            "Fix: get a free Starter key at https://app.pinecone.io and put it in .env",
        )
        ok = False

    return ok


def check_pinecone(settings) -> bool:
    """Verify the Pinecone key authenticates and report index state.

    Also compares the configured embedding dimensionality against any
    existing index, since a mismatch is a common and confusing failure
    when switching embedding providers.

    Args:
        settings: Loaded application settings.

    Returns:
        True if Pinecone authenticated successfully.
    """
    try:
        from pinecone import Pinecone

        client = Pinecone(api_key=settings.pinecone_api_key)
        indexes = list(client.list_indexes())
        _print(OK, "Pinecone authentication succeeded")

        names = {idx["name"] for idx in indexes}
        target = settings.pinecone_index_name
        if target in names:
            described = client.describe_index(target)
            existing_dim = described.dimension
            if existing_dim == settings.embedding_dimensions:
                _print(OK, f"Index '{target}' exists", f"dimension {existing_dim} - matches config")
            else:
                _print(
                    FAIL,
                    f"Index '{target}' dimension MISMATCH",
                    f"Index is {existing_dim}-dim but EMBEDDING_DIMENSIONS={settings.embedding_dimensions}.\n"
                    f"       Fix: delete the index in the Pinecone console, then re-run "
                    f"scripts/ingest_knowledge_base.py to recreate it.",
                )
                return False
        else:
            _print(
                WARN,
                f"Index '{target}' does not exist yet",
                "This is expected on first run - ingest_knowledge_base.py will create it.",
            )
        return True

    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        hint = ""
        if "401" in message or "Unauthorized" in message:
            hint = (
                "\n       This is an authentication failure - the key is wrong, not the code.\n"
                "       Fix: re-copy the key from https://app.pinecone.io (API Keys section).\n"
                "       Check for a stray leading/trailing character from the paste."
            )
        _print(FAIL, "Pinecone check failed", f"{message}{hint}")
        return False


# Stable, dated fallback candidates to probe when the configured model is
# exhausted or invalid. Deliberately excludes "-latest" aliases (see the
# quota note in .env) and is checked live rather than assumed, since which
# models exist/have quota left changes over time and a hardcoded single
# recommendation goes stale (and can even echo back the same exhausted
# model the user is already on).
_FALLBACK_MODEL_CANDIDATES = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.1-flash-lite",
]


def _probe_model(model: str, api_key: str) -> int:
    """Send one minimal generateContent request and return its status code.

    Args:
        model: Bare Gemini model name (no "google:" prefix).
        api_key: Gemini API key to authenticate with.

    Returns:
        The HTTP status code, or -1 if the request itself raised.
    """
    import httpx

    try:
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key},
            json={"contents": [{"parts": [{"text": "hi"}]}]},
            timeout=15.0,
        )
        return response.status_code
    except Exception:  # noqa: BLE001
        return -1


def _find_models_with_quota(api_key: str, exclude: str) -> list[str]:
    """Probe the fallback candidates and return which ones respond 200 now.

    Args:
        api_key: Gemini API key to authenticate with.
        exclude: Model name to skip (the one that just failed).

    Returns:
        Bare model names (no prefix) that are currently reachable and not
        rate-limited.
    """
    available = []
    for candidate in _FALLBACK_MODEL_CANDIDATES:
        if candidate == exclude:
            continue
        if _probe_model(candidate, api_key) == 200:
            available.append(candidate)
    return available


def check_gemini(settings) -> bool:
    """Verify the Gemini key authenticates and still has daily quota left.

    Spends exactly one minimal request in the common case. Distinguishes
    an auth failure (bad key) from a 429 quota exhaustion (valid key, no
    requests left today), because the two need completely different
    fixes. On quota exhaustion or an invalid model name, additionally
    probes a short list of alternate stable models live and reports which
    ones currently have quota, rather than suggesting a fixed name that
    may itself already be exhausted.

    Args:
        settings: Loaded application settings.

    Returns:
        True if Gemini responded successfully.
    """
    if not settings.llm_model_router.startswith("google:"):
        _print(WARN, "Gemini check skipped", "LLM_MODEL_ROUTER is not a google:* model.")
        return True

    try:
        model = settings.llm_model_router.split(":", 1)[1]
        status_code = _probe_model(model, settings.gemini_api_key)

        if status_code == 200:
            _print(OK, f"Gemini model '{model}' reachable, quota available")
            return True

        if status_code == 429:
            alternatives = _find_models_with_quota(settings.gemini_api_key, exclude=model)
            if alternatives:
                switch_lines = "\n".join(
                    f"              LLM_MODEL_ROUTER=google:{m}\n"
                    f"              LLM_MODEL_REASONING=google:{m}"
                    for m in alternatives[:1]
                )
                fix_a = (
                    f"a) Switch .env to a model with quota left right now "
                    f"(checked live just now: {', '.join(alternatives)}):\n{switch_lines}"
                )
            else:
                fix_a = (
                    "a) All probed fallback models are also rate-limited right now - "
                    "this key may be near its combined daily cap. Try again in a "
                    "few minutes."
                )
            _print(
                FAIL,
                f"Gemini model '{model}' QUOTA EXHAUSTED (429)",
                f"The key is valid but this model's free-tier requests are used up "
                f"(daily or per-minute cap - see the message below).\n"
                f"       Fixes (any one):\n"
                f"         {fix_a}\n"
                "         b) Wait for the daily reset (midnight Pacific) or a minute "
                "for a per-minute cap.\n"
                "         c) Create a second free key under a different Google account.\n"
                "       Check your live per-model limits at:\n"
                "         https://aistudio.google.com/rate-limit",
            )
            return False

        if status_code in (401, 403):
            _print(
                FAIL,
                "Gemini authentication failed",
                "Fix: re-copy your key from https://aistudio.google.com/apikey",
            )
            return False

        if status_code == 404:
            alternatives = _find_models_with_quota(settings.gemini_api_key, exclude=model)
            hint = (
                f"Known-good right now: google:{alternatives[0]}"
                if alternatives
                else "Check https://aistudio.google.com/apikey for currently valid model names."
            )
            _print(
                FAIL,
                f"Gemini model '{model}' not found (404)",
                f"That model name isn't valid for this API version.\n       Fix: {hint}",
            )
            return False

        _print(FAIL, f"Gemini returned HTTP {status_code}")
        return False

    except Exception as exc:  # noqa: BLE001
        _print(FAIL, "Gemini check failed", str(exc))
        return False


def main() -> None:
    """Run every preflight check and exit non-zero if any hard check failed."""
    print("=" * 72)
    print("  Nectar Voice Agent - setup check")
    print("=" * 72)

    results: list[bool] = []

    results.append(check_env_file())
    settings = check_settings()
    if settings is None:
        print("\nCannot continue until the configuration loads. Fix the error above.")
        sys.exit(1)

    print()
    results.append(check_keys_present(settings))
    print()
    results.append(check_pinecone(settings))
    print()
    results.append(check_gemini(settings))

    print()
    print("=" * 72)
    if all(results):
        print("  All checks passed. Next:")
        print("    python scripts/ingest_knowledge_base.py")
        print("    python scripts/run_demo.py --text")
        sys.exit(0)
    else:
        print("  Some checks failed - see the fixes listed above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
