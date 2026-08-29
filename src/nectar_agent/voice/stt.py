"""Speech-to-text wrapper (Task 1).

Provides one function, ``transcribe``, that hides the choice of STT
backend behind a single interface driven by ``settings.stt_provider``.
``main.py`` and ``scripts/run_demo.py`` call this without needing to
know which backend is configured.

Three backends are supported:

- ``"faster-whisper"`` (default): runs Whisper locally with no API key
  and no network access after the first model download - the free
  option.
- ``"whisper"``: OpenAI's hosted Whisper transcription endpoint.
- ``"deepgram"``: Deepgram's hosted transcription API.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from nectar_agent.config import get_settings


def transcribe(audio_path: str | Path) -> str:
    """Transcribe an audio file to text using the configured STT provider.

    Args:
        audio_path: Path to the audio file to transcribe (e.g. a WAV
            recording of the facility operator's voice input).

    Returns:
        The transcribed text.

    Raises:
        ValueError: If ``settings.stt_provider`` is not a supported value.
    """
    settings = get_settings()
    if settings.stt_provider == "faster-whisper":
        return _transcribe_faster_whisper(audio_path)
    if settings.stt_provider == "whisper":
        return _transcribe_whisper(audio_path)
    if settings.stt_provider == "deepgram":
        return _transcribe_deepgram(audio_path)
    raise ValueError(f"Unsupported stt_provider: {settings.stt_provider}")


def _transcribe_faster_whisper(audio_path: str | Path) -> str:
    """Transcribe audio locally using faster-whisper (no API key required).

    This is the default backend. It runs Whisper entirely on the local
    machine via CTranslate2, so it needs no credentials and no network
    once the model is cached. The model is downloaded automatically on
    first use (roughly 75 MB for "base", larger for bigger sizes) and
    cached under the user's Hugging Face cache directory thereafter.

    Args:
        audio_path: Path to the audio file.

    Returns:
        The transcribed text, with leading/trailing whitespace stripped.
    """
    from faster_whisper import WhisperModel

    settings = get_settings()
    model = _get_faster_whisper_model(settings.faster_whisper_model)
    segments, _info = model.transcribe(str(audio_path))
    return " ".join(segment.text for segment in segments).strip()


@lru_cache
def _get_faster_whisper_model(model_size: str):
    """Load (once) and cache a faster-whisper model instance.

    Model loading is expensive (hundreds of milliseconds to seconds), so
    the instance is cached per model size for the life of the process
    rather than reloaded on every transcription.

    Args:
        model_size: Model size identifier, e.g. "base" or "small".

    Returns:
        A loaded ``WhisperModel`` running on CPU with int8 quantization,
        which is the most portable configuration (no GPU required).
    """
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _transcribe_whisper(audio_path: str | Path) -> str:
    """Transcribe audio using OpenAI's Whisper transcription endpoint.

    Args:
        audio_path: Path to the audio file.

    Returns:
        The transcribed text.
    """
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return result.text


def _transcribe_deepgram(audio_path: str | Path) -> str:
    """Transcribe audio using the Deepgram REST API.

    Implemented with a plain HTTP call (via ``httpx``) rather than the
    Deepgram SDK to avoid adding a second heavyweight dependency purely
    for the alternate-provider code path.

    Args:
        audio_path: Path to the audio file.

    Returns:
        The transcribed text.
    """
    import httpx

    settings = get_settings()
    with open(audio_path, "rb") as audio_file:
        response = httpx.post(
            "https://api.deepgram.com/v1/listen",
            headers={
                "Authorization": f"Token {settings.deepgram_api_key}",
                "Content-Type": "audio/wav",
            },
            content=audio_file.read(),
            timeout=30.0,
        )
    response.raise_for_status()
    payload = response.json()
    return payload["results"]["channels"][0]["alternatives"][0]["transcript"]
