"""Text-to-speech wrapper (Task 1).

Mirrors ``stt.py``: one ``synthesize`` function that hides the choice of
TTS backend behind ``settings.tts_provider``.

Three backends are supported:

- ``"edge-tts"`` (default): Microsoft Edge's neural voice service via
  the ``edge-tts`` package. Requires no API key and no signup - the free
  option.
- ``"elevenlabs"``: ElevenLabs' hosted TTS API.
- ``"aws"``: AWS Polly.
"""

from __future__ import annotations

from pathlib import Path

from nectar_agent.config import get_settings
from nectar_agent.voice.audio_utils import write_audio_bytes


def synthesize(text: str, output_path: str | Path) -> Path:
    """Convert text to speech and write the resulting audio to disk.

    Args:
        text: The agent's natural-language response to speak aloud.
        output_path: File path to write the synthesized audio to.

    Returns:
        The path the audio was written to.

    Raises:
        ValueError: If ``settings.tts_provider`` is not a supported value.
    """
    settings = get_settings()
    if settings.tts_provider == "edge-tts":
        audio_bytes = _synthesize_edge_tts(text)
    elif settings.tts_provider == "elevenlabs":
        audio_bytes = _synthesize_elevenlabs(text)
    elif settings.tts_provider == "aws":
        audio_bytes = _synthesize_aws_polly(text)
    else:
        raise ValueError(f"Unsupported tts_provider: {settings.tts_provider}")
    return write_audio_bytes(audio_bytes, output_path)


async def _collect_edge_tts_audio(text: str, voice: str) -> bytes:
    """Stream synthesized audio chunks from edge-tts and concatenate them.

    Args:
        text: Text to synthesize.
        voice: Edge TTS voice name, e.g. "en-US-AriaNeural".

    Returns:
        The complete MP3 payload as bytes.
    """
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])
    return bytes(chunks)


def _synthesize_edge_tts(text: str) -> bytes:
    """Synthesize speech using Microsoft Edge's free neural voice service.

    This is the default backend and requires no API key or account.

    ``edge-tts`` is async-only, but ``synthesize`` presents a synchronous
    interface (matching the other backends) and is itself called from
    inside an already-running event loop in ``main.process_voice_turn``.
    Calling ``asyncio.run`` directly would therefore raise "cannot be
    called from a running event loop", so when a loop is already running
    the coroutine is dispatched to a short-lived worker thread with its
    own loop instead.

    Args:
        text: Text to synthesize.

    Returns:
        Raw audio bytes (MP3).
    """
    import asyncio
    import concurrent.futures

    settings = get_settings()
    coro_factory = lambda: _collect_edge_tts_audio(text, settings.edge_tts_voice)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running - safe to drive one ourselves.
        return asyncio.run(coro_factory())

    # A loop is already running in this thread; run the coroutine to
    # completion on its own loop in a worker thread and block for it.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_factory())).result()


def _synthesize_elevenlabs(text: str) -> bytes:
    """Synthesize speech using the ElevenLabs REST API.

    Args:
        text: Text to synthesize.

    Returns:
        Raw audio bytes (MP3).
    """
    import httpx

    settings = get_settings()
    response = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}",
        headers={
            "xi-api-key": settings.elevenlabs_api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.content


def _synthesize_aws_polly(text: str) -> bytes:
    """Synthesize speech using AWS Polly.

    Args:
        text: Text to synthesize.

    Returns:
        Raw audio bytes (MP3).
    """
    import boto3

    polly = boto3.client("polly")
    response = polly.synthesize_speech(
        Text=text, OutputFormat="mp3", VoiceId="Joanna", Engine="neural"
    )
    return response["AudioStream"].read()
