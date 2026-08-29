"""Entry point for the voice interaction loop (Task 1).

Ties the voice I/O layer to the orchestrator: record audio path in ->
STT -> orchestrator.handle_turn -> TTS -> audio path out. Run this
directly for a live, fully-spoken demo loop (microphone in, speakers
out - see ``_voice_demo_loop``/``process_voice_turn_live``), or import
``process_voice_turn`` from a real telephony/voice-gateway integration
that already hands you recorded audio files.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path

from nectar_agent.orchestration.orchestrator_agent import handle_turn
from nectar_agent.voice.audio_utils import play_audio, record_from_microphone
from nectar_agent.voice.stt import transcribe
from nectar_agent.voice.tts import synthesize


async def process_voice_turn(
    session_id: str, input_audio_path: str | Path, output_audio_path: str | Path
) -> str:
    """Process one full voice turn: audio in -> agent response -> audio out.

    Returns the agent's response text (also spoken to ``output_audio_path``).
    """
    user_text = transcribe(input_audio_path)
    response_text = await handle_turn(session_id, user_text)
    synthesize(response_text, output_audio_path)
    return response_text


async def process_text_turn(session_id: str, user_text: str) -> str:
    """Process one turn using text input directly, bypassing STT.

    Useful for the demo script and automated tests, where driving the
    conversation via text is faster and doesn't require recorded audio
    fixtures, while still exercising the full router -> agents ->
    confirmation pipeline.
    """
    return await handle_turn(session_id, user_text)


async def process_voice_turn_live(session_id: str, sample_rate: int = 16000) -> str:
    """Process one full turn using the real microphone and speakers.

    This is the fully-voice path: records push-to-talk audio from the
    default microphone, transcribes it, runs it through the orchestrator,
    synthesizes the spoken reply, and plays it back through the default
    speaker output. Temp audio files are cleaned up afterwards.
    """
    input_path = record_from_microphone(sample_rate)
    try:
        user_text = transcribe(input_path)
    finally:
        input_path.unlink(missing_ok=True)

    print(f"You said: {user_text}")
    response_text = await handle_turn(session_id, user_text)

    output_path = Path(tempfile.gettempdir()) / f"nectar_reply_{uuid.uuid4().hex}.mp3"
    try:
        synthesize(response_text, output_path)
        play_audio(output_path)
    finally:
        output_path.unlink(missing_ok=True)

    return response_text


async def _demo_loop() -> None:
    """Run a simple interactive text-based demo loop in the terminal.

    Reads lines from stdin, feeds them through the full orchestration
    pipeline, and prints the agent's response - useful for manually
    exercising the system without wiring up real audio I/O.
    """
    session_id = str(uuid.uuid4())
    print("Nectar Facility Agent - text demo (Ctrl+C to exit)")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        response = await process_text_turn(session_id, user_text)
        print(f"Agent: {response}")


async def _voice_demo_loop() -> None:
    """Run a fully-spoken interactive demo loop: mic in, speakers out.

    Each turn is push-to-talk: press Enter to start talking, press Enter
    again to stop, and the agent speaks its reply back before prompting
    for the next turn. Ctrl+C exits.
    """
    session_id = str(uuid.uuid4())
    print("Nectar Facility Agent - live voice demo")
    print("Press Enter to start speaking, Enter again to stop. Ctrl+C to exit.")
    while True:
        try:
            input("\nPress Enter to speak...")
        except (EOFError, KeyboardInterrupt):
            break
        try:
            response = await process_voice_turn_live(session_id)
        except Exception as exc:  # noqa: BLE001 - keep the demo loop alive on one bad turn
            print(f"Error during voice turn: {exc}")
            continue
        print(f"Agent: {response}")


if __name__ == "__main__":
    asyncio.run(_voice_demo_loop())
