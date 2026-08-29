"""Demo runner for the end-to-end agent (Task 5).

Three modes:
    python scripts/run_demo.py --text
        Interactive text-based conversation in the terminal (fastest way
        to exercise routing/RAG/MCP/action logic without audio).

    python scripts/run_demo.py --voice --input in.wav --output out.mp3
        Single-turn voice mode: transcribes ``in.wav``, runs it through
        the full pipeline, and writes the spoken response to ``out.mp3``.

    python scripts/run_demo.py --live
        Fully-spoken interactive loop: speak into the microphone
        (push-to-talk - Enter to start, Enter to stop) and hear the
        agent's reply through the speakers, turn after turn.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nectar_agent.main import _demo_loop, _voice_demo_loop, process_voice_turn  # noqa: E402

# google-genai (used internally by pydantic-ai's Gemini integration) logs a
# one-time advisory - "Direct use of automatic function calling (AFC) ..." -
# at WARNING level every time it makes a function-calling request, which is
# every turn in this app. Since nothing else in the process configures
# logging, Python's default "handler of last resort" prints any WARNING+
# log record straight to stderr, which is what puts that line in the
# terminal between "You:" and "Agent:". It is not an error - raise this
# logger's threshold so only genuine problems (ERROR+) surface here.
logging.getLogger("google_genai").setLevel(logging.ERROR)


def main() -> None:
    """Parse CLI arguments and run the requested demo mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", action="store_true", help="Run the interactive text demo.")
    mode.add_argument("--voice", action="store_true", help="Run a single-turn voice demo.")
    mode.add_argument(
        "--live",
        action="store_true",
        help="Run a live, fully-spoken loop (microphone in, speakers out).",
    )
    parser.add_argument("--input", type=str, help="Input audio file path (voice mode).")
    parser.add_argument("--output", type=str, help="Output audio file path (voice mode).")
    args = parser.parse_args()

    if args.text:
        asyncio.run(_demo_loop())
        return

    if args.live:
        asyncio.run(_voice_demo_loop())
        return

    if not args.input or not args.output:
        parser.error("--voice mode requires --input and --output.")

    session_id = str(uuid.uuid4())
    response = asyncio.run(process_voice_turn(session_id, args.input, args.output))
    print(f"Agent response: {response}")
    print(f"Audio written to: {args.output}")


if __name__ == "__main__":
    main()
