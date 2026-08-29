"""Small shared audio helpers used by the STT/TTS wrappers.

Kept minimal and dependency-light; heavier audio processing (resampling,
VAD, noise suppression) is intentionally out of scope for this
prototype and would live here if added later.

Also provides the two pieces needed for a genuinely *live* voice loop
(real microphone input, real speaker output) rather than the
file-in/file-out demo: ``record_from_microphone`` (push-to-talk capture
via ``sounddevice``) and ``play_audio`` (playback via ``sounddevice`` +
``miniaudio``, decoding whatever ``voice.tts.synthesize`` produced -
typically MP3 - to raw PCM in memory). ``miniaudio`` was chosen over the
more obvious ``playsound`` because ``playsound``'s sdist has a
long-standing, still-unfixed build bug (its ``setup.py`` calls
``inspect.getsource`` on itself, which raises ``OSError: could not get
source code`` under pip's isolated build environment on this machine -
every published version, 1.0.0 through 1.3.0, fails the same way here).
``miniaudio`` ships prebuilt wheels (no compiler, no system ffmpeg
needed) and decodes MP3 directly to PCM samples that ``sounddevice`` can
play. All of these imports are deferred into the functions so that
importing this module - and therefore ``stt``/``tts`` - never requires
the optional ``sounddevice``/``miniaudio`` packages unless the live loop
is actually used.
"""

from __future__ import annotations

import tempfile
import uuid
import wave
from pathlib import Path


def read_audio_bytes(path: str | Path) -> bytes:
    """Read raw audio bytes from a file path.

    Args:
        path: Path to an audio file (e.g. WAV/MP3) captured from the
            user's microphone.

    Returns:
        Raw file bytes, ready to hand to an STT provider.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    return file_path.read_bytes()


def write_audio_bytes(data: bytes, path: str | Path) -> Path:
    """Write raw audio bytes (e.g. TTS output) to a file.

    Args:
        data: Raw audio bytes to write.
        path: Destination file path.

    Returns:
        The path the audio was written to, as a ``Path``.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(data)
    return file_path


def record_from_microphone(sample_rate: int = 16000) -> Path:
    """Record push-to-talk audio from the default microphone to a temp WAV file.

    Recording starts immediately and continues until the caller presses
    Enter again (blocking on ``input()`` in the calling thread while
    ``sounddevice`` captures frames on its own callback thread in the
    background) - this avoids guessing a fixed recording duration, which
    would either cut the operator off mid-sentence or force an awkward
    silent wait.

    Args:
        sample_rate: Capture sample rate in Hz. 16 kHz matches what
            faster-whisper (and most STT backends) expect, so no
            resampling step is needed downstream.

    Returns:
        Path to a newly written mono 16-bit PCM WAV file in the system
        temp directory. Caller is responsible for deleting it once done.
    """
    import sounddevice as sd

    frames: list[bytes] = []

    def _callback(indata, _frame_count, _time_info, _status) -> None:
        frames.append(bytes(indata))

    print("Recording... press Enter to stop.")
    with sd.RawInputStream(
        samplerate=sample_rate, channels=1, dtype="int16", callback=_callback
    ):
        input()

    output_path = Path(tempfile.gettempdir()) / f"nectar_mic_{uuid.uuid4().hex}.wav"
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # int16 = 2 bytes/sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(frames))
    return output_path


def play_audio(path: str | Path) -> None:
    """Play an audio file (MP3 or WAV) through the default speaker output.

    Decodes the file to raw PCM in memory with ``miniaudio`` (no external
    ffmpeg binary required - it bundles its own decoders) and plays it
    with ``sounddevice``, blocking until playback finishes so the live
    voice loop naturally waits for the agent's spoken response before
    prompting for the next turn.

    Args:
        path: Path to the audio file to play (e.g. the MP3 written by
            ``voice.tts.synthesize``).
    """
    import miniaudio
    import numpy as np
    import sounddevice as sd

    decoded = miniaudio.decode_file(str(path))
    samples = np.asarray(decoded.samples, dtype=np.int16).reshape(-1, decoded.nchannels)
    sd.play(samples, samplerate=decoded.sample_rate)
    sd.wait()
