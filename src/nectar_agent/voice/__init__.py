"""Voice I/O layer (Task 1): speech-to-text and text-to-speech wrappers.

Submodules:
    stt: ``transcribe(audio_path) -> str``, backend chosen by config.
    tts: ``synthesize(text, output_path) -> Path``, backend chosen by config.
    audio_utils: Small file read/write helpers shared by both.
"""
