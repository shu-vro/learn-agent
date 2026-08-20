"""Text-to-speech voices offered in the read-aloud feature.

Voice ids are edge-tts short names (``edge-tts --list-voices``).
"""

DEFAULT_VOICE_ID = "en-US-AriaNeural"

VOICES: list[dict[str, str]] = [
    {"id": "en-US-AriaNeural", "label": "Aria — US English, female"},
    {"id": "en-US-GuyNeural", "label": "Guy — US English, male"},
    {"id": "en-US-JennyNeural", "label": "Jenny — US English, female"},
    {"id": "en-GB-SoniaNeural", "label": "Sonia — British English, female"},
    {"id": "en-GB-RyanNeural", "label": "Ryan — British English, male"},
    {"id": "en-IN-NeerjaNeural", "label": "Neerja — Indian English, female"},
    {"id": "en-IN-PrabhatNeural", "label": "Prabhat — Indian English, male"},
    {"id": "en-AU-NatashaNeural", "label": "Natasha — Australian English, female"},
]

VOICE_IDS = frozenset(voice["id"] for voice in VOICES)


def resolve_voice_id(voice_id: str | None) -> str:
    """Return a known voice id, falling back to the default for unknown input."""
    return voice_id if voice_id in VOICE_IDS else DEFAULT_VOICE_ID


__all__ = ["VOICES", "VOICE_IDS", "DEFAULT_VOICE_ID", "resolve_voice_id"]
