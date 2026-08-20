"""Read-aloud voice: id resolution, S3 key layout, and preference updates."""

from src.config.voice_config import DEFAULT_VOICE_ID, VOICE_IDS, resolve_voice_id
from src.schemas.preferences import (
    ChatModelPreferencesUpdate,
    UserPreferencesUpdate,
    apply_preferences_update,
)


class _Prefs:
    """Stand-in for the Preferences row (no DB needed for this logic)."""

    default_voice_id = None


def test_resolve_voice_id_accepts_known_voices():
    assert resolve_voice_id("en-GB-SoniaNeural") == "en-GB-SoniaNeural"
    assert DEFAULT_VOICE_ID in VOICE_IDS


def test_resolve_voice_id_falls_back_for_unknown_input():
    # The id reaches edge-tts and the S3 key, so unknown values must not pass.
    assert resolve_voice_id(None) == DEFAULT_VOICE_ID
    assert resolve_voice_id("") == DEFAULT_VOICE_ID
    assert resolve_voice_id("../../etc/passwd") == DEFAULT_VOICE_ID


def test_preference_update_stores_validated_voice():
    prefs = _Prefs()
    apply_preferences_update(
        prefs,
        UserPreferencesUpdate(chat=ChatModelPreferencesUpdate(voice="en-US-GuyNeural")),
    )
    assert prefs.default_voice_id == "en-US-GuyNeural"

    apply_preferences_update(
        prefs,
        UserPreferencesUpdate(chat=ChatModelPreferencesUpdate(voice="bogus-voice")),
    )
    assert prefs.default_voice_id == DEFAULT_VOICE_ID


def test_preference_update_leaves_voice_alone_when_omitted():
    prefs = _Prefs()
    prefs.default_voice_id = "en-IN-NeerjaNeural"
    apply_preferences_update(
        prefs,
        UserPreferencesUpdate(chat=ChatModelPreferencesUpdate(default_model="x")),
    )
    assert prefs.default_voice_id == "en-IN-NeerjaNeural"
