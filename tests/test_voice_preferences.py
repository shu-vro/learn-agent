"""Read-aloud voice: id resolution, S3 key layout, and preference updates."""

from src.config.voice_config import DEFAULT_VOICE_ID, VOICE_IDS, resolve_voice_id
from src.schemas.preferences import (
    ChatModelPreferencesUpdate,
    UserPreferencesUpdate,
    apply_preferences_update,
)
from src.utils.markdown_speech import markdown_to_speech


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


def test_markdown_is_not_read_as_syntax():
    spoken = markdown_to_speech("**it is completely normal**, see `x` here.")
    assert spoken == "it is completely normal, see x here."


def test_markdown_blocks_are_separated_but_sentences_are_not():
    spoken = markdown_to_speech(
        "# Title\n\nA [link](http://a.b) inline.\n\n- one\n- two"
    )
    assert spoken == "Title\nA link inline.\none\ntwo"


def test_code_fences_are_dropped():
    spoken = markdown_to_speech("Before.\n\n```py\nprint(1)\n```\n\nAfter.")
    assert spoken == "Before.\nAfter."


def test_empty_and_syntax_only_markdown_yield_nothing():
    assert markdown_to_speech("") == ""
    assert markdown_to_speech("---") == ""


def test_latex_delimiters_are_not_read_aloud():
    assert markdown_to_speech("The energy is $$E = mc^2$$ here.") == (
        "The energy is E = mc squared here."
    )
    assert markdown_to_speech("Inline \\(a + b\\) and \\[c = d\\].") == (
        "Inline a + b and c = d ."
    )


def test_latex_operators_become_words():
    spoken = markdown_to_speech("We know $$\\frac{a}{b} \\le \\alpha$$ holds.")
    assert spoken == "We know a over b less than or equal to alpha holds."


def test_subscripts_survive_markdown_emphasis():
    # Without the math pass, markdown reads `_1$$ and $$x_` as emphasis.
    assert markdown_to_speech("Let $$x_1$$ and $$x_2$$ differ.") == (
        "Let x sub 1 and x sub 2 differ."
    )


def test_currency_is_left_alone():
    assert markdown_to_speech("It costs $5 and $10 total.") == (
        "It costs $5 and $10 total."
    )
