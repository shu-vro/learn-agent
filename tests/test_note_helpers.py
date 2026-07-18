from __future__ import annotations

from src.side_agents.note_generator_agent import _clean_markdown, _normalize_math


def test_clean_markdown_passes_through_plain_markdown() -> None:
    note = "## Introduction\n\nIntro text\n\n## Summary\n\nSum"
    assert _clean_markdown(note) == note


def test_clean_markdown_strips_wrapping_fence() -> None:
    fenced = "```markdown\n## Introduction\n\nIntro text\n```"
    assert _clean_markdown(fenced) == "## Introduction\n\nIntro text"


def test_clean_markdown_keeps_inner_code_fences() -> None:
    note = "## Description\n\n```python\nprint(1)\n```\n\nmore text"
    assert _clean_markdown(note) == note


def test_normalize_math_converts_inline_latex_delimiters() -> None:
    assert _normalize_math(r"the value \(x^2 + 1\) is positive") == (
        "the value $$x^2 + 1$$ is positive"
    )


def test_normalize_math_converts_block_latex_delimiters() -> None:
    assert _normalize_math(r"\[E = mc^2\]") == "\n$$\nE = mc^2\n$$\n"


def test_normalize_math_leaves_existing_dollar_math_untouched() -> None:
    note = "inline $$a+b$$ and block\n\n$$\nc+d\n$$"
    assert _normalize_math(note) == note
