"""Turn assistant markdown into text that sounds right when read aloud."""

import re

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from pylatexenc.latex2text import LatexNodes2Text

_md = MarkdownIt("commonmark").enable("table")
_latex = LatexNodes2Text(math_mode="text")

# Tags whose content should end with a pause when spoken.
_BLOCK_TAGS = (
    "p, h1, h2, h3, h4, h5, h6, li, tr, blockquote, pre, div, hr, table"
).split(", ")

# The agent prompts mandate `$$…$$` (see src/agent/prompts/main_agent.md), but
# models drift into the other delimiters, so all four are handled.
_MATH_DELIMITERS = (
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    # Single `$`: content must not start or end with a space, which keeps
    # currency ("$5 and $10 total") from being read as one math expression.
    re.compile(r"\$(?!\s)([^$\n]{1,200}?)(?<!\s)\$"),
)

# Symbols pylatexenc leaves behind that a TTS voice either skips or mangles.
# Applied only inside a math expression, never to ordinary prose.
_MATH_WORDS = {
    "≤": " less than or equal to ",
    "≥": " greater than or equal to ",
    "≠": " not equal to ",
    "≈": " approximately ",
    "±": " plus or minus ",
    "×": " times ",
    "÷": " divided by ",
    "⋅": " times ",
    "∞": " infinity ",
    "∑": " sum of ",
    "∏": " product of ",
    "∫": " integral of ",
    "∂": " partial ",
    "√": " square root of ",
    "∈": " in ",
    "→": " to ",
    "/": " over ",
    "α": " alpha ",
    "β": " beta ",
    "γ": " gamma ",
    "δ": " delta ",
    "Δ": " delta ",
    "ε": " epsilon ",
    "θ": " theta ",
    "λ": " lambda ",
    "μ": " mu ",
    "π": " pi ",
    "ρ": " rho ",
    "σ": " sigma ",
    "τ": " tau ",
    "φ": " phi ",
    "ω": " omega ",
}

_EXPONENT_WORDS = {"2": "squared", "3": "cubed"}


def _exponent(power: str) -> str:
    if power in _EXPONENT_WORDS:
        return f" {_EXPONENT_WORDS[power]} "
    return f" to the power of {power} "


def _speak_math(match: re.Match) -> str:
    """Render one LaTeX expression as words.

    ponytail: good enough for the operators that show up in chat answers, not a
    full math-to-speech engine. Extend _MATH_WORDS when something reads wrong.
    """
    expression = match.group(1).strip()
    try:
        spoken = _latex.latex_to_text(expression)
    except Exception:
        spoken = expression

    spoken = re.sub(r"\^\{?(\w+)\}?", lambda m: _exponent(m.group(1)), spoken)
    spoken = re.sub(r"_\{?(\w+)\}?", r" sub \1", spoken)
    for symbol, word in _MATH_WORDS.items():
        spoken = spoken.replace(symbol, word)

    # Pad so the expression never fuses with the words around it.
    return f" {re.sub(r'\s+', ' ', spoken).strip()} "


def strip_math(text: str) -> str:
    """Replace LaTeX expressions with their spoken form, delimiters included."""
    for pattern in _MATH_DELIMITERS:
        text = pattern.sub(_speak_math, text)
    return text


def markdown_to_speech(text: str) -> str:
    """Strip markdown syntax so TTS reads prose, not asterisks and backticks.

    Rendering to HTML first lets the parser decide what is syntax and what is
    content, instead of guessing with regexes. Fenced code blocks are dropped —
    read aloud they are noise, not information.
    """
    # Math runs first: markdown would otherwise eat `_` in `$$x_1$$` as emphasis.
    soup = BeautifulSoup(_md.render(strip_math(text or "")), "html.parser")

    for block in soup.find_all("pre"):
        block.decompose()

    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append("\n")
    for cell in soup.find_all(["td", "th"]):
        cell.append(" ")
    for br in soup.find_all("br"):
        br.replace_with("\n")

    spoken = soup.get_text("")
    spoken = re.sub(r"[ \t]+", " ", spoken)
    spoken = re.sub(r" *\n\s*", "\n", spoken)
    return spoken.strip()


__all__ = ["markdown_to_speech", "strip_math"]
