"""Turn assistant markdown into text that sounds right when read aloud."""

import re

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark").enable("table")

# Tags whose content should end with a pause when spoken.
_BLOCK_TAGS = (
    "p, h1, h2, h3, h4, h5, h6, li, tr, blockquote, pre, div, hr, table"
).split(", ")


def markdown_to_speech(text: str) -> str:
    """Strip markdown syntax so TTS reads prose, not asterisks and backticks.

    Rendering to HTML first lets the parser decide what is syntax and what is
    content, instead of guessing with regexes. Fenced code blocks are dropped —
    read aloud they are noise, not information.
    """
    soup = BeautifulSoup(_md.render(text or ""), "html.parser")

    for block in soup.find_all("pre"):
        block.decompose()

    # Separate blocks (and table cells) so sentences don't run together, while
    # inline markup (bold, links, inline code) stays part of its sentence.
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


__all__ = ["markdown_to_speech"]
