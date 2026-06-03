from src.utils.time_utils import measure_time
import re

import nltk
from langchain_text_splitters import NLTKTextSplitter

_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)\n]+\)")
_SPLIT_IMAGE_PATTERN = re.compile(
    r"!\s*\n+\s*(\[[^\]]*\]\([^)\n]+\))",
    re.MULTILINE,
)


def _repair_split_markdown_images(text: str) -> str:
    """Rejoin markdown image syntax broken across lines by upstream export/splitting."""
    return _SPLIT_IMAGE_PATTERN.sub(r"!\1", text)


def _protect_markdown_images(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}

    def _replacement(match: re.Match[str]) -> str:
        token = f"__MDIMG{len(placeholders)}__"
        placeholders[token] = match.group(0)
        return token

    protected = _MARKDOWN_IMAGE_PATTERN.sub(_replacement, text)
    return protected, placeholders


def _restore_markdown_images(text: str, placeholders: dict[str, str]) -> str:
    restored = text
    for token, image_markdown in placeholders.items():
        restored = restored.replace(token, image_markdown)
    return restored


@measure_time
def _ensure_nltk_resources() -> bool:
    "basically checks if nltk tokenizers are available"
    resources = ["tokenizers/punkt", "tokenizers/punkt_tab"]
    for resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            return False

    return True


@measure_time
def chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 250
) -> list[str]:
    """Splits the input text into chunks of specified size with overlap, using NLTK sentence tokenizer if available.
    nltk takes ![image](path) as 2 tokens, ! and [image]. this means they get to join using \\n\\n. so we write extra code
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    normalized_text = _repair_split_markdown_images(text)
    protected_text, image_placeholders = _protect_markdown_images(normalized_text)

    has_nltk_resources = _ensure_nltk_resources()
    text_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if has_nltk_resources:
        chunks = text_splitter.split_text(protected_text)
    else:
        sentence_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        sentence_splits = sentence_tokenizer.tokenize(protected_text)
        chunks = text_splitter._merge_splits(sentence_splits, "\n\n")

    return [
        _restore_markdown_images(chunk.strip(), image_placeholders)
        for chunk in chunks
        if chunk.strip()
    ]
