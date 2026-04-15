from src.utils.time_utils import measure_time
import nltk
from langchain_text_splitters import NLTKTextSplitter


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
    text: str, chunk_size: int = 1800, chunk_overlap: int = 250
) -> list[str]:
    "Splits the input text into chunks of specified size with overlap, using NLTK sentence tokenizer if available."
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    has_nltk_resources = _ensure_nltk_resources()
    text_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if has_nltk_resources:
        chunks = text_splitter.split_text(text)
    else:
        sentence_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        sentence_splits = sentence_tokenizer.tokenize(text)
        chunks = text_splitter._merge_splits(sentence_splits, "\n\n")

    return [chunk.strip() for chunk in chunks if chunk.strip()]
