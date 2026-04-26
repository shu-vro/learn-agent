from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from src.config.constants import DEFAULT_EMBEDDING_MODEL


@lru_cache(maxsize=8)
def build_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)
