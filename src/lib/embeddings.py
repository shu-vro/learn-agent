from langchain_huggingface import HuggingFaceEmbeddings
from src.config.constants import DEFAULT_EMBEDDING_MODEL


def build_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)
