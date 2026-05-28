from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from src.config.constants import DEFAULT_EMBEDDING_MODEL
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from src.config.embedding_model_config import EMBEDDING_MODELS


@lru_cache(maxsize=8)
def build_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> OllamaEmbeddings:
    selected_model = EMBEDDING_MODELS[model_name]
    provider, model = selected_model["provider"], selected_model["model"]
    if provider == "hf":
        return HuggingFaceEmbeddings(model_name=model)
    elif provider == "ollama":
        return OllamaEmbeddings(model_name=model)
    elif provider == "openai":
        return OpenAIEmbeddings(model=model)
    else:
        raise ValueError(f"Invalid provider: {provider}")


__all__ = ["build_embeddings"]
