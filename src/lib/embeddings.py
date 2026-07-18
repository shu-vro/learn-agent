from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config.constants import DEFAULT_EMBEDDING_MODEL
from src.config.embedding_model_config import EMBEDDING_MODELS
from src.config.model_config import OMLX_API_KEY, OMLX_BASE_URL
from src.lib.omlx import OmlxEmbeddings


@lru_cache(maxsize=8)
def build_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Embeddings:
    selected_model = EMBEDDING_MODELS[model_name]
    provider, model = selected_model["provider"], selected_model["model"]
    if provider == "hf":
        return HuggingFaceEmbeddings(model_name=model)
    if provider == "ollama":
        return OllamaEmbeddings(model_name=model)
    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    if provider == "omlx":
        return OmlxEmbeddings(
            model=model,
            base_url=OMLX_BASE_URL,
            api_key=OMLX_API_KEY or None,
        )
    raise ValueError(f"Invalid provider: {provider}")


__all__ = ["build_embeddings"]
