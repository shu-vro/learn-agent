from __future__ import annotations

import os
from types import MappingProxyType
from typing import TypedDict, Unpack

from langchain.chat_models import init_chat_model, BaseChatModel

OMLX_BASE_URL = os.environ.get("OMLX_BASE_URL", "http://localhost:9999/v1")
OMLX_API_KEY = os.environ.get("OMLX_API_KEY", "")

MODEL_CONFIGS = MappingProxyType(
    {
        "ollama:gemma4:e2b": {
            "provider": "ollama",
            "model": "gemma4:e2b",
            "input": 0,
            "output": 0,
            "context_window": 128000,
            "temperature": 1,
        },
        "omlx:gemma-4-e4b-it-4bit": {
            "provider": "omlx",
            "model": "gemma-4-e4b-it-4bit",
            "input": 0,
            "output": 0,
            "context_window": 128000,
            "temperature": 1,
        },
    }
)


class InitChatModelKwargs(TypedDict, total=False):
    model_provider: str | None
    config_prefix: str | None
    temperature: float
    callbacks: list[object]
    base_url: str | None
    api_key: str


def model_selector(
    model_name: str, **kwargs: Unpack[InitChatModelKwargs]
) -> BaseChatModel:
    selected_model = MODEL_CONFIGS[model_name]
    provider, model = selected_model["provider"], selected_model["model"]

    model_kwargs: InitChatModelKwargs = {
        "temperature": selected_model.get("temperature", 0),
        **kwargs,
    }
    if provider == "omlx":
        # OMLX exposes an OpenAI-compatible API on the local machine.
        provider = "openai"
        model_kwargs["base_url"] = OMLX_BASE_URL
        model_kwargs.setdefault("api_key", OMLX_API_KEY)
    return init_chat_model(
        model=model,
        model_provider=provider,
        **model_kwargs,
    )


__all__ = ["model_selector", "MODEL_CONFIGS"]
