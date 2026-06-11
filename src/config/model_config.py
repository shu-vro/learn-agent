from __future__ import annotations

import os
from types import MappingProxyType
from typing import TypedDict, Unpack
from enum import Enum

from langchain.chat_models import init_chat_model, BaseChatModel

from src.lib.omlx_chat import ChatOmlx

OMLX_BASE_URL = os.environ.get("OMLX_BASE_URL", "http://localhost:9999/v1")
OMLX_API_KEY = os.environ.get("OMLX_API_KEY", "")


class ReasoningEffort(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NONE = None


class Model:
    def __init__(
        self,
        provider: str,
        model: str,
        model_name: str,
        pricing: dict[str, float],
        context_window: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None = None,
    ):
        self.provider = provider
        self.model = model
        self.model_name = model_name
        self.pricing = pricing
        self.context_window = context_window
        self.temperature = temperature
        self.reasoning_effort = (
            reasoning_effort if reasoning_effort else ReasoningEffort.NONE
        )


MODEL_CONFIGS = MappingProxyType(
    {
        "ollama:gemma4:e2b": Model(
            provider="ollama",
            model="gemma4:e2b",
            model_name="gemma4:e2b",
            pricing={"input": 0, "output": 0, "input_cache_read": 0},
            context_window=128000,
            temperature=1,
        ),
        "omlx:gemma-4-e4b-it-4bit": Model(
            provider="omlx",
            model="gemma-4-e4b-it-4bit",
            model_name="gemma4:e4b-it-4bit",
            pricing={"input": 0, "output": 0, "input_cache_read": 0},
            context_window=128000,
            temperature=1,
            reasoning_effort=ReasoningEffort.HIGH,
        ),
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
        return ChatOmlx(
            model=model,
            base_url=OMLX_BASE_URL,
            api_key=model_kwargs.get("api_key") or OMLX_API_KEY,
            temperature=model_kwargs.get("temperature", 0),
            callbacks=model_kwargs.get("callbacks"),
        )
    return init_chat_model(
        model=model,
        model_provider=provider,
        **model_kwargs,
    )


__all__ = ["model_selector", "MODEL_CONFIGS", "ReasoningEffort"]
