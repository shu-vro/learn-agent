from __future__ import annotations

from types import MappingProxyType
from typing import Literal, TypedDict, Unpack

from langchain.chat_models import init_chat_model

MODEL_CONFIGS = MappingProxyType(
    {
        "ollama:gemma4:e2b": {
            "provider": "ollama",
            "model": "gemma4:e2b",
            "input": 0,
            "output": 0,
            "context_window": 128000,
            "temperature": 1,
        }
    }
)


class InitChatModelKwargs(TypedDict, total=False):
    model_provider: str | None
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None
    config_prefix: str | None
    temperature: float
    callbacks: list[object]


def model_selector(model_name: str, **kwargs: Unpack[InitChatModelKwargs]):
    selected_model = MODEL_CONFIGS[model_name]
    provider, model = selected_model["provider"], selected_model["model"]
    model_kwargs: InitChatModelKwargs = {
        "temperature": selected_model.get("temperature", 0),
        "configurable_fields": "any",
        **kwargs,
    }
    return init_chat_model(
        model=model,
        model_provider=provider,
        **model_kwargs,
    )


__all__ = ["model_selector", "MODEL_CONFIGS"]
