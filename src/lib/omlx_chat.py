from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

OMLX_REASONING_EFFORT = os.environ.get("OMLX_REASONING_EFFORT", "high")


class ChatOmlx(ChatOpenAI):
    """OpenAI-compatible client for OMLX that preserves reasoning stream tokens."""

    def __init__(self, **kwargs: Any) -> None:
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        extra_body.setdefault("reasoning", {"effort": OMLX_REASONING_EFFORT})
        super().__init__(extra_body=extra_body, **kwargs)

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type[BaseMessageChunk],
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None

        choices = chunk.get("choices", []) or chunk.get("chunk", {}).get("choices", [])
        if not choices:
            return generation_chunk

        delta = choices[0].get("delta") or {}
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content and isinstance(generation_chunk.message, AIMessageChunk):
            generation_chunk.message.additional_kwargs["reasoning_content"] = (
                reasoning_content
            )

        return generation_chunk
