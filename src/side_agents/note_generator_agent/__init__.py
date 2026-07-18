from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config.constants import DEFAULT_NOTE_GENERATION_MODEL, DEFAULT_VISION_MODEL
from src.config.model_config import model_selector
from src.db.models.chunk import IMAGE_CHUNK_TYPE
from src.utils.usage_aggregator_callback import UsageAggregatorCallback

PROMPT_PATH = Path(__file__).with_name("prompt.md")
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()

EMPTY_CHUNK: dict[str, Any] = {
    "id": "",
    "type": "text_chunk",
    "content": "",
    "extra": {},
}

# Strips a single set of markdown/code fences wrapping the *entire* response,
# e.g. when a model wraps the whole note in ```markdown ... ```.
_WRAPPING_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n(.*)\n```$", re.DOTALL)

# Streamdown renders math with `$$` delimiters only. Models often default to the
# standard LaTeX `\( ... \)` (inline) and `\[ ... \]` (block) delimiters, so we
# rewrite those to the expected format. These delimiters are unambiguous, so the
# conversion is safe (unlike touching single `$`, which can be currency).
_LATEX_BLOCK = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_LATEX_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def _normalize_math(text: str) -> str:
    text = _LATEX_BLOCK.sub(lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", text)
    text = _LATEX_INLINE.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    return text


def _chunk_context(label: str, chunk: dict[str, Any]) -> str:
    chunk_type = str(chunk.get("type") or "text_chunk")
    extra = chunk.get("extra") or {}
    metadata = json.dumps(extra, ensure_ascii=False, sort_keys=True, default=str)
    content = str(chunk.get("content") or "").strip()
    lines = [
        f"[{label}]",
        f"chunk_id: {chunk.get('id') or 'n/a'}",
        f"type: {chunk_type}",
        f"metadata: {metadata}",
        "content:",
        content if content else "[empty]",
    ]
    if chunk_type == IMAGE_CHUNK_TYPE:
        caption = extra.get("caption")
        if caption:
            lines.append(f"caption: {caption}")
        path = extra.get("path")
        if path:
            lines.append(f"image_path: {path}")
    return "\n".join(lines)


def _clean_markdown(raw_content: object) -> str:
    text = raw_content if isinstance(raw_content, str) else str(raw_content)
    text = text.strip()
    match = _WRAPPING_FENCE.match(text)
    if match:
        text = match.group(1).strip()
    return _normalize_math(text)


def _build_human_text(
    prev_chunk: dict[str, Any],
    target_chunk: dict[str, Any],
    next_chunk: dict[str, Any],
) -> str:
    return (
        "Write study notes for the CENTER (target) chunk using the sliding window "
        "below. Output Markdown only.\n\n"
        f"{_chunk_context('Previous chunk', prev_chunk)}\n\n"
        f"{_chunk_context('Target chunk (CENTER — notes are about this chunk)', target_chunk)}\n\n"
        f"{_chunk_context('Next chunk', next_chunk)}"
    )


def _image_path_for_chunk(chunk: dict[str, Any]) -> Path | None:
    extra = chunk.get("extra") or {}
    raw_path = extra.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path.strip())
    if path.is_file():
        return path
    return None


def generate_chunk_note(
    prev_chunk: dict[str, Any] | None,
    target_chunk: dict[str, Any],
    next_chunk: dict[str, Any] | None,
) -> str:
    """Generate a Markdown study note for the target chunk.

    The model writes the full Markdown note directly (introduction, description,
    summary, and optional analytical questions), which is returned verbatim.
    """
    usage_aggregator = UsageAggregatorCallback("note_generator_agent_usage")
    prev = prev_chunk or EMPTY_CHUNK
    nxt = next_chunk or EMPTY_CHUNK
    human_text = _build_human_text(prev, target_chunk, nxt)

    target_type = str(target_chunk.get("type") or "text_chunk")
    image_path = (
        _image_path_for_chunk(target_chunk) if target_type == IMAGE_CHUNK_TYPE else None
    )

    if image_path is not None:
        with image_path.open("rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("ascii")
        llm = model_selector(
            DEFAULT_VISION_MODEL,
            temperature=0,
            callbacks=[usage_aggregator],
        )
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": human_text},
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_b64}",
                },
            ]
        )
    else:
        llm = model_selector(
            DEFAULT_NOTE_GENERATION_MODEL,
            temperature=0,
            callbacks=[usage_aggregator],
        )
        human_message = HumanMessage(content=human_text)

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            human_message,
        ]
    )

    note = _clean_markdown(response.content)
    if not note:
        raise ValueError("Model returned an empty note for the target chunk.")
    return note


__all__ = [
    "EMPTY_CHUNK",
    "generate_chunk_note",
]
