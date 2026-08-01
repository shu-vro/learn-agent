"""Shared RAG agent factory and event stream used by CLI and API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import models as qdrant_models

from src.agent.prompts import main_agent_system_prompt
from src.agent.tools.builtin_tools import youtube_search
from src.agent.tools.document_retriever import retrieve_context_tool
from src.agent.tools.duckduckgo_search import duckduckgo_search
from src.agent.tools.web_fetch import fetch_url
from src.config.constants import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_OCR_LIB,
    DEFAULT_PAPER_SOURCES,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_VISION_MODEL,
)
from src.config.model_config import model_selector
from src.utils.usage_aggregator_callback import UsageAggregatorCallback

SUMMARIZATION_AGGREGATOR_KEY = "summarization_calls"


@dataclass(slots=True)
class RagAppConfig:
    sources: list[str] = field(default_factory=lambda: list(DEFAULT_PAPER_SOURCES))
    collection_name: str = DEFAULT_QDRANT_COLLECTION
    artifacts_root: Path = DEFAULT_ARTIFACTS_DIR
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    equation_ocr_lib: str = DEFAULT_OCR_LIB
    top_k: int = 5


@dataclass(frozen=True, slots=True)
class RagStreamEvent:
    """Typed event emitted while streaming a RAG agent run."""

    type: Literal["thinking", "token", "tool", "summarizing", "done"]
    data: dict[str, Any] = field(default_factory=dict)


# Demo content hashes (Qdrant metadata.doc_id / documents.sha256) for CLI only.
DEFAULT_CLI_DOC_IDS: list[str] = [
    "444673994328f7be8aee9d96fb240596b6f254f06ebaa53a2673413a244198c9",  # pragma: allowlist secret
    "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",  # pragma: allowlist secret
]


def chunk_reasoning_text(token: Any) -> str:
    additional_kwargs = getattr(token, "additional_kwargs", None) or {}
    reasoning = additional_kwargs.get("reasoning_content")
    if reasoning is None:
        return ""
    return reasoning if isinstance(reasoning, str) else str(reasoning)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                    continue
                if isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
                    continue
            text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def format_human_prompt(
    question: str,
    *,
    selection: str | None = None,
    reference_id: str | None = None,
    has_images: bool = False,
) -> str:
    """Build the human message text sent to the model (may include selection)."""
    q = (question or "").strip()
    if not q and has_images:
        q = "Please analyze the attached image(s)."
    parts = [f"Question:\n{q}\n"]
    if selection and reference_id:
        parts.append(
            "Referenced selection "
            f"(from message {reference_id}):\n"
            f'"""\n{selection}\n"""\n'
        )
    return "\n".join(parts) + "\n"


def build_human_message(
    question: str,
    *,
    selection: str | None = None,
    reference_id: str | None = None,
    image_data_urls: list[str] | None = None,
) -> HumanMessage:
    """Build a text or multimodal HumanMessage for the agent."""
    from src.utils.chat_images import ensure_model_image_data_urls

    text = format_human_prompt(
        question,
        selection=selection,
        reference_id=reference_id,
        has_images=bool(image_data_urls),
    )
    normalized = ensure_model_image_data_urls(image_data_urls)
    if not normalized:
        return HumanMessage(content=text)

    # Image-first matches OMLX chat UI / VLM template expectations.
    content: list[dict[str, Any]] = [
        *[{"type": "image_url", "image_url": {"url": url}} for url in normalized],
        {"type": "text", "text": text},
    ]
    return HumanMessage(content=content)


def qdrant_filter_for_doc_ids(doc_ids: list[str] | None) -> qdrant_models.Filter | dict:
    """Build a Qdrant filter on ``metadata.doc_id``.

    ``doc_ids`` must be content fingerprints (``documents.sha256``), which is
    what ingestion writes as ``metadata.doc_id`` — not ``documents.id`` UUIDs.

    An empty list returns a never-matching filter so the tool cannot leak
    other tenants' vectors. Pass ``None`` only for callers that intentionally
    want unscoped retrieval (not used by the API).
    """
    if doc_ids is None:
        return {}
    if not doc_ids:
        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.doc_id",
                    match=qdrant_models.MatchValue(value="__no_project_documents__"),
                ),
            ]
        )
    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="metadata.doc_id",
                match=qdrant_models.MatchAny(any=list(doc_ids)),
            ),
        ]
    )


def build_rag_agent(
    config: RagAppConfig,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    qdrant_filter: qdrant_models.Filter | dict | None = None,
    usage_aggregator: UsageAggregatorCallback | None = None,
    summarization_aggregator: UsageAggregatorCallback | None = None,
    extra_middleware: list[Any] | None = None,
) -> tuple[CompiledStateGraph, UsageAggregatorCallback | None]:
    """Create the shared RAG agent graph.

    Returns ``(agent, summarization_aggregator)`` so callers can read usage.
    """
    if summarization_aggregator is None:
        summarization_aggregator = UsageAggregatorCallback(SUMMARIZATION_AGGREGATOR_KEY)

    system_prompt = SystemMessage(content=main_agent_system_prompt)

    llm = model_selector(
        config.llm_model,
        callbacks=[usage_aggregator] if usage_aggregator else None,
    )
    summarization_llm = model_selector(
        config.llm_model,
        temperature=0,
        callbacks=[summarization_aggregator] if summarization_aggregator else None,
    )

    filter_value: qdrant_models.Filter | dict
    if qdrant_filter is None:
        filter_value = qdrant_filter_for_doc_ids(DEFAULT_CLI_DOC_IDS)
    else:
        filter_value = qdrant_filter

    retrieve_context = retrieve_context_tool(filters=filter_value)
    tools = [retrieve_context, duckduckgo_search, youtube_search, fetch_url]

    middleware: list[Any] = list(extra_middleware or [])
    middleware.append(
        SummarizationMiddleware(
            model=summarization_llm,
            trigger=("tokens", 20000),
            keep=("messages", 10),
        )
    )

    agent = create_agent(
        llm,
        tools=tools,
        middleware=middleware,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )
    return agent, summarization_aggregator


def stream_rag_events(
    agent: CompiledStateGraph,
    *,
    question: str,
    thread_id: str,
    human_prompt: str | HumanMessage | None = None,
    image_data_urls: list[str] | None = None,
) -> Iterator[RagStreamEvent]:
    """Stream typed events from a RAG agent run for the given thread.

    Handles multi-round agent loops (think → tools → think → … → answer).
    Each contiguous reasoning segment gets a monotonic ``step`` index so
    callers can persist/render discrete thinking blocks between tool calls.

    ``human_prompt`` may be plain text or a multimodal ``HumanMessage``.
    ``image_data_urls`` attaches images when ``human_prompt`` is text/omitted.
    """
    runnable_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    if isinstance(human_prompt, HumanMessage):
        message_input: Any = human_prompt
    else:
        from src.utils.chat_images import ensure_model_image_data_urls

        text = (
            human_prompt
            if human_prompt is not None
            else format_human_prompt(question, has_images=bool(image_data_urls))
        )
        normalized = ensure_model_image_data_urls(image_data_urls)
        if normalized:
            # Image-first — OMLX rejects remote URLs and is happier with this order.
            message_input = HumanMessage(
                content=[
                    *[
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in normalized
                    ],
                    {"type": "text", "text": text},
                ]
            )
        else:
            message_input = text

    stream_messages: Any = (
        [message_input] if isinstance(message_input, HumanMessage) else message_input
    )

    answer_text = ""
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    # Contiguous reasoning between tool rounds. -1 until the first think delta.
    thinking_step = -1
    # After tools, bump step on the next reasoning delta.
    start_new_thinking_step = False

    for chunk in agent.stream(
        {"messages": stream_messages},
        config=runnable_config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if metadata.get("langgraph_node") == "model":
                reasoning_delta = chunk_reasoning_text(token)
                content_delta = content_to_text(token.content)

                if reasoning_delta:
                    if thinking_step < 0 or start_new_thinking_step:
                        thinking_step += 1
                        start_new_thinking_step = False
                    yield RagStreamEvent(
                        type="thinking",
                        data={"delta": reasoning_delta, "step": thinking_step},
                    )

                if content_delta:
                    answer_text += content_delta
                    yield RagStreamEvent(type="token", data={"delta": content_delta})

        if chunk["type"] == "updates":
            update = chunk["data"]
            if update.get("SummarizationMiddleware.before_model"):
                yield RagStreamEvent(type="summarizing", data={})

            model_message = (update.get("model") or {}).get("messages", [None])[-1]
            if isinstance(model_message, AIMessage):
                if model_message.tool_calls:
                    # Boundary: next reasoning segment is a new thinking step.
                    start_new_thinking_step = True
                    for tool_call in model_message.tool_calls:
                        tool_call_id = tool_call.get("id")
                        if tool_call_id:
                            pending_tool_calls[tool_call_id] = {
                                "name": tool_call.get("name", "unknown"),
                                "args": tool_call.get("args", {}),
                            }
                            yield RagStreamEvent(
                                type="tool",
                                data={
                                    "phase": "start",
                                    "id": tool_call_id,
                                    "name": tool_call.get("name", "unknown"),
                                    "args": tool_call.get("args", {}),
                                    "step": thinking_step,
                                },
                            )

                # Final answer text only when this model turn has no tool calls.
                if not model_message.tool_calls and not answer_text:
                    model_text = content_to_text(model_message.content)
                    if model_text:
                        answer_text += model_text
                        yield RagStreamEvent(type="token", data={"delta": model_text})

            tools_update = update.get("tools") or {}
            tool_messages = tools_update.get("messages") or []
            for tool_msg in tool_messages:
                tool_call_id = getattr(tool_msg, "tool_call_id", None)
                pending = (
                    pending_tool_calls.pop(tool_call_id, None) if tool_call_id else None
                )
                start_new_thinking_step = True
                yield RagStreamEvent(
                    type="tool",
                    data={
                        "phase": "result",
                        "id": tool_call_id,
                        "name": (pending or {}).get("name", "unknown"),
                        "args": (pending or {}).get("args", {}),
                        "result": content_to_text(
                            getattr(tool_msg, "content", tool_msg)
                        ),
                        "step": thinking_step,
                    },
                )

    yield RagStreamEvent(
        type="done",
        data={"answer": answer_text},
    )


def truncate_checkpoint_before_turn(
    agent: CompiledStateGraph,
    *,
    thread_id: str,
    keep_human_turns: int,
) -> None:
    """Remove checkpoint messages from the Nth human turn onward (1-based).

    After truncation, the checkpointer holds history for turns ``1..N-1`` only.
    The caller should then re-run the agent with the Nth human question.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    messages = list((state.values or {}).get("messages") or [])
    if not messages:
        return

    human_indices = [
        i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)
    ]
    if keep_human_turns < 1 or keep_human_turns > len(human_indices):
        # Nothing to keep from this turn — clear everything if regenerating turn 1
        # with no prior humans, or invalid count.
        if keep_human_turns <= 0:
            to_remove = [
                RemoveMessage(id=msg.id) for msg in messages if getattr(msg, "id", None)
            ]
            if to_remove:
                agent.update_state(config, {"messages": to_remove})
        return

    cutoff = human_indices[keep_human_turns - 1]
    to_remove = [
        RemoveMessage(id=msg.id)
        for msg in messages[cutoff:]
        if getattr(msg, "id", None)
    ]
    if to_remove:
        agent.update_state(config, {"messages": to_remove})
