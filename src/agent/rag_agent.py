"""CLI adapters for the shared RAG agent core."""

from __future__ import annotations

from typing import Any, Literal

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver

from src.agent.rag_core import (
    RagAppConfig,
    build_rag_agent,
    content_to_text,
    stream_rag_events,
)
from src.db import CONN_URL
from src.utils.time_utils import measure_time
from src.utils.usage_aggregator_callback import UsageAggregatorCallback

# Re-export for callers that import RagAppConfig from this module.
__all__ = [
    "RagAppConfig",
    "answer_question",
    "interactive_chat",
]

_DIM = "\033[2m"
_RESET = "\033[0m\n\n"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_ANSI_RESET = "\033[0m"

SUMMARIZATION_AGGREGATOR_KEY = "summarization_calls"


def _format_tool_args(args: Any) -> str:
    if isinstance(args, dict):
        if not args:
            return "(no args)"
        return ", ".join(f"{key}={value!r}" for key, value in args.items())
    return repr(args)


def _preview(text: str, limit: int = 500) -> str:
    text = text.strip()
    if len(text) > limit:
        return f"{text[:limit]}… ({len(text)} chars total)"
    return text


@wrap_tool_call
def trace_tool_calls(request, handler):
    """Print each tool execution to the terminal with a formatted trace."""
    tool_call = request.tool_call
    name = tool_call.get("name", "unknown")
    args = tool_call.get("args", {})

    print(
        f"\n{_CYAN}{_BOLD}┌─ tool call → {name}{_ANSI_RESET}",
        flush=True,
    )
    print(f"{_CYAN}│  args: {_ANSI_RESET}{_format_tool_args(args)}", flush=True)

    try:
        result = handler(request)
    except Exception as exc:  # noqa: BLE001
        print(
            f"{_RED}└─ error ← {name}: {exc}{_ANSI_RESET}\n",
            flush=True,
        )
        raise

    output = getattr(result, "content", result)
    print(
        f"{_GREEN}└─ result ← {name}{_ANSI_RESET}\n"
        f"{_DIM}{_preview(content_to_text(output))}{_ANSI_RESET}\n",
        flush=True,
    )
    return result


@measure_time
def answer_question(
    question: str,
    config: RagAppConfig,
    mode: Literal["ask", "chat"] = "ask",
    checkpointer: BaseCheckpointSaver | None = None,
    messages: list[HumanMessage | AIMessage] | None = None,
    usage_aggregator: UsageAggregatorCallback | None = UsageAggregatorCallback(
        "rag_agent_calls"
    ),
    thread_id: str = "1",
):
    """Answer a question via the shared RAG agent (CLI output)."""
    checkpointer = checkpointer if mode == "chat" else None

    agent, summarization_aggregator = build_rag_agent(
        config,
        checkpointer=checkpointer,
        usage_aggregator=usage_aggregator,
        extra_middleware=[trace_tool_calls],
    )

    answer_text = ""
    reasoning_section_open = False
    current_thinking_step: int | None = None

    for event in stream_rag_events(
        agent,
        question=question,
        thread_id=thread_id,
    ):
        if event.type == "thinking":
            delta = event.data.get("delta", "")
            step = int(event.data.get("step") or 0)
            if current_thinking_step != step:
                if reasoning_section_open:
                    print(f"\n---{_RESET}", flush=True)
                print(
                    f"\n{_DIM}--- thinking (step {step + 1}) ---\n",
                    end="",
                    flush=True,
                )
                reasoning_section_open = True
                current_thinking_step = step
            print(delta, end="", flush=True)
        elif event.type == "token":
            delta = event.data.get("delta", "")
            if reasoning_section_open:
                print(f"\n---{_RESET}\n", flush=True)
                reasoning_section_open = False
                current_thinking_step = None
            print(delta, end="", flush=True)
            answer_text += delta
        elif event.type == "tool":
            if event.data.get("phase") == "start" and reasoning_section_open:
                print(f"\n---{_RESET}\n", flush=True)
                reasoning_section_open = False
        elif event.type == "summarizing":
            print("\n---------Summarizing Past Messages---------\n")
        elif event.type == "done":
            answer_text = event.data.get("answer", answer_text)

    if reasoning_section_open:
        print(f"\n---{_RESET}", flush=True)

    if mode == "ask":
        print(
            "\nAggregated Usage Metadata:",
            usage_aggregator.get_aggregated_usage() if usage_aggregator else "N/A",
        )
        if summarization_aggregator:
            summarize_usage = summarization_aggregator.get_aggregated_usage().get(
                SUMMARIZATION_AGGREGATOR_KEY, []
            )
            if summarize_usage:
                print(
                    "\nAggregated Summarization Metadata:",
                    summarize_usage,
                )

    if messages is not None:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer_text))


def interactive_chat(config: RagAppConfig) -> None:
    with PostgresSaver.from_conn_string(CONN_URL) as checkpointer:
        checkpointer.setup()
        messages: list[BaseMessage] = []
        global_usage_aggregator = UsageAggregatorCallback("rag_agent_calls")

        while True:
            question = input("\n> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            answer_question(
                question,
                config=config,
                messages=messages,
                mode="chat",
                checkpointer=checkpointer,
                usage_aggregator=global_usage_aggregator,
            )

        print(
            "\nAggregated Usage Metadata:",
            global_usage_aggregator.get_aggregated_usage(),
        )

        for msg in messages:
            msg.pretty_print()
