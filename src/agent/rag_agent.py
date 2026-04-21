from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
)
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from src.config.constants import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OCR_LIB,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_PAPER_SOURCES,
    DEFAULT_VISION_MODEL,
    DEFAULT_LLM_MODEL,
)
from src.utils.usage_aggregator_callback import UsageAggregatorCallback
from src.utils.time_utils import measure_time
from src.agent.tools.document_retriever import retrieve_context
from src.agent.prompts import main_agent_system_prompt


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


def _source_summary_lines(documents: list[Document]) -> list[str]:
    """this is to print in console"""
    lines: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata
        lines.append(
            " | ".join(
                [
                    f"#{idx}",
                    f"type={meta.get('type', 'unknown')}",
                    f"source={meta.get('source', 'unknown')}",
                    f"page={meta.get('page', 'n/a')}",
                    f"image={meta.get('path', 'n/a')}",
                    f"similarity_score={meta.get('similarity_score', 'n/a')}",
                ]
            )
        )
    return lines


def _content_to_text(content: Any) -> str:
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


@measure_time
def answer_question(
    question: str,
    config: RagAppConfig,
    mode: Literal["ask", "chat"] = "ask",
    checkpointer: InMemorySaver | None = None,
    messages: list[HumanMessage | AIMessage] | None = None,
    usage_aggregator: UsageAggregatorCallback | None = UsageAggregatorCallback(
        "rag_agent_calls"
    ),
):
    """
    Answers a question using retrieved context from the vector store and an LLM.
    Args:
        question: The question to answer.
        config: RagAppConfig object containing configuration parameters.
        mode: "ask" for one-off question answering, "chat" for interactive chat mode.
        messages: Optional list to append the question and answer messages for chat mode.
        usage_aggregator: Optional UsageAggregatorCallback to collect LLM usage metadata.
    Returns:
    """

    prompt = main_agent_system_prompt
    system_prompt = SystemMessage(content=prompt)

    SUMMARIZATION_AGGREGATOR_KEY = "summarization_calls"
    summarization_aggregator: UsageAggregatorCallback = UsageAggregatorCallback(
        SUMMARIZATION_AGGREGATOR_KEY
    )

    llm = init_chat_model(
        model=config.llm_model,
        temperature=0,
        configurable_fields="any",
        callbacks=[usage_aggregator] if usage_aggregator else None,
    )

    summarization_llm = init_chat_model(
        model=config.llm_model,
        temperature=0,
        configurable_fields="any",
        callbacks=[summarization_aggregator] if summarization_aggregator else None,
    )

    tools = [retrieve_context]

    agent = create_agent(
        llm,
        tools=tools,
        middleware=[
            SummarizationMiddleware(
                model=summarization_llm, trigger=("tokens", 4000), keep=("messages", 10)
            )
        ],
        system_prompt=system_prompt,
        checkpointer=checkpointer if checkpointer else None,
    )

    runnable_config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    # response = agent.invoke(
    #     {"messages": (f"Question:\n{question}\n\n" f"Context:\n{context}\n\n")},
    #     config=runnable_config,
    # )
    # print(response)

    # answer_text = response.content if hasattr(response, "content") else str(response)

    answer_text = ""
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    for chunk in agent.stream(
        {"messages": (f"Question:\n{question}\n\n")},
        config=runnable_config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        # print(chunk)
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if metadata["langgraph_node"] and metadata["langgraph_node"] == "model":
                print(token.content, end="", flush=True)

        if chunk["type"] == "updates":
            token = chunk["data"]
            if token.get("SummarizationMiddleware.before_model"):
                print("\n---------Summarizing Past Messages---------\n")
                continue

            model_message = (token.get("model") or {}).get("messages", [None])[-1]
            if isinstance(model_message, AIMessage):
                if model_message.tool_calls:
                    for tool_call in model_message.tool_calls:
                        tool_call_id = tool_call.get("id")
                        if tool_call_id:
                            pending_tool_calls[tool_call_id] = {
                                "name": tool_call.get("name", "unknown"),
                                "args": tool_call.get("args", {}),
                            }

                model_text = _content_to_text(model_message.content)
                if model_text:
                    answer_text += model_text

            tool_message = (token.get("tools") or {}).get("messages", [None])[-1]
            if isinstance(tool_message, ToolMessage):
                tool_meta = pending_tool_calls.get(tool_message.tool_call_id, {})
                tool_name = tool_meta.get(
                    "name", getattr(tool_message, "name", "unknown")
                )
                tool_args = tool_meta.get("args", {})
                # tool_response = _content_to_text(tool_message.content)
                artifact = (
                    tool_message["artifact"] if "artifact" in tool_message else None
                )

                print("\n---------tool fired---------")
                print(f"-name: {tool_name}")
                print(f"-args (as you got): {tool_args}")
                _source_summary_lines(artifact) if artifact else None

    print("\n\nSources:")

    if mode == "ask":
        print(
            "\nAggregated Usage Metadata:",
            usage_aggregator.get_aggregated_usage() if usage_aggregator else "N/A",
        )
        if summarization_aggregator:
            summarize_usage = summarization_aggregator.get_aggregated_usage()[0].get(
                SUMMARIZATION_AGGREGATOR_KEY, []
            )
            if len(summarize_usage[SUMMARIZATION_AGGREGATOR_KEY]):
                print(
                    "\nAggregated Summarization Metadata:",
                    summarize_usage,
                )

    if messages is not None:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer_text))
    pass


def interactive_chat(config: RagAppConfig) -> None:
    checkpointer = InMemorySaver()
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
    pass
