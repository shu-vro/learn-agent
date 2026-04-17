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
    AIMessageChunk,
)
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
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
from src.vector_store.qdrant_store import vector_store
from src.utils.time_utils import measure_time


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


def _format_context(documents: list[Document]) -> str:
    """this is for llm prompt."""
    context_blocks: list[str] = []

    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata
        block_header = (
            f"[Source {idx}] type={meta.get('type', 'unknown')}, "
            f"source={meta.get('source', 'unknown')}, "
            f"page={meta.get('page', 'n/a')}, "
            f"image_path={meta.get('path', 'n/a')}"
        )
        context_blocks.append(f"{block_header}\n{doc.page_content}")

    return "\n\n".join(context_blocks)


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
                ]
            )
        )
    return lines


@measure_time
def _context_making_strategy(
    question: str, messages: list[BaseMessage], config: RagAppConfig
) -> list[Document]:
    retrieved_docs = vector_store.similarity_search(
        question, k=config.top_k, score_threshold=0.5
    )
    retrieved_docs = retrieved_docs[: config.top_k]
    return retrieved_docs


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

    retrieved_docs = _context_making_strategy(question, messages or [], config)
    context = _format_context(retrieved_docs)

    prompt = (
        "You are a strict research-paper QA assistant. "
        "Answer only from the provided context extracted from the indexed papers. "
        "If the answer is not present in context, explicitly say you could not find it in the indexed paper context.\n\n"
        "Provide a concise answer followed by evidence bullets that reference source numbers."
    )
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

    agent = create_agent(
        llm,
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=summarization_llm, trigger=("tokens", 500), keep=("messages", 2)
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
    for chunk in agent.stream(
        {"messages": (f"Question:\n{question}\n\n" f"Context:\n{context}\n\n")},
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
            ai_message = (token.get("model") or {}).get("messages", [None])[0]
            if ai_message and ai_message.content:
                answer_text += (
                    ai_message.content if ai_message and ai_message.content else ""
                )
            elif token.get("SummarizationMiddleware.before_model"):
                print("\n---------Summarizing Past Messages---------\n")
            else:
                print(token)

    print("\n\nSources:")
    for line in _source_summary_lines(retrieved_docs):
        print(f"- {line}")

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
