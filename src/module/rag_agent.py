from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from src.vector_store.faiss_store import faiss_index_exists, load_faiss_index
from src.module.upload_docs import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_PAPER_SOURCE,
    DEFAULT_VISION_MODEL,
    ingest_paper_to_faiss,
)


@dataclass(slots=True)
class RagAppConfig:
    source: str = DEFAULT_PAPER_SOURCE
    index_dir: Path = DEFAULT_INDEX_DIR
    artifacts_root: Path = DEFAULT_ARTIFACTS_DIR
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_model: str = "gemma4:e4b"
    vision_model: str = DEFAULT_VISION_MODEL
    top_k: int = 5


def _format_context(documents: list[Document]) -> str:
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


def _balance_context_documents(documents: list[Document], top_k: int) -> list[Document]:
    if len(documents) <= top_k:
        return documents

    text_docs = [doc for doc in documents if doc.metadata.get("type") != "image"]
    image_docs = [doc for doc in documents if doc.metadata.get("type") == "image"]

    if top_k <= 1:
        return documents[:1]

    selected: list[Document] = []
    selected.extend(text_docs[: max(top_k - 1, 1)])

    if image_docs:
        selected.append(image_docs[0])

    if len(selected) < top_k:
        for doc in documents:
            if doc in selected:
                continue
            selected.append(doc)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


def _get_vectorstore(
    config: RagAppConfig,
    rebuild_index: bool,
    use_vision_model: bool,
    auto_pull_models: bool,
) -> tuple[Any, dict[str, Any]]:
    if rebuild_index or not faiss_index_exists(config.index_dir):
        ingestion_info = ingest_paper_to_faiss(
            source=config.source,
            index_dir=config.index_dir,
            artifacts_root=config.artifacts_root,
            embedding_model_name=config.embedding_model,
            vision_model_name=config.vision_model,
            use_vision_model=use_vision_model,
            auto_pull_models=auto_pull_models,
        )
        return ingestion_info["vectorstore"], ingestion_info

    loaded_vectorstore = load_faiss_index(
        embedding_model_name=config.embedding_model,
        index_dir=config.index_dir,
    )
    return loaded_vectorstore, {
        "vectorstore": loaded_vectorstore,
        "source": config.source,
        "index_dir": str(config.index_dir),
        "artifacts_root": str(config.artifacts_root),
        "documents_indexed": 0,
        "embedding_model": config.embedding_model,
        "vision_model": config.vision_model if use_vision_model else "disabled",
    }


def answer_question(
    question: str,
    config: RagAppConfig,
    rebuild_index: bool = False,
    use_vision_model: bool = True,
    auto_pull_models: bool = True,
) -> dict[str, Any]:
    vectorstore, ingestion_info = _get_vectorstore(
        config=config,
        rebuild_index=rebuild_index,
        use_vision_model=use_vision_model,
        auto_pull_models=auto_pull_models,
    )

    retrieved_docs = vectorstore.similarity_search(
        question, k=max(config.top_k * 3, config.top_k)
    )
    retrieved_docs = _balance_context_documents(retrieved_docs, config.top_k)
    context = _format_context(retrieved_docs)

    llm = ChatOllama(model=config.llm_model, temperature=0)
    prompt = (
        "You are a strict research-paper QA assistant. "
        "Answer only from the provided context extracted from the paper 'Attention Is All You Need'. "
        "If the answer is not present in context, explicitly say you could not find it in the indexed paper context.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Provide a concise answer followed by evidence bullets that reference source numbers."
    )

    response = llm.invoke(prompt)
    answer_text = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer_text,
        "sources": retrieved_docs,
        "source_summaries": _source_summary_lines(retrieved_docs),
        "ingestion": ingestion_info,
    }


def interactive_chat(
    config: RagAppConfig,
    rebuild_index: bool = False,
    use_vision_model: bool = True,
    auto_pull_models: bool = True,
) -> None:
    vectorstore, _ = _get_vectorstore(
        config=config,
        rebuild_index=rebuild_index,
        use_vision_model=use_vision_model,
        auto_pull_models=auto_pull_models,
    )
    llm = ChatOllama(model=config.llm_model, temperature=0)

    print("RAG chat is ready. Type a question, or 'exit' to stop.")
    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        retrieved_docs = vectorstore.similarity_search(
            question, k=max(config.top_k * 3, config.top_k)
        )
        retrieved_docs = _balance_context_documents(retrieved_docs, config.top_k)
        context = _format_context(retrieved_docs)

        prompt = (
            "You are a strict research-paper QA assistant. "
            "Answer only from the provided context extracted from the paper 'Attention Is All You Need'.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Give a concise answer and include source references by source number."
        )

        response = llm.invoke(prompt)
        answer_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        print("\nAnswer:\n")
        print(answer_text)
        print("\nSources:")
        for line in _source_summary_lines(retrieved_docs):
            print(f"- {line}")
