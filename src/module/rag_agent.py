from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model

from src.config.env import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OCR_LIB,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_PAPER_SOURCES,
    DEFAULT_VISION_MODEL,
    DEFAULT_LLM_MODEL,
)
from src.module.upload_docs import ingest_paper_to_qdrant


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
    use_image_descriptions: bool,
    use_formula_transcription: bool,
    equation_ocr_lib: str,
    auto_pull_models: bool,
) -> tuple[Any, dict[str, Any]]:
    if not config.sources:
        raise ValueError("RagAppConfig.sources cannot be empty.")

    ingestion_info = ingest_paper_to_qdrant(
        source=config.sources,
        collection_name=config.collection_name,
        artifacts_root=config.artifacts_root,
        embedding_model_name=config.embedding_model,
        vision_model_name=config.vision_model,
        equation_ocr_lib=equation_ocr_lib,
        use_vision_model=use_vision_model,
        use_image_descriptions=use_image_descriptions,
        use_formula_transcription=use_formula_transcription,
        auto_pull_models=auto_pull_models,
        recreate_collection=rebuild_index,
    )
    return ingestion_info["vectorstore"], ingestion_info


def answer_question(
    question: str,
    config: RagAppConfig,
    rebuild_index: bool = False,
    use_vision_model: bool = True,
    use_image_descriptions: bool = True,
    use_formula_transcription: bool = True,
    equation_ocr_lib: str = DEFAULT_OCR_LIB,
    auto_pull_models: bool = True,
) -> dict[str, Any]:
    vectorstore, ingestion_info = _get_vectorstore(
        config=config,
        rebuild_index=rebuild_index,
        use_vision_model=use_vision_model,
        use_image_descriptions=use_image_descriptions,
        use_formula_transcription=use_formula_transcription,
        equation_ocr_lib=equation_ocr_lib,
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
        "Answer only from the provided context extracted from the indexed papers. "
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
    use_image_descriptions: bool = True,
    use_formula_transcription: bool = True,
    equation_ocr_lib: str = DEFAULT_OCR_LIB,
    auto_pull_models: bool = True,
) -> None:
    vectorstore, _ = _get_vectorstore(
        config=config,
        rebuild_index=rebuild_index,
        use_vision_model=use_vision_model,
        use_image_descriptions=use_image_descriptions,
        use_formula_transcription=use_formula_transcription,
        equation_ocr_lib=equation_ocr_lib,
        auto_pull_models=auto_pull_models,
    )
    # llm = ChatOllama(model=config.llm_model, temperature=0)
    llm = init_chat_model(model=config.llm_model, temperature=0)

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
            "Answer only from the provided context extracted from the paper.\n\n"
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
