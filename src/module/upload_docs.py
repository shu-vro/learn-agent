from collections.abc import Sequence
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

from src.lib.docling_lib import docling_pdf_extractor
from src.lib.paper_fingerprint import PaperFingerprint, fingerprint_paper_source
from src.vector_store.qdrant_store import (
    create_qdrant_index,
    load_qdrant_index,
    qdrant_paper_hash_exists,
)
from src.lib.ollama_vision import OllamaVisionClient
from src.lib.pix2tex_ocr import Pix2TexFormulaTranscriber
from src.utils.time_utils import measure_time
from src.config.env import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OCR_LIB,
    DEFAULT_PAPER_SOURCES,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_VISION_MODEL,
)


def _tag_documents_with_paper_hash(
    documents: list[Document],
    source: str,
    paper_sha256: str,
) -> None:
    for doc in documents:
        metadata = dict(doc.metadata or {})
        metadata["source"] = source
        metadata["paper_sha256"] = paper_sha256
        doc.metadata = metadata


def _normalize_sources(source: str | Sequence[str]) -> list[str]:
    if isinstance(source, str):
        sources = [source]
    else:
        sources = [item for item in source if item]

    if not sources:
        raise ValueError("At least one paper source is required for ingestion.")

    return sources


def _normalize_equation_ocr_lib(equation_ocr_lib: str) -> str:
    normalized = equation_ocr_lib.strip().lower()
    if normalized not in {"local", "llm"}:
        raise ValueError("equation_ocr_lib must be either 'local' or 'llm'.")
    return normalized


@measure_time
def ingest_paper_to_qdrant(
    source: str | Sequence[str] = DEFAULT_PAPER_SOURCES,
    collection_name: str = DEFAULT_QDRANT_COLLECTION,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_DIR,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    vision_model_name: str = DEFAULT_VISION_MODEL,
    equation_ocr_lib: str = DEFAULT_OCR_LIB,
    use_vision_model: bool = True,
    use_image_descriptions: bool = True,
    use_formula_transcription: bool = True,
    auto_pull_models: bool = True,
    recreate_collection: bool = False,
    paper_fingerprint: PaperFingerprint | None = None,
) -> dict[str, Any]:
    sources = _normalize_sources(source)

    if paper_fingerprint is not None and len(sources) != 1:
        raise ValueError(
            "paper_fingerprint is only supported for single-source ingestion."
        )

    selected_equation_ocr_lib = _normalize_equation_ocr_lib(equation_ocr_lib)

    image_describer = None
    formula_transcriber = None
    if not use_vision_model:
        use_image_descriptions = False
        use_formula_transcription = False

    needs_llm_vision = use_image_descriptions or (
        use_formula_transcription and selected_equation_ocr_lib == "llm"
    )

    vision_client = None
    if needs_llm_vision:
        vision_client = OllamaVisionClient(
            model=vision_model_name,
            auto_pull=auto_pull_models,
        )

    if use_image_descriptions and vision_client is not None:
        image_describer = vision_client.describe_image

    if use_formula_transcription:
        if selected_equation_ocr_lib == "local":
            formula_client = Pix2TexFormulaTranscriber()
            formula_transcriber = formula_client.transcribe_formula_latex
        elif vision_client is not None:
            formula_transcriber = vision_client.transcribe_formula_latex

    print(
        f"{use_vision_model=}, {use_image_descriptions=}, "
        f"{use_formula_transcription=}, {selected_equation_ocr_lib=}, {auto_pull_models=}"
    )

    source_results: list[dict[str, Any]] = []
    total_documents_indexed = 0
    vectorstore = None

    for source_index, current_source in enumerate(sources):
        recreate_for_source = recreate_collection and source_index == 0
        resolved_paper = (
            paper_fingerprint
            if source_index == 0 and paper_fingerprint is not None
            else fingerprint_paper_source(current_source)
        )
        paper_sha256 = resolved_paper.sha256

        if not recreate_for_source and qdrant_paper_hash_exists(
            collection_name=collection_name,
            paper_sha256=paper_sha256,
        ):
            print(
                f"Paper already indexed. paper_sha256={paper_sha256}. Skipping ingestion."
            )
            if vectorstore is None:
                vectorstore = load_qdrant_index(
                    embedding_model_name=embedding_model_name,
                    collection_name=collection_name,
                )
            source_results.append(
                {
                    "source": current_source,
                    "source_file": str(resolved_paper.local_path),
                    "paper_sha256": paper_sha256,
                    "documents_indexed": 0,
                    "skipped_existing_paper": True,
                }
            )
            continue

        print(f"New paper hash detected: {paper_sha256}")

        documents = docling_pdf_extractor(
            file_path=str(resolved_paper.local_path),
            artifacts_root=artifacts_root,
            image_describer=image_describer,
            formula_transcriber=formula_transcriber,
        )
        _tag_documents_with_paper_hash(documents, current_source, paper_sha256)

        vectorstore = create_qdrant_index(
            documents=documents,
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
            recreate=recreate_for_source,
        )

        documents_indexed = len(documents)
        total_documents_indexed += documents_indexed
        source_results.append(
            {
                "source": current_source,
                "source_file": str(resolved_paper.local_path),
                "paper_sha256": paper_sha256,
                "documents_indexed": documents_indexed,
                "skipped_existing_paper": False,
            }
        )

    if vectorstore is None:
        vectorstore = load_qdrant_index(
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
        )

    vision_model = vision_model_name if use_vision_model else "disabled"
    effective_equation_ocr_lib = (
        selected_equation_ocr_lib if use_formula_transcription else "disabled"
    )
    if len(source_results) == 1:
        single_result = source_results[0]
        return {
            "vectorstore": vectorstore,
            "source": single_result["source"],
            "source_file": single_result["source_file"],
            "paper_sha256": single_result["paper_sha256"],
            "collection_name": collection_name,
            "artifacts_root": str(artifacts_root),
            "documents_indexed": single_result["documents_indexed"],
            "embedding_model": embedding_model_name,
            "vision_model": vision_model,
            "equation_ocr_lib": effective_equation_ocr_lib,
            "skipped_existing_paper": single_result["skipped_existing_paper"],
        }

    return {
        "vectorstore": vectorstore,
        "source": source_results[0]["source"],
        "source_file": source_results[0]["source_file"],
        "paper_sha256": source_results[0]["paper_sha256"],
        "sources": [result["source"] for result in source_results],
        "source_files": [result["source_file"] for result in source_results],
        "paper_sha256_list": [result["paper_sha256"] for result in source_results],
        "source_results": source_results,
        "collection_name": collection_name,
        "artifacts_root": str(artifacts_root),
        "documents_indexed": total_documents_indexed,
        "embedding_model": embedding_model_name,
        "vision_model": vision_model,
        "equation_ocr_lib": effective_equation_ocr_lib,
        "skipped_existing_paper": all(
            result["skipped_existing_paper"] for result in source_results
        ),
    }
