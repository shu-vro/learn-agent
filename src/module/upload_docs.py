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
from src.utils.time_utils import measure_time
from src.config.env import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PAPER_SOURCE,
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


@measure_time
def ingest_paper_to_qdrant(
    source: str = DEFAULT_PAPER_SOURCE,
    collection_name: str = DEFAULT_QDRANT_COLLECTION,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_DIR,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    vision_model_name: str = DEFAULT_VISION_MODEL,
    use_vision_model: bool = True,
    auto_pull_models: bool = True,
    recreate_collection: bool = False,
    paper_fingerprint: PaperFingerprint | None = None,
) -> dict[str, Any]:
    resolved_paper = paper_fingerprint or fingerprint_paper_source(source)
    paper_sha256 = resolved_paper.sha256

    if not recreate_collection and qdrant_paper_hash_exists(
        collection_name=collection_name,
        paper_sha256=paper_sha256,
    ):
        print(
            f"Paper already indexed. paper_sha256={paper_sha256}. Skipping ingestion."
        )
        loaded_vectorstore = load_qdrant_index(
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
        )
        return {
            "vectorstore": loaded_vectorstore,
            "source": source,
            "source_file": str(resolved_paper.local_path),
            "paper_sha256": paper_sha256,
            "collection_name": collection_name,
            "artifacts_root": str(artifacts_root),
            "documents_indexed": 0,
            "embedding_model": embedding_model_name,
            "vision_model": vision_model_name if use_vision_model else "disabled",
            "skipped_existing_paper": True,
        }

    print(f"New paper hash detected: {paper_sha256}")

    image_describer = None
    formula_transcriber = None
    if use_vision_model:
        vision_client = OllamaVisionClient(
            model=vision_model_name,
            auto_pull=auto_pull_models,
        )
        image_describer = vision_client.describe_image
        formula_transcriber = vision_client.transcribe_formula_latex

    print(f"{use_vision_model=}, {auto_pull_models=}")

    documents = docling_pdf_extractor(
        file_path=str(resolved_paper.local_path),
        artifacts_root=artifacts_root,
        image_describer=image_describer,
        formula_transcriber=formula_transcriber,
    )
    _tag_documents_with_paper_hash(documents, source, paper_sha256)

    vectorstore = create_qdrant_index(
        documents=documents,
        embedding_model_name=embedding_model_name,
        collection_name=collection_name,
        recreate=recreate_collection,
    )

    return {
        "vectorstore": vectorstore,
        "source": source,
        "source_file": str(resolved_paper.local_path),
        "paper_sha256": paper_sha256,
        "collection_name": collection_name,
        "artifacts_root": str(artifacts_root),
        "documents_indexed": len(documents),
        "embedding_model": embedding_model_name,
        "vision_model": vision_model_name if use_vision_model else "disabled",
        "skipped_existing_paper": False,
    }
