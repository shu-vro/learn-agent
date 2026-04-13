from pathlib import Path
from typing import Any

from src.lib.docling_lib import docling_pdf_extractor
from src.vector_store.faiss_store import create_faiss_index
from src.lib.ollama_vision import OllamaVisionClient
from src.utils.time_utils import measure_time
from src.config.env import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_PAPER_SOURCE,
    DEFAULT_VISION_MODEL,
)


@measure_time
def ingest_paper_to_faiss(
    source: str = DEFAULT_PAPER_SOURCE,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_DIR,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    vision_model_name: str = DEFAULT_VISION_MODEL,
    use_vision_model: bool = True,
    auto_pull_models: bool = True,
) -> dict[str, Any]:
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
        file_path=source,
        artifacts_root=artifacts_root,
        image_describer=image_describer,
        formula_transcriber=formula_transcriber,
    )

    vectorstore = create_faiss_index(
        documents=documents,
        embedding_model_name=embedding_model_name,
        index_dir=index_dir,
    )

    return {
        "vectorstore": vectorstore,
        "source": source,
        "index_dir": str(index_dir),
        "artifacts_root": str(artifacts_root),
        "documents_indexed": len(documents),
        "embedding_model": embedding_model_name,
        "vision_model": vision_model_name if use_vision_model else "disabled",
    }
