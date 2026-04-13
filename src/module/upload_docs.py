from pathlib import Path
from typing import Any

from src.lib.docling_lib import docling_pdf_extractor
from src.lib.faiss_store import create_faiss_index
from src.lib.ollama_vision import OllamaVisionClient

DEFAULT_PAPER_SOURCE = "https://arxiv.org/pdf/1706.03762"
DEFAULT_INDEX_DIR = Path("data/faiss_db")
DEFAULT_ARTIFACTS_DIR = Path("data/artifacts")
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_VISION_MODEL = "moondream"


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
    if use_vision_model:
        vision_client = OllamaVisionClient(
            model=vision_model_name,
            auto_pull=auto_pull_models,
        )
        image_describer = vision_client.describe_image

    documents = docling_pdf_extractor(
        file_path=source,
        artifacts_root=artifacts_root,
        image_describer=image_describer,
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
