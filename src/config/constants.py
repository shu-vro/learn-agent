import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Strict environment configurations
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").lower()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
DEFAULT_PAPER_SOURCES = [
    "https://arxiv.org/pdf/1706.03762",
    "https://arxiv.org/pdf/2603.15031",
]
DEFAULT_ARTIFACTS_DIR = Path("data/artifacts")
DEFAULT_DOWNLOADS_DIR = Path("data/downloads")
DEFAULT_EMBEDDING_MODEL = "Octen/Octen-Embedding-0.6B"
DEFAULT_VISION_MODEL = "ollama:gemma4:e2b"
DEFAULT_LLM_MODEL = "ollama:gemma4:e2b"
DEFAULT_OCR_LIB = os.environ.get("DEFAULT_OCR_LIB", "local").strip().lower()
if DEFAULT_OCR_LIB not in {"local", "llm"}:
    DEFAULT_OCR_LIB = "local"
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
DEFAULT_QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION_NAME", "store")
