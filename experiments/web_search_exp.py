from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

import src.config.bootstrap  # noqa: F401
from src.agent.tools.duckduckgo_search import duckduckgo_search


data = duckduckgo_search.invoke("LangChain web search retriever")

print(data)
