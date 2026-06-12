from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

import src.config.bootstrap  # noqa: F401

from src.lib.ddg_lib import search_ddg, pretty_print

data = search_ddg("LangChain web search retriever", max_results=5)
pretty_print(data)

data = DuckDuckGoSearchResults(num_results=5, output_format="list").invoke(
    "LangChain web search retriever"
)
print(data)
