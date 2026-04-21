from pathlib import Path
from langchain_mineru import MinerULoader
import sys
import src.config.bootstrap  # noqa: F401

# Ensure top-level `src` imports resolve when this file is executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


loader = MinerULoader(source="https://arxiv.org/pdf/2603.15031", mode="flash")
docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)
