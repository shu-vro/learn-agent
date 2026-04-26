# from pathlib import Path
# from langchain_mineru import MinerULoader
# import sys
# from markitdown import MarkItDown

# # Ensure top-level `src` imports resolve when this file is executed as a script.
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))


# import src.config.bootstrap  # noqa: F401


# md = MarkItDown()
# result = md.convert("https://arxiv.org/pdf/2603.15031")

# with open("fk", "w") as f:
#     f.write(result.markdown)
