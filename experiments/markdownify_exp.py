from pathlib import Path
import sys
import requests
import markdownify

# Ensure top-level `src` imports resolve when this file is executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config.bootstrap  # noqa: F401

url = "https://shirshen.is-a.dev"

response = requests.get(url)
html = response.text
md = markdownify.markdownify(html)
print(md)
