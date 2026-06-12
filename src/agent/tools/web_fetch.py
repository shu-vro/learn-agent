import requests
from langchain_core.tools import tool
from markdownify import markdownify


@tool
def fetch_url(url: str) -> str:
    """Fetch text content from a URL"""
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return markdownify(response.text)


__all__ = ["fetch_url"]
