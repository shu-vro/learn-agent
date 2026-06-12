from langchain_core.tools import tool

from src.utils.fetch_web_content import fetch_page_markdown


@tool
def fetch_url(url: str) -> str:
    """Fetch text content from a URL"""
    return fetch_page_markdown(url)


__all__ = ["fetch_url"]
