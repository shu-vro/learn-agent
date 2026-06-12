import re

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_STRIP_TAGS = {
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "noscript",
    "form",
    "iframe",
    "svg",
}

_MAIN_SELECTORS = ("main", "article", '[role="main"]', "#content", "#main", ".content")


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    main_el = None
    for selector in _MAIN_SELECTORS:
        main_el = soup.select_one(selector)
        if main_el:
            break

    content_root = main_el or soup.body or soup
    markdown = markdownify(str(content_root), heading_style="ATX")
    return _collapse_whitespace(markdown)


def fetch_page_markdown(url: str, timeout: float = 10.0) -> str:
    response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return html_to_markdown(response.text)
