"""DuckDuckGo image search tool for educational / diagram images."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote

import requests
from langchain_core.tools import tool

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_VQD_PATTERNS = (
    re.compile(r'vqd=["\']?([\d-]+)["\']?'),
    re.compile(r'"vqd":"([\d-]+)"'),
    re.compile(r'vqd\\?":\\?"([\d-]+)\\?"'),
)

_PAGE_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_API_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

_DEFAULT_LIMIT = 3
_TIMEOUT = 15.0


def extract_vqd(html: str) -> str | None:
    for pattern in _VQD_PATTERNS:
        match = pattern.search(html)
        if match:
            return match.group(1)
    return None


def duckduckgo_images(
    query: str,
    limit: int = _DEFAULT_LIMIT,
    *,
    timeout: float = _TIMEOUT,
) -> list[dict[str, str]]:
    """Fetch image search results from DuckDuckGo.

    Uses browser-like headers and a session so requests are not blocked (403).
    Appends "educational diagram" to bias results toward teaching visuals.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    safe_limit = max(1, min(int(limit), 20))
    safe_query = f"{query.strip()} educational diagram"
    search_url = f"https://duckduckgo.com/?q={quote(safe_query)}"

    with requests.Session() as session:
        page_res = session.get(
            search_url,
            headers=_PAGE_HEADERS,
            timeout=timeout,
        )
        page_res.raise_for_status()

        vqd = extract_vqd(page_res.text)
        if not vqd:
            raise RuntimeError("Could not get search token")

        api_url = (
            "https://duckduckgo.com/i.js?"
            f"l=us-en&o=json&q={quote(safe_query)}"
            f"&vqd={vqd}&f=,,,&p=1&s=1&type=photo&license=public&size=medium"
        )
        api_headers = {
            **_API_HEADERS,
            "Referer": search_url,
        }
        api_res = session.get(api_url, headers=api_headers, timeout=timeout)
        body = api_res.text

        if not api_res.ok:
            raise RuntimeError(
                f"DuckDuckGo image API HTTP {api_res.status_code}: {body[:120]}"
            )

        try:
            data: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"DuckDuckGo image API returned non-JSON: {body[:120]}"
            ) from exc

    results: list[dict[str, str]] = []
    for img in (data.get("results") or [])[:safe_limit]:
        if not isinstance(img, dict):
            continue
        url = (img.get("image") or "").strip()
        title = (img.get("title") or "").strip() or url
        if not url:
            continue
        results.append({"url": url, "title": title})
    return results


def _format_image_results(query: str, images: list[dict[str, str]]) -> str:
    if not images:
        return f'No image results found for query: "{query}"'

    lines = [
        f'Image search results for "{query}" (biased toward educational diagrams):',
        "",
        "Embed useful images in your answer as markdown: ![title](url)",
        "",
    ]
    for idx, img in enumerate(images, start=1):
        lines.append(f"{idx}. {img['title']}")
        lines.append(f"   url: {img['url']}")
        lines.append(f"   markdown: ![{img['title']}]({img['url']})")
        lines.append("")
    return "\n".join(lines).rstrip()


@tool
def duckduckgo_image_search(query: str, limit: int = _DEFAULT_LIMIT) -> str:
    """Search DuckDuckGo for educational / diagram images.

    Use when a visual explanation would help (diagrams, schematics, labeled
    figures). Returns image titles and direct URLs — embed them in answers as
    markdown images: ![title](url).
    """
    try:
        images = duckduckgo_images(query, limit=limit)
    except Exception as exc:
        print(
            f"duckduckgo_image_search failed for query {query!r}", log_level="WARNING"
        )
        return (
            f'Image search failed for query "{query}": '
            f"{type(exc).__name__}: {exc}. Answer without images."
        )
    return _format_image_results(query, images)


__all__ = ["duckduckgo_image_search", "duckduckgo_images", "extract_vqd"]
