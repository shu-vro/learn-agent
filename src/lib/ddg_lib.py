"""deprecated, use DuckDuckGoSearchResults from langchain_community.tools instead"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode, urljoin, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup


DDG_URL = "https://html.duckduckgo.com/html/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xhtml;q=0.9,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://duckduckgo.com/",
    "Content-Type": "application/x-www-form-urlencoded",
}


@dataclass
class SearchResult:
    rank: int
    title: str
    url: str
    displayUrl: str
    snippet: str


def search_ddg(
    query: str,
    max_results: int = 10,
    region: str = "wt-wt",
    timeout: int = 15,
) -> dict:
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    body = urlencode(
        {
            "q": query,
            "b": "",
            "kl": region,
        }
    )

    response = requests.post(
        DDG_URL,
        headers=HEADERS,
        data=body,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"DDG returned HTTP {response.status_code}: {response.reason}"
        )

    return parse_ddg_html(response.text, query, max_results)


def parse_ddg_html(html: str, query: str, max_results: int) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    results: list[SearchResult] = []

    for el in soup.select(".result:not(.result--ad)"):
        if len(results) >= max_results:
            break

        title_el = el.select_one(".result__title a")
        snippet_el = el.select_one(".result__snippet")
        url_el = el.select_one(".result__url")

        title = title_el.get_text(strip=True) if title_el else ""
        raw_href = title_el.get("href", "") if title_el else ""
        url = resolve_url(raw_href)
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        display_url = url_el.get_text(" ", strip=True) if url_el else ""

        if not title or not url:
            continue

        results.append(
            SearchResult(
                rank=len(results) + 1,
                title=title,
                url=url,
                displayUrl=display_url,
                snippet=snippet,
            )
        )

    related = []

    for el in soup.select(
        ".related-searches__item a, .badge--ad + .result .result__title a"
    ):
        text = el.get_text(strip=True)
        if text:
            related.append(text)

    did_you_mean_el = soup.select_one(".msg--suggestion a")
    did_you_mean: Optional[str] = (
        did_you_mean_el.get_text(strip=True) if did_you_mean_el else None
    )

    return {
        "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "totalFound": len(results),
        "results": [asdict(r) for r in results],
        "related": list(dict.fromkeys(related))[:8],
        "didYouMean": did_you_mean,
    }


def resolve_url(href: str) -> str:
    if not href:
        return ""

    try:
        if href.startswith("/l/"):
            full_url = urljoin("https://duckduckgo.com", href)
            parsed = urlparse(full_url)
            params = parse_qs(parsed.query)

            wrapped = (
                params.get("uddg", [None])[0] or params.get("kh", [None])[0] or href
            )

            return unquote(wrapped)
    except Exception:
        pass

    if href.startswith("http"):
        return href

    return urljoin("https://duckduckgo.com", href)


def pretty_print(data: dict) -> None:
    line = lambda char, n=60: char * n

    print("\n" + line("═"))
    print("  DDG Scraper Results")
    print(f'  Query    : "{data["query"]}"')
    print(f"  Time     : {data['timestamp']}")
    print(f"  Found    : {data['totalFound']} organic results")

    if data.get("didYouMean"):
        print(f'  Did you mean: "{data["didYouMean"]}"')

    print(line("═") + "\n")

    for r in data["results"]:
        print(f"[{r['rank']}] {r['title']}")
        print(f"    URL: {r['url']}")

        if r.get("snippet"):
            print(f"    Snippet: {r['snippet']}")

        print()

    if data.get("related"):
        print(line("─"))
        print("  Related searches:")
        for s in data["related"]:
            print(f"    • {s}")
        print(line("─"))
