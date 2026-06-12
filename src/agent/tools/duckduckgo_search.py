"""
Search API gets top 5 URLs
 ↓
Fetch pages using requests + markdownify, stripping nav/header/footer
 ↓
Split into chunks
 ↓
BM25 keeps top 20 chunks
 ↓
Return ranked excerpts with source URLs for the agent to cite
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

from langchain_community.tools import DuckDuckGoSearchResults

from src.utils.fetch_web_content import fetch_page_markdown
from src.utils.textsplitters import chunk_text

_MAX_SEARCH_RESULTS = 5
_TOP_CHUNKS = 10
_FETCH_TIMEOUT = 10.0
_MIN_CHUNK_CHARS = 150
_MAX_PAGE_CHARS = 120_000

_ddg_search = DuckDuckGoSearchResults(
    num_results=_MAX_SEARCH_RESULTS,
    output_format="list",
)


@dataclass(frozen=True, slots=True)
class _RankedChunk:
    text: str
    url: str
    title: str
    bm25_score: float
    chunk_index: int


def _tokenize(text: str) -> list[str]:
    return [token for token in text.split(" ") if token]


def _result_url(result: dict) -> str:
    """DuckDuckGoSearchResults uses `link`; search_ddg uses `url`."""
    return (result.get("link") or result.get("url") or "").strip()


def _normalize_search_results(raw: object) -> list[dict]:
    if isinstance(raw, tuple):
        raw = raw[0]
    if not isinstance(raw, list):
        return []

    normalized: list[dict] = []
    for result in raw:
        if not isinstance(result, dict):
            continue
        url = _result_url(result)
        if not url:
            continue
        normalized.append(
            {
                **result,
                "url": url,
                "title": (result.get("title") or url).strip(),
                "snippet": (result.get("snippet") or "").strip(),
            }
        )
    return normalized


def _search_urls(query: str, max_results: int = _MAX_SEARCH_RESULTS) -> list[dict]:
    if max_results != _ddg_search.max_results:
        searcher = DuckDuckGoSearchResults(
            num_results=max_results,
            output_format="list",
        )
    else:
        searcher = _ddg_search
    return _normalize_search_results(searcher.invoke(query))


def _fetch_chunks_for_result(result: dict) -> list[_RankedChunk]:
    url = _result_url(result)
    title = result.get("title", url)
    if not url:
        return []

    try:
        markdown = fetch_page_markdown(url, timeout=_FETCH_TIMEOUT)
    except Exception:
        snippet = result.get("snippet", "").strip()
        if not snippet:
            return []
        return [
            _RankedChunk(
                text=snippet,
                url=url,
                title=title,
                bm25_score=0.0,
                chunk_index=0,
            )
        ]

    if len(markdown) > _MAX_PAGE_CHARS:
        markdown = markdown[:_MAX_PAGE_CHARS]

    chunks = [
        chunk.strip()
        for chunk in chunk_text(markdown)
        if len(chunk.strip()) >= _MIN_CHUNK_CHARS
    ]
    if not chunks:
        snippet = result.get("snippet", "").strip()
        if snippet:
            chunks = [snippet]
        else:
            return []

    return [
        _RankedChunk(
            text=chunk,
            url=url,
            title=title,
            bm25_score=0.0,
            chunk_index=idx,
        )
        for idx, chunk in enumerate(chunks)
    ]


def _rank_chunks(
    query: str, chunks: list[_RankedChunk], top_n: int
) -> list[_RankedChunk]:
    if not chunks:
        return []

    if len(chunks) <= top_n:
        return chunks

    corpus = [chunk.text for chunk in chunks]
    tokenized_corpus = [_tokenize(text) for text in corpus]
    if not any(tokenized_corpus):
        return chunks[:top_n]

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )[:top_n]

    return [
        _RankedChunk(
            text=chunks[idx].text,
            url=chunks[idx].url,
            title=chunks[idx].title,
            bm25_score=float(scores[idx]),
            chunk_index=chunks[idx].chunk_index,
        )
        for idx in ranked_indices
        if scores[idx] > 0
    ] or [
        _RankedChunk(
            text=chunks[idx].text,
            url=chunks[idx].url,
            title=chunks[idx].title,
            bm25_score=float(scores[idx]),
            chunk_index=chunks[idx].chunk_index,
        )
        for idx in ranked_indices
    ]


def _format_results(
    query: str,
    search_results: list[dict],
    ranked_chunks: list[_RankedChunk],
) -> str:
    if not search_results and not ranked_chunks:
        return f"No web results found for query: {query}"

    lines = [
        f'Web search results for "{query}"',
        "",
        "Search hits:",
    ]

    for result in search_results:
        title = result.get("title", "Untitled")
        url = _result_url(result)
        lines.append(f"- [{title}]({url})")
        snippet = result.get("snippet", "").strip()
        if snippet:
            lines.append(f"  snippet: {snippet}")

    if not ranked_chunks:
        lines.append("")
        lines.append(
            "No page content could be fetched. Use the search-hit URLs/snippets above."
        )
        return "\n".join(lines)

    lines.extend(["", f"Top {len(ranked_chunks)} BM25-ranked excerpts:"])
    for idx, chunk in enumerate(ranked_chunks, start=1):
        lines.append(
            "\n".join(
                [
                    f"[Web {idx}] [{chunk.title}]({chunk.url}), "
                    f"bm25_score={chunk.bm25_score:.4f}, "
                    f"chunk={chunk.chunk_index}",
                    chunk.text,
                ]
            )
        )

    return "\n\n".join(lines)


@tool
def duckduckgo_search(query: str) -> str:
    """Search the public web for a query.

    Fetches the top search results, extracts page content, ranks the most
    relevant passages with BM25, and returns excerpts with source URLs.
    Always cite sources using markdown links: `[title](url)`.
    """
    search_results = _search_urls(query)
    all_chunks: list[_RankedChunk] = []
    for result in search_results:
        all_chunks.extend(_fetch_chunks_for_result(result))

    ranked_chunks = _rank_chunks(query, all_chunks, _TOP_CHUNKS)
    return _format_results(query, search_results, ranked_chunks)
