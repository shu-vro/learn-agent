"""Collect citable artifacts (documents, websites, videos, images) from RAG tool calls.

The agent already fetched every resource it could cite; this turns the raw tool
results streamed out of ``stream_rag_events`` into rows for
``chat_message_artifacts``.
"""

from __future__ import annotations

import json
import re
from typing import Any

# `previous_reference_id=` in next_chunk headers must not match.
_DOC_REF = re.compile(
    r"(?<![\w])reference_id=([^\s:,]+):(\d+)"
    r"(?:,\s*type=(?P<type>[^\s,]+))?"
    r"(?:,\s*similarity_score=(?P<score>[^\s,]+))?"
    r"(?:,\s*page=(?P<page>[^\s,]+))?"
)
_MD_LINK = re.compile(r"!?\[([^\]\n]*)\]\((https?://[^\s)]+)\)")
_BARE_URL = re.compile(r"https?://[^\s'\"<>)\]]+")
# duckduckgo_search excerpts embed arbitrary page links — only take its own hits.
_WEB_HIT = re.compile(
    r"^(?:- |\[Web \d+\] )\[([^\]\n]*)\]\((https?://[^\s)]+)\)", re.MULTILINE
)

_DOC_TOOLS = ("retrieve_context", "next_chunk")
_TOOL_ARTIFACT_TYPE = {
    "duckduckgo_search": "website",
    "fetch_url": "website",
    "duckduckgo_image_search": "image",
    "youtube_search": "video",
}


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _links_for(
    tool_name: str, text: str, args: dict[str, Any]
) -> list[tuple[str, str]]:
    """(title, url) pairs a tool result offers, in first-seen order."""
    if tool_name == "fetch_url":
        url = str(args.get("url") or "").strip()
        return [("", url)] if url.startswith("http") else []
    if tool_name == "duckduckgo_search":
        return [(title, url) for title, url in _WEB_HIT.findall(text)]
    if tool_name == "duckduckgo_image_search":
        return [(title, url) for title, url in _MD_LINK.findall(text)]
    return [("", url.rstrip(".,")) for url in _BARE_URL.findall(text)]


def collect_artifacts(
    tools: list[dict[str, Any]], answer: str = ""
) -> list[dict[str, Any]]:
    """Build ``Artifacts`` kwargs from completed tool calls, deduped by url.

    ``artifact_url`` is the citation target: a real URL, or
    ``reference_id=<doc_id>:<chunk_id>`` for document chunks — the same string
    the agent is told to cite, so the UI can match a citation to its artifact.
    """
    seen: dict[str, dict[str, Any]] = {}

    def add(url: str, artifact_type: str, metadata: dict[str, Any]) -> None:
        if not url or url in seen:
            return
        seen[url] = {
            "artifact_type": artifact_type,
            "artifact_url": url,
            "artifact_metadata": {
                k: v for k, v in metadata.items() if v not in (None, "")
            },
        }

    for tool in tools:
        name = str(tool.get("name") or "")
        text = _as_text(tool.get("result"))
        args = tool.get("args") if isinstance(tool.get("args"), dict) else {}

        if name in _DOC_TOOLS:
            for match in _DOC_REF.finditer(text):
                doc_id, chunk_id = match.group(1), match.group(2)
                add(
                    f"reference_id={doc_id}:{chunk_id}",
                    "document",
                    {
                        "doc_id": doc_id,
                        "chunk_id": int(chunk_id),
                        "chunk_type": match.group("type"),
                        "score": match.group("score"),
                        "page": match.group("page"),
                        "tool": name,
                    },
                )
            continue

        artifact_type = _TOOL_ARTIFACT_TYPE.get(name)
        if not artifact_type:
            continue
        for title, url in _links_for(name, text, args):
            add(
                url,
                artifact_type,
                {"title": title, "tool": name, "query": args.get("query")},
            )

    for url, artifact in seen.items():
        artifact["artifact_metadata"]["cited"] = url in answer
    return list(seen.values())


__all__ = ["collect_artifacts"]
