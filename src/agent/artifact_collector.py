"""Collect citable artifacts (documents, websites, videos, images) from RAG tool calls.

The agent already fetched every resource it could cite; this turns the raw tool
results streamed out of ``stream_rag_events`` into rows for
``chat_message_artifacts``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.chunk import Chunk, documents_chunks
from src.db.models.document import Document

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


async def resolve_document_artifacts(
    session: AsyncSession, artifacts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Add ``document_id`` / ``chunk_uuid`` / ``document_name`` to document artifacts.

    Citations carry ``doc_id`` (``documents.sha256``) and a 1-based ``chunk_id``;
    the UI addresses chunks by ``documents.id`` + ``chunks.id``. Resolving here
    keeps that join out of the client. Artifacts whose chunk no longer exists are
    dropped — a citation the preview cannot open is worse than none.
    """
    wanted = {
        (meta["doc_id"], meta["chunk_id"] - 1)
        for artifact in artifacts
        if artifact["artifact_type"] == "document"
        for meta in [artifact["artifact_metadata"]]
        if meta.get("doc_id") and isinstance(meta.get("chunk_id"), int)
    }
    if not wanted:
        return artifacts

    rows = (
        await session.execute(
            select(
                Document.sha256,
                Document.id,
                Document.name,
                documents_chunks.c.order,
                Chunk.id,
            )
            .select_from(documents_chunks)
            .join(Chunk, documents_chunks.c.chunks_id == Chunk.id)
            .join(Document, documents_chunks.c.document_id == Document.id)
            .where(tuple_(Document.sha256, documents_chunks.c.order).in_(wanted))
        )
    ).all()
    resolved = {
        (sha256, order): (document_id, name, chunk_uuid)
        for sha256, document_id, name, order, chunk_uuid in rows
    }

    kept: list[dict[str, Any]] = []
    for artifact in artifacts:
        meta = artifact["artifact_metadata"]
        if artifact["artifact_type"] != "document":
            kept.append(artifact)
            continue
        match = resolved.get((meta.get("doc_id"), (meta.get("chunk_id") or 0) - 1))
        if match is None:
            continue
        document_id, name, chunk_uuid = match
        meta["document_id"] = document_id
        meta["document_name"] = name
        meta["chunk_uuid"] = chunk_uuid
        kept.append(artifact)
    return kept


__all__ = ["collect_artifacts", "resolve_document_artifacts"]
