"""Fetch the next sequential document chunk after a retrieve_context hit."""

from __future__ import annotations

from langchain_core.tools import tool
from sqlalchemy import or_, select

from src.db import sync_session_factory
from src.db.models import Chunk, Document, documents_chunks


def _fetch_next_chunk(doc_id: str, chunk_id: int) -> tuple[str, str, int, str] | None:
    """Return (fingerprint, content, next_chunk_id, chunk_type) after ``chunk_id``.

    ``doc_id`` is the sha256 fingerprint from ``reference_id`` (also accepts
    ``documents.id`` UUID). ``chunk_id`` is the 1-based index from
    ``reference_id`` / Qdrant metadata; DB ``order`` is 0-based, so
    ``order == chunk_id`` selects the next chunk.
    """
    query = (
        select(
            Document.sha256,
            Document.id,
            Chunk.content,
            documents_chunks.c.order,
            Chunk.type,
        )
        .select_from(documents_chunks)
        .join(Chunk, documents_chunks.c.chunks_id == Chunk.id)
        .join(Document, documents_chunks.c.document_id == Document.id)
        .where(
            or_(Document.sha256 == doc_id, Document.id == doc_id),
            documents_chunks.c.order == chunk_id,
        )
    )

    with sync_session_factory()() as session:
        row = session.execute(query).one_or_none()

    if row is None:
        return None

    sha256, document_uuid, content, order, chunk_type = row
    fingerprint = sha256 or document_uuid or doc_id
    next_chunk_id = int(order) + 1
    return (
        str(fingerprint),
        content or "",
        next_chunk_id,
        chunk_type or "text_chunk",
    )


@tool
def next_chunk(doc_id: str, chunk_id: int) -> str:
    """Fetch the next chunk after a document hit from retrieve_context.

    When a retrieved passage is cut off mid-thought, parse its
    ``reference_id=<doc_id>:<chunk_id>`` and call this tool with those values
    to read the immediately following chunk in the same document.

    Args:
        doc_id: Document fingerprint from ``reference_id`` (sha256 hex string).
        chunk_id: The integer after the colon in ``reference_id`` (1-based).
            Passing ``4`` from ``…:4`` returns the next chunk (``…:5``).
    """
    if not doc_id or not str(doc_id).strip():
        return "Invalid doc_id: expected the document fingerprint from reference_id."

    try:
        current_chunk_id = int(chunk_id)
    except (TypeError, ValueError):
        return f"Invalid chunk_id: expected an integer, got {chunk_id!r}."

    if current_chunk_id < 1:
        return "Invalid chunk_id: must be >= 1 (from reference_id)."

    try:
        result = _fetch_next_chunk(str(doc_id).strip(), current_chunk_id)
    except Exception as exc:
        return f"Error fetching next chunk for {doc_id}:{current_chunk_id}: {exc}"

    if result is None:
        return (
            f"No next chunk found after reference_id={doc_id}:{current_chunk_id} "
            "(end of document or unknown id)."
        )

    fingerprint, content, next_chunk_id, chunk_type = result
    header = (
        f"[Next chunk] reference_id={fingerprint}:{next_chunk_id}, "
        f"type={chunk_type}, "
        f"previous_reference_id={fingerprint}:{current_chunk_id}"
    )
    return f"{header}\n{content}"


__all__ = ["next_chunk"]
