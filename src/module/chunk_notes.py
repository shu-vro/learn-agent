"""Helpers for on-demand chunk note generation and Qdrant persistence.

Notes are generated per-chunk when a user explicitly requests one from the UI.
Each note is stored as a ``NOTE_CHUNK_TYPE`` row in Postgres (for display) and
as a document in Qdrant (for retrieval). Qdrant documents are intentionally kept
out of the main ``/artifacts`` response.
"""

from __future__ import annotations

from langchain_core.documents import Document as LangchainDocument
from qdrant_client.http.models import FieldCondition, Filter, FilterSelector, MatchValue

from src.config.constants import DEFAULT_QDRANT_COLLECTION
from src.db.models.chunk import NOTE_CHUNK_TYPE
from src.vector_store.qdrant_store import build_hybrid_qdrant_store, client


def upsert_chunk_note_in_qdrant(
    *,
    document_id: str,
    source_chunk_id: str,
    note_content: str,
    doc_sha256: str | None = None,
    collection_name: str = DEFAULT_QDRANT_COLLECTION,
) -> None:
    """Replace any existing Qdrant note for the source chunk, then add the new one.

    Notes are scoped to the chunk/document only (documents are globally
    deduplicated), so the identity is ``document_id + source_chunk_id`` with no
    project/user dimension. This is a blocking call (network + embeddings); run
    it in a thread when invoked from async code.
    """
    delete_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.type",
                match=MatchValue(value=NOTE_CHUNK_TYPE),
            ),
            FieldCondition(
                key="metadata.document_id",
                match=MatchValue(value=document_id),
            ),
            FieldCondition(
                key="metadata.source_chunk_id",
                match=MatchValue(value=source_chunk_id),
            ),
        ]
    )
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=delete_filter),
        )
    except Exception:
        # Best-effort cleanup: a missing prior note should not block the new one.
        pass

    vectorstore = build_hybrid_qdrant_store(collection_name=collection_name)
    vectorstore.add_documents(
        [
            LangchainDocument(
                page_content=note_content,
                metadata={
                    "type": NOTE_CHUNK_TYPE,
                    "source_chunk_id": source_chunk_id,
                    "document_id": document_id,
                    "doc_id": doc_sha256,
                },
            )
        ]
    )


__all__ = ["upsert_chunk_note_in_qdrant"]
