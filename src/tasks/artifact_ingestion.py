from __future__ import annotations

from pathlib import Path
from typing import List

from celery.signals import task_failure
from loguru import logger
from sqlalchemy import insert
from langchain_core.documents import Document as LangchainDocument

from src.config.constants import DEFAULT_OCR_LIB, DEFAULT_QDRANT_COLLECTION
from src.db import sync_session_factory
from src.db.models.chunk import (
    Chunk,
    documents_chunks,
    DEFAULT_CHUNK_TYPE,
)
from src.db.models.document import Document
from src.lib.celery_lib import celery_app
from src.utils.ingestion_progress import (
    clear_ingestion_progress,
    set_ingestion_progress,
)
from src.module.upload_docs import ingest_uploaded_pdf_to_qdrant
from src.db.models.project import Project
from sqlalchemy import select


def _resolve_chunk_type(
    metadata: dict | None,
    *,
    fallback: str = DEFAULT_CHUNK_TYPE,
) -> str:
    """Parses chunk type from extra."""
    if metadata:
        raw = metadata.get("type")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return fallback


def _chunk_extra_with_type(
    metadata: dict | None,
    *,
    chunk_type: str | None = None,
) -> dict:
    """Adds chunk type to extra."""
    extra = dict(metadata or {})
    resolved = chunk_type or _resolve_chunk_type(extra)
    extra["type"] = resolved
    return extra


def _try_update_project_metadata_from_document(
    project_id: str,
    document_id: str,
    *,
    chunks: list[LangchainDocument] | None = None,
) -> None:
    """Attempt to generate and update project name/description if empty.

    If chunks are provided (from fresh extraction), use them directly.
    Otherwise, fetches the first 3 chunks from the DB. Updates the project
    if both name and description are empty. Silently fails (logs exception) if
    generation fails.
    """
    session = sync_session_factory()()
    try:
        project = session.get(Project, project_id)
        if project is None:
            return

        name_empty = (not project.name) or (str(project.name).strip() == "")
        desc_empty = (not project.description) or (
            str(project.description).strip() == ""
        )
        if not (name_empty and desc_empty):
            return

        # Use provided chunks or fetch from database
        if chunks is None:
            chunk_stmt = (
                select(Chunk)
                .join(documents_chunks, documents_chunks.c.chunks_id == Chunk.id)
                .where(documents_chunks.c.document_id == document_id)
                .order_by(documents_chunks.c.order.asc())
                .limit(3)
            )
            chunk_result = session.execute(chunk_stmt)
            db_chunks = list(chunk_result.scalars())
            if not db_chunks:
                logger.debug(
                    "No chunks found for document %s, skipping metadata generation",
                    document_id,
                )
                return
            # Convert DB Chunk objects to document-like format
            chunks = [
                LangchainDocument(page_content=ch.content, metadata=ch.extra or {})
                for ch in db_chunks
            ]

        if not chunks:
            logger.debug(
                "No chunks provided for document %s, skipping metadata generation",
                document_id,
            )
            return

        logger.info(
            "Project %s has empty name and description, generating from chunks",
            project_id,
        )

        try:
            from src.side_agents.update_project_name_agent import (
                generate_project_name_and_description,
            )

            result = generate_project_name_and_description(chunks)
            project.name = result.get("name", project.name)
            project.description = result.get("description", project.description)
            session.commit()
            logger.info(
                "Updated project metadata from document: project_id=%s document_id=%s",
                project_id,
                document_id,
            )
        except Exception as gen_err:
            logger.exception(
                "Failed to generate project name/description for project_id=%s: %s",
                project_id,
                gen_err,
            )
    except Exception:
        logger.exception(
            "Error while attempting to update project metadata for project_id=%s",
            project_id,
        )
    finally:
        session.close()


def _mark_document_failed(document_id: str, error: str) -> None:
    session = sync_session_factory()()
    try:
        document = session.get(Document, document_id)
        if document is None or document.ingestion_status != "processing":
            return
        document.ingestion_status = "failed"
        document.ingestion_error = error[:1000]
        session.commit()
        clear_ingestion_progress(document_id)
    finally:
        session.close()


@task_failure.connect
def _on_artifact_ingestion_failure(
    sender=None,
    task_id=None,
    exception=None,
    kwargs=None,
    **kw,
) -> None:
    if sender is None or sender.name != "artifact.process_upload":
        return
    document_id = (kwargs or {}).get("document_id")
    if not document_id:
        return
    message = str(exception) if exception is not None else "Ingestion task failed"
    logger.error(
        "Artifact ingestion task failed: document_id={} task_id={} error={}",
        document_id,
        task_id,
        message,
    )
    _mark_document_failed(document_id, message)


@celery_app.task(name="artifact.process_upload", bind=True, max_retries=0)
def process_artifact_upload(
    self,
    *,
    document_id: str,
    file_path: str,
    project_id: str,
    original_url: str | None,
    collection_name: str = DEFAULT_QDRANT_COLLECTION,
    equation_ocr_lib: str = DEFAULT_OCR_LIB,
    use_vision_model: bool = True,
    use_image_descriptions: bool = True,
    use_formula_transcription: bool = True,
    recreate_collection: bool = False,
) -> dict[str, str]:
    logger.info(
        "Artifact ingestion started: document_id={} project_id={} file={}",
        document_id,
        project_id,
        file_path,
    )
    session = sync_session_factory()()
    try:
        set_ingestion_progress(
            document_id,
            stage="extracting",
            label="Parsing PDF and extracting content",
            progress=35,
        )
        extraction_result = ingest_uploaded_pdf_to_qdrant(
            file_path=Path(file_path),
            document_id=document_id,
            project_id=project_id,
            original_url=original_url,
            collection_name=collection_name,
            equation_ocr_lib=equation_ocr_lib,
            use_vision_model=use_vision_model,
            use_image_descriptions=use_image_descriptions,
            use_formula_transcription=use_formula_transcription,
            recreate_collection=recreate_collection,
        )
        extracted_documents: List[LangchainDocument] = extraction_result["documents"]
        document = session.get(Document, document_id)
        if document is None:
            raise ValueError(f"Document {document_id} not found")

        set_ingestion_progress(
            document_id,
            stage="indexing",
            label="Indexing content in vector store",
            progress=70,
        )

        for order, extracted in enumerate(extracted_documents):
            metadata = dict(extracted.metadata or {})
            chunk_type = _resolve_chunk_type(metadata)
            chunk = Chunk(
                content=extracted.page_content,
                type=chunk_type,
                extra=_chunk_extra_with_type(metadata, chunk_type=chunk_type),
            )
            session.add(chunk)
            session.flush()
            session.execute(
                insert(documents_chunks).values(
                    document_id=document.id,
                    chunks_id=chunk.id,
                    order=order,
                )
            )
        set_ingestion_progress(
            document_id,
            stage="saving",
            label="Saving extracted content",
            progress=90,
        )
        document.ingestion_status = "completed"
        document.ingestion_error = None
        session.commit()
        clear_ingestion_progress(document_id)

        # Try to update project metadata from the freshly extracted chunks
        _try_update_project_metadata_from_document(
            project_id, document_id, chunks=extracted_documents[:3]
        )
        logger.info(
            "Artifact ingestion completed: document_id={} chunks={}",
            document_id,
            len(extracted_documents),
        )
        return {"document_id": document_id, "status": "completed"}
    except Exception as err:
        session.rollback()
        logger.exception(
            "Artifact ingestion failed: document_id={} error={}",
            document_id,
            err,
        )
        document = session.get(Document, document_id)
        if document is not None:
            document.ingestion_status = "failed"
            document.ingestion_error = str(err)[:1000]
            session.commit()
        clear_ingestion_progress(document_id)
        raise err
    finally:
        session.close()


__all__ = ["process_artifact_upload"]
