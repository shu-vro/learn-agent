from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any, Dict
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.constants import DEFAULT_QDRANT_COLLECTION
from src.config.env import ASSET_UPLOAD_ROOT
from src.db import get_session
from src.db.models.chunk import Chunk, documents_chunks
from src.db.models.document import Document
from src.db.models.project import Project
from src.db.models.project_document import ProjectDocument
from src.module.upload_docs import ingest_uploaded_pdf_to_qdrant
from src.utils.api.BaseResponse import BaseResponse
from src.utils.api.artifact_markdown_fixer import rewrite_chunk_image_urls

ALLOWED_SUFFIXES = frozenset[str]({".pdf"})
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

router = APIRouter(prefix="/projects", tags=["artifacts"])


class ArtifactRead(BaseModel):
    id: str
    name: str
    chunks: Dict[str, Any] = {}
    ingestion_status: str = "completed"


ArtifactsListResponse = BaseResponse[list[ArtifactRead]]
ArtifactResponse = BaseResponse[ArtifactRead]


def _suffix_for_upload(filename: str) -> str:
    return Path(filename).suffix.lower()


def _sanitize_stored_filename(filename: str) -> str:
    base = Path(filename).name
    if not base or base in {".", ".."}:
        return "file"
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return cleaned[:200] or "file"


async def _get_document_chunks(
    document_id: str,
    session: AsyncSession,
) -> list[Chunk]:
    stmt = (
        select(Chunk)
        .join(documents_chunks, documents_chunks.c.chunks_id == Chunk.id)
        .where(
            documents_chunks.c.document_id == document_id,
            Chunk.extra["type"].as_string() == "text_chunk",
        )
        .order_by(documents_chunks.c.order.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


def _chunks_to_dict(chunks: list[Chunk], *, doc_sha256: str | None) -> Dict[str, Any]:
    return {
        chunk.id: rewrite_chunk_image_urls(chunk.content, doc_sha256)
        for chunk in chunks
    }


def _require_user(request: Request):
    """Equivalent to request.state.user but safer."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


async def _ensure_project_document_link(
    session: AsyncSession,
    *,
    project_id: str,
    document_id: str,
) -> None:
    existing_link_stmt = select(ProjectDocument.id).where(
        ProjectDocument.project_id == project_id,
        ProjectDocument.document_id == document_id,
    )
    existing_link = await session.execute(existing_link_stmt)
    if existing_link.scalar_one_or_none() is None:
        session.add(ProjectDocument(project_id=project_id, document_id=document_id))


@router.get("/{project_id}/artifacts", response_model=ArtifactsListResponse)
async def list_project_artifacts(
    request: Request,
    project_id: str,
    session: AsyncSession = Depends(get_session),
) -> ArtifactsListResponse:
    user = _require_user(request)
    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = (
        select(Document)
        .join(ProjectDocument, ProjectDocument.document_id == Document.id)
        .join(Project, Project.id == ProjectDocument.project_id)
        .where(ProjectDocument.project_id == project_id, Project.user_id == user.id)
        .order_by(Document.name.asc())
    )
    result = await session.execute(stmt)
    docs = result.scalars().all()
    items: list[ArtifactRead] = []
    for doc in docs:
        doc_chunks = await _get_document_chunks(doc.id, session)
        items.append(
            ArtifactRead(
                id=doc.id,
                name=doc.name,
                chunks=_chunks_to_dict(doc_chunks, doc_sha256=doc.sha256),
                ingestion_status=doc.ingestion_status,
            )
        )
    return ArtifactsListResponse.ok(data=items)


@router.post("/{project_id}/artifacts", response_model=ArtifactResponse)
async def upload_project_artifact(
    request: Request,
    project_id: str,
    session: AsyncSession = Depends(get_session),
    file: UploadFile = File(...),
) -> ArtifactResponse:
    user = _require_user(request)
    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    raw_name = file.filename or "upload"
    suffix = _suffix_for_upload(raw_name)
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Only .pdf files are allowed.",
        )

    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    doc_id = str(uuid.uuid4())
    safe = _sanitize_stored_filename(Path(raw_name).name)
    rel_path = f"{project_id}/{doc_id}_{safe}"
    root = ASSET_UPLOAD_ROOT
    dest = root / rel_path
    await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(dest.write_bytes, body)
    source_file_url = dest.resolve().as_uri()
    upload_sha256 = hashlib.sha256(body).hexdigest()

    should_process = False
    existing_stmt = select(Document).where(Document.sha256 == upload_sha256)
    existing_result = await session.execute(existing_stmt)
    document = existing_result.scalar_one_or_none()

    if document is None:
        document = Document(
            id=doc_id,
            source="uploaded",
            url=rel_path,
            original_url=source_file_url,
            name=Path(raw_name).name,
            sha256=upload_sha256,
            mime_type=file.content_type,
            file_size=len(body),
            ingestion_status="processing",
        )
        session.add(document)
        await _ensure_project_document_link(
            session,
            project_id=project_id,
            document_id=doc_id,
        )
        try:
            await session.commit()
            should_process = True
        except IntegrityError:
            await session.rollback()
            race_result = await session.execute(existing_stmt)
            document = race_result.scalar_one()
            await _ensure_project_document_link(
                session,
                project_id=project_id,
                document_id=document.id,
            )
            await session.commit()
    else:
        await _ensure_project_document_link(
            session,
            project_id=project_id,
            document_id=document.id,
        )
        await session.commit()

    if not should_process:
        if dest.exists():
            await asyncio.to_thread(dest.unlink)
        if document.ingestion_status == "processing":
            return ArtifactResponse.ok(
                data=ArtifactRead(
                    id=document.id,
                    name=document.name,
                    chunks={},
                    ingestion_status=document.ingestion_status,
                )
            )
        if document.ingestion_status == "completed":
            doc_chunks = await _get_document_chunks(document.id, session)
            return ArtifactResponse.ok(
                data=ArtifactRead(
                    id=document.id,
                    name=document.name,
                    chunks=_chunks_to_dict(doc_chunks, doc_sha256=document.sha256),
                    ingestion_status=document.ingestion_status,
                )
            )
        return ArtifactResponse.ok(
            data=ArtifactRead(
                id=document.id,
                name=document.name,
                chunks={},
                ingestion_status=document.ingestion_status,
            )
        )

    try:
        extraction_result = await asyncio.to_thread(
            ingest_uploaded_pdf_to_qdrant,
            file_path=dest,
            document_id=document.id,
            project_id=project_id,
            original_url=source_file_url,
            collection_name=DEFAULT_QDRANT_COLLECTION,
        )
        extracted_documents = extraction_result["documents"]

        for order, extracted in enumerate(extracted_documents):
            chunk = Chunk(
                content=extracted.page_content,
                extra=dict(extracted.metadata or {}),
            )
            session.add(chunk)
            await session.flush()
            await session.execute(
                insert(documents_chunks).values(
                    document_id=document.id,
                    chunks_id=chunk.id,
                    order=order,
                )
            )
        document.ingestion_status = "completed"
        document.ingestion_error = None
        await session.commit()
        await session.refresh(document)
    except Exception as err:
        await session.rollback()
        document.ingestion_status = "failed"
        document.ingestion_error = str(err)[:1000]
        session.add(document)
        await session.commit()
        raise HTTPException(
            status_code=500,
            detail="Failed to process uploaded PDF.",
        ) from err

    doc_chunks = await _get_document_chunks(document.id, session)
    return ArtifactResponse.ok(
        data=ArtifactRead(
            id=document.id,
            name=document.name,
            chunks=_chunks_to_dict(doc_chunks, doc_sha256=document.sha256),
            ingestion_status=document.ingestion_status,
        )
    )


__all__ = ["router"]
