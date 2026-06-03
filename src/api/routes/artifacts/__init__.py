from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any, Dict
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document as LangchainDocument

from src.config.constants import DEFAULT_OCR_LIB, DEFAULT_QDRANT_COLLECTION
from src.db.models.preferences import Preferences
from src.schemas.preferences import UserPreferencesPublic, resolve_ingestion_flags
from src.config.env import ASSET_UPLOAD_ROOT
from src.db import get_session
from src.db.models.chunk import (
    Chunk,
    coalesced_chunk_type,
    DEFAULT_CHUNK_TYPE,
    documents_chunks,
)
from src.db.models.document import Document
from src.db.models.project import Project
from src.db.models.project_document import ProjectDocument
from src.tasks.artifact_ingestion import process_artifact_upload
from src.utils.ingestion_progress import get_ingestion_progress, set_ingestion_progress
from src.utils.api.BaseResponse import BaseResponse
from src.utils.api.artifact_markdown_fixer import rewrite_chunk_image_urls

ALLOWED_SUFFIXES = frozenset[str]({".pdf"})
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

router = APIRouter(prefix="/projects", tags=["artifacts"])


class ArtifactRead(BaseModel):
    id: str
    name: str
    chunks: Dict[str, Any] = {}
    ingestion_status: str = "completed"
    ingestion_stage: str | None = None
    ingestion_stage_label: str | None = None
    ingestion_progress: int | None = None


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
) -> list[tuple[Chunk, int]]:
    """Just fetches chunks from document."""
    stmt = (
        select(Chunk, documents_chunks.c.order)
        .join(documents_chunks, documents_chunks.c.chunks_id == Chunk.id)
        .where(
            documents_chunks.c.document_id == document_id,
            coalesced_chunk_type() == DEFAULT_CHUNK_TYPE,
        )
        .order_by(documents_chunks.c.order.asc())
    )
    result = await session.execute(stmt)
    return list(result.all())


def _chunks_to_dict(
    chunks: list[tuple[Chunk, int]],
    *,
    doc_sha256: str | None,
) -> Dict[str, Any]:
    ordered: Dict[str, Any] = {}
    for chunk, order in chunks:
        ordered[chunk.id] = {
            "content": rewrite_chunk_image_urls(chunk.content, doc_sha256),
            "order": order,
        }
    return ordered


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
        items.append(await _artifact_read_for_document(doc, session))
    return ArtifactsListResponse.ok(data=items)


def _form_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def _artifact_read_for_document(
    document: Document,
    session: AsyncSession,
) -> ArtifactRead:
    chunks: dict[str, Any] = {}
    if document.ingestion_status == "completed":
        doc_chunks = await _get_document_chunks(document.id, session)
        chunks = _chunks_to_dict(doc_chunks, doc_sha256=document.sha256)

    progress = (
        get_ingestion_progress(document.id)
        if document.ingestion_status == "processing"
        else None
    )
    return ArtifactRead(
        id=document.id,
        name=document.name,
        chunks=chunks,
        ingestion_status=document.ingestion_status,
        ingestion_stage=progress.get("stage") if progress else None,
        ingestion_stage_label=progress.get("label") if progress else None,
        ingestion_progress=progress.get("progress") if progress else None,
    )


@router.get(
    "/{project_id}/artifacts/{artifact_id}/ingestion-status",
    response_model=ArtifactResponse,
)
async def get_artifact_ingestion_status(
    request: Request,
    project_id: str,
    artifact_id: str,
    session: AsyncSession = Depends(get_session),
) -> ArtifactResponse:
    user = _require_user(request)
    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = (
        select(Document)
        .join(ProjectDocument, ProjectDocument.document_id == Document.id)
        .where(
            ProjectDocument.project_id == project_id,
            Document.id == artifact_id,
        )
    )
    result = await session.execute(stmt)
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Artifact not found")

    return ArtifactResponse.ok(
        data=await _artifact_read_for_document(document, session)
    )


@router.delete(
    "/{project_id}/artifacts/{artifact_id}", response_model=BaseResponse[None]
)
async def delete_project_artifact(
    request: Request,
    project_id: str,
    artifact_id: str,
    session: AsyncSession = Depends(get_session),
) -> BaseResponse[None]:
    user = _require_user(request)
    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = select(ProjectDocument).where(
        ProjectDocument.project_id == project_id,
        ProjectDocument.document_id == artifact_id,
    )
    result = await session.execute(stmt)
    project_document = result.scalar_one_or_none()
    if not project_document:
        raise HTTPException(status_code=404, detail="Artifact not found")

    await session.delete(project_document)
    await session.commit()
    return BaseResponse[None].ok(data=None)


@router.post("/{project_id}/artifacts", response_model=ArtifactResponse)
async def upload_project_artifact(
    request: Request,
    project_id: str,
    session: AsyncSession = Depends(get_session),
    file: UploadFile = File(...),
    use_vision_model: str | None = Form(None),
    use_image_descriptions: str | None = Form(None),
    use_formula_transcription: str | None = Form(None),
    equation_ocr_lib: str | None = Form(None),
    rebuild: str | None = Form(None),
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
        if document.ingestion_status == "failed":
            document.ingestion_status = "processing"
            document.ingestion_error = None
            document.url = rel_path
            document.original_url = source_file_url
            session.add(document)
            should_process = True
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
            # Try to update project metadata from existing completed document
            from src.tasks.artifact_ingestion import (
                _try_update_project_metadata_from_document,
            )

            doc_chunks = await _get_document_chunks(document.id, session)

            try:
                _try_update_project_metadata_from_document(
                    project_id,
                    document.id,
                    chunks=list(
                        map(
                            lambda c: LangchainDocument(
                                page_content=c[0].content, metadata=c[0].extra or {}
                            ),
                            doc_chunks[:3],
                        )
                    ),
                )
                # Refresh project to get updated name/description
                await session.refresh(project)
            except Exception:
                pass  # Non-critical; metadata update failures don't block reuse
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

    prefs_row = await Preferences.get_or_create(session, user.id)
    base_ingestion = UserPreferencesPublic.from_model(prefs_row).ingestion
    ingestion = resolve_ingestion_flags(
        base_ingestion,
        use_vision_model=_form_bool(use_vision_model),
        use_image_descriptions=_form_bool(use_image_descriptions),
        use_formula_transcription=_form_bool(use_formula_transcription),
        equation_ocr_lib=equation_ocr_lib if equation_ocr_lib else None,
    )
    recreate_collection = _form_bool(rebuild) or False

    try:
        process_artifact_upload.delay(
            document_id=document.id,
            file_path=str(dest),
            project_id=project_id,
            original_url=source_file_url,
            collection_name=DEFAULT_QDRANT_COLLECTION,
            equation_ocr_lib=ingestion.equation_ocr_lib or DEFAULT_OCR_LIB,
            use_vision_model=ingestion.use_vision_model,
            use_image_descriptions=ingestion.use_image_descriptions,
            use_formula_transcription=ingestion.use_formula_transcription,
            recreate_collection=recreate_collection,
        )
        set_ingestion_progress(
            document.id,
            stage="queued",
            label="Queued for processing",
            progress=10,
        )
    except Exception as err:
        document.ingestion_status = "failed"
        document.ingestion_error = str(err)[:1000]
        session.add(document)
        await session.commit()
        raise HTTPException(
            status_code=503,
            detail="Failed to queue PDF for processing.",
        ) from err

    return ArtifactResponse.ok(
        data=await _artifact_read_for_document(document, session)
    )


__all__ = ["router"]
