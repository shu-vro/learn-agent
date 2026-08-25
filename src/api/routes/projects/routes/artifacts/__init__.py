from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.constants import (
    DEFAULT_OCR_LIB,
    DEFAULT_QDRANT_COLLECTION,
    SUPPORTED_UPLOAD_SUFFIXES,
)
from src.db.models.preferences import Preferences
from src.schemas.preferences import (
    IngestionPreferences,
    UserPreferencesPublic,
    resolve_ingestion_flags,
)
from src.config.env import ASSET_UPLOAD_ROOT
from src.db import get_session
from src.db.models.chunk import (
    Chunk,
    coalesced_chunk_type,
    DEFAULT_CHUNK_TYPE,
    IMAGE_CHUNK_TYPE,
    NOTE_CHUNK_TYPE,
    documents_chunks,
)
from src.db.models.document import Document
from src.db.models.project import Project
from src.db.models.project_document import ProjectDocument
from src.module.chunk_notes import upsert_chunk_note_in_qdrant
from src.side_agents.note_generator_agent import generate_chunk_note
from src.tasks.artifact_ingestion import process_artifact_upload
from src.utils.ingestion_progress import get_ingestion_progress, set_ingestion_progress
from src.utils.api.BaseResponse import BaseResponse
from src.utils.api.artifact_markdown_fixer import rewrite_chunk_image_urls

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

router = APIRouter(tags=["artifacts"])


class ArtifactRead(BaseModel):
    id: str
    name: str
    # Content fingerprint; document citations address chunks by it
    # (``reference_id=<sha256>:<chunk_id>``).
    sha256: str | None = None
    chunks: Dict[str, Any] = {}
    ingestion_status: str = "completed"
    ingestion_stage: str | None = None
    ingestion_stage_label: str | None = None
    ingestion_progress: int | None = None


class ChunkNoteRead(BaseModel):
    chunk_id: str
    content: str


class ArtifactNotesRead(BaseModel):
    notes: Dict[str, str] = {}


ArtifactsListResponse = BaseResponse[list[ArtifactRead]]
ArtifactResponse = BaseResponse[ArtifactRead]
ChunkNoteResponse = BaseResponse[ChunkNoteRead]
ArtifactNotesResponse = BaseResponse[ArtifactNotesRead]


def _suffix_for_upload(filename: str) -> str:
    return Path(filename).suffix.lower()


def _sanitize_stored_filename(filename: str) -> str:
    base = Path(filename).name
    if not base or base in {".", ".."}:
        return "file"
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return cleaned[:200] or "file"


async def _get_source_chunks(
    document_id: str,
    session: AsyncSession,
) -> list[tuple[Chunk, int]]:
    """Fetch displayable source chunks (text and image) in document order."""
    stmt = (
        select(Chunk, documents_chunks.c.order)
        .join(documents_chunks, documents_chunks.c.chunks_id == Chunk.id)
        .where(
            documents_chunks.c.document_id == document_id,
            coalesced_chunk_type().in_((DEFAULT_CHUNK_TYPE, IMAGE_CHUNK_TYPE)),
        )
        .order_by(documents_chunks.c.order.asc())
    )
    result = await session.execute(stmt)
    return list(result.all())


async def _get_notes_for_document(
    document_id: str,
    session: AsyncSession,
) -> dict[str, Chunk]:
    stmt = select(Chunk).where(
        Chunk.type == NOTE_CHUNK_TYPE,
        Chunk.extra["document_id"].as_string() == document_id,
    )
    result = await session.execute(stmt)
    notes_by_source: dict[str, Chunk] = {}
    for note in result.scalars().all():
        extra = note.extra or {}
        source_id = extra.get("source_chunk_id")
        if isinstance(source_id, str) and source_id:
            notes_by_source[source_id] = note
    return notes_by_source


def _chunks_to_dict(
    chunks: list[tuple[Chunk, int]],
    *,
    doc_sha256: str | None,
) -> Dict[str, Any]:
    """Serialize source chunks for the artifacts response.

    Notes are intentionally excluded here; they are fetched and generated
    on demand via the dedicated notes endpoints.
    """
    ordered: Dict[str, Any] = {}
    for chunk, order in chunks:
        ordered[chunk.id] = {
            "content": rewrite_chunk_image_urls(chunk.content, doc_sha256),
            "order": order,
            "type": chunk.type or DEFAULT_CHUNK_TYPE,
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


@dataclass(frozen=True)
class _ParsedUpload:
    raw_name: str
    body: bytes
    content_type: str | None


def _validate_upload_file(file: UploadFile, body: bytes) -> _ParsedUpload:
    raw_name = file.filename or "upload"
    suffix = _suffix_for_upload(raw_name)
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type ({raw_name!r}). Allowed: "
                f"{', '.join(sorted(SUPPORTED_UPLOAD_SUFFIXES))}."
            ),
        )
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({raw_name!r}).",
        )
    return _ParsedUpload(
        raw_name=raw_name,
        body=body,
        content_type=file.content_type,
    )


async def _upload_single_artifact(
    session: AsyncSession,
    *,
    project: Project,
    project_id: str,
    upload: _ParsedUpload,
    ingestion: IngestionPreferences,
    recreate_collection: bool,
) -> tuple[ArtifactRead, bool]:
    doc_id = str(uuid.uuid4())
    safe = _sanitize_stored_filename(Path(upload.raw_name).name)
    rel_path = f"{project_id}/{doc_id}_{safe}"
    root = ASSET_UPLOAD_ROOT
    dest = root / rel_path
    await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(dest.write_bytes, upload.body)
    source_file_url = dest.resolve().as_uri()
    upload_sha256 = hashlib.sha256(upload.body).hexdigest()

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
            name=Path(upload.raw_name).name,
            sha256=upload_sha256,
            mime_type=upload.content_type,
            file_size=len(upload.body),
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
            return (
                ArtifactRead(
                    id=document.id,
                    name=document.name,
                    sha256=document.sha256,
                    chunks={},
                    ingestion_status=document.ingestion_status,
                ),
                False,
            )
        if document.ingestion_status == "completed":
            from src.tasks.artifact_ingestion import _try_update_project_metadata

            try:
                _try_update_project_metadata(project_id)
                await session.refresh(project)
            except Exception:
                pass
            return (
                await _artifact_read_for_document(document, session),
                False,
            )
        return (
            ArtifactRead(
                id=document.id,
                name=document.name,
                sha256=document.sha256,
                chunks={},
                ingestion_status=document.ingestion_status,
            ),
            False,
        )

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
            detail=f"Failed to queue file for processing ({upload.raw_name!r}).",
        ) from err

    return await _artifact_read_for_document(document, session), True


async def _artifact_read_for_document(
    document: Document,
    session: AsyncSession,
) -> ArtifactRead:
    chunks: dict[str, Any] = {}
    if document.ingestion_status == "completed":
        doc_chunks = await _get_source_chunks(document.id, session)
        chunks = _chunks_to_dict(doc_chunks, doc_sha256=document.sha256)

    ingestion_progress = (
        get_ingestion_progress(document.id)
        if document.ingestion_status == "processing"
        else None
    )
    return ArtifactRead(
        id=document.id,
        name=document.name,
        sha256=document.sha256,
        chunks=chunks,
        ingestion_status=document.ingestion_status,
        ingestion_stage=ingestion_progress.get("stage") if ingestion_progress else None,
        ingestion_stage_label=(
            ingestion_progress.get("label") if ingestion_progress else None
        ),
        ingestion_progress=(
            ingestion_progress.get("progress") if ingestion_progress else None
        ),
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


@router.post("/{project_id}/artifacts", response_model=ArtifactsListResponse)
async def upload_project_artifacts(
    request: Request,
    project_id: str,
    session: AsyncSession = Depends(get_session),
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
    use_vision_model: str | None = Form(None),
    use_image_descriptions: str | None = Form(None),
    use_formula_transcription: str | None = Form(None),
    equation_ocr_lib: str | None = Form(None),
    rebuild: str | None = Form(None),
) -> ArtifactsListResponse:
    user = _require_user(request)
    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    uploads = list(files)
    if file is not None:
        uploads.append(file)
    if not uploads:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    parsed_uploads: list[_ParsedUpload] = []
    for upload_file in uploads:
        body = await upload_file.read()
        parsed_uploads.append(_validate_upload_file(upload_file, body))

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
    recreate_for_next = recreate_collection

    items: list[ArtifactRead] = []
    for parsed in parsed_uploads:
        artifact, queued = await _upload_single_artifact(
            session,
            project=project,
            project_id=project_id,
            upload=parsed,
            ingestion=ingestion,
            recreate_collection=recreate_for_next,
        )
        items.append(artifact)
        if queued and recreate_for_next:
            recreate_for_next = False

    return ArtifactsListResponse.ok(data=items)


async def _load_scoped_document(
    session: AsyncSession,
    *,
    project_id: str,
    artifact_id: str,
) -> Document:
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
    return document


def _chunk_to_window_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "type": chunk.type or DEFAULT_CHUNK_TYPE,
        "content": chunk.content,
        "extra": dict(chunk.extra or {}),
    }


@router.get(
    "/{project_id}/artifacts/{artifact_id}/notes",
    response_model=ArtifactNotesResponse,
)
async def get_artifact_notes(
    request: Request,
    project_id: str,
    artifact_id: str,
    session: AsyncSession = Depends(get_session),
) -> ArtifactNotesResponse:
    user = _require_user(request)
    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    document = await _load_scoped_document(
        session, project_id=project_id, artifact_id=artifact_id
    )
    notes_by_source = await _get_notes_for_document(document.id, session)
    notes = {source_id: note.content for source_id, note in notes_by_source.items()}
    return ArtifactNotesResponse.ok(data=ArtifactNotesRead(notes=notes))


@router.post(
    "/{project_id}/artifacts/{artifact_id}/chunks/{chunk_id}/note",
    response_model=ChunkNoteResponse,
)
async def generate_chunk_note_endpoint(
    request: Request,
    project_id: str,
    artifact_id: str,
    chunk_id: str,
    regenerate: bool = False,
    session: AsyncSession = Depends(get_session),
) -> ChunkNoteResponse:
    user = _require_user(request)
    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    document = await _load_scoped_document(
        session, project_id=project_id, artifact_id=artifact_id
    )
    if document.ingestion_status != "completed":
        raise HTTPException(
            status_code=409, detail="Artifact is not ready for note generation."
        )

    source_chunks = await _get_source_chunks(document.id, session)
    target_index = next(
        (idx for idx, (chunk, _) in enumerate(source_chunks) if chunk.id == chunk_id),
        None,
    )
    if target_index is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    # Notes belong to the chunk/document, not to a project or user: documents
    # (and their chunks) are globally deduplicated by the ingestion pipeline.
    # A plain generate is idempotent — if a note already exists we return it
    # as-is. An explicit ?regenerate=true forces a fresh note that replaces the
    # shared one (Postgres row + Qdrant point).
    existing_notes = await _get_notes_for_document(document.id, session)
    existing_note = existing_notes.get(chunk_id)
    if existing_note is not None and not regenerate:
        return ChunkNoteResponse.ok(
            data=ChunkNoteRead(chunk_id=chunk_id, content=existing_note.content)
        )

    window = [_chunk_to_window_dict(chunk) for chunk, _ in source_chunks]
    prev_chunk = window[target_index - 1] if target_index > 0 else None
    target_chunk = window[target_index]
    next_chunk = window[target_index + 1] if target_index < len(window) - 1 else None

    try:
        note_content = await asyncio.to_thread(
            generate_chunk_note, prev_chunk, target_chunk, next_chunk
        )
    except Exception as err:
        logger.exception(
            "Failed to generate note for chunk_id={} (document_id={})",
            chunk_id,
            document.id,
        )
        raise HTTPException(
            status_code=502, detail=f"Failed to generate note: {err}"
        ) from err

    # Regenerate replaces the existing note in place (Qdrant is replaced below
    # via delete-then-add keyed on document_id + source_chunk_id).
    if existing_note is not None:
        await session.delete(existing_note)

    note_chunk = Chunk(
        content=note_content,
        type=NOTE_CHUNK_TYPE,
        extra={
            "type": NOTE_CHUNK_TYPE,
            "source_chunk_id": chunk_id,
            "document_id": document.id,
        },
    )
    session.add(note_chunk)
    await session.commit()

    try:
        await asyncio.to_thread(
            upsert_chunk_note_in_qdrant,
            document_id=document.id,
            source_chunk_id=chunk_id,
            note_content=note_content,
            doc_sha256=document.sha256,
        )
    except Exception:
        # Note is persisted in Postgres; a Qdrant failure should not fail the request.
        logger.exception(
            "Failed to upsert note into Qdrant for chunk_id={} (document_id={})",
            chunk_id,
            document.id,
        )

    return ChunkNoteResponse.ok(
        data=ChunkNoteRead(chunk_id=chunk_id, content=note_content)
    )


__all__ = ["router"]
