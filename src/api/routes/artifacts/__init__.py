from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.env import ASSET_UPLOAD_ROOT
from src.db import get_session
from src.db.models.document import Document
from src.db.models.project import Project
from src.db.models.project_document import ProjectDocument
from src.utils.api.BaseResponse import BaseResponse

ALLOWED_SUFFIXES = frozenset({".pdf", ".md", ".markdown"})
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

router = APIRouter(prefix="/projects", tags=["artifacts"])


class ArtifactRead(BaseModel):
    id: str
    name: str
    content: str


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


def _artifact_preview_text(root: Path, doc: Document) -> str:
    rel = doc.url
    target = (root / rel).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return "_Invalid asset path._"
    if not target.is_file():
        return "_File missing on disk._"
    suffix = _suffix_for_upload(doc.source)
    if suffix == ".pdf":
        return (
            f"_PDF **{doc.source}**. Inline preview is not available here; "
            "the file is stored on the server._"
        )
    if suffix in {".md", ".markdown"}:
        try:
            return target.read_text(encoding="utf-8")
        except OSError:
            return "_Could not read markdown file._"
    return f"_Unsupported type for **{doc.source}**._"


def _require_user(request: Request):
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


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
        .order_by(Document.source.asc())
    )
    result = await session.execute(stmt)
    docs = result.scalars().all()
    root = ASSET_UPLOAD_ROOT
    items: list[ArtifactRead] = []
    for doc in docs:
        content = await asyncio.to_thread(_artifact_preview_text, root, doc)
        items.append(ArtifactRead(id=doc.id, name=doc.source, content=content))
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
            detail="Only .pdf and .md / .markdown files are allowed.",
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

    document = Document(id=doc_id, source=Path(raw_name).name, url=rel_path)
    link = ProjectDocument(project_id=project_id, document_id=doc_id)
    session.add(document)
    session.add(link)
    await session.commit()
    await session.refresh(document)

    content = await asyncio.to_thread(_artifact_preview_text, root, document)
    return ArtifactResponse.ok(
        data=ArtifactRead(id=document.id, name=document.source, content=content)
    )


__all__ = ["router"]
