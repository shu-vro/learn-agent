from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_session
from src.db.models.project import Project
from src.db.models.project_document import ProjectDocument
from src.utils.api.BaseResponse import BaseResponse
from src.utils.db.read_schema import read_schema_for_orm_columns
from pydantic import BaseModel, Field
from src.api.routes.projects.routes.artifacts import router as artifacts_router
from src.api.routes.projects.routes.threads import router as threads_router

# Pydantic shape is derived from ``Project`` columns — add DB fields only there.
ProjectRead = read_schema_for_orm_columns(Project, name="ProjectRead")
ProjectsResponse = BaseResponse[list[ProjectRead]]
ProjectResponse = BaseResponse[ProjectRead]

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("/", response_model=ProjectsResponse)
async def get_projects(
    request: Request, session: AsyncSession = Depends(get_session)
) -> ProjectsResponse:
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    projects = await Project.get_by_user_id(session, user.id)
    payload = [ProjectRead.model_validate(p) for p in projects]
    return ProjectsResponse.ok(data=payload)


class ProjectCreate(BaseModel):
    name: str = ""
    description: str = ""
    extra: dict = Field(default_factory=dict)


@router.post("/", response_model=ProjectResponse)
async def create_project(
    request: Request,
    payload: ProjectCreate,
    session: AsyncSession = Depends(get_session),
) -> ProjectResponse:
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    project = await Project.create(
        session,
        user_id=user.id,
        name=payload.name,
        description=payload.description,
        extra=payload.extra or None,
    )
    return ProjectResponse.ok(data=ProjectRead.model_validate(project))


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    request: Request,
    project_id: str,
    payload: ProjectCreate,
    session: AsyncSession = Depends(get_session),
) -> ProjectResponse:
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    project.name = payload.name if payload.name is not None else project.name
    project.description = (
        payload.description if payload.description is not None else project.description
    )
    await session.commit()
    await session.refresh(project)
    return ProjectResponse.ok(data=ProjectRead.model_validate(project))


@router.delete("/{project_id}", response_model=BaseResponse[None])
async def delete_project(
    request: Request,
    project_id: str,
    session: AsyncSession = Depends(get_session),
) -> BaseResponse[None]:
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    await session.execute(
        delete(ProjectDocument).where(ProjectDocument.project_id == project_id)
    )
    await session.delete(project)
    await session.commit()
    return BaseResponse.ok()


router.include_router(artifacts_router)
router.include_router(threads_router)
