from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_session
from src.db.models.project import Project
from src.utils.api.BaseResponse import BaseResponse
from src.utils.db.read_schema import read_schema_for_orm_columns
from pydantic import BaseModel

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
    title: str
    description: str


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
        session, user_id=user.id, title=payload.title, description=payload.description
    )
    return ProjectResponse.ok(data=ProjectRead.model_validate(project))
