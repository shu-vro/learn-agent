from datetime import datetime
from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from src.db import get_session
from src.db.models.thread import Thread
from src.db.models.project import Project
from src.utils.api.BaseResponse import BaseResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from src.api.routes.projects.routes.threads.routes.chat import router as chat_router


class ThreadRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    thread_name: str
    extra: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @field_validator("extra", mode="before")
    @classmethod
    def coerce_extra(cls, value: object) -> dict:
        return value if isinstance(value, dict) else {}


ThreadsResponse = BaseResponse[List[ThreadRead]]
ThreadResponse = BaseResponse[ThreadRead]


class ThreadUpdatePayload(BaseModel):
    thread_name: str = Field(min_length=1, max_length=255)


router = APIRouter(tags=["threads"])


@router.get("/{project_id}/threads")
async def get_threads(
    request: Request,
    project_id: str,
    session: AsyncSession = Depends(get_session),
) -> ThreadsResponse:
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    threads = await Thread.get_by_project_id_for_user_visible(
        session, project_id, user.id
    )
    payload = [ThreadRead.model_validate(t) for t in threads]
    return ThreadsResponse.ok(data=payload)


@router.post("/{project_id}/threads")
async def create_thread(
    request: Request,
    project_id: str,
    session: AsyncSession = Depends(get_session),
) -> ThreadResponse:
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=403, detail="Project not found")

    thread = await Thread.create(session, project_id, user.id)
    return ThreadResponse.ok(data=ThreadRead.model_validate(thread))


@router.patch("/{project_id}/threads/{thread_id}")
async def update_thread(
    request: Request,
    project_id: str,
    thread_id: str,
    payload: ThreadUpdatePayload,
    session: AsyncSession = Depends(get_session),
) -> ThreadResponse:
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    thread = await Thread.get_by_id_for_user(session, thread_id, user.id)
    if not thread or thread.project_id != project_id:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread.thread_name = payload.thread_name.strip()
    await session.commit()
    await session.refresh(thread)
    return ThreadResponse.ok(data=ThreadRead.model_validate(thread))


@router.delete("/{project_id}/threads/{thread_id}")
async def delete_thread(
    request: Request,
    project_id: str,
    thread_id: str,
    session: AsyncSession = Depends(get_session),
) -> BaseResponse[None]:
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    project = await Project.get_by_id_for_user(session, project_id, user.id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    thread = await Thread.get_by_id_for_user(session, thread_id, user.id)
    if not thread or thread.project_id != project_id:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread.visibility = False
    await session.commit()
    return BaseResponse[None].ok()


router.include_router(chat_router)
