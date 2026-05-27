from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from typing import List, Optional

from src.db.models.base import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    extra = Column(JSON)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user = relationship("User", back_populates="projects")
    threads = relationship("Thread", back_populates="project")
    documents = relationship("ProjectDocument", back_populates="project")

    @classmethod
    async def get_by_user_id(
        cls, session: AsyncSession, user_id: str
    ) -> List["Project"]:
        stmt = select(cls).where(cls.user_id == user_id)
        result = await session.execute(stmt)
        return result.scalars().all()

    @classmethod
    async def get_by_id_for_user(
        cls, session: AsyncSession, project_id: str, user_id: str
    ) -> Optional["Project"]:
        stmt = select(cls).where(cls.id == project_id, cls.user_id == user_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @classmethod
    async def create(
        cls,
        session: AsyncSession,
        *,
        user_id: str,
        title: str,
        description: str,
        extra: dict | None = None,
    ) -> "Project":
        project = cls(
            user_id=user_id,
            name=title,
            description=description,
            extra=extra,
        )
        session.add(project)
        await session.commit()
        await session.refresh(project)
        return project
