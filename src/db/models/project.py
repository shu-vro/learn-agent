from __future__ import annotations

import uuid

from sqlalchemy import Column, String, JSON, ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import relationship
from src.db.models.base import Base
from typing import List
from sqlalchemy import select


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    extra = Column(JSON)

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
