import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, JSON, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.models.base import Base
from typing import List, Optional


class Thread(Base):
    __tablename__ = "threads"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    thread_name = Column(String, nullable=False, default="")
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    visibility = Column(Boolean, nullable=False, default=True)
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

    user = relationship("User", back_populates="threads")
    project = relationship("Project", back_populates="threads")
    chats = relationship("Chat", back_populates="thread")

    @classmethod
    async def get_by_project_id(
        cls, session: AsyncSession, project_id: str
    ) -> List["Thread"]:
        stmt = select(cls).where(cls.project_id == project_id)
        result = await session.execute(stmt)
        return result.scalars().all()

    @classmethod
    async def get_by_id_for_user(
        cls, session: AsyncSession, thread_id: str, user_id: str
    ) -> Optional["Thread"]:
        stmt = select(cls).where(cls.id == thread_id, cls.user_id == user_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @classmethod
    async def get_by_project_id_for_user(
        cls, session: AsyncSession, project_id: str, user_id: str
    ) -> List["Thread"]:
        stmt = select(cls).where(cls.project_id == project_id, cls.user_id == user_id)
        result = await session.execute(stmt)
        return result.scalars().all()

    @classmethod
    async def get_by_project_id_for_user_visible(
        cls, session: AsyncSession, project_id: str, user_id: str
    ) -> List["Thread"]:
        stmt = select(cls).where(
            cls.project_id == project_id, cls.user_id == user_id, cls.visibility
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    @classmethod
    async def create(
        cls,
        session: AsyncSession,
        project_id: str,
        user_id: str,
        thread_name: str = "",
        extra: dict | None = None,
    ) -> "Thread":
        thread = cls(
            project_id=project_id,
            user_id=user_id,
            thread_name=thread_name,
            extra=extra,
        )
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        return thread
