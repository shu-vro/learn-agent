from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import Column, DateTime, String, select
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import relationship

from src.db.models.base import Base
from src.db.models.preferences import Preferences
from src.utils.argon2_utils import hash_password


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    temp_password = Column(String, nullable=True)
    temp_password_expires_at = Column(DateTime, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    projects = relationship("Project", back_populates="user")
    threads = relationship("Thread", back_populates="user")
    chats = relationship("Chat", back_populates="user")
    preferences = relationship(
        "Preferences",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )

    @classmethod
    async def get_by_email(cls, session: AsyncSession, email: str) -> Optional["User"]:
        stmt = select(cls).where(cls.email == email)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @classmethod
    async def create(
        cls,
        session: AsyncSession,
        *,
        name: str,
        email: str,
        password: str,
    ) -> "User":
        user = cls(name=name, email=email, password=hash_password(password))
        session.add(user)
        await session.flush()
        session.add(Preferences(user_id=user.id))
        await session.commit()
        await session.refresh(user)
        return user

    def __repr__(self):
        print(dict(self.__dict__))
