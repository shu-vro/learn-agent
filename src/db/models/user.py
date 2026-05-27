from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import Column, String, select, DateTime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import relationship

from src.db.models.base import Base
from src.utils.argon2_utils import hash_password


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    temp_password = Column(String, nullable=True)
    temp_password_expires_at = Column(DateTime, nullable=True)

    projects = relationship("Project", back_populates="user")
    threads = relationship("Thread", back_populates="user")
    chats = relationship("Chat", back_populates="user")

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
        await session.commit()
        await session.refresh(user)
        return user

    def __repr__(self):
        print(dict(self.__dict__))
