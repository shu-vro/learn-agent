from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.config.constants import DEFAULT_OCR_LIB
from src.db.models.base import Base


class Preferences(Base):
    __tablename__ = "preferences"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    theme = Column(String, nullable=False, server_default="system")
    use_vision_model = Column(Boolean, nullable=False, server_default="false")
    use_image_descriptions = Column(Boolean, nullable=False, server_default="false")
    use_formula_transcription = Column(Boolean, nullable=False, server_default="false")
    equation_ocr_lib = Column(String, nullable=False, server_default=DEFAULT_OCR_LIB)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user = relationship("User", back_populates="preferences")

    @classmethod
    async def get_by_user_id(
        cls, session: AsyncSession, user_id: str
    ) -> Optional["Preferences"]:
        stmt = select(cls).where(cls.user_id == user_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @classmethod
    async def get_or_create(cls, session: AsyncSession, user_id: str) -> "Preferences":
        prefs = await cls.get_by_user_id(session, user_id)
        if prefs is not None:
            return prefs
        prefs = cls(user_id=user_id)
        session.add(prefs)
        await session.commit()
        await session.refresh(prefs)
        return prefs
