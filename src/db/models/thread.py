import uuid

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.models.base import Base


class Thread(Base):
    __tablename__ = "threads"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    thread_name = Column(String, nullable=False)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
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
