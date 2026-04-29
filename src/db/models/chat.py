import uuid

from sqlalchemy import (
    Column,
    String,
    JSON,
    ForeignKey,
    Integer,
    Table,
)
from sqlalchemy.orm import relationship
from src.db.models.base import Base

chats_chunks = Table(
    "chats_chunks",
    Base.metadata,
    Column("id", String, primary_key=True, default=lambda: str(uuid.uuid4())),
    Column("chats_id", String, ForeignKey("chats.id", ondelete="CASCADE")),
    Column("chunks_id", String, ForeignKey("chunks.id", ondelete="CASCADE")),
)


class Chat(Base):
    __tablename__ = "chats"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, ForeignKey("threads.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    type = Column(String, nullable=False)
    message = Column(String, nullable=False)
    input_token = Column(Integer, default=0)
    output_token = Column(Integer, default=0)
    total_token = Column(Integer, default=0)
    group_id = Column(String)
    extra = Column(JSON)

    thread = relationship("Thread", back_populates="chats")
    user = relationship("User", back_populates="chats")
    chunks = relationship("Chunk", secondary=chats_chunks, back_populates="chats")
