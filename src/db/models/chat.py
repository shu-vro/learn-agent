import uuid

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Table,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.models.base import Base

chat_messages_chunks = Table(
    "chat_messages_chunks",
    Base.metadata,
    Column("id", String, primary_key=True, default=lambda: str(uuid.uuid4())),
    Column(
        "chat_message_id",
        String,
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
    ),
    Column("chunk_id", String, ForeignKey("chunks.id", ondelete="CASCADE")),
)


class Chat(Base):
    __tablename__ = "chats"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, ForeignKey("threads.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    type = Column(String, nullable=False)
    group_id = Column(String)
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

    thread = relationship("Thread", back_populates="chats")
    user = relationship("User", back_populates="chats")
    # One chat -> many messages (initial generation + regenerated outputs).
    messages = relationship(
        "ChatMessage",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(
        String,
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
    )
    message = Column(String, nullable=False)
    input_token = Column(Integer, default=0)
    output_token = Column(Integer, default=0)
    total_token = Column(Integer, default=0)

    # Read-only recommended resources (document chunks, YouTube videos, etc.)
    # produced by the RAG agent for this message.
    chunks = relationship(
        "Chunk", secondary=chat_messages_chunks, back_populates="chat_messages"
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    chat = relationship("Chat", back_populates="messages")
