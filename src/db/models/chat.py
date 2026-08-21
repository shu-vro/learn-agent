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
from sqlalchemy.dialects.postgresql import ARRAY
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
    type = Column(String, nullable=False)  # user, assistant
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
    selection = Column(String, nullable=True)  # selection text from the message
    reference_id = Column(String, nullable=True)  # reference id of the message
    input_token = Column(Integer, default=0)
    cache_token = Column(Integer, default=0)
    output_token = Column(Integer, default=0)
    total_token = Column(Integer, default=0)
    # Per-iteration token breakdown for this generation (streaming-derived).
    usage_detail = Column(JSON, nullable=True)
    image_urls = Column(ARRAY(String), nullable=True)

    # Read-only recommended resources (document chunks, YouTube videos, etc.)
    # produced by the RAG agent for this message.
    chunks = relationship(
        "Chunk", secondary=chat_messages_chunks, back_populates="chat_messages"
    )
    # One message -> many thinking / tool steps produced while generating it.
    thinking_messages = relationship(
        "ThinkingMessage",
        back_populates="chat_message",
        cascade="all, delete-orphan",
        order_by="ThinkingMessage.created_at",
    )
    tool_messages = relationship(
        "ToolMessage",
        back_populates="chat_message",
        cascade="all, delete-orphan",
        order_by="ToolMessage.created_at",
    )
    # collected artifacts (documents, videos, websites) produced by the RAG agent for this message.
    artifacts = relationship(
        "Artifacts",
        back_populates="chat_message",
        cascade="all, delete-orphan",
        order_by="Artifacts.created_at",
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


class ThinkingMessage(Base):
    __tablename__ = "thinking_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_message_id = Column(
        String, ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False
    )
    thinking = Column(String, nullable=False)

    chat_message = relationship("ChatMessage", back_populates="thinking_messages")

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


class ToolMessage(Base):
    __tablename__ = "tool_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_message_id = Column(
        String, ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False
    )
    tool_name = Column(String, nullable=False)
    tool_parameters = Column(JSON, nullable=False)
    tool_result = Column(JSON, nullable=False)

    chat_message = relationship("ChatMessage", back_populates="tool_messages")

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


class Artifacts(Base):
    __tablename__ = "chat_message_artifacts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_message_id = Column(
        String, ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False
    )
    artifact_type = Column(
        String, nullable=False
    )  # e.g., "document", "video", "website"
    artifact_url = Column(String, nullable=False)
    artifact_metadata = Column(JSON, nullable=True)

    chat_message = relationship("ChatMessage", back_populates="artifacts")

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
