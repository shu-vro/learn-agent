import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.models.base import Base

DEFAULT_CHUNK_TYPE = "text_chunk"
IMAGE_CHUNK_TYPE = "image"


documents_chunks = Table(
    "documents_chunks",
    Base.metadata,
    Column("id", String, primary_key=True, default=lambda: str(uuid.uuid4())),
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE")),
    Column("chunks_id", String, ForeignKey("chunks.id", ondelete="CASCADE")),
    Column("order", Integer, nullable=False, default=0),
)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(String, nullable=False)
    type = Column(String, nullable=True, default=DEFAULT_CHUNK_TYPE)
    extra = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    chats = relationship("Chat", secondary="chats_chunks", back_populates="chunks")
    documents = relationship(
        "Document", secondary=documents_chunks, back_populates="chunks"
    )


def coalesced_chunk_type():
    """SQL expression: prefer ``chunks.type``, fall back to legacy ``extra.type``."""
    return func.coalesce(
        Chunk.type,
        Chunk.extra["type"].as_string(),
        DEFAULT_CHUNK_TYPE,
    )
