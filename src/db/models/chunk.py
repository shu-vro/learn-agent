import uuid

from sqlalchemy import Column, String, ForeignKey, Table
from sqlalchemy.orm import relationship
from src.db.models.base import Base


documents_chunks = Table(
    "documents_chunks",
    Base.metadata,
    Column("id", String, primary_key=True, default=lambda: str(uuid.uuid4())),
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE")),
    Column("chunks_id", String, ForeignKey("chunks.id", ondelete="CASCADE")),
)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    chats = relationship("Chat", secondary="chats_chunks", back_populates="chunks")
    documents = relationship(
        "Document", secondary=documents_chunks, back_populates="chunks"
    )
