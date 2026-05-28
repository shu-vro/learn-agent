import uuid

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.db.models.base import Base
from src.db.models.chunk import documents_chunks


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String, nullable=False)  # url or "uploaded"
    url = Column(
        String, nullable=False
    )  # file path in upload directory after processed.
    original_url = Column(String, nullable=True)  # original document url.
    name = Column(String, nullable=False)  # original file name
    sha256 = Column(
        String, nullable=True, unique=True
    )  # binary fingerprint of uploaded file
    mime_type = Column(String, nullable=True)
    file_size = Column(Integer, nullable=False, default=0)
    ingestion_status = Column(String, nullable=False, default="processing")
    ingestion_error = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    chunks = relationship(
        "Chunk",
        secondary=documents_chunks,
        back_populates="documents",
        cascade="all, delete",
    )
    projects = relationship(
        "ProjectDocument", back_populates="document", cascade="all, delete"
    )
