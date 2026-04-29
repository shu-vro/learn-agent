import uuid

from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
from src.db.models.base import Base
from src.db.models.chunk import documents_chunks


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String, nullable=False)
    url = Column(String, nullable=False)

    chunks = relationship(
        "Chunk",
        secondary=documents_chunks,
        back_populates="documents",
        cascade="all, delete",
    )
    projects = relationship(
        "ProjectDocument", back_populates="document", cascade="all, delete"
    )
