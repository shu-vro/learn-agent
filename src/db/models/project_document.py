import uuid

from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.orm import relationship
from src.db.models.base import Base


class ProjectDocument(Base):
    __tablename__ = "projects_documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)

    project = relationship("Project", back_populates="documents")
    document = relationship("Document", back_populates="projects")
