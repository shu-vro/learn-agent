import uuid

from sqlalchemy import Column, String, JSON, ForeignKey
from sqlalchemy.orm import relationship
from src.db.models.base import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    extra = Column(JSON)

    user = relationship("User", back_populates="projects")
    threads = relationship("Thread", back_populates="project")
    documents = relationship("ProjectDocument", back_populates="project")
