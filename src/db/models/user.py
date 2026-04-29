import uuid

from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
from src.db.models.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

    projects = relationship("Project", back_populates="user")
    threads = relationship("Thread", back_populates="user")
    chats = relationship("Chat", back_populates="user")
