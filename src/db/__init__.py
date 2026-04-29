from __future__ import annotations
from functools import lru_cache
from typing import Sequence
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase
from src.config.env import (
    DATABASE_PORT,
    DATABASE_NAME,
    DATABASE_USER,
    DATABASE_PASSWORD,
    DATABASE_HOST,
)
from src.db.models import (
    Base,
    User,
    Project,
    Thread,
    Chat,
    Chunk,
    Document,
    ProjectDocument,
)

__all__ = [
    "Base",
    "User",
    "Project",
    "Thread",
    "Chat",
    "Chunk",
    "Document",
    "ProjectDocument",
    "db_engine",
    "create_all_tables",
]

CONN_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

AllTables = User, Project, Thread, Chat, Chunk, Document, ProjectDocument


@lru_cache(maxsize=1)
def db_engine():
    engine = create_engine(
        CONN_URL,
    )
    engine.connect()
    return engine


def create_all_tables(tables: Sequence[type[DeclarativeBase]] = AllTables) -> None:
    engine = db_engine()
    for table in tables:
        table.__table__.create(engine, checkfirst=True)
