from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker
from src.config.env import (
    DATABASE_HOST,
    DATABASE_NAME,
    DATABASE_PASSWORD,
    DATABASE_PORT,
    DATABASE_USER,
)
from src.db.models import (
    Base,
    Chat,
    Chunk,
    Document,
    Project,
    ProjectDocument,
    Thread,
    User,
)

# Sync URL is still consumed by langgraph PostgresSaver (psycopg2-based),
# so we keep it alongside the async URL used by SQLAlchemy / FastAPI.
_USER = quote_plus(DATABASE_USER)
_PASSWORD = quote_plus(DATABASE_PASSWORD)
_NETLOC = f"{_USER}:{_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

CONN_URL = f"postgresql://{_NETLOC}"
ASYNC_CONN_URL = f"postgresql+asyncpg://{_NETLOC}"


def _alembic_ini_path() -> Path:
    return Path(__file__).resolve().parents[2] / "alembic.ini"


def upgrade_schema_to_head() -> None:
    """Apply all pending Alembic migrations (sync; safe to call via asyncio.to_thread)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_alembic_ini_path()))
    command.upgrade(cfg, "head")


@lru_cache(maxsize=1)
def engine() -> AsyncEngine:
    # pool_pre_ping avoids stale connections after Postgres restarts; the
    # asyncpg driver is fully non-blocking so multiple FastAPI requests share
    # the pool without serializing on the event loop.
    return create_async_engine(
        ASYNC_CONN_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


@lru_cache(maxsize=1)
def session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine(),
        expire_on_commit=False,
        autoflush=False,
    )


@lru_cache(maxsize=1)
def sync_engine() -> Engine:
    return create_engine(CONN_URL, pool_pre_ping=True)


@lru_cache(maxsize=1)
def sync_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=sync_engine(),
        expire_on_commit=False,
        autoflush=False,
    )


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped AsyncSession."""
    async with session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def create_all_tables() -> None:
    """Bring the database schema to the latest revision (creates and alters tables).

    Model changes are delivered through Alembic revisions under ``alembic/versions/``.
    After editing models, run ``uv run alembic revision --autogenerate -m \"...\"``,
    review the script, then commit; this function runs ``alembic upgrade head``.
    """
    await asyncio.to_thread(upgrade_schema_to_head)


__all__ = [
    "Base",
    "User",
    "Project",
    "Thread",
    "Chat",
    "Chunk",
    "Document",
    "ProjectDocument",
    "engine",
    "session_factory",
    "get_session",
    "sync_engine",
    "sync_session_factory",
    "create_all_tables",
    "CONN_URL",
    "ASYNC_CONN_URL",
]
