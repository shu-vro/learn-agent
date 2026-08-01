from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from dotenv import load_dotenv

load_dotenv()

from src.db.models import (  # noqa: F401 — register tables on Base.metadata
    Base,
    Chat,
    Chunk,
    Document,
    Project,
    ProjectDocument,
    Thread,
    User,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# checkpointer tables are created at runtime
# wont versioned by Alembic autogenerate.
_IGNORED_TABLES = frozenset(
    {
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
    }
)


def include_object(object, name, type_, reflected, compare_to) -> bool:
    if type_ == "table" and name in _IGNORED_TABLES:
        return False
    if type_ == "index" and getattr(object, "table", None) is not None:
        if object.table.name in _IGNORED_TABLES:
            return False
    return True


def _async_url() -> str:
    from src.db import ASYNC_CONN_URL

    return ASYNC_CONN_URL


def run_migrations_offline() -> None:
    context.configure(
        url=_async_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _async_url()
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
