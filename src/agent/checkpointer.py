"""Process-wide LangGraph Postgres checkpointer for the API."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.db import CONN_URL

_pool: ConnectionPool | None = None
_checkpointer: PostgresSaver | None = None


def init_checkpointer() -> PostgresSaver:
    """Open a shared connection pool and PostgresSaver (idempotent)."""
    global _pool, _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    print("Initializing checkpointer...", log_level="INFO")

    _pool = ConnectionPool(
        conninfo=CONN_URL,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        min_size=1,
        max_size=10,
        open=True,
    )
    # PostgresSaver expects a connection; use pool connection as context-managed
    # long-lived saver via the pool's getconn pattern wrapped by from_conn.
    # Prefer ConnectionPool directly when supported:
    try:
        _checkpointer = PostgresSaver(_pool)  # type: ignore[arg-type]
    except TypeError:
        # Fallback: hold one dedicated connection for the process lifetime.
        conn = Connection.connect(
            CONN_URL,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        _checkpointer = PostgresSaver(conn)

    _checkpointer.setup()
    print("Checkpointer initialized successfully.", log_level="SUCCESS")
    return _checkpointer


def close_checkpointer() -> None:
    global _pool, _checkpointer
    if _pool is not None:
        _pool.close()
    _pool = None
    _checkpointer = None


@contextmanager
def checkpointer_session() -> Iterator[PostgresSaver]:
    """CLI-style context manager that yields the shared API checkpointer."""
    yield init_checkpointer()
