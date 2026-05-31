from __future__ import annotations

import json
from typing import Any

from src.lib.redis_lib import redis_client

_KEY_PREFIX = "ingestion:progress:"
_TTL_SECONDS = 86_400


def set_ingestion_progress(
    document_id: str,
    *,
    stage: str,
    label: str,
    progress: int,
) -> None:
    payload = {"stage": stage, "label": label, "progress": progress}
    redis_client.setex(
        f"{_KEY_PREFIX}{document_id}",
        _TTL_SECONDS,
        json.dumps(payload),
    )


def get_ingestion_progress(document_id: str) -> dict[str, Any] | None:
    raw = redis_client.get(f"{_KEY_PREFIX}{document_id}")
    if not raw:
        return None
    return json.loads(raw)


def clear_ingestion_progress(document_id: str) -> None:
    redis_client.delete(f"{_KEY_PREFIX}{document_id}")


__all__ = [
    "set_ingestion_progress",
    "get_ingestion_progress",
    "clear_ingestion_progress",
]
