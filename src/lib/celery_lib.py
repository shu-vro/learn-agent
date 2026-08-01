import sys

from celery import Celery
from src.lib.redis_lib import REDIS_CONN_URL

celery_app = Celery(
    "artifact_upload_queue",
    broker=REDIS_CONN_URL,
    backend=REDIS_CONN_URL,
)

# Docling crash with SIGSEGV inside Celery prefork workers on macOS.
_DEFAULT_WORKER_POOL = "solo" if sys.platform == "darwin" else "prefork"

celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_pool=_DEFAULT_WORKER_POOL,
    imports=("src.tasks.artifact_ingestion", "src.tasks.chat_images"),
)

__all__ = ["celery_app"]
