from celery import Celery
from src.lib.redis_lib import REDIS_CONN_URL

celery_app = Celery(
    "artifact_upload_queue",
    broker=REDIS_CONN_URL,
    backend=REDIS_CONN_URL,
)

celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

__all__ = ["celery_app"]
