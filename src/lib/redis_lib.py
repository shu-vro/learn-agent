from redis import Redis
from src.config.env import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD


REDIS_CONN_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
redis_client = Redis.from_url(REDIS_CONN_URL, decode_responses=True)

__all__ = ["redis_client", "REDIS_CONN_URL"]
