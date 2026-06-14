from redis import Redis
from src.config.env import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
from fakeredis import FakeRedis


REDIS_CONN_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"


def get_redis_client():
    try:
        client = Redis.from_url(REDIS_CONN_URL, decode_responses=True)
        client.ping()
        print("Successfully connected to Redis.", log_level="SUCCESS")
        return client
    except Exception as e:
        print(f"Failed to connect to Redis: {e}", log_level="ERROR")
        fr = FakeRedis(decode_responses=True)
        return fr


redis_client = get_redis_client()

__all__ = ["redis_client", "REDIS_CONN_URL"]
