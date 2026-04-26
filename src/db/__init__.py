from functools import lru_cache
from sqlalchemy import create_engine
from src.config.env import (
    DATABASE_PORT,
    DATABASE_NAME,
    DATABASE_USER,
    DATABASE_PASSWORD,
    DATABASE_HOST,
)

CONN_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"


@lru_cache(maxsize=1)
def db_engine():
    engine = create_engine(
        CONN_URL,
    )
    engine.connect()
    return engine
