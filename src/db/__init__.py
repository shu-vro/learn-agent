from sqlalchemy import create_engine
from src.config.env import (
    DATABASE_NAME,
    DATABASE_USER,
    DATABASE_PASSWORD,
    DATABASE_PORT,
    DATABASE_HOST,
)

CONNECTION_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"


def get_connection():
    engine = create_engine(CONNECTION_URL)
    return engine


try:
    engine = get_connection()
    engine.connect()
    print(
        f"Connection to the {DATABASE_HOST} for user {DATABASE_USER} created successfully."
    )

except Exception as ex:
    print("Connection could not be made due to the following error:\n", ex)
    exit(1)
