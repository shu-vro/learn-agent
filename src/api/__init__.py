from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.env import CORS_ALLOW_ORIGINS
from src.config.constants import ENVIRONMENT
from src.utils.api.exception_handlers import register_exception_handlers


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from src.agent.checkpointer import close_checkpointer, init_checkpointer

    init_checkpointer()
    try:
        yield
    finally:
        close_checkpointer()


def create_api() -> FastAPI:
    app = FastAPI(title="RAG Agent API", version="1.0", lifespan=_lifespan)
    register_exception_handlers(app)

    # Import and include your API routes here
    from src.api.routes import router as api_router

    origins = CORS_ALLOW_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    if ENVIRONMENT == "development":
        from src.utils.helper import main

        main()

    return app
