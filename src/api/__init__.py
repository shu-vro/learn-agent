from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.env import CORS_ALLOW_ORIGINS


def create_api() -> FastAPI:
    app = FastAPI(title="RAG Agent API", version="1.0")

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

    return app
