from fastapi import FastAPI


def create_api() -> FastAPI:
    app = FastAPI(title="RAG Agent API", version="1.0")

    # Import and include your API routes here
    from src.api.routes import router as api_router

    app.include_router(api_router, prefix="/api/v1")

    return app
