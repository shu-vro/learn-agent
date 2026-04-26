from fastapi import APIRouter
from src.api.routes.chat import router as chat_router

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


router.include_router(chat_router, prefix="/v1")
