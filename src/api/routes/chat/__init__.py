from fastapi import APIRouter


router = APIRouter()


@router.post("/v1/chat")
async def chat_endpoint():
    return {"message": "Hello, World!"}
