from fastapi import APIRouter


router = APIRouter()


@router.post("/chats")
async def chat_endpoint():
    return {"message": "Hello, World!"}
