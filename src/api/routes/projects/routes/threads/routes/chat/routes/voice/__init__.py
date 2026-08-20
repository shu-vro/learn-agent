import asyncio

import edge_tts
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_session
from src.db.models.chat import Chat, ChatMessage
from src.lib.aws import (
    download_user_asset_bytes,
    upload_bytes_to_s3,
)
from src.config.voice_config import voice_dict, VoiceConfig

router = APIRouter(tags=["chats"])

VOICE_BUCKET = "uservoices"


def voice_s3_key(user_id: str, message_id: str) -> str:
    return f"{user_id}/{message_id}.mp3"


async def tts_stream(text: str, s3_key: str, voice: VoiceConfig = voice_dict["female"]):
    """Stream TTS audio to the client and persist the full clip to S3."""
    communicate = edge_tts.Communicate(text, voice.voice_name)
    parts: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            parts.append(chunk["data"])
            yield chunk["data"]

    if not parts:
        return

    audio = b"".join(parts)
    try:
        await asyncio.to_thread(
            upload_bytes_to_s3,
            audio,
            s3_key,
            content_type="audio/mpeg",
            bucket_name=VOICE_BUCKET,
        )
    except Exception as exc:  # playback already succeeded; don't fail the request
        logger.warning(f"failed to store voice clip {s3_key}: {exc}")


@router.get("/")
async def stream_voice(
    request: Request,
    chat_id: str,
    message_id: str,
    session: AsyncSession = Depends(get_session),
    voice_id: str = "female",
):
    user = request.state.user
    print(user)
    return None
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    query = (
        select(ChatMessage.message)
        .join(Chat, ChatMessage.chat_id == Chat.id)
        .where(
            ChatMessage.id == message_id,
            ChatMessage.chat_id == chat_id,
            Chat.user_id == user.id,
        )
    )

    res = await session.execute(query)
    message = res.scalar_one_or_none()

    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    headers = {
        "Content-Disposition": 'inline; filename="output.mp3"',
        "Cache-Control": "no-cache",
    }
    s3_key = voice_s3_key(user.id, message_id)

    try:
        audio, _ = await asyncio.to_thread(
            download_user_asset_bytes, s3_key, bucket_name=VOICE_BUCKET
        )
    except Exception:
        audio = None

    if audio:
        return StreamingResponse(
            iter([audio]), media_type="audio/mpeg", headers=headers
        )

    return StreamingResponse(
        tts_stream(message, s3_key),
        media_type="audio/mpeg",
        headers=headers,
    )


__all__ = ["router"]
