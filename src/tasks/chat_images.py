"""Persist chat-uploaded images to the userassets S3 bucket."""

from __future__ import annotations

import logging
from typing import Any

from src.db import sync_session_factory
from src.db.models.chat import ChatMessage
from src.lib.aws import (
    generate_public_url,
    upload_bytes_to_s3,
    user_asset_s3_key,
)
from src.lib.celery_lib import celery_app
from src.utils.chat_images import convert_image_bytes_to_webp, validate_chat_image

logger = logging.getLogger(__name__)


@celery_app.task(name="chat.save_user_images", bind=True, max_retries=0)
def save_chat_user_images(
    self,
    *,
    message_id: str,
    user_id: str,
    images: list[str],
) -> dict[str, Any]:
    """Decode base64 chat images, upload to S3, store stable public URLs on the message."""
    if not images:
        return {"message_id": message_id, "image_urls": []}

    uploaded_urls: list[str] = []
    for index, payload in enumerate(images):
        try:
            validated = validate_chat_image(payload)
            webp_bytes = convert_image_bytes_to_webp(validated.raw_bytes)
            s3_key = user_asset_s3_key(
                user_id,
                message_id,
                index,
                extension="webp",
            )
            upload_bytes_to_s3(
                webp_bytes,
                s3_key,
                content_type="image/webp",
                bucket_name="userassets",
                public=True,
            )
            uploaded_urls.append(generate_public_url(s3_key, bucket_name="userassets"))
        except Exception:
            logger.exception(
                "Failed to upload chat image %s for message_id=%s",
                index,
                message_id,
            )

    session = sync_session_factory()()
    try:
        message = session.get(ChatMessage, message_id)
        if message is None:
            logger.warning(
                "ChatMessage %s missing after image upload; urls=%s",
                message_id,
                uploaded_urls,
            )
            return {"message_id": message_id, "image_urls": uploaded_urls}

        message.image_urls = uploaded_urls or None
        session.commit()
        logger.info(
            "Saved %s chat image(s) for message_id=%s",
            len(uploaded_urls),
            message_id,
        )
        return {"message_id": message_id, "image_urls": uploaded_urls}
    except Exception:
        session.rollback()
        logger.exception("Failed to persist image_urls for message_id=%s", message_id)
        raise
    finally:
        session.close()


__all__ = ["save_chat_user_images"]
