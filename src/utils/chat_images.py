"""Validate and normalize chat image payloads (data URLs / raw base64)."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from src.config.env import MAIN_CHAT_MAX_IMAGE_BYTES, MAIN_CHAT_MAX_IMAGES_PER_MESSAGE

# Lossy WebP quality for durable S3 storage (good size/quality tradeoff).
_WEBP_QUALITY = 80
_WEBP_METHOD = 6

# data:image/png;base64,AAAA…  or  data:image/jpeg;base64,…
_DATA_URL = re.compile(
    r"^data:(image/(?:png|jpeg|jpg|webp|gif));base64,(.+)$",
    re.IGNORECASE | re.DOTALL,
)

_MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}

# Magic-byte sniffers (first bytes of decoded payload).
_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"RIFF", "image/webp"),  # refined below for WEBP
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]

MAX_IMAGES_PER_MESSAGE = MAIN_CHAT_MAX_IMAGES_PER_MESSAGE
MAX_IMAGE_BYTES = MAIN_CHAT_MAX_IMAGE_BYTES


@dataclass(frozen=True, slots=True)
class ValidatedChatImage:
    """Normalized image ready for multimodal prompts and S3 upload."""

    mime_type: str
    extension: str
    data_url: str
    raw_bytes: bytes


def _sniff_mime(raw: bytes) -> str | None:
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    for magic, mime in _MAGIC:
        if magic != b"RIFF" and raw.startswith(magic):
            return mime
    return None


def validate_chat_image(value: str) -> ValidatedChatImage:
    """Parse a data URL or raw base64 image; raise ValueError on failure."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Image payload must be a non-empty string")

    text = value.strip()
    mime: str | None = None
    b64_part = text

    match = _DATA_URL.match(text)
    if match:
        mime = match.group(1).lower()
        if mime == "image/jpg":
            mime = "image/jpeg"
        b64_part = match.group(2)

    # Strip whitespace/newlines often added by transports.
    b64_part = re.sub(r"\s+", "", b64_part)
    try:
        raw = base64.b64decode(b64_part, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid base64 image data") from exc

    if not raw:
        raise ValueError("Image data is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image exceeds max size of {MAX_IMAGE_BYTES // (1024 * 1024)} MiB"
        )

    sniffed = _sniff_mime(raw)
    if sniffed is None:
        raise ValueError("Unrecognized image format (expected png/jpeg/webp/gif)")
    if (
        mime
        and mime != sniffed
        and not (mime == "image/jpeg" and sniffed == "image/jpeg")
    ):
        # Trust magic bytes over a mismatched declared MIME.
        mime = sniffed
    mime = mime or sniffed

    extension = _MIME_TO_EXT.get(mime, "bin")
    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    return ValidatedChatImage(
        mime_type=mime,
        extension=extension,
        data_url=data_url,
        raw_bytes=raw,
    )


def validate_chat_images(values: list[str] | None) -> list[ValidatedChatImage]:
    if not values:
        return []
    if len(values) > MAX_IMAGES_PER_MESSAGE:
        raise ValueError(
            f"At most {MAX_IMAGES_PER_MESSAGE} images are allowed per message"
        )
    return [validate_chat_image(v) for v in values]


def convert_image_bytes_to_webp(
    raw: bytes,
    *,
    quality: int = _WEBP_QUALITY,
) -> bytes:
    """Re-encode image bytes as WebP for smaller durable storage.

    Animated sources are flattened to the first frame. Alpha is preserved when
    present; otherwise RGB is used.
    """
    if not raw:
        raise ValueError("Image data is empty")

    with Image.open(BytesIO(raw)) as img:
        img.load()
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            converted = img.convert("RGBA")
        else:
            converted = img.convert("RGB")

        buf = BytesIO()
        converted.save(
            buf,
            format="WEBP",
            quality=quality,
            method=_WEBP_METHOD,
        )
        return buf.getvalue()


def _sniff_mime_from_bytes(raw: bytes) -> str:
    sniffed = _sniff_mime(raw)
    if sniffed is None:
        raise ValueError("Unrecognized image format (expected png/jpeg/webp/gif)")
    return sniffed


def bytes_to_data_url(raw: bytes, mime_type: str | None = None) -> str:
    mime = mime_type or _sniff_mime_from_bytes(raw)
    if mime == "image/jpg":
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def upload_chat_images(
    images: list[ValidatedChatImage],
    *,
    user_id: str,
    message_id: str,
) -> list[str]:
    """Convert validated chat images to WebP and upload to S3, returning public URLs."""
    from src.lib.aws import generate_public_url, upload_bytes_to_s3, user_asset_s3_key

    urls: list[str] = []
    for index, image in enumerate(images):
        webp_bytes = convert_image_bytes_to_webp(image.raw_bytes)
        s3_key = user_asset_s3_key(user_id, message_id, index, extension="webp")
        upload_bytes_to_s3(
            webp_bytes,
            s3_key,
            content_type="image/webp",
            bucket_name="userassets",
            public=True,
        )
        urls.append(generate_public_url(s3_key, bucket_name="userassets"))
    return urls


def ensure_model_image_data_urls(urls: list[str] | None) -> list[str]:
    """Normalize image refs to data URIs for providers that reject remote URLs.

    OMLX only accepts ``data:image/...;base64,...``. OpenAI accepts both, so
    converting everything to data URIs is safe for either provider.
    """
    if not urls:
        return []

    from src.lib.aws import download_user_asset_bytes, parse_user_asset_public_url

    out: list[str] = []
    for value in urls:
        if not value or not str(value).strip():
            continue
        text = value.strip()
        if text.startswith("data:image/"):
            out.append(validate_chat_image(text).data_url)
            continue

        parsed = parse_user_asset_public_url(text)
        if parsed is not None:
            bucket_name, s3_key = parsed
            raw, content_type = download_user_asset_bytes(
                s3_key, bucket_name=bucket_name
            )
            mime = (
                content_type
                if content_type and content_type.startswith("image/")
                else None
            )
            out.append(bytes_to_data_url(raw, mime))
            continue

        # Last resort: HTTP(S) fetch (may fail for private hosts).
        if text.startswith("http://") or text.startswith("https://"):
            import urllib.request

            with urllib.request.urlopen(text, timeout=30) as resp:  # noqa: S310
                raw = resp.read()
                content_type = resp.headers.get("Content-Type")
            mime = (
                content_type.split(";")[0].strip()
                if content_type and content_type.startswith("image/")
                else None
            )
            out.append(bytes_to_data_url(raw, mime))
            continue

        # Treat as raw base64.
        out.append(validate_chat_image(text).data_url)

    if len(out) > MAX_IMAGES_PER_MESSAGE:
        raise ValueError(
            f"At most {MAX_IMAGES_PER_MESSAGE} images are allowed per message"
        )
    return out


__all__ = [
    "MAX_IMAGE_BYTES",
    "MAX_IMAGES_PER_MESSAGE",
    "ValidatedChatImage",
    "bytes_to_data_url",
    "convert_image_bytes_to_webp",
    "ensure_model_image_data_urls",
    "upload_chat_images",
    "validate_chat_image",
    "validate_chat_images",
]
