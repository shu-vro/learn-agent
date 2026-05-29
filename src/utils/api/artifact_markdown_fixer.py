import re
from pathlib import Path

from src.lib.aws import generate_disposable_url

_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\n]+)\)")


def _image_path_to_s3_key(doc_sha256: str, image_path: str) -> str:
    normalized = image_path.strip().replace("\\", "/")
    if normalized.startswith("artifacts/"):
        return normalized

    relative = normalized.lstrip("/")
    if relative.startswith("images/"):
        return f"artifacts/{doc_sha256}/{relative}"

    filename = Path(relative).name
    return f"artifacts/{doc_sha256}/images/{filename}"


def rewrite_chunk_image_urls(content: str, doc_sha256: str | None) -> str:
    """Replace local artifact image paths in markdown with CloudFront/S3 URLs."""
    if not doc_sha256 or not content:
        return content

    def _replacement(match: re.Match[str]) -> str:
        alt = match.group("alt")
        path = match.group("path").strip()
        if path.startswith(("http://", "https://")):
            return match.group(0)

        s3_key = _image_path_to_s3_key(doc_sha256, path)
        signed_url = generate_disposable_url(s3_key)
        return f"![{alt}]({signed_url})"

    return _MARKDOWN_IMAGE_PATTERN.sub(_replacement, content)
