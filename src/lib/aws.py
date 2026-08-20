from datetime import datetime, timedelta, timezone
from pathlib import Path
import mimetypes

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import boto3
from botocore.config import Config
from botocore.signers import CloudFrontSigner

from src.config.env import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    AWS_S3_ENDPOINT,
    AWS_S3_USE_PATH_STYLE,
    AWS_S3_BUCKET,
    AWS_S3_USER_ASSETS_BUCKET,
    AWS_S3_USER_VOICES_BUCKET,
    AWS_S3_USER_ASSETS_PUBLIC_BASE_URL,
    AWS_CLOUDFRONT_DOMAIN,
    AWS_CLOUDFRONT_PUBLIC_KEY_ID,
    AWS_CLOUDFRONT_PRIVATE_KEY_PATH,
)

# Configuration (Use the output from your Terraform apply)
BUCKET_NAME = {
    "main": AWS_S3_BUCKET,
    "userassets": AWS_S3_USER_ASSETS_BUCKET,
    "uservoices": AWS_S3_USER_VOICES_BUCKET,
}
CLOUDFRONT_DOMAIN = AWS_CLOUDFRONT_DOMAIN
KEY_ID = AWS_CLOUDFRONT_PUBLIC_KEY_ID
PRIVATE_KEY_PATH = AWS_CLOUDFRONT_PRIVATE_KEY_PATH
EXPIRES_IN_MINUTES = 5

# Initialize local S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url=AWS_S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
    config=Config(
        signature_version="s3v4",
        s3={
            "addressing_style": (
                "path" if str(AWS_S3_USE_PATH_STYLE).lower() == "true" else "virtual"
            )
        },
    ),
)


def _is_local_endpoint() -> bool:
    """True when S3 traffic goes to LocalStack / a custom path-style endpoint."""
    endpoint = (AWS_S3_ENDPOINT or "").rstrip("/")
    if not endpoint:
        return False
    return (
        "localhost" in endpoint
        or "127.0.0.1" in endpoint
        or "localstack" in endpoint.lower()
        or str(AWS_S3_USE_PATH_STYLE).lower() == "true"
    )


def upload_file_to_s3(
    local_path: str | Path,
    s3_key: str,
    bucket_name: str = "main",
) -> str:
    """Upload a file to S3. No public ACLs are applied."""
    path = Path(local_path)
    bucket = BUCKET_NAME[bucket_name]
    extra_args: dict[str, str] = {}
    content_type, _ = mimetypes.guess_type(path.name)
    if content_type:
        extra_args["ContentType"] = content_type

    upload_kwargs: dict[str, object] = {}
    if extra_args:
        upload_kwargs["ExtraArgs"] = extra_args

    s3_client.upload_file(str(path), bucket, s3_key, **upload_kwargs)
    return s3_key


def upload_bytes_to_s3(
    body: bytes,
    s3_key: str,
    *,
    content_type: str = "application/octet-stream",
    bucket_name: str = "userassets",
    public: bool = False,
) -> str:
    """Upload in-memory bytes to S3.

    When ``public=True``, objects are uploaded with a public-read ACL so
    ``generate_public_url`` links remain stable (non-expiring).
    """
    bucket = BUCKET_NAME[bucket_name]
    put_kwargs: dict[str, object] = {
        "Bucket": bucket,
        "Key": s3_key,
        "Body": body,
        "ContentType": content_type,
    }
    if public:
        put_kwargs["ACL"] = "public-read"
    s3_client.put_object(**put_kwargs)
    return s3_key


def generate_public_url(s3_key: str, bucket_name: str = "userassets") -> str:
    """Build a stable, non-expiring public URL for an S3 object.

    Prefer a configured public base (CDN / CloudFront). Otherwise use the
    S3 endpoint in path-style (LocalStack) or virtual-hosted AWS style.
    """
    bucket = BUCKET_NAME[bucket_name]
    key = s3_key.lstrip("/")

    # Optional override, e.g. https://cdn.example.com or https://dxxx.cloudfront.net
    if AWS_S3_USER_ASSETS_PUBLIC_BASE_URL and bucket_name == "userassets":
        return f"{AWS_S3_USER_ASSETS_PUBLIC_BASE_URL}/{key}"

    # Local / custom endpoint (LocalStack): path-style public URL.
    if _is_local_endpoint():
        endpoint = (AWS_S3_ENDPOINT or "").rstrip("/")
        return f"{endpoint}/{bucket}/{key}"

    # Standard AWS virtual-hosted–style public URL.
    return f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{key}"


def parse_user_asset_public_url(url: str) -> tuple[str, str] | None:
    """If ``url`` points at a known bucket object, return ``(bucket_alias, key)``."""
    text = (url or "").strip()
    if not text:
        return None

    for alias, bucket in BUCKET_NAME.items():
        # Path-style: http://endpoint/bucket/key
        marker = f"/{bucket}/"
        idx = text.find(marker)
        if idx >= 0:
            return alias, text[idx + len(marker) :].split("?", 1)[0]

        # Virtual-hosted: https://bucket.s3.region.amazonaws.com/key
        host_marker = f"https://{bucket}.s3."
        if text.startswith(host_marker):
            path = text.split("://", 1)[1].split("/", 1)
            if len(path) == 2:
                return alias, path[1].split("?", 1)[0]

        if AWS_S3_USER_ASSETS_PUBLIC_BASE_URL and alias == "userassets":
            base = AWS_S3_USER_ASSETS_PUBLIC_BASE_URL.rstrip("/") + "/"
            if text.startswith(base):
                return alias, text[len(base) :].split("?", 1)[0]

    return None


def download_user_asset_bytes(
    s3_key: str,
    *,
    bucket_name: str = "userassets",
) -> tuple[bytes, str | None]:
    """Download object bytes and optional ContentType from S3."""
    bucket = BUCKET_NAME[bucket_name]
    obj = s3_client.get_object(Bucket=bucket, Key=s3_key.lstrip("/"))
    body = obj["Body"].read()
    content_type = obj.get("ContentType")
    return body, content_type if isinstance(content_type, str) else None


def user_asset_s3_key(
    user_id: str,
    message_id: str,
    index: int,
    *,
    extension: str = "png",
) -> str:
    ext = extension.lstrip(".").lower() or "bin"
    return f"userassets/{user_id}/{message_id}/{index}.{ext}"


def upload_artifacts_directory_to_s3(
    local_dir: str | Path,
    doc_id: str,
    *,
    s3_prefix: str = "artifacts",
    bucket_name: str = "main",
) -> list[str]:
    """Upload a hashed artifact directory to S3, preserving relative paths."""
    root = Path(local_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {root}")

    uploaded_keys: list[str] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        relative_key = file_path.relative_to(root).as_posix()
        s3_key = f"{s3_prefix}/{doc_id}/{relative_key}"
        upload_file_to_s3(file_path, s3_key, bucket_name=bucket_name)
        uploaded_keys.append(s3_key)

    return uploaded_keys


def artifact_s3_prefix(doc_id: str, *, s3_prefix: str = "artifacts") -> str:
    return f"{s3_prefix}/{doc_id}"


def artifact_markdown_s3_key(doc_id: str, *, s3_prefix: str = "artifacts") -> str:
    return f"{artifact_s3_prefix(doc_id, s3_prefix=s3_prefix)}/{doc_id}.md"


def artifact_pdf_s3_key(doc_id: str, *, s3_prefix: str = "artifacts") -> str:
    return f"{artifact_s3_prefix(doc_id, s3_prefix=s3_prefix)}/{doc_id}.pdf"


def rsa_signer(message):
    """Cryptographically sign the CloudFront policy structure using your private key."""
    with open(PRIVATE_KEY_PATH, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(), password=None, backend=default_backend()
        )
    return private_key.sign(message, padding.PKCS1v15(), hashes.SHA1())


def generate_disposable_url(
    s3_key, expires_in_minutes=EXPIRES_IN_MINUTES, bucket_name="main"
):
    """Generates a secure, expiring link for object download."""
    # LocalStack generally cannot emulate CloudFront signed URL flow end-to-end.
    # Fallback to native S3 pre-signed URL when CloudFront signing is not configured.
    if (
        not KEY_ID
        or KEY_ID == "YOUR_PUBLIC_KEY_ID_FROM_TERRAFORM"
        or "localhost" in str(CLOUDFRONT_DOMAIN).lower()
    ):
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET_NAME[bucket_name], "Key": s3_key},
            ExpiresIn=expires_in_minutes * 60,
        )

    # Build complete resource locator path
    # Locally, it mirrors: http://localhost:4566/cloudfront/<dist_id>/path/to/object
    resource_url = f"{CLOUDFRONT_DOMAIN}/cloudfront/DistributionIdPlaceholder/{BUCKET_NAME[bucket_name]}/{s3_key}"

    # Calculate target time boundary
    expire_epoch = int(
        (datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)).timestamp()
    )

    # Custom canned policy layout mapping
    cloudfront_signer = CloudFrontSigner(KEY_ID, rsa_signer)

    signed_url = cloudfront_signer.generate_presigned_url(
        resource_url,
        date_less_than=datetime.fromtimestamp(expire_epoch, tz=timezone.utc),
    )
    return signed_url
