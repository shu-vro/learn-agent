import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# CHAT CONFIG
MAIN_CHAT_MAX_IMAGE_BYTES = int(
    os.environ.get("MAIN_CHAT_MAX_IMAGE_BYTES", 8388608)
)  # 8 MiB
MAIN_CHAT_MAX_IMAGES_PER_MESSAGE = int(
    os.environ.get("MAIN_CHAT_MAX_IMAGES_PER_MESSAGE", 6)
)


# Stored uploads for project-bound documents (PDF / markdown).
_default_upload_root = Path(__file__).resolve().parents[2] / "var" / "uploads"
ASSET_UPLOAD_ROOT = Path(
    os.environ.get("ASSET_UPLOAD_ROOT", str(_default_upload_root))
).resolve()

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# Postgres database connection parameters
DATABASE_PORT = os.environ.get("DATABASE_PORT", "5433")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "postgres")
DATABASE_USER = os.environ.get("DATABASE_USER", "pguser")
DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD", "")
DATABASE_HOST = os.environ.get("DATABASE_HOST", "localhost")

# JWT
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
)

# CORS
CORS_ALLOW_ORIGINS = os.environ.get(
    "CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")


# AWS
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "mock_key")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "mock_secret")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT", "http://localhost:4566")
AWS_S3_USE_PATH_STYLE = os.environ.get("AWS_S3_USE_PATH_STYLE", "true")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "my-disposable-assets-bucket")
AWS_S3_USER_ASSETS_BUCKET = os.environ.get("AWS_S3_USER_ASSETS_BUCKET", "userassets")
AWS_S3_USER_VOICES_BUCKET = os.environ.get("AWS_S3_USER_VOICES_BUCKET", "user_voices")
# Optional CDN / stable public base for userassets (no trailing slash).
# Example: https://cdn.example.com  or leave empty to use S3 endpoint URLs.
AWS_S3_USER_ASSETS_PUBLIC_BASE_URL = os.environ.get(
    "AWS_S3_USER_ASSETS_PUBLIC_BASE_URL", ""
).rstrip("/")
AWS_CLOUDFRONT_DOMAIN = os.environ.get("AWS_CLOUDFRONT_DOMAIN", "http://localhost:4566")
AWS_CLOUDFRONT_PUBLIC_KEY_ID = os.environ.get(
    "AWS_CLOUDFRONT_PUBLIC_KEY_ID", "YOUR_PUBLIC_KEY_ID_FROM_TERRAFORM"
)
AWS_CLOUDFRONT_PRIVATE_KEY_PATH = os.environ.get(
    "AWS_CLOUDFRONT_PRIVATE_KEY_PATH", "private_key.pem"
)

# Redis
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
