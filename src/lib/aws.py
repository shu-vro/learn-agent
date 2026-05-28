from datetime import datetime, timedelta, timezone
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
    AWS_CLOUDFRONT_DOMAIN,
    AWS_CLOUDFRONT_PUBLIC_KEY_ID,
    AWS_CLOUDFRONT_PRIVATE_KEY_PATH,
)

# Configuration (Use the output from your Terraform apply)
BUCKET_NAME = {"main": AWS_S3_BUCKET}
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


def upload_file_to_s3(local_path, s3_key, bucket_name="main"):
    """Uploads a file securely to S3. No public ACLs allowed."""
    print(f"Uploading {local_path} to S3 bucket...")
    s3_client.upload_file(local_path, BUCKET_NAME[bucket_name], s3_key)
    print("Upload Complete.")


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


if __name__ == "__main__":
    # 1. Prep a dummy file locally
    with open("secret_report.pdf", "w") as f:
        f.write("Classified data stream.")

    # 2. Push up to S3
    file_key = "reports/secret_report.pdf"
    upload_file_to_s3("secret_report.pdf", file_key)

    # 3. Generate a url valid for only 2 minutes
    disposable_url = generate_disposable_url(file_key, expires_in_minutes=2)
    print(f"\n[Generated Disposable Link]:\n{disposable_url}")
