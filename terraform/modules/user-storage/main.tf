locals {
  user_assets_bucket = (
    var.user_assets_bucket_name != ""
    ? var.user_assets_bucket_name
    : "${var.project}-${var.environment}-userassets"
  )
  user_voices_bucket = (
    var.user_voices_bucket_name != ""
    ? var.user_voices_bucket_name
    : "${var.project}-${var.environment}-user-voices"
  )
}

# ---------------------------------------------------------------------------
# Chat images. The browser fetches these straight from S3, so objects are
# uploaded with a public-read ACL and the bucket must permit that.
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "user_assets" {
  bucket = local.user_assets_bucket
}

# ACLs are rejected under the AWS default (BucketOwnerEnforced); the app sets
# public-read per object, so ownership has to allow ACLs.
resource "aws_s3_bucket_ownership_controls" "user_assets" {
  bucket = aws_s3_bucket.user_assets.id

  rule {
    object_ownership = "ObjectWriter"
  }
}

resource "aws_s3_bucket_public_access_block" "user_assets" {
  bucket = aws_s3_bucket.user_assets.id

  # Public object ACLs are the delivery mechanism for chat images, so the ACL
  # blocks stay off. Bucket-wide public policies remain blocked.
  block_public_acls       = false
  ignore_public_acls      = false
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_cors_configuration" "user_assets" {
  bucket = aws_s3_bucket.user_assets.id

  cors_rule {
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = var.cors_allowed_origins
    allowed_headers = ["*"]
    max_age_seconds = 3600
  }
}

resource "aws_s3_bucket_versioning" "user_assets" {
  count  = var.enable_bucket_versioning ? 1 : 0
  bucket = aws_s3_bucket.user_assets.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ---------------------------------------------------------------------------
# Read-aloud audio clips. Served only through the authenticated API, never
# directly to a browser, so this bucket stays fully private.
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "user_voices" {
  bucket = local.user_voices_bucket
}

resource "aws_s3_bucket_public_access_block" "user_voices" {
  bucket = aws_s3_bucket.user_voices.id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "user_voices" {
  bucket = aws_s3_bucket.user_voices.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "user_voices" {
  count  = var.enable_bucket_versioning ? 1 : 0
  bucket = aws_s3_bucket.user_voices.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Clips are a regenerable cache of the TTS output — expire them instead of
# paying to store every message forever.
resource "aws_s3_bucket_lifecycle_configuration" "user_voices" {
  bucket = aws_s3_bucket.user_voices.id

  rule {
    id     = "expire-voice-clips"
    status = "Enabled"

    filter {}

    expiration {
      days = 90
    }
  }
}
