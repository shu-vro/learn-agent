locals {
  bucket_name = var.bucket_name != "" ? var.bucket_name : "${var.project}-${var.environment}-assets"
  name_prefix = "${var.project}-${var.environment}"

  use_acm_certificate = var.acm_certificate_arn != ""
  is_prod_environment = var.environment == "prod"

  # The local emulator (floci 2.0.1) answers CreateDistribution/GetDistribution
  # with a DefaultCacheBehavior that drops ForwardedValues and MinTTL, which makes
  # the AWS provider panic on the read after create. Skip CloudFront there.
  enable_cloudfront = !var.use_localstack

  enable_signing = (
    local.enable_cloudfront
    && var.enable_cloudfront_signing
    && var.cloudfront_public_key != null
    && var.cloudfront_public_key != ""
  )
}

# Lifecycle prevent_destroy cannot use variables; prod uses a separate resource with literal true.
resource "aws_s3_bucket" "private_storage" {
  count  = local.is_prod_environment ? 0 : 1
  bucket = local.bucket_name
}

resource "aws_s3_bucket" "private_storage_prod" {
  count  = local.is_prod_environment ? 1 : 0
  bucket = local.bucket_name

  lifecycle {
    prevent_destroy = true
  }
}

locals {
  private_storage = local.is_prod_environment ? aws_s3_bucket.private_storage_prod[0] : aws_s3_bucket.private_storage[0]
}

resource "aws_s3_bucket_public_access_block" "private_storage" {
  bucket = local.private_storage.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "private_storage" {
  bucket = local.private_storage.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "private_storage" {
  count  = var.enable_bucket_versioning ? 1 : 0
  bucket = local.private_storage.id

  versioning_configuration {
    status = "Suspended" # or Enabled
  }
}

resource "aws_cloudfront_public_key" "signing_key" {
  count       = local.enable_signing ? 1 : 0
  comment     = "Key used to sign disposable URLs (${local.name_prefix})"
  encoded_key = var.cloudfront_public_key
  name        = "${local.name_prefix}-signing-key"
}

resource "aws_cloudfront_key_group" "signing_group" {
  count   = local.enable_signing ? 1 : 0
  comment = "Key group for backend signers (${local.name_prefix})"
  items   = [aws_cloudfront_public_key.signing_key[0].id]
  name    = "${local.name_prefix}-signer-group"
}

resource "aws_cloudfront_origin_access_control" "s3_oac" {
  count                             = local.enable_cloudfront ? 1 : 0
  name                              = "${local.name_prefix}-s3-oac"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "cdn" {
  count               = local.enable_cloudfront ? 1 : 0
  enabled             = true
  is_ipv6_enabled     = true
  wait_for_deployment = var.wait_for_deployment
  aliases             = local.use_acm_certificate ? var.cloudfront_aliases : []

  origin {
    domain_name              = local.private_storage.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.s3_oac[0].id
    origin_id                = "S3Origin"
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3Origin"
    viewer_protocol_policy = var.viewer_protocol_policy
    trusted_key_groups     = local.enable_signing ? [aws_cloudfront_key_group.signing_group[0].id] : []

    forwarded_values {
      query_string = true
      cookies {
        forward = "none"
      }
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn            = local.use_acm_certificate ? var.acm_certificate_arn : null
    cloudfront_default_certificate = local.use_acm_certificate ? false : true
    ssl_support_method             = local.use_acm_certificate ? "sni-only" : null
    minimum_protocol_version       = local.use_acm_certificate ? "TLSv1.2_2021" : null
  }
}

resource "aws_s3_bucket_policy" "allow_cloudfront" {
  count  = local.enable_cloudfront ? 1 : 0
  bucket = local.private_storage.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCloudFrontServicePrincipalReadOnly"
        Effect = "Allow"
        Principal = {
          Service = "cloudfront.amazonaws.com"
        }
        Action   = "s3:GetObject"
        Resource = "${local.private_storage.arn}/*"
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = aws_cloudfront_distribution.cdn[0].arn
          }
        }
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.private_storage]
}
