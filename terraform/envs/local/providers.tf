provider "aws" {
  region = var.aws_region

  # LocalStack only — do not use in production.
  access_key                  = var.access_key
  secret_key                  = var.secret_key
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  skip_region_validation      = true
  s3_use_path_style           = true

  endpoints {
    s3         = var.localstack_endpoint
    cloudfront = var.localstack_endpoint
  }
}
