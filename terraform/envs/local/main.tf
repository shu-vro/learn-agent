module "disposable_assets" {
  source = "../../modules/disposable-assets"

  project     = var.project
  environment = "local"
  aws_region  = var.aws_region
  bucket_name = var.bucket_name

  use_localstack            = true
  enable_cloudfront_signing = false
  viewer_protocol_policy    = "allow-all"
  wait_for_deployment       = false
  enable_bucket_versioning  = false
}

module "user_storage" {
  source = "../../modules/user-storage"

  project     = var.project
  environment = "local"
  aws_region  = var.aws_region

  # Match the backend defaults so LocalStack works without extra env vars.
  user_assets_bucket_name = var.user_assets_bucket_name
  user_voices_bucket_name = var.user_voices_bucket_name

  cors_allowed_origins     = var.cors_allowed_origins
  enable_bucket_versioning = false
}

output "bucket_name" {
  value = module.disposable_assets.bucket_name
}

output "user_assets_bucket_name" {
  value = module.user_storage.user_assets_bucket_name
}

output "user_voices_bucket_name" {
  value = module.user_storage.user_voices_bucket_name
}

output "cloudfront_domain" {
  value = module.disposable_assets.cloudfront_domain
}

output "cloudfront_distribution_id" {
  value = module.disposable_assets.cloudfront_distribution_id
}

output "public_key_id" {
  value     = module.disposable_assets.public_key_id
  sensitive = true
}
