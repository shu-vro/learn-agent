module "disposable_assets" {
  source = "../../modules/disposable-assets"

  project    = var.project
  environment = var.environment
  aws_region = var.aws_region
  bucket_name = var.bucket_name

  cloudfront_public_key   = var.cloudfront_public_key
  acm_certificate_arn     = var.acm_certificate_arn
  cloudfront_aliases      = var.cloudfront_aliases
  viewer_protocol_policy  = var.viewer_protocol_policy

  use_localstack            = false
  enable_cloudfront_signing = true
  wait_for_deployment      = true
  enable_bucket_versioning = true
}

output "bucket_name" {
  value = module.disposable_assets.bucket_name
}

output "bucket_arn" {
  value = module.disposable_assets.bucket_arn
}

output "cloudfront_domain" {
  value = module.disposable_assets.cloudfront_domain
}

output "cloudfront_distribution_arn" {
  value = module.disposable_assets.cloudfront_distribution_arn
}

output "cloudfront_distribution_id" {
  value = module.disposable_assets.cloudfront_distribution_id
}

output "public_key_id" {
  value     = module.disposable_assets.public_key_id
  sensitive = false
}
