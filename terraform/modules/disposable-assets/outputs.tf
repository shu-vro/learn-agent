output "bucket_name" {
  description = "Private S3 bucket name."
  value       = local.private_storage.id
}

output "bucket_arn" {
  description = "Private S3 bucket ARN."
  value       = local.private_storage.arn
}

output "cloudfront_domain" {
  description = "CloudFront distribution domain name."
  value       = aws_cloudfront_distribution.cdn.domain_name
}

output "cloudfront_distribution_arn" {
  description = "CloudFront distribution ARN."
  value       = aws_cloudfront_distribution.cdn.arn
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID."
  value       = aws_cloudfront_distribution.cdn.id
}

output "public_key_id" {
  description = "CloudFront public key ID for signed URLs (null when signing is disabled)."
  value       = local.enable_signing ? aws_cloudfront_public_key.signing_key[0].id : null
}

output "origin_access_control_id" {
  description = "CloudFront origin access control ID for the S3 origin."
  value       = aws_cloudfront_origin_access_control.s3_oac.id
}
