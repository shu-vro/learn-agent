output "user_assets_bucket_name" {
  description = "Name of the chat images bucket."
  value       = aws_s3_bucket.user_assets.id
}

output "user_assets_bucket_arn" {
  description = "ARN of the chat images bucket."
  value       = aws_s3_bucket.user_assets.arn
}

output "user_voices_bucket_name" {
  description = "Name of the read-aloud audio bucket."
  value       = aws_s3_bucket.user_voices.id
}

output "user_voices_bucket_arn" {
  description = "ARN of the read-aloud audio bucket."
  value       = aws_s3_bucket.user_voices.arn
}
