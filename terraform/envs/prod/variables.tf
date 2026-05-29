variable "project" {
  description = "Project name used in resource naming."
  type        = string
}

variable "environment" {
  description = "Deployment environment (e.g. staging, prod)."
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS region for regional resources."
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Optional override for the S3 bucket name."
  type        = string
  default     = ""
}

variable "cloudfront_public_key" {
  description = "PEM-encoded CloudFront public key. Supply via TF_VAR, SSM, or Secrets Manager in CI — never commit."
  type        = string
  sensitive   = true
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN in us-east-1 for the CloudFront custom domain."
  type        = string
}

variable "cloudfront_aliases" {
  description = "DNS names for the CloudFront distribution (must match the ACM certificate)."
  type        = list(string)
}

variable "viewer_protocol_policy" {
  description = "CloudFront viewer protocol policy for production."
  type        = string
  default     = "redirect-to-https"

  validation {
    condition     = contains(["redirect-to-https", "https-only"], var.viewer_protocol_policy)
    error_message = "Production should use redirect-to-https or https-only."
  }
}
