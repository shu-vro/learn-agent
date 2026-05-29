variable "project" {
  description = "Project name used in resource naming."
  type        = string
}

variable "environment" {
  description = "Deployment environment (e.g. local, staging, prod)."
  type        = string
}

variable "aws_region" {
  description = "AWS region for regional resources."
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "S3 bucket name. Defaults to {project}-{environment}-assets when empty."
  type        = string
  default     = ""
}

variable "use_localstack" {
  description = "Whether this stack targets LocalStack (affects CloudFront deployment wait behavior)."
  type        = bool
  default     = false
}

variable "enable_cloudfront_signing" {
  description = "Create CloudFront public key and key group for signed URLs."
  type        = bool
  default     = true
}

variable "cloudfront_public_key" {
  description = "PEM-encoded CloudFront public key for signed URLs. Pass via CI secrets in production."
  type        = string
  sensitive   = true
  default     = null
}

variable "viewer_protocol_policy" {
  description = "CloudFront viewer protocol policy (allow-all, redirect-to-https, https-only)."
  type        = string
  default     = "redirect-to-https"

  validation {
    condition     = contains(["allow-all", "redirect-to-https", "https-only"], var.viewer_protocol_policy)
    error_message = "viewer_protocol_policy must be allow-all, redirect-to-https, or https-only."
  }
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN in us-east-1 for custom CloudFront domain. Leave empty for default certificate."
  type        = string
  default     = ""
}

variable "cloudfront_aliases" {
  description = "Alternate domain names (CNAMEs) for the CloudFront distribution."
  type        = list(string)
  default     = []
}

variable "enable_bucket_versioning" {
  description = "Enable S3 bucket versioning."
  type        = bool
  default     = true
}

variable "wait_for_deployment" {
  description = "Wait for CloudFront distribution deployment to complete."
  type        = bool
  default     = true
}
