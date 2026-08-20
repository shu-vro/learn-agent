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

variable "user_assets_bucket_name" {
  description = "Bucket holding chat images. Defaults to {project}-{environment}-userassets when empty."
  type        = string
  default     = ""
}

variable "user_voices_bucket_name" {
  description = "Bucket holding read-aloud audio clips. Defaults to {project}-{environment}-user-voices when empty."
  type        = string
  default     = ""
}

variable "cors_allowed_origins" {
  description = "Origins allowed to read chat images directly from S3."
  type        = list(string)
  default     = ["*"]
}

variable "enable_bucket_versioning" {
  description = "Enable S3 bucket versioning on both buckets."
  type        = bool
  default     = false
}
