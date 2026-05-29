variable "access_key" {
  description = "LocalStack access key (not used for real AWS)."
  type        = string
  default     = "mock_key"
  sensitive   = true
}

variable "secret_key" {
  description = "LocalStack secret key (not used for real AWS)."
  type        = string
  default     = "mock_secret"
  sensitive   = true
}

variable "project" {
  description = "Project name used in resource naming."
  type        = string
  default     = "learn-agent"
}

variable "aws_region" {
  description = "AWS region passed to the provider."
  type        = string
  default     = "us-east-1"
}

variable "localstack_endpoint" {
  description = "LocalStack gateway URL."
  type        = string
  default     = "http://localhost:4566"
}

variable "bucket_name" {
  description = "S3 bucket name (defaults to legacy local bucket for existing LocalStack state)."
  type        = string
  default     = "my-disposable-assets-bucket"
}
