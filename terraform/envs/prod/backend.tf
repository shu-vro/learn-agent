terraform {
  backend "s3" {
    bucket       = "CHANGE_ME-terraform-state-prod"
    key          = "disposable-assets/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}
