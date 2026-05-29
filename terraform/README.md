# Disposable assets (S3 + CloudFront)

Infrastructure is split by environment:

```text
terraform/
├── modules/disposable-assets/   # Shared S3, OAC, CloudFront, signing resources
├── envs/local/                  # LocalStack (mock credentials, path-style S3)
└── envs/prod/                   # Real AWS (remote state, ACM, signed URLs)
```

## Local (LocalStack)

```bash
cd terraform/envs/local
terraform init
terraform plan
terraform apply
```

Outputs feed application config:

- `bucket_name` → `AWS_S3_BUCKET`
- `cloudfront_domain` → `AWS_CLOUDFRONT_DOMAIN`
- `public_key_id` is `null` locally (app falls back to S3 presigned URLs)

LocalStack provider settings (mock keys, endpoint overrides) live only in `envs/local/providers.tf`.

## Production (AWS)

1. Create a **versioned** S3 bucket for Terraform state and update `envs/prod/backend.tf`.
2. Configure AWS credentials via profile, environment variables, or CI OIDC — never in HCL.
3. Issue an ACM certificate in **us-east-1** for your CloudFront aliases.
4. Copy `terraform.tfvars.example` → `terraform.tfvars` (gitignored) and set `project`, `acm_certificate_arn`, `cloudfront_aliases`.
5. Pass the CloudFront **public** key without committing it:

```bash
export TF_VAR_cloudfront_public_key="$(cat /path/to/public_key.pem)"
cd terraform/envs/prod
terraform init
terraform plan
terraform apply
```

Keep the **private** signing key out of Terraform state; store it in Secrets Manager / SSM and mount it for the app (`AWS_CLOUDFRONT_PRIVATE_KEY_PATH`).

## Production safeguards (module)

- S3 Block Public Access, SSE-S3, optional versioning
- Bucket policy allowing only the CloudFront distribution (OAC)
- `redirect-to-https` (configurable to `https-only`)
- `prevent_destroy` on the bucket in prod
- Named resources: `{project}-{environment}-*`

## Migrating from the old root `main.tf`

State was tied to `terraform/main.tf`. For local, either:

- `terraform state mv` after `init` in `envs/local`, or
- `terraform import` / fresh `apply` against LocalStack (dev-only)

Old root-level `terraform.tfstate*` files can be removed after migration.
