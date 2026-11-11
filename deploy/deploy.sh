#!/usr/bin/env bash
# Deploy Supersede to Cloud Run in three steps:
#   1. Terraform: APIs, registry, network, service accounts (no services yet).
#   2. Cloud Build: build and push both images.
#   3. Terraform: both Cloud Run services, then a smoke test.
# Needs: gcloud (logged in: gcloud auth login && gcloud auth application-default login),
# terraform, and infra/terraform/terraform.tfvars (see terraform.tfvars.example).
# Usage: deploy/deploy.sh [tag]      (default tag v1; use a new tag to roll out code changes)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-v1}"
TF="terraform -chdir=infra/terraform"

$TF init -input=false
echo "== 1/3 base infrastructure"
$TF apply -input=false -auto-approve -var deploy_services=false -var image_tag="$TAG"

REGISTRY="$($TF output -raw registry)"
BUILD_SA="$($TF output -raw build_service_account)"
PROJECT="$(echo "$REGISTRY" | cut -d/ -f2)"

echo "== 2/3 images -> $REGISTRY (tag $TAG; the LLM image is ~7 GB, allow ~20 min)"
gcloud builds submit . --project "$PROJECT" --region "$(echo "$REGISTRY" | cut -d- -f1-2)" \
  --config deploy/cloudbuild.yaml --service-account "$BUILD_SA" \
  --substitutions "_REGISTRY=$REGISTRY,_TAG=$TAG"

echo "== 3/3 services"
$TF apply -input=false -auto-approve -var deploy_services=true -var image_tag="$TAG"

API="$($TF output -raw api_url)"
echo "API: $API"
curl -fsS "$API/healthz" && echo
curl -fsS "$API/search?q=TCP%20Congestion%20Control&k=3" | head -c 400 && echo
echo "Ambiguous-pair check (cold GPU start can take a minute or two):"
curl -fsS --max-time 300 -X POST "$API/assess" -H 'content-type: application/json' \
  -d '{"old": "RFC2246", "new": "RFC4346"}' && echo
