#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
REGION="${REGION:-asia-northeast3}"
ZONE="${ZONE:-asia-northeast3-b}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-mirip-runtime}"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
INSTANCE_TEMPLATE_NAME="${INSTANCE_TEMPLATE_NAME:-mirip-worker-template}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required. Set PROJECT_ID or run gcloud config set project <project-id>." >&2
  exit 1
fi

if [[ -z "${LOCAL_DEV_TOKEN:-}" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    LOCAL_DEV_TOKEN="$(openssl rand -hex 24)"
  else
    LOCAL_DEV_TOKEN="mirip-$(date +%s)"
  fi
fi
export LOCAL_DEV_TOKEN

PROJECT_ID="${PROJECT_ID}" \
REGION="${REGION}" \
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME}" \
ENSURE_FIRESTORE="${ENSURE_FIRESTORE:-true}" \
"${SCRIPT_DIR}/bootstrap_gcp.sh"

PROJECT_ID="${PROJECT_ID}" \
REGION="${REGION}" \
ZONE="${ZONE}" \
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL}" \
INSTANCE_TEMPLATE_NAME="${INSTANCE_TEMPLATE_NAME}" \
WORKER_MODE=stub \
DATA_BACKEND=firestore \
STORAGE_BACKEND=fake \
"${SCRIPT_DIR}/deploy_worker_template.sh"

PROJECT_ID="${PROJECT_ID}" \
REGION="${REGION}" \
ZONE="${ZONE}" \
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL}" \
WORKER_SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL}" \
INSTANCE_TEMPLATE_NAME="${INSTANCE_TEMPLATE_NAME}" \
DEPLOY_PROFILE=stub-spot \
LOCAL_DEV_TOKEN="${LOCAL_DEV_TOKEN}" \
"${SCRIPT_DIR}/deploy_api.sh"
