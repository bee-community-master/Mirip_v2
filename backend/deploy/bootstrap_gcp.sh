#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
REGION="${REGION:-asia-northeast3}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-mirip}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-mirip-runtime}"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
FIRESTORE_LOCATION="${FIRESTORE_LOCATION:-asia-northeast3}"
ENSURE_FIRESTORE="${ENSURE_FIRESTORE:-true}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required. Set PROJECT_ID or run gcloud config set project <project-id>." >&2
  exit 1
fi

echo "Bootstrapping GCP project ${PROJECT_ID} in ${REGION}..."

gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  compute.googleapis.com \
  firestore.googleapis.com \
  iam.googleapis.com \
  run.googleapis.com \
  --project "${PROJECT_ID}"

if ! gcloud artifacts repositories describe "${ARTIFACT_REPOSITORY}" \
  --location "${REGION}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${ARTIFACT_REPOSITORY}" \
    --location "${REGION}" \
    --repository-format docker \
    --description "Mirip backend images" \
    --project "${PROJECT_ID}"
fi

if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --display-name "Mirip runtime" \
    --project "${PROJECT_ID}"
fi

for role in \
  roles/artifactregistry.reader \
  roles/compute.instanceAdmin.v1 \
  roles/datastore.user \
  roles/iam.serviceAccountUser \
  roles/logging.logWriter \
  roles/monitoring.metricWriter \
  roles/storage.objectAdmin
do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role "${role}" >/dev/null
done

if [[ "${ENSURE_FIRESTORE}" == "true" ]]; then
  if ! gcloud firestore databases describe \
    --database='(default)' \
    --project "${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud firestore databases create \
      --database='(default)' \
      --location "${FIRESTORE_LOCATION}" \
      --project "${PROJECT_ID}"
  fi
fi

echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "ARTIFACT_REPOSITORY=${ARTIFACT_REPOSITORY}"
echo "SERVICE_ACCOUNT_EMAIL=${SERVICE_ACCOUNT_EMAIL}"
