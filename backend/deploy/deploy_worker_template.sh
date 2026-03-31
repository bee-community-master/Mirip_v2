#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
REGION="${REGION:-asia-northeast3}"
ZONE="${ZONE:-asia-northeast3-b}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-mirip}"
IMAGE_NAME="${IMAGE_NAME:-mirip-v2-worker}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${BACKEND_DIR}" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
INSTANCE_TEMPLATE_NAME="${INSTANCE_TEMPLATE_NAME:-mirip-worker-template}"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL:-}"
SUBNETWORK="${SUBNETWORK:-}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-30GB}"
WORKER_MODE="${WORKER_MODE:-stub}"
DATA_BACKEND="${DATA_BACKEND:-firestore}"
STORAGE_BACKEND="${STORAGE_BACKEND:-fake}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required. Set PROJECT_ID or run gcloud config set project <project-id>." >&2
  exit 1
fi

if [[ -z "${SERVICE_ACCOUNT_EMAIL}" ]]; then
  echo "SERVICE_ACCOUNT_EMAIL is required." >&2
  exit 1
fi

ensure_repository() {
  if ! gcloud artifacts repositories describe "${ARTIFACT_REPOSITORY}" \
    --location "${REGION}" \
    --project "${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud artifacts repositories create "${ARTIFACT_REPOSITORY}" \
      --location "${REGION}" \
      --repository-format docker \
      --description "Mirip backend images" \
      --project "${PROJECT_ID}"
  fi
}

build_image() {
  local config_file
  config_file="$(mktemp)"
  cat > "${config_file}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - -f
      - deploy/Dockerfile.worker
      - -t
      - ${IMAGE_URI}
      - .
images:
  - ${IMAGE_URI}
EOF
  gcloud builds submit "${BACKEND_DIR}" \
    --config "${config_file}" \
    --project "${PROJECT_ID}"
  rm -f "${config_file}"
}

make_worker_env_file() {
  local env_file
  env_file="$(mktemp)"
  cat > "${env_file}" <<EOF
MIRIP_APP_ENV=prod
MIRIP_DATA_BACKEND=${DATA_BACKEND}
MIRIP_STORAGE_BACKEND=${STORAGE_BACKEND}
MIRIP_FIREBASE__PROJECT_ID=${FIREBASE_PROJECT_ID:-${PROJECT_ID}}
MIRIP_WORKER__MODE=${WORKER_MODE}
MIRIP_WORKER__RUN_ONCE=true
MIRIP_WORKER__LOCAL_MODEL_CACHE_DIR=${LOCAL_MODEL_CACHE_DIR:-/var/lib/mirip/model-cache}
MIRIP_COMPUTE__ENABLED=true
MIRIP_COMPUTE__PROJECT_ID=${PROJECT_ID}
MIRIP_COMPUTE__ZONE=${ZONE}
MIRIP_COMPUTE__INSTANCE_TEMPLATE=global/instanceTemplates/${INSTANCE_TEMPLATE_NAME}
EOF

  if [[ -n "${WORKER_MODEL_URI:-}" ]]; then
    echo "MIRIP_WORKER__MODEL_URI=${WORKER_MODEL_URI}" >> "${env_file}"
  fi

  if [[ "${STORAGE_BACKEND}" == "gcs" ]]; then
    echo "MIRIP_GCS__PROJECT_ID=${GCS_PROJECT_ID:-${PROJECT_ID}}" >> "${env_file}"
    echo "MIRIP_GCS__BUCKET_NAME=${GCS_BUCKET_NAME:?GCS_BUCKET_NAME is required when STORAGE_BACKEND=gcs}" >> "${env_file}"
  fi

  echo "${env_file}"
}

recreate_template() {
  local worker_env_file="$1"
  if gcloud compute instance-templates describe "${INSTANCE_TEMPLATE_NAME}" \
    --project "${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud compute instance-templates delete "${INSTANCE_TEMPLATE_NAME}" \
      --project "${PROJECT_ID}" \
      --quiet
  fi

  local args=(
    compute instance-templates create "${INSTANCE_TEMPLATE_NAME}"
    --project "${PROJECT_ID}"
    --machine-type "${MACHINE_TYPE}"
    --provisioning-model SPOT
    --instance-termination-action DELETE
    --maintenance-policy TERMINATE
    --service-account "${SERVICE_ACCOUNT_EMAIL}"
    --scopes https://www.googleapis.com/auth/cloud-platform
    --image-family cos-stable
    --image-project cos-cloud
    --boot-disk-size "${BOOT_DISK_SIZE}"
    --metadata "image_uri=${IMAGE_URI}"
    --metadata-from-file "startup-script=${SCRIPT_DIR}/spot-worker-startup.sh,worker-env=${worker_env_file}"
    --tags mirip-worker
  )

  if [[ -n "${SUBNETWORK}" ]]; then
    args+=(--subnet "${SUBNETWORK}" --no-address)
  fi

  gcloud "${args[@]}"
}

ensure_repository
build_image
WORKER_ENV_FILE="$(make_worker_env_file)"
recreate_template "${WORKER_ENV_FILE}"
rm -f "${WORKER_ENV_FILE}"

echo "INSTANCE_TEMPLATE_NAME=${INSTANCE_TEMPLATE_NAME}"
echo "IMAGE_URI=${IMAGE_URI}"
