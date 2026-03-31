#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
REGION="${REGION:-asia-northeast3}"
SERVICE_NAME="${SERVICE_NAME:-mirip-v2-api}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-mirip}"
IMAGE_NAME="${IMAGE_NAME:-mirip-v2-api}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${BACKEND_DIR}" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
DEPLOY_PROFILE="${DEPLOY_PROFILE:-preview}"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL:-}"
CORS_ORIGINS_JSON="${CORS_ORIGINS_JSON:-[\"http://localhost:5173\"]}"
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-true}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
CPU="${CPU:-1}"
MEMORY="${MEMORY:-1Gi}"
TIMEOUT="${TIMEOUT:-60s}"
INGRESS="${INGRESS:-all}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required. Set PROJECT_ID or run gcloud config set project <project-id>." >&2
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
      - deploy/Dockerfile.api
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

append_env() {
  local file_path="$1"
  local key="$2"
  local value="$3"
  printf '%s: "%s"\n' "${key}" "${value//\"/\\\"}" >> "${file_path}"
}

make_env_file() {
  local env_file
  env_file="$(mktemp)"
  append_env "${env_file}" "MIRIP_APP_ENV" "prod"
  append_env "${env_file}" "MIRIP_API__CORS_ORIGINS" "${CORS_ORIGINS_JSON}"
  append_env "${env_file}" "MIRIP_OBSERVABILITY__JSON_LOGS" "true"
  append_env "${env_file}" "MIRIP_OBSERVABILITY__LOG_LEVEL" "${LOG_LEVEL:-INFO}"

  case "${DEPLOY_PROFILE}" in
    preview)
      append_env "${env_file}" "MIRIP_DATA_BACKEND" "memory"
      append_env "${env_file}" "MIRIP_STORAGE_BACKEND" "fake"
      append_env "${env_file}" "MIRIP_WORKER__MODE" "stub"
      ;;
    stub-spot)
      append_env "${env_file}" "MIRIP_DATA_BACKEND" "firestore"
      append_env "${env_file}" "MIRIP_STORAGE_BACKEND" "fake"
      append_env "${env_file}" "MIRIP_WORKER__MODE" "stub"
      append_env "${env_file}" "MIRIP_FIREBASE__PROJECT_ID" "${FIREBASE_PROJECT_ID:-${PROJECT_ID}}"
      append_env "${env_file}" "MIRIP_FIREBASE__ALLOW_INSECURE_DEV_AUTH" "${ALLOW_INSECURE_DEV_AUTH_FOR_DEV:-true}"
      append_env "${env_file}" "MIRIP_FIREBASE__LOCAL_DEV_TOKEN" "${LOCAL_DEV_TOKEN:?LOCAL_DEV_TOKEN is required for stub-spot profile}"
      append_env "${env_file}" "MIRIP_COMPUTE__ENABLED" "true"
      append_env "${env_file}" "MIRIP_COMPUTE__PROJECT_ID" "${PROJECT_ID}"
      append_env "${env_file}" "MIRIP_COMPUTE__ZONE" "${ZONE:?ZONE is required for stub-spot profile}"
      append_env "${env_file}" "MIRIP_COMPUTE__INSTANCE_TEMPLATE" "global/instanceTemplates/${INSTANCE_TEMPLATE_NAME:?INSTANCE_TEMPLATE_NAME is required for stub-spot profile}"
      append_env "${env_file}" "MIRIP_COMPUTE__SERVICE_ACCOUNT" "${WORKER_SERVICE_ACCOUNT_EMAIL:-${SERVICE_ACCOUNT_EMAIL}}"
      if [[ -n "${SUBNETWORK:-}" ]]; then
        append_env "${env_file}" "MIRIP_COMPUTE__SUBNETWORK" "${SUBNETWORK}"
      fi
      ;;
    full)
      append_env "${env_file}" "MIRIP_DATA_BACKEND" "firestore"
      append_env "${env_file}" "MIRIP_STORAGE_BACKEND" "gcs"
      append_env "${env_file}" "MIRIP_WORKER__MODE" "${WORKER_MODE:-cpu_onnx}"
      append_env "${env_file}" "MIRIP_WORKER__MODEL_URI" "${WORKER_MODEL_URI:?WORKER_MODEL_URI is required for full profile}"
      append_env "${env_file}" "MIRIP_FIREBASE__PROJECT_ID" "${FIREBASE_PROJECT_ID:-${PROJECT_ID}}"
      append_env "${env_file}" "MIRIP_GCS__PROJECT_ID" "${GCS_PROJECT_ID:-${PROJECT_ID}}"
      append_env "${env_file}" "MIRIP_GCS__BUCKET_NAME" "${GCS_BUCKET_NAME:?GCS_BUCKET_NAME is required for full profile}"
      append_env "${env_file}" "MIRIP_COMPUTE__ENABLED" "true"
      append_env "${env_file}" "MIRIP_COMPUTE__PROJECT_ID" "${PROJECT_ID}"
      append_env "${env_file}" "MIRIP_COMPUTE__ZONE" "${ZONE:?ZONE is required for full profile}"
      append_env "${env_file}" "MIRIP_COMPUTE__INSTANCE_TEMPLATE" "global/instanceTemplates/${INSTANCE_TEMPLATE_NAME:?INSTANCE_TEMPLATE_NAME is required for full profile}"
      append_env "${env_file}" "MIRIP_COMPUTE__SERVICE_ACCOUNT" "${WORKER_SERVICE_ACCOUNT_EMAIL:-${SERVICE_ACCOUNT_EMAIL}}"
      if [[ -n "${SUBNETWORK:-}" ]]; then
        append_env "${env_file}" "MIRIP_COMPUTE__SUBNETWORK" "${SUBNETWORK}"
      fi
      ;;
    *)
      echo "Unsupported DEPLOY_PROFILE: ${DEPLOY_PROFILE}" >&2
      exit 1
      ;;
  esac

  echo "${env_file}"
}

deploy_service() {
  local env_file="$1"
  local allow_flag="--no-allow-unauthenticated"
  if [[ "${ALLOW_UNAUTHENTICATED}" == "true" ]]; then
    allow_flag="--allow-unauthenticated"
  fi

  local args=(
    run deploy "${SERVICE_NAME}"
    --project "${PROJECT_ID}"
    --region "${REGION}"
    --image "${IMAGE_URI}"
    --platform managed
    --port 8080
    --execution-environment gen2
    --concurrency 80
    --min-instances "${MIN_INSTANCES}"
    --max-instances "${MAX_INSTANCES}"
    --cpu "${CPU}"
    --memory "${MEMORY}"
    --timeout "${TIMEOUT}"
    --ingress "${INGRESS}"
    --env-vars-file "${env_file}"
    "${allow_flag}"
    --format "value(status.url)"
  )

  if [[ -n "${SERVICE_ACCOUNT_EMAIL}" ]]; then
    args+=(--service-account "${SERVICE_ACCOUNT_EMAIL}")
  fi

  gcloud "${args[@]}"
}

ensure_repository
build_image
ENV_FILE="$(make_env_file)"
SERVICE_URL="$(deploy_service "${ENV_FILE}")"
rm -f "${ENV_FILE}"

echo "SERVICE_URL=${SERVICE_URL}"
echo "IMAGE_URI=${IMAGE_URI}"
if [[ -n "${LOCAL_DEV_TOKEN:-}" ]]; then
  echo "LOCAL_DEV_TOKEN=${LOCAL_DEV_TOKEN}"
fi
