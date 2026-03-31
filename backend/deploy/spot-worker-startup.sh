#!/usr/bin/env bash
set -euo pipefail

BASE_ENV_FILE="${BASE_ENV_FILE:-/etc/mirip/worker.env}"
GENERATED_ENV_FILE="${GENERATED_ENV_FILE:-/tmp/mirip-worker.env}"
METADATA_ROOT="http://metadata.google.internal/computeMetadata/v1/instance/attributes"

metadata() {
  local key="$1"
  curl -fsS -H "Metadata-Flavor: Google" "${METADATA_ROOT}/${key}" 2>/dev/null || true
}

IMAGE_URI="${IMAGE_URI:-$(metadata image_uri)}"
if [[ -z "${IMAGE_URI}" ]]; then
  echo "IMAGE_URI is required either as an environment variable or instance metadata" >&2
  exit 1
fi

if command -v systemctl >/dev/null 2>&1; then
  systemctl start docker >/dev/null 2>&1 || true
fi

for _ in {1..30}; do
  if command -v docker >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on the worker VM" >&2
  exit 1
fi

{
  if [[ -f "${BASE_ENV_FILE}" ]]; then
    cat "${BASE_ENV_FILE}"
  else
    metadata worker-env
  fi
  echo "MIRIP_WORKER__RUN_ONCE=$(metadata run_once)"
  echo "MIRIP_WORKER__TARGET_JOB_ID=$(metadata target_job_id)"
  echo "MIRIP_WORKER__MODE=$(metadata worker_mode)"
  echo "MIRIP_WORKER__MODEL_URI=$(metadata model_uri)"
  echo "MIRIP_COMPUTE__INSTANCE_NAME=$(metadata instance_name)"
  echo "MIRIP_COMPUTE__ZONE=$(metadata zone)"
} > "${GENERATED_ENV_FILE}"

docker pull "${IMAGE_URI}"
docker run --rm --env-file "${GENERATED_ENV_FILE}" "${IMAGE_URI}"
