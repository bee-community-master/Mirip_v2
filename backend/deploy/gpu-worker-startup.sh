#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-/etc/mirip/worker.env}"
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

if [[ ! -f "${ENV_FILE}" ]]; then
  metadata worker-env > "${ENV_FILE}"
fi

if command -v systemctl >/dev/null 2>&1; then
  systemctl start docker >/dev/null 2>&1 || true
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on the worker VM" >&2
  exit 1
fi

docker pull "${IMAGE_URI}"
docker run --rm --gpus all --env-file "${ENV_FILE}" "${IMAGE_URI}"
