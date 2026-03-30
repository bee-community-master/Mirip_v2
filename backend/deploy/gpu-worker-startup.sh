#!/usr/bin/env bash
set -euo pipefail

IMAGE_URI="${IMAGE_URI:?IMAGE_URI is required}"
ENV_FILE="${ENV_FILE:-/etc/mirip/worker.env}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on the worker VM" >&2
  exit 1
fi

docker pull "${IMAGE_URI}"
docker run --rm --gpus all --env-file "${ENV_FILE}" "${IMAGE_URI}"
