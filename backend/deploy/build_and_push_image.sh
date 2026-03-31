#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${BACKEND_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
REGION="${REGION:-asia-northeast3}"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-${BACKEND_DIR}}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:?DOCKERFILE_PATH is required}"
IMAGE_URI="${IMAGE_URI:?IMAGE_URI is required}"
BUILD_STRATEGY="${BUILD_STRATEGY:-local-first}"
LOCAL_DOCKER_CONTEXT="${LOCAL_DOCKER_CONTEXT:-colima-x86}"
LOCAL_PLATFORM="${LOCAL_PLATFORM:-linux/amd64}"
COLIMA_PROFILE="${COLIMA_PROFILE:-x86}"
COLIMA_ARCH="${COLIMA_ARCH:-x86_64}"
WORKER_DEPENDENCY_PROFILE="${WORKER_DEPENDENCY_PROFILE:-}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required. Set PROJECT_ID or run gcloud config set project <project-id>." >&2
  exit 1
fi

log() {
  printf '[build] %s\n' "$*"
}

ensure_artifact_registry_auth() {
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet >/dev/null
}

ensure_local_docker_context() {
  if docker --context "${LOCAL_DOCKER_CONTEXT}" version >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v colima >/dev/null 2>&1; then
    echo "Local Docker context ${LOCAL_DOCKER_CONTEXT} is unavailable and colima is not installed." >&2
    return 1
  fi

  log "Starting colima profile ${COLIMA_PROFILE} for context ${LOCAL_DOCKER_CONTEXT}"
  if [[ "${COLIMA_PROFILE}" == "default" ]]; then
    colima start
  else
    colima start --profile "${COLIMA_PROFILE}" --arch "${COLIMA_ARCH}"
  fi

  docker --context "${LOCAL_DOCKER_CONTEXT}" version >/dev/null
}

build_local() {
  ensure_local_docker_context
  ensure_artifact_registry_auth

  log "Building ${IMAGE_URI} locally via ${LOCAL_DOCKER_CONTEXT} for ${LOCAL_PLATFORM}"
  local docker_args=(
    --context "${LOCAL_DOCKER_CONTEXT}" build
    --platform "${LOCAL_PLATFORM}"
    -f "${DOCKERFILE_PATH}"
    -t "${IMAGE_URI}"
  )
  if [[ -n "${WORKER_DEPENDENCY_PROFILE}" ]]; then
    docker_args+=(--build-arg "WORKER_DEPENDENCY_PROFILE=${WORKER_DEPENDENCY_PROFILE}")
  fi
  docker_args+=("${BUILD_CONTEXT_DIR}")
  docker "${docker_args[@]}"

  log "Pushing ${IMAGE_URI} to Artifact Registry"
  docker --context "${LOCAL_DOCKER_CONTEXT}" push "${IMAGE_URI}"
}

build_cloud() {
  local config_file
  config_file="$(mktemp)"
  cat > "${config_file}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - -f
      - ${DOCKERFILE_PATH}
EOF
  if [[ -n "${WORKER_DEPENDENCY_PROFILE}" ]]; then
    cat >> "${config_file}" <<EOF
      - --build-arg
      - WORKER_DEPENDENCY_PROFILE=${WORKER_DEPENDENCY_PROFILE}
EOF
  fi
  cat >> "${config_file}" <<EOF
      - -t
      - ${IMAGE_URI}
      - .
images:
  - ${IMAGE_URI}
EOF

  log "Falling back to Cloud Build for ${IMAGE_URI}"
  gcloud builds submit "${BUILD_CONTEXT_DIR}" \
    --config "${config_file}" \
    --project "${PROJECT_ID}"
  rm -f "${config_file}"
}

case "${BUILD_STRATEGY}" in
  local-first)
    if ! build_local; then
      build_cloud
      echo "BUILD_METHOD=cloud"
      echo "IMAGE_URI=${IMAGE_URI}"
      exit 0
    fi
    echo "BUILD_METHOD=local"
    ;;
  local-only)
    build_local
    echo "BUILD_METHOD=local"
    ;;
  cloud)
    build_cloud
    echo "BUILD_METHOD=cloud"
    ;;
  *)
    echo "Unsupported BUILD_STRATEGY: ${BUILD_STRATEGY}" >&2
    exit 1
    ;;
esac

echo "IMAGE_URI=${IMAGE_URI}"
