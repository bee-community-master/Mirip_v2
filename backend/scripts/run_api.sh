#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run uvicorn mirip_backend.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
