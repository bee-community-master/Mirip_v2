# Mirip v2 Backend

FastAPI backend scaffold for Mirip v2.

## Architecture

- Cloud Run API for authenticated REST endpoints
- Firebase Auth token verification
- Firestore as the primary domain datastore
- Google Cloud Storage for uploads and generated assets
- Spot GPU worker for async diagnosis jobs
- Spot VM worker orchestration for one-job-per-instance CPU ONNX inference

## Quick Start

```bash
uv sync --group dev
uv run uvicorn mirip_backend.api.app:create_app --factory --reload
```

## Local GCS Setup

This pass supports a local runtime with:

- `MIRIP_DATA_BACKEND=memory`
- `MIRIP_STORAGE_BACKEND=gcs`
- mocked auth enabled for local/test
- stub worker inference only

Use [`backend/.env.example`](./.env.example) as the starting point. The intended GCP setup is:

- project display name: `mirip_v2`
- project ID: `mirip-v2`
- one GCS bucket such as `mirip-v2-assets`
- one service account JSON with GCS object permissions

Set `MIRIP_GCS__PROJECT_ID`, `MIRIP_GCS__BUCKET_NAME`, and `MIRIP_GCS__CREDENTIALS_PATH` in `.env`.
`MIRIP_FIREBASE__ALLOW_INSECURE_DEV_AUTH=true` keeps authenticated routes usable without an
`Authorization` header in local/test mode.

This mode is for local API validation only. Memory-backed domain data is not shared across
processes, so multi-process deployments still need Firestore for durable API/worker coordination.

## Commands

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run python -m mirip_backend.worker.main
uv lock
```

## Environment

Key settings are grouped around:

- `MIRIP_API__*`
- `MIRIP_FIREBASE__*`
- `MIRIP_GCS__*`
- `MIRIP_JOB__*`
- `MIRIP_WORKER__*`
- `MIRIP_COMPUTE__*`
- `MIRIP_OBSERVABILITY__*`

See `src/mirip_backend/infrastructure/config/settings.py` for the full shape.

## Runtime Notes

- `POST /v1/uploads/{upload_id}/complete` marks a signed upload as usable by the rest of the API.
- `GET /v1/uploads` lists the current user's uploads and supports `category` and `status` filters.
- `GET /v1/profiles/me`, `POST /v1/profiles/me/portfolio-items`, and
  `GET /v1/profiles/me/portfolio-items` complete the editable profile flow.
- Diagnosis jobs, competition submissions, and portfolio items now require uploads that have been
  explicitly completed.
- Diagnosis jobs can launch a Spot VM from a Compute Engine instance template when
  `MIRIP_COMPUTE__ENABLED=true` and `MIRIP_WORKER__MODE=cpu_onnx`.
- CPU ONNX workers expect a serving bundle manifest and fail fast if the bundle does not include
  diagnosis extras (`diagnosis_head`, `anchors`).
