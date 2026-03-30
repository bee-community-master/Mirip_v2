# Mirip v2 Backend

FastAPI backend scaffold for Mirip v2.

## Architecture

- Cloud Run API for authenticated REST endpoints
- Firebase Auth token verification
- Firestore as the primary domain datastore
- Google Cloud Storage for uploads and generated assets
- Spot GPU worker for async diagnosis jobs

## Quick Start

```bash
uv sync --group dev
uv run uvicorn mirip_backend.api.app:create_app --factory --reload
```

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
- `MIRIP_OBSERVABILITY__*`

See `src/mirip_backend/infrastructure/config/settings.py` for the full shape.
