# Mirip v2 Backend Plan

## Summary

- Source reference: `Mirip/backend`
- Runtime target: `Cloud Run API + Firebase(Auth/Firestore) + GCS + Spot GPU VM worker`
- Ownership rule: frontend talks only to backend APIs; backend owns Firestore and GCS writes
- Delivery scope: full-domain backend scaffold for diagnosis, competitions, credentials, and profiles

## Public API Shape

### Core diagnosis APIs

- `POST /v1/uploads`
- `POST /v1/diagnosis/jobs`
- `GET /v1/diagnosis/jobs/{job_id}`
- `GET /v1/diagnosis/history`

### Extended domain APIs

- `GET /v1/competitions`
- `POST /v1/competitions/{competition_id}/submissions`
- `PUT /v1/profiles/me`
- `GET /v1/profiles/{handle}`
- `POST /v1/credentials`
- `GET /v1/credentials/{credential_id}`

## Structural Decisions

- API stays stateless and never loads ML models at startup.
- Worker owns heavyweight model loading and job execution.
- Firestore is the system of record for domain state.
- GCS stores binaries and generated assets only.
- Diagnosis is async only. Sync inference is out of scope for v2.
- Job statuses are fixed to `queued`, `leased`, `running`, `succeeded`, `failed`, `expired`.

## Folder Strategy

- `api/`: FastAPI app factory, dependencies, routes, schemas, middleware, exception mapping
- `domain/`: pure entities, value objects, repository protocols
- `usecases/`: orchestration for commands and queries
- `infrastructure/`: Firebase auth, Firestore adapters, GCS helpers, job queue, config, logging, observability
- `worker/`: polling, claiming, inference execution, result persistence
- `tests/`: unit, integration, and e2e flow coverage
- `deploy/`: Cloud Run and GPU worker bootstrap assets

## Runtime Boundaries

- Legacy training and data pipeline code are not part of the runtime package.
- ML worker dependencies are isolated behind the `worker` optional dependency group.
- API routes depend on usecases, never directly on Firestore or GCS clients.

## Validation Targets

- Upload session creation enforces authenticated user ownership.
- Job creation validates upload ownership and persists a queued job.
- Worker lease flow is idempotent and supports retry after lease expiration.
- Job status and history endpoints enforce per-user access.
- Competition and credential APIs validate references through repositories rather than route-layer code.
