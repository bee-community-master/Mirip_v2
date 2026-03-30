# Mirip v2 Workspace

`Mirip_v2` is the shared workspace for the Mirip v2 frontend mock, backend scaffold, and training experiments.
The repository is organized so each area can evolve independently while still sharing one root for docs and operational assets.

## Frontend

- `frontend/` contains the SvelteKit mock application used to validate the Mirip v2 UX flows.
- The mock currently covers the home, diagnosis, competitions, and portfolio screens with local fixtures and reusable UI components.
- Run the standard frontend scripts from `frontend/`: `npm run check`, `npm run test`, and `npm run build`.
- For diagnosis API integration, create `frontend/.env` from `frontend/.env.example` and set
  `PUBLIC_MIRIP_API_BASE_URL` to the backend base URL. The default local value is
  `http://localhost:8000`.

## Backend

The backend scaffold is designed for:

- Cloud Run API
- Firebase Auth + Firestore
- Google Cloud Storage
- Spot GPU VM workers for heavyweight inference jobs

See [BACKEND_V2_PLAN.md](./BACKEND_V2_PLAN.md) for the implementation plan and [backend/README.md](./backend/README.md) for local backend commands.

## Local Diagnosis Flow

Run the diagnosis slice locally with three processes:

```bash
cd backend
cp .env.example .env
# set `MIRIP_STORAGE_BACKEND=fake` in .env if you do not have local GCS credentials
uv sync --group dev
uv run uvicorn mirip_backend.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

```bash
cd backend
uv run python -m mirip_backend.worker.main
```

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

The frontend diagnosis screen talks directly to the backend API, creates upload sessions, completes
uploads, creates diagnosis jobs, and polls for worker results.

If you want to exercise real signed uploads instead of fake mode, keep `MIRIP_STORAGE_BACKEND=gcs`
and provide valid `MIRIP_GCS__*` credentials in `backend/.env`.

## Training

The training workspace under `train/` prepares `#1 DINOv2 baseline reproduction` and `#2 DINOv3 teacher training` on top of Vast.ai-managed GPU environments.

- [train/docs/vast_ai_preparation.md](./train/docs/vast_ai_preparation.md): Vast.ai operation notes
- [train/training/dinov3_vast_plan.md](./train/training/dinov3_vast_plan.md): DINOv3 training and evaluation plan
- [train/scripts/vast_ai_control.py](./train/scripts/vast_ai_control.py): Vast.ai REST API + SSH/rsync wrapper
- [train/scripts/vast_ai_training_runner.py](./train/scripts/vast_ai_training_runner.py): remote bootstrap/validate-upload/smoke/full stage runner

### Training quick start

```bash
cd <repo-root>
cp .env.example .env
export $(grep -v '^#' .env | xargs)

python3 train/scripts/vast_ai_control.py search --config train/configs/vast_rtx_pro_4500_blackwell_32gb_ondemand.toml
python3 train/scripts/vast_ai_control.py create --config train/configs/vast_rtx_pro_4500_blackwell_32gb_ondemand.toml --attach-ssh
python3 train/scripts/vast_ai_control.py wait --instance-id <INSTANCE_ID>
python3 train/scripts/vast_ai_control.py ssh --instance-id <INSTANCE_ID>
```

### Training pipeline

```bash
python3 train/training/validate_training_readiness.py
python3 train/training/prepare_snapshot.py
python3 train/training/build_pairs.py --manifest train/training/data/snapshot_manifest.csv --output-dir train/training/data
python3 train/training/validate_training_readiness.py --mode prepared
python3 train/scripts/vast_ai_training_runner.py print-command --stage bootstrap
python3 train/scripts/vast_ai_training_runner.py print-command --stage validate-upload
python3 train/scripts/vast_ai_training_runner.py print-command --stage smoke
```
