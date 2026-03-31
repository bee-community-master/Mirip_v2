# Mirip v2 Workspace

`Mirip_v2` is the shared workspace for the Mirip v2 frontend mock, backend scaffold, and training experiments.
The repository is organized so each area can evolve independently while still sharing one root for docs and operational assets.

## Frontend

- `frontend/` contains the SvelteKit mock application used to validate the Mirip v2 UX flows.
- The mock currently covers the home, diagnosis, competitions, and portfolio screens with local fixtures and reusable UI components.
- Run the standard frontend scripts from `frontend/`: `npm run check`, `npm run test`, and `npm run build`.
- The frontend is configured for static Vercel deployment as an SPA fallback via `@sveltejs/adapter-static`.

## Backend

The backend scaffold is designed for:

- Cloud Run API
- Firebase Auth + Firestore
- Google Cloud Storage
- Spot GPU VM workers for heavyweight inference jobs

See [BACKEND_V2_PLAN.md](./BACKEND_V2_PLAN.md) for the implementation plan and [backend/README.md](./backend/README.md) for local backend commands.

### Deployment targets

- Frontend: deploy `frontend/` to Vercel
- Backend (current recommended profile): deploy `backend/` to GCP as `Cloud Run API + Firestore + Spot VM stub workers`
- Backend (future model profile): switch the worker stack to `cpu_onnx` + GCS model bundles when the serving bundle is ready

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
