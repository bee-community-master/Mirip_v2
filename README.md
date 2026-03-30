# Mirip v2

Mirip v2 is the next-generation application workspace for the Mirip product line.
This change set introduces the backend foundation for the new runtime architecture.

## Backend

The backend is designed for:

- Cloud Run API
- Firebase Auth + Firestore
- Google Cloud Storage
- Spot GPU VM workers for heavyweight inference jobs

See [BACKEND_V2_PLAN.md](./BACKEND_V2_PLAN.md) for the implementation plan and [backend/README.md](./backend/README.md) for local backend commands.
