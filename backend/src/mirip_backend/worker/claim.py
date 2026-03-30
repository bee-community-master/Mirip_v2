"""Worker claim helpers."""

from __future__ import annotations

import os

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.infrastructure.jobs.queue import JobQueueService


def build_worker_id() -> str:
    hostname = os.getenv("HOSTNAME", "local-worker")
    pid = os.getpid()
    return f"{hostname}-{pid}"


async def claim_next_job(queue: JobQueueService, worker_id: str) -> DiagnosisJob | None:
    return await queue.lease_next(worker_id=worker_id)
