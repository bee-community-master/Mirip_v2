"""Tests for job poller heartbeat behavior."""

from __future__ import annotations

import asyncio

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.infrastructure.config.settings import JobSettings
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import (
    DocumentDiagnosisJobRepository,
    DocumentDiagnosisResultRepository,
)
from mirip_backend.infrastructure.jobs.queue import JobQueueService
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import JobStatus
from mirip_backend.worker.poller import JobPoller
from mirip_backend.worker.result_writer import DiagnosisResultWriter


class SlowInferenceService:
    def __init__(self, repository: DocumentDiagnosisJobRepository) -> None:
        self._repository = repository

    async def evaluate(self, job: DiagnosisJob):  # type: ignore[no-untyped-def]
        await asyncio.sleep(1.05)
        current = await self._repository.get(job.id)
        assert current is not None
        assert current.lease_expires_at is not None
        assert job.lease_expires_at is not None
        assert current.lease_expires_at > job.lease_expires_at
        return type(
            "Output",
            (),
            {
                "tier": "A",
                "scores": {"composition": 90.0},
                "probabilities": [],
                "feedback": None,
                "summary": "ok",
            },
        )()


async def test_job_poller_keeps_lease_alive_during_long_evaluation() -> None:
    store = MemoryDocumentStore()
    job_repository = DocumentDiagnosisJobRepository(store)
    result_repository = DocumentDiagnosisResultRepository(store)
    now = utc_now()
    job = DiagnosisJob(
        id="job-heartbeat",
        user_id="user-1",
        upload_ids=["upl-1"],
        job_type="evaluate",
        department="visual_design",
        include_feedback=True,
        theme=None,
        language="ko",
        status=JobStatus.QUEUED,
        created_at=now,
        updated_at=now,
    )
    await job_repository.create(job)

    poller = JobPoller(
        worker_id="worker-1",
        queue=JobQueueService(
            JobSettings(lease_seconds=1, heartbeat_interval_seconds=1, max_attempts=3),
            job_repository,
        ),
        inference_service=SlowInferenceService(job_repository),
        result_writer=DiagnosisResultWriter(result_repository),
    )

    completed = await poller.process_once()

    assert completed is not None
    assert completed.status == JobStatus.SUCCEEDED
    stored_result = await result_repository.get_by_job_id("job-heartbeat")
    assert stored_result is not None
