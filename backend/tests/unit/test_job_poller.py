"""Tests for worker poller failure handling."""

from __future__ import annotations

import pytest

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
from mirip_backend.worker.inference.service import NonRetryableInferenceError
from mirip_backend.worker.poller import JobPoller
from mirip_backend.worker.result_writer import DiagnosisResultWriter


class FailingInferenceService:
    async def evaluate(self, job: DiagnosisJob):  # type: ignore[no-untyped-def]
        raise RuntimeError("s3cret stack detail")


class NonRetryableFailingInferenceService:
    async def evaluate(self, job: DiagnosisJob):  # type: ignore[no-untyped-def]
        raise NonRetryableInferenceError("bundle is invalid")


async def test_job_poller_hides_internal_failure_reason() -> None:
    store = MemoryDocumentStore()
    job_repository = DocumentDiagnosisJobRepository(store)
    result_repository = DocumentDiagnosisResultRepository(store)
    now = utc_now()
    job = DiagnosisJob(
        id="job-fail",
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
        queue=JobQueueService(JobSettings(), job_repository),
        inference_service=FailingInferenceService(),
        result_writer=DiagnosisResultWriter(result_repository),
    )

    with pytest.raises(RuntimeError):
        await poller.process_once()

    stored = await job_repository.get("job-fail")
    assert stored is not None
    assert stored.status in {JobStatus.EXPIRED, JobStatus.FAILED}
    assert stored.failure_reason == "Worker execution failed"


async def test_job_poller_marks_non_retryable_inference_failures_as_failed() -> None:
    store = MemoryDocumentStore()
    job_repository = DocumentDiagnosisJobRepository(store)
    result_repository = DocumentDiagnosisResultRepository(store)
    now = utc_now()
    job = DiagnosisJob(
        id="job-fail-hard",
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
        queue=JobQueueService(JobSettings(), job_repository),
        inference_service=NonRetryableFailingInferenceService(),
        result_writer=DiagnosisResultWriter(result_repository),
    )

    with pytest.raises(NonRetryableInferenceError):
        await poller.process_once()

    stored = await job_repository.get("job-fail-hard")
    assert stored is not None
    assert stored.status == JobStatus.FAILED
    assert stored.failure_reason == "Internal model error"
