"""Tests for job queue retry behavior."""

from __future__ import annotations

from datetime import timedelta

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.infrastructure.config.settings import JobSettings
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import DocumentDiagnosisJobRepository
from mirip_backend.infrastructure.jobs.queue import JobQueueService
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import JobStatus


async def test_queue_reclaims_running_job_after_lease_expiry() -> None:
    repository = DocumentDiagnosisJobRepository(MemoryDocumentStore())
    now = utc_now()
    job = DiagnosisJob(
        id="job-1",
        user_id="user-1",
        upload_ids=["upl-1"],
        job_type="evaluate",
        department="visual_design",
        include_feedback=True,
        theme=None,
        language="ko",
        status=JobStatus.RUNNING,
        created_at=now - timedelta(minutes=10),
        updated_at=now - timedelta(minutes=10),
        attempts=1,
        lease_owner="worker-a",
        lease_expires_at=now - timedelta(seconds=1),
    )
    await repository.create(job)

    queue = JobQueueService(JobSettings(lease_seconds=60, max_attempts=5), repository)

    leased = await queue.lease_next(worker_id="worker-b")

    assert leased is not None
    assert leased.id == "job-1"
    assert leased.status == JobStatus.LEASED
    assert leased.lease_owner == "worker-b"
    assert leased.attempts == 2


async def test_queue_marks_failed_after_max_attempts() -> None:
    repository = DocumentDiagnosisJobRepository(MemoryDocumentStore())
    now = utc_now()
    job = DiagnosisJob(
        id="job-2",
        user_id="user-1",
        upload_ids=["upl-1"],
        job_type="evaluate",
        department="visual_design",
        include_feedback=True,
        theme=None,
        language="ko",
        status=JobStatus.RUNNING,
        created_at=now,
        updated_at=now,
        attempts=3,
        lease_owner="worker-a",
        lease_expires_at=now + timedelta(minutes=1),
    )
    await repository.create(job)

    queue = JobQueueService(JobSettings(lease_seconds=60, max_attempts=3), repository)
    failed = await queue.mark_failed(job, reason="boom")

    assert failed.status == JobStatus.FAILED
    assert failed.failure_reason == "boom"
    assert failed.lease_owner is None


async def test_queue_heartbeat_extends_lease_without_changing_status() -> None:
    repository = DocumentDiagnosisJobRepository(MemoryDocumentStore())
    now = utc_now()
    job = DiagnosisJob(
        id="job-3",
        user_id="user-1",
        upload_ids=["upl-1"],
        job_type="evaluate",
        department="visual_design",
        include_feedback=True,
        theme=None,
        language="ko",
        status=JobStatus.RUNNING,
        created_at=now,
        updated_at=now,
        attempts=1,
        lease_owner="worker-a",
        lease_expires_at=now,
    )
    await repository.create(job)

    queue = JobQueueService(
        JobSettings(lease_seconds=60, heartbeat_interval_seconds=10, max_attempts=3),
        repository,
    )
    heartbeated = await queue.heartbeat(job)

    assert heartbeated.status == JobStatus.RUNNING
    assert heartbeated.lease_owner == "worker-a"
    assert heartbeated.lease_expires_at is not None
    assert heartbeated.lease_expires_at > now
