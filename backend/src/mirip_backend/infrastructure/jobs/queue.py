"""Diagnosis job queue helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.domain.diagnosis.repositories import DiagnosisJobRepository
from mirip_backend.infrastructure.config.settings import JobSettings
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import JobStatus


@dataclass(slots=True)
class JobQueueService:
    settings: JobSettings
    repository: DiagnosisJobRepository

    async def lease_next(self, *, worker_id: str) -> DiagnosisJob | None:
        lease_until = utc_now() + timedelta(seconds=self.settings.lease_seconds)
        return await self.repository.lease_next_ready_job(
            worker_id=worker_id, lease_until=lease_until
        )

    async def mark_running(self, job: DiagnosisJob) -> DiagnosisJob:
        updated = replace(job, status=JobStatus.RUNNING, updated_at=utc_now())
        return await self.repository.update(updated)

    async def mark_succeeded(self, job: DiagnosisJob) -> DiagnosisJob:
        updated = replace(
            job,
            status=JobStatus.SUCCEEDED,
            updated_at=utc_now(),
            lease_owner=None,
            lease_expires_at=None,
            failure_reason=None,
        )
        return await self.repository.update(updated)

    async def mark_failed(self, job: DiagnosisJob, *, reason: str) -> DiagnosisJob:
        next_status = (
            JobStatus.FAILED if job.attempts >= self.settings.max_attempts else JobStatus.EXPIRED
        )
        updated = replace(
            job,
            status=next_status,
            updated_at=utc_now(),
            lease_owner=None,
            lease_expires_at=None,
            failure_reason=reason,
        )
        return await self.repository.update(updated)
