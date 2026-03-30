"""Get diagnosis job status usecase."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.diagnosis.entities import DiagnosisJob, DiagnosisResult
from mirip_backend.domain.diagnosis.repositories import (
    DiagnosisJobRepository,
    DiagnosisResultRepository,
)
from mirip_backend.shared.exceptions import AuthorizationError, NotFoundError


@dataclass(slots=True, frozen=True)
class DiagnosisJobStatusView:
    job: DiagnosisJob
    result: DiagnosisResult | None


class GetDiagnosisJobStatusUseCase:
    """Load a job and the attached result for the authenticated user."""

    def __init__(
        self,
        job_repository: DiagnosisJobRepository,
        result_repository: DiagnosisResultRepository,
    ) -> None:
        self._job_repository = job_repository
        self._result_repository = result_repository

    async def execute(self, *, actor: AuthenticatedUser, job_id: str) -> DiagnosisJobStatusView:
        job = await self._job_repository.get(job_id)
        if job is None:
            raise NotFoundError("Diagnosis job not found")
        if job.user_id != actor.user_id:
            raise AuthorizationError()
        result = await self._result_repository.get_by_job_id(job_id)
        return DiagnosisJobStatusView(job=job, result=result)
