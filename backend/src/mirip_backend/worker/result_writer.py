"""Result persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.diagnosis.entities import DiagnosisJob, DiagnosisResult
from mirip_backend.domain.diagnosis.repositories import DiagnosisResultRepository
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.ids import new_id
from mirip_backend.worker.inference.service import InferenceOutput


@dataclass(slots=True)
class DiagnosisResultWriter:
    repository: DiagnosisResultRepository

    async def write_success(self, *, job: DiagnosisJob, output: InferenceOutput) -> DiagnosisResult:
        result = DiagnosisResult(
            id=new_id("res"),
            job_id=job.id,
            user_id=job.user_id,
            tier=output.tier,
            scores=output.scores,
            probabilities=output.probabilities,
            feedback=output.feedback,
            created_at=utc_now(),
            summary=output.summary,
        )
        return await self.repository.create(result)
