"""Worker polling loop."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.infrastructure.jobs.queue import JobQueueService
from mirip_backend.worker.claim import claim_next_job
from mirip_backend.worker.inference.service import GpuInferenceService
from mirip_backend.worker.result_writer import DiagnosisResultWriter


@dataclass(slots=True)
class JobPoller:
    worker_id: str
    queue: JobQueueService
    inference_service: GpuInferenceService
    result_writer: DiagnosisResultWriter

    async def process_once(self) -> DiagnosisJob | None:
        job = await claim_next_job(self.queue, self.worker_id)
        if job is None:
            return None

        running_job = await self.queue.mark_running(job)
        try:
            output = await self.inference_service.evaluate(running_job)
            await self.result_writer.write_success(job=running_job, output=output)
            return await self.queue.mark_succeeded(running_job)
        except Exception as exc:
            await self.queue.mark_failed(running_job, reason=str(exc))
            raise
