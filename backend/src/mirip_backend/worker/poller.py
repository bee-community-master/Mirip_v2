"""Worker polling loop."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.infrastructure.jobs.queue import JobQueueService
from mirip_backend.worker.claim import claim_next_job
from mirip_backend.worker.inference.service import InferenceService
from mirip_backend.worker.result_writer import DiagnosisResultWriter

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class JobPoller:
    worker_id: str
    queue: JobQueueService
    inference_service: InferenceService
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
        except Exception:
            logger.exception("worker.job_failed", job_id=running_job.id)
            await self.queue.mark_failed(running_job, reason="Worker execution failed")
            raise
