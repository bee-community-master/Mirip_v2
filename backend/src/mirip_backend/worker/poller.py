"""Worker polling loop."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field

import structlog

from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.infrastructure.jobs.queue import JobQueueService
from mirip_backend.worker.claim import claim_job, claim_next_job
from mirip_backend.worker.inference.service import InferenceOutput, InferenceService
from mirip_backend.worker.result_writer import DiagnosisResultWriter

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class JobPoller:
    worker_id: str
    queue: JobQueueService
    inference_service: InferenceService
    result_writer: DiagnosisResultWriter
    target_job_id: str | None = None
    _active_job: DiagnosisJob | None = field(init=False, default=None, repr=False)
    _active_task: asyncio.Task[InferenceOutput] | None = field(init=False, default=None, repr=False)

    async def process_once(self) -> DiagnosisJob | None:
        job = await self._claim_job()
        if job is None:
            return None

        running_job = await self.queue.mark_running(job)
        try:
            output, running_job = await self._evaluate_with_heartbeat(running_job)
            await self.result_writer.write_success(job=running_job, output=output)
            return await self.queue.mark_succeeded(running_job)
        except Exception:
            logger.exception("worker.job_failed", job_id=running_job.id)
            await self.queue.mark_failed(running_job, reason="Worker execution failed")
            raise

    async def request_shutdown(self, *, reason: str) -> None:
        if self._active_task is None or self._active_job is None:
            return
        task = self._active_task
        current_job = self._active_job
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await self.queue.mark_failed(current_job, reason=reason)
        self._active_task = None
        self._active_job = None

    async def _claim_job(self) -> DiagnosisJob | None:
        if self.target_job_id:
            return await claim_job(self.queue, self.worker_id, self.target_job_id)
        return await claim_next_job(self.queue, self.worker_id)

    async def _evaluate_with_heartbeat(
        self,
        job: DiagnosisJob,
    ) -> tuple[InferenceOutput, DiagnosisJob]:
        heartbeat_interval = max(
            1,
            min(self.queue.settings.heartbeat_interval_seconds, self.queue.settings.lease_seconds),
        )
        task = asyncio.create_task(self.inference_service.evaluate(job))
        self._active_task = task
        self._active_job = job
        current_job = job
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=heartbeat_interval)
                if task in done:
                    return task.result(), current_job
                current_job = await self.queue.heartbeat(current_job)
                self._active_job = current_job
        except Exception:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            raise
        finally:
            self._active_task = None
            self._active_job = None
