"""Worker entrypoint."""

from __future__ import annotations

import asyncio

import structlog

from mirip_backend.infrastructure.config.container import build_container
from mirip_backend.infrastructure.config.settings import get_settings
from mirip_backend.infrastructure.logging.setup import configure_logging
from mirip_backend.worker.claim import build_worker_id
from mirip_backend.worker.inference.service import GpuInferenceService
from mirip_backend.worker.poller import JobPoller
from mirip_backend.worker.result_writer import DiagnosisResultWriter

logger = structlog.get_logger(__name__)


async def run_worker() -> None:
    settings = get_settings()
    configure_logging(settings)
    container = await build_container(settings)
    worker_id = build_worker_id()
    logger.info("worker.startup", worker_id=worker_id, mode=settings.worker.mode)
    poller = JobPoller(
        worker_id=worker_id,
        queue=container.job_queue,
        inference_service=GpuInferenceService(
            mode=settings.worker.mode,
            model_uri=settings.worker.model_uri,
        ),
        result_writer=DiagnosisResultWriter(container.diagnosis_result_repository),
    )

    if settings.worker.run_once:
        await poller.process_once()
        return

    while True:
        try:
            handled = await poller.process_once()
        except Exception:
            logger.warning("worker.loop_iteration_failed", worker_id=worker_id)
            await asyncio.sleep(settings.job.worker_poll_interval_seconds)
            continue
        if handled is None:
            await asyncio.sleep(settings.job.worker_poll_interval_seconds)


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
