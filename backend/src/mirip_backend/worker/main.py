"""Worker entrypoint."""

from __future__ import annotations

import asyncio
import signal
from contextlib import suppress

import structlog

from mirip_backend.infrastructure.config.container import build_container
from mirip_backend.infrastructure.config.settings import get_settings
from mirip_backend.infrastructure.logging.setup import configure_logging
from mirip_backend.worker.claim import build_worker_id
from mirip_backend.worker.inference.service import WorkerInferenceService
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
        inference_service=WorkerInferenceService(
            mode=settings.worker.mode,
            model_uri=settings.worker.model_uri,
            storage_service=container.storage_service,
            local_model_cache_dir=settings.worker.local_model_cache_dir,
        ),
        result_writer=DiagnosisResultWriter(container.diagnosis_result_repository),
        target_job_id=settings.worker.target_job_id,
    )
    shutdown_event = asyncio.Event()

    def _handle_shutdown_signal() -> None:
        if shutdown_event.is_set():
            return
        shutdown_event.set()
        asyncio.create_task(
            poller.request_shutdown(reason="Worker interrupted before completion")
        )

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signum, _handle_shutdown_signal)

    try:
        if settings.worker.run_once:
            await poller.process_once()
            return

        while not shutdown_event.is_set():
            try:
                handled = await poller.process_once()
            except Exception:
                logger.warning("worker.loop_iteration_failed", worker_id=worker_id)
                await asyncio.sleep(settings.job.worker_poll_interval_seconds)
                continue
            if handled is None:
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=settings.job.worker_poll_interval_seconds,
                    )
                except TimeoutError:
                    pass
    finally:
        try:
            await poller.request_shutdown(reason="Worker interrupted before completion")
        finally:
            if (
                container.compute_launcher is not None
                and settings.compute.delete_self_on_completion
                and settings.compute.instance_name
                and settings.compute.zone
            ):
                try:
                    await container.compute_launcher.delete_instance(
                        instance_name=settings.compute.instance_name,
                        zone=settings.compute.zone,
                    )
                except Exception:
                    logger.warning(
                        "worker.instance_cleanup_failed",
                        instance_name=settings.compute.instance_name,
                        zone=settings.compute.zone,
                    )


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
