"""Worker entrypoint."""

from __future__ import annotations

import asyncio
import signal
from contextlib import suppress

import structlog

from mirip_backend.infrastructure.config.container import ApplicationContainer, build_container
from mirip_backend.infrastructure.config.settings import Settings, get_settings
from mirip_backend.infrastructure.logging.setup import configure_logging
from mirip_backend.worker.claim import build_worker_id
from mirip_backend.worker.inference.service import WorkerInferenceService
from mirip_backend.worker.poller import JobPoller
from mirip_backend.worker.result_writer import DiagnosisResultWriter

logger = structlog.get_logger(__name__)
SHUTDOWN_REASON = "Worker interrupted before completion"


def _build_poller(
    *,
    container: ApplicationContainer,
    settings: Settings,
    worker_id: str,
) -> JobPoller:
    return JobPoller(
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


def _install_shutdown_handlers(
    *,
    poller: JobPoller,
    shutdown_event: asyncio.Event,
) -> None:
    def _handle_shutdown_signal() -> None:
        if shutdown_event.is_set():
            return
        shutdown_event.set()
        asyncio.create_task(poller.request_shutdown(reason=SHUTDOWN_REASON))

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signum, _handle_shutdown_signal)


async def _run_worker_loop(
    *,
    poller: JobPoller,
    shutdown_event: asyncio.Event,
    worker_id: str,
    poll_interval_seconds: int,
    run_once: bool,
) -> None:
    if run_once:
        await poller.process_once()
        return

    while not shutdown_event.is_set():
        try:
            handled = await poller.process_once()
        except Exception:
            logger.warning("worker.loop_iteration_failed", worker_id=worker_id)
            await asyncio.sleep(poll_interval_seconds)
            continue
        if handled is None:
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=poll_interval_seconds,
                )
            except TimeoutError:
                pass


async def _cleanup_instance(
    *,
    container: ApplicationContainer,
    settings: Settings,
) -> None:
    if (
        container.compute_launcher is None
        or not settings.compute.delete_self_on_completion
        or not settings.compute.instance_name
        or not settings.compute.zone
    ):
        return
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


async def run_worker() -> None:
    settings = get_settings()
    configure_logging(settings)
    container = await build_container(settings)
    worker_id = build_worker_id()
    logger.info("worker.startup", worker_id=worker_id, mode=settings.worker.mode)
    poller = _build_poller(container=container, settings=settings, worker_id=worker_id)
    shutdown_event = asyncio.Event()
    _install_shutdown_handlers(poller=poller, shutdown_event=shutdown_event)

    try:
        await _run_worker_loop(
            poller=poller,
            shutdown_event=shutdown_event,
            worker_id=worker_id,
            poll_interval_seconds=settings.job.worker_poll_interval_seconds,
            run_once=settings.worker.run_once,
        )
    finally:
        try:
            await poller.request_shutdown(reason=SHUTDOWN_REASON)
        finally:
            await _cleanup_instance(container=container, settings=settings)


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
