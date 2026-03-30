"""FastAPI lifespan hooks."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from mirip_backend.infrastructure.config.container import build_container
from mirip_backend.infrastructure.config.settings import get_settings
from mirip_backend.infrastructure.logging.setup import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    logger = structlog.get_logger(__name__)
    app.state.settings = settings
    app.state.container = await build_container(settings)
    logger.info("api.startup", env=settings.app_env, data_backend=settings.data_backend)
    yield
    logger.info("api.shutdown")
