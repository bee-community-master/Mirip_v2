"""FastAPI application factory."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mirip_backend.api.errors.handlers import install_exception_handlers
from mirip_backend.api.lifespan import lifespan
from mirip_backend.api.middleware.request_context import install_request_context_middleware
from mirip_backend.api.routes import competitions, credentials, diagnosis, health, profiles, uploads
from mirip_backend.infrastructure.config.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_request_context_middleware(app)
    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(uploads.router)
    app.include_router(diagnosis.router)
    app.include_router(competitions.router)
    app.include_router(credentials.router)
    app.include_router(profiles.router)
    return app


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "mirip_backend.api.app:create_app",
        factory=True,
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
    )
