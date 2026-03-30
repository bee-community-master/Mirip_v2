"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from mirip_backend.api.app import create_app
from mirip_backend.infrastructure.config.settings import get_settings


@pytest.fixture(autouse=True)
def _configure_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIRIP_APP_ENV", "test")
    monkeypatch.setenv("MIRIP_DATA_BACKEND", "memory")
    monkeypatch.setenv("MIRIP_STORAGE_BACKEND", "fake")
    monkeypatch.setenv("MIRIP_FIREBASE__ALLOW_INSECURE_DEV_AUTH", "true")
    monkeypatch.setenv("MIRIP_FIREBASE__LOCAL_DEV_TOKEN", "local-dev-token")
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def app() -> AsyncIterator[FastAPI]:
    instance = create_app()
    async with LifespanManager(instance):
        yield instance
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer local-dev-token"}
