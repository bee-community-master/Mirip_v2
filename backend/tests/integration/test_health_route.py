"""Integration tests for health routes."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from mirip_backend.api.app import create_app
from mirip_backend.infrastructure.config.settings import get_settings
from mirip_backend.infrastructure.gcs.service import GCSStorageService


async def test_health_route_returns_dependency_snapshot(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert {item["name"] for item in payload["dependencies"]} == {"firebase_auth", "gcs"}


@pytest_asyncio.fixture
async def gcs_client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    class FakeBucket:
        def exists(self) -> bool:
            return True

    class FakeClient:
        def bucket(self, _: str) -> FakeBucket:
            return FakeBucket()

    monkeypatch.setenv("MIRIP_APP_ENV", "test")
    monkeypatch.setenv("MIRIP_DATA_BACKEND", "memory")
    monkeypatch.setenv("MIRIP_STORAGE_BACKEND", "gcs")
    monkeypatch.setenv("MIRIP_FIREBASE__ALLOW_INSECURE_DEV_AUTH", "true")
    monkeypatch.setenv("MIRIP_GCS__PROJECT_ID", "mirip-v2")
    monkeypatch.setenv("MIRIP_GCS__BUCKET_NAME", "mirip-v2-assets")
    monkeypatch.setenv("MIRIP_GCS__CREDENTIALS_PATH", "/tmp/fake-gcs-service-account.json")
    monkeypatch.setattr(GCSStorageService, "_client", lambda self: FakeClient())
    get_settings.cache_clear()

    app: FastAPI = create_app()
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client
    get_settings.cache_clear()


async def test_health_route_boots_with_memory_data_and_gcs_storage(gcs_client: AsyncClient) -> None:
    response = await gcs_client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    dependency_map = {item["name"]: item for item in payload["dependencies"]}
    assert dependency_map["gcs"]["status"] == "healthy"
    assert dependency_map["gcs"]["detail"] == "mirip-v2-assets"


@pytest_asyncio.fixture
async def broken_gcs_client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    class BrokenClient:
        def bucket(self, _: str) -> None:
            raise RuntimeError("missing credentials")

    monkeypatch.setenv("MIRIP_APP_ENV", "test")
    monkeypatch.setenv("MIRIP_DATA_BACKEND", "memory")
    monkeypatch.setenv("MIRIP_STORAGE_BACKEND", "gcs")
    monkeypatch.setenv("MIRIP_FIREBASE__ALLOW_INSECURE_DEV_AUTH", "true")
    monkeypatch.setenv("MIRIP_GCS__PROJECT_ID", "mirip-v2")
    monkeypatch.setenv("MIRIP_GCS__BUCKET_NAME", "mirip-v2-assets")
    monkeypatch.setattr(GCSStorageService, "_client", lambda self: BrokenClient())
    get_settings.cache_clear()

    app: FastAPI = create_app()
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client
    get_settings.cache_clear()


async def test_health_route_degrades_when_gcs_is_misconfigured(
    broken_gcs_client: AsyncClient,
) -> None:
    response = await broken_gcs_client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    dependency_map = {item["name"]: item for item in payload["dependencies"]}
    assert payload["status"] == "degraded"
    assert dependency_map["gcs"]["status"] == "unhealthy"
