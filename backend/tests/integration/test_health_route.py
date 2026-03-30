"""Integration tests for health routes."""

from __future__ import annotations

from httpx import AsyncClient


async def test_health_route_returns_dependency_snapshot(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert {item["name"] for item in payload["dependencies"]} == {"firebase_auth", "gcs"}
