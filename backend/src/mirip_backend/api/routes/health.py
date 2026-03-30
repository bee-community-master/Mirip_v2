"""Health routes."""

from __future__ import annotations

from fastapi import APIRouter

from mirip_backend.api.deps.services import ContainerDep
from mirip_backend.api.schemas.common import HealthDependencyResponse, HealthResponse
from mirip_backend.shared.clock import utc_now

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
@router.get("/v1/health", response_model=HealthResponse)
async def health(container: ContainerDep) -> HealthResponse:
    dependencies = await container.health_reporter.report()
    return HealthResponse(
        status="healthy",
        version=container.settings.api.version,
        timestamp=utc_now(),
        dependencies=[
            HealthDependencyResponse(name=item.name, status=item.status, detail=item.detail)
            for item in dependencies
        ],
    )
