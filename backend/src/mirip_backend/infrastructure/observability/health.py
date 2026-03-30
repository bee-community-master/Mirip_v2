"""Health aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field

from mirip_backend.domain.common.models import HealthDependency
from mirip_backend.domain.common.ports import HealthCheckPort


@dataclass(slots=True)
class HealthReporter:
    checks: list[HealthCheckPort] = field(default_factory=list)

    async def report(self) -> list[HealthDependency]:
        return [await check.check() for check in self.checks]
