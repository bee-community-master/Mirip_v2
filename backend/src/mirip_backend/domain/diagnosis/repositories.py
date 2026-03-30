"""Diagnosis repository contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from mirip_backend.domain.common.models import Page
from mirip_backend.domain.diagnosis.entities import DiagnosisJob, DiagnosisResult


class DiagnosisJobRepository(Protocol):
    async def create(self, job: DiagnosisJob) -> DiagnosisJob: ...

    async def get(self, job_id: str) -> DiagnosisJob | None: ...

    async def update(self, job: DiagnosisJob) -> DiagnosisJob: ...

    async def list_by_user(
        self, user_id: str, *, limit: int, offset: int
    ) -> Page[DiagnosisJob]: ...

    async def lease_next_ready_job(
        self,
        *,
        worker_id: str,
        lease_until: datetime,
    ) -> DiagnosisJob | None: ...

    async def lease_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_until: datetime,
    ) -> DiagnosisJob | None: ...


class DiagnosisResultRepository(Protocol):
    async def create(self, result: DiagnosisResult) -> DiagnosisResult: ...

    async def get(self, result_id: str) -> DiagnosisResult | None: ...

    async def get_by_job_id(self, job_id: str) -> DiagnosisResult | None: ...

    async def list_by_user(
        self, user_id: str, *, limit: int, offset: int
    ) -> Page[DiagnosisResult]: ...
