"""Common domain ports."""

from __future__ import annotations

from typing import Protocol

from mirip_backend.domain.common.models import HealthDependency, SignedUploadSession


class UploadSessionSigner(Protocol):
    async def create_upload_session(
        self,
        *,
        object_name: str,
        content_type: str,
        metadata: dict[str, str],
    ) -> SignedUploadSession: ...


class HealthCheckPort(Protocol):
    async def check(self) -> HealthDependency: ...
