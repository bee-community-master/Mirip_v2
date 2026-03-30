"""Upload repository contracts."""

from __future__ import annotations

from typing import Protocol

from mirip_backend.domain.uploads.entities import UploadAsset


class UploadRepository(Protocol):
    async def create(self, upload: UploadAsset) -> UploadAsset: ...

    async def get(self, upload_id: str) -> UploadAsset | None: ...

    async def list_by_user(self, user_id: str) -> list[UploadAsset]: ...

    async def update(self, upload: UploadAsset) -> UploadAsset: ...
