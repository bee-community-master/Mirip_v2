"""List uploads usecase."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.common.models import Page
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.domain.uploads.repositories import UploadRepository
from mirip_backend.shared.enums import UploadStatus


@dataclass(slots=True, frozen=True)
class ListUploadsQuery:
    limit: int = 50
    offset: int = 0
    category: str | None = None
    status: UploadStatus | None = None


class ListUploadsUseCase:
    """List uploads for the authenticated user with lightweight filtering."""

    def __init__(self, upload_repository: UploadRepository) -> None:
        self._upload_repository = upload_repository

    async def execute(
        self,
        *,
        actor: AuthenticatedUser,
        query: ListUploadsQuery,
    ) -> Page[UploadAsset]:
        uploads = await self._upload_repository.list_by_user(actor.user_id)
        if query.category is not None:
            uploads = [
                item
                for item in uploads
                if item.metadata.get("category") == query.category.lower().strip()
            ]
        if query.status is not None:
            uploads = [item for item in uploads if item.status == query.status]
        return Page(
            items=uploads[query.offset : query.offset + query.limit],
            total=len(uploads),
            limit=query.limit,
            offset=query.offset,
        )
