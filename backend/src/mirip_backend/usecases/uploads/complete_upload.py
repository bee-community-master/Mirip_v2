"""Complete upload usecase."""

from __future__ import annotations

from dataclasses import replace

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.domain.uploads.repositories import UploadRepository
from mirip_backend.shared.enums import UploadStatus
from mirip_backend.shared.exceptions import AuthorizationError, NotFoundError, ValidationError


class CompleteUploadUseCase:
    """Mark a signed-upload target as uploaded for the authenticated user."""

    def __init__(self, upload_repository: UploadRepository) -> None:
        self._upload_repository = upload_repository

    async def execute(self, *, actor: AuthenticatedUser, upload_id: str) -> UploadAsset:
        upload = await self._upload_repository.get(upload_id)
        if upload is None:
            raise NotFoundError("Upload not found")
        if upload.user_id != actor.user_id:
            raise AuthorizationError("Upload does not belong to the authenticated user")
        if upload.status == UploadStatus.CONSUMED:
            raise ValidationError("Consumed uploads cannot be marked as uploaded")
        if upload.status == UploadStatus.UPLOADED:
            return upload

        completed = replace(upload, status=UploadStatus.UPLOADED)
        return await self._upload_repository.update(completed)
