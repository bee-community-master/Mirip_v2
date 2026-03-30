"""Complete upload usecase."""

from __future__ import annotations

from dataclasses import replace

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.common.ports import UploadSessionSigner
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.domain.uploads.repositories import UploadRepository
from mirip_backend.shared.enums import UploadStatus
from mirip_backend.shared.exceptions import ValidationError
from mirip_backend.usecases.uploads.validation import load_owned_upload


class CompleteUploadUseCase:
    """Mark a signed-upload target as uploaded for the authenticated user."""

    def __init__(
        self,
        upload_repository: UploadRepository,
        storage_service: UploadSessionSigner,
    ) -> None:
        self._upload_repository = upload_repository
        self._storage_service = storage_service

    async def execute(self, *, actor: AuthenticatedUser, upload_id: str) -> UploadAsset:
        upload = await load_owned_upload(
            upload_repository=self._upload_repository,
            actor=actor,
            upload_id=upload_id,
        )
        if upload.status == UploadStatus.CONSUMED:
            raise ValidationError("Consumed uploads cannot be marked as uploaded")
        if upload.status == UploadStatus.UPLOADED:
            return upload
        if not await self._storage_service.object_exists(object_name=upload.object_name):
            raise ValidationError("Uploaded object not found in storage")

        completed = replace(upload, status=UploadStatus.UPLOADED)
        return await self._upload_repository.update(completed)
