"""Shared validation helpers for upload-centric usecases."""

from __future__ import annotations

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.domain.uploads.repositories import UploadRepository
from mirip_backend.shared.enums import UploadStatus
from mirip_backend.shared.exceptions import AuthorizationError, NotFoundError, ValidationError


async def load_owned_upload(
    *,
    upload_repository: UploadRepository,
    actor: AuthenticatedUser,
    upload_id: str,
    not_found_message: str = "Upload not found",
    ownership_message: str = "Upload does not belong to the authenticated user",
) -> UploadAsset:
    upload = await upload_repository.get(upload_id)
    if upload is None:
        raise NotFoundError(not_found_message)
    if upload.user_id != actor.user_id:
        raise AuthorizationError(ownership_message)
    return upload


def require_uploaded_asset(
    upload: UploadAsset,
    *,
    message: str,
) -> None:
    if upload.status != UploadStatus.UPLOADED:
        raise ValidationError(message)
