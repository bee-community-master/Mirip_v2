"""Create upload session usecase."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.common.models import SignedUploadSession
from mirip_backend.domain.common.ports import UploadSessionSigner
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.domain.uploads.repositories import UploadRepository
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import UploadStatus
from mirip_backend.shared.exceptions import ValidationError
from mirip_backend.shared.ids import new_id

_SAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(filename: str) -> str:
    name = PurePosixPath(filename).name.strip()
    safe_name = _SAFE_PATH_CHARS.sub("-", name).strip(".-")
    if not safe_name:
        raise ValidationError("Filename must contain at least one safe character")
    return safe_name


def _sanitize_category(category: str) -> str:
    safe_category = _SAFE_PATH_CHARS.sub("-", category.strip().lower()).strip(".-")
    if not safe_category:
        raise ValidationError("Category must contain at least one safe character")
    return safe_category


@dataclass(slots=True, frozen=True)
class CreateUploadSessionCommand:
    filename: str
    content_type: str
    size_bytes: int
    category: str = "diagnosis"


@dataclass(slots=True, frozen=True)
class CreateUploadSessionResult:
    upload: UploadAsset
    session: SignedUploadSession


class CreateUploadSessionUseCase:
    """Create upload metadata and a signed upload target."""

    def __init__(
        self,
        upload_repository: UploadRepository,
        signer: UploadSessionSigner,
    ) -> None:
        self._upload_repository = upload_repository
        self._signer = signer

    async def execute(
        self,
        *,
        actor: AuthenticatedUser,
        command: CreateUploadSessionCommand,
    ) -> CreateUploadSessionResult:
        upload_id = new_id("upl")
        safe_filename = _sanitize_filename(command.filename)
        safe_category = _sanitize_category(command.category)
        object_name = f"users/{actor.user_id}/{safe_category}/{upload_id}/{safe_filename}"
        upload = UploadAsset(
            id=upload_id,
            user_id=actor.user_id,
            filename=command.filename,
            content_type=command.content_type,
            size_bytes=command.size_bytes,
            object_name=object_name,
            status=UploadStatus.PENDING,
            created_at=utc_now(),
            metadata={"category": safe_category, "original_filename": command.filename},
        )
        stored = await self._upload_repository.create(upload)
        session = await self._signer.create_upload_session(
            object_name=object_name,
            content_type=command.content_type,
            metadata={"upload_id": stored.id, "user_id": actor.user_id},
        )
        return CreateUploadSessionResult(upload=stored, session=session)
