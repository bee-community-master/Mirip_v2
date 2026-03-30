"""Create upload session usecase."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.common.models import SignedUploadSession
from mirip_backend.domain.common.ports import UploadSessionSigner
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.domain.uploads.repositories import UploadRepository
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import UploadStatus
from mirip_backend.shared.ids import new_id


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
        object_name = f"users/{actor.user_id}/{command.category}/{upload_id}/{command.filename}"
        upload = UploadAsset(
            id=upload_id,
            user_id=actor.user_id,
            filename=command.filename,
            content_type=command.content_type,
            size_bytes=command.size_bytes,
            object_name=object_name,
            status=UploadStatus.PENDING,
            created_at=utc_now(),
            metadata={"category": command.category},
        )
        stored = await self._upload_repository.create(upload)
        session = await self._signer.create_upload_session(
            object_name=object_name,
            content_type=command.content_type,
            metadata={"upload_id": stored.id, "user_id": actor.user_id},
        )
        return CreateUploadSessionResult(upload=stored, session=session)
