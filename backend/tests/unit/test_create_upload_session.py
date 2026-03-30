"""Tests for the upload session usecase."""

from __future__ import annotations

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.common.models import SignedUploadSession
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import DocumentUploadRepository
from mirip_backend.usecases.uploads.create_upload_session import (
    CreateUploadSessionCommand,
    CreateUploadSessionUseCase,
)


class FakeSigner:
    async def create_upload_session(
        self,
        *,
        object_name: str,
        content_type: str,
        metadata: dict[str, str],
    ) -> SignedUploadSession:
        return SignedUploadSession(
            upload_url=f"https://example.invalid/{object_name}",
            method="PUT",
            object_name=object_name,
            headers={"content-type": content_type, **metadata},
        )


async def test_create_upload_session_uses_user_scoped_object_path() -> None:
    repository = DocumentUploadRepository(MemoryDocumentStore())
    usecase = CreateUploadSessionUseCase(repository, FakeSigner())
    actor = AuthenticatedUser(user_id="user-123")

    result = await usecase.execute(
        actor=actor,
        command=CreateUploadSessionCommand(
            filename="sample.png",
            content_type="image/png",
            size_bytes=1024,
            category="diagnosis",
        ),
    )

    assert result.upload.object_name.startswith("users/user-123/diagnosis/")
    assert result.session.object_name == result.upload.object_name
