"""Tests for the upload session usecase."""

from __future__ import annotations

import pytest

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.common.models import SignedUploadSession
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import DocumentUploadRepository
from mirip_backend.shared.exceptions import ValidationError
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

    async def object_exists(self, *, object_name: str) -> bool:
        return True


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


async def test_create_upload_session_sanitizes_object_path_segments() -> None:
    repository = DocumentUploadRepository(MemoryDocumentStore())
    usecase = CreateUploadSessionUseCase(repository, FakeSigner())

    result = await usecase.execute(
        actor=AuthenticatedUser(user_id="user-123"),
        command=CreateUploadSessionCommand(
            filename="../../my concept.png",
            content_type="image/png",
            size_bytes=1024,
            category="Competition Entries",
        ),
    )

    assert "/Competition Entries/" not in result.upload.object_name
    assert result.upload.object_name.endswith("/my-concept.png")
    assert "/competition-entries/" in result.upload.object_name
    assert result.upload.metadata["original_filename"] == "../../my concept.png"


async def test_create_upload_session_rejects_empty_safe_filename() -> None:
    repository = DocumentUploadRepository(MemoryDocumentStore())
    usecase = CreateUploadSessionUseCase(repository, FakeSigner())

    with pytest.raises(ValidationError):
        await usecase.execute(
            actor=AuthenticatedUser(user_id="user-123"),
            command=CreateUploadSessionCommand(
                filename="////",
                content_type="image/png",
                size_bytes=1024,
                category="diagnosis",
            ),
        )
