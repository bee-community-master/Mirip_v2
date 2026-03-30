"""Tests for upload completion and filtering."""

from __future__ import annotations

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import DocumentUploadRepository
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import UploadStatus
from mirip_backend.usecases.uploads.complete_upload import CompleteUploadUseCase
from mirip_backend.usecases.uploads.list_uploads import ListUploadsQuery, ListUploadsUseCase


async def test_complete_upload_marks_pending_upload_as_uploaded() -> None:
    repository = DocumentUploadRepository(MemoryDocumentStore())
    await repository.create(
        UploadAsset(
            id="upl-1",
            user_id="user-1",
            filename="piece.png",
            content_type="image/png",
            size_bytes=128,
            object_name="users/user-1/diagnosis/upl-1/piece.png",
            status=UploadStatus.PENDING,
            created_at=utc_now(),
            metadata={"category": "diagnosis"},
        )
    )

    usecase = CompleteUploadUseCase(repository)
    upload = await usecase.execute(actor=AuthenticatedUser(user_id="user-1"), upload_id="upl-1")

    assert upload.status == UploadStatus.UPLOADED


async def test_list_uploads_filters_by_category_and_status() -> None:
    repository = DocumentUploadRepository(MemoryDocumentStore())
    await repository.create(
        UploadAsset(
            id="upl-1",
            user_id="user-1",
            filename="piece.png",
            content_type="image/png",
            size_bytes=128,
            object_name="users/user-1/diagnosis/upl-1/piece.png",
            status=UploadStatus.UPLOADED,
            created_at=utc_now(),
            metadata={"category": "diagnosis"},
        )
    )
    await repository.create(
        UploadAsset(
            id="upl-2",
            user_id="user-1",
            filename="poster.png",
            content_type="image/png",
            size_bytes=128,
            object_name="users/user-1/portfolio/upl-2/poster.png",
            status=UploadStatus.PENDING,
            created_at=utc_now(),
            metadata={"category": "portfolio"},
        )
    )

    usecase = ListUploadsUseCase(repository)
    page = await usecase.execute(
        actor=AuthenticatedUser(user_id="user-1"),
        query=ListUploadsQuery(category="diagnosis", status=UploadStatus.UPLOADED),
    )

    assert page.total == 1
    assert page.items[0].id == "upl-1"
