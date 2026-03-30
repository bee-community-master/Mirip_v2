"""Tests for diagnosis job creation."""

from __future__ import annotations

import pytest

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import (
    DocumentDiagnosisJobRepository,
    DocumentUploadRepository,
)
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import JobStatus, UploadStatus
from mirip_backend.shared.exceptions import AuthorizationError
from mirip_backend.usecases.diagnosis.create_job import (
    CreateDiagnosisJobCommand,
    CreateDiagnosisJobUseCase,
)


async def test_create_diagnosis_job_stores_queued_job() -> None:
    store = MemoryDocumentStore()
    upload_repository = DocumentUploadRepository(store)
    job_repository = DocumentDiagnosisJobRepository(store)
    upload = UploadAsset(
        id="upl-1",
        user_id="user-123",
        filename="piece.jpg",
        content_type="image/jpeg",
        size_bytes=2048,
        object_name="users/user-123/diagnosis/upl-1/piece.jpg",
        status=UploadStatus.UPLOADED,
        created_at=utc_now(),
    )
    await upload_repository.create(upload)

    usecase = CreateDiagnosisJobUseCase(upload_repository, job_repository)
    job = await usecase.execute(
        actor=AuthenticatedUser(user_id="user-123"),
        command=CreateDiagnosisJobCommand(upload_ids=["upl-1"], department="fine_art"),
    )

    assert job.status == JobStatus.QUEUED
    assert job.user_id == "user-123"


async def test_create_diagnosis_job_rejects_foreign_upload() -> None:
    store = MemoryDocumentStore()
    upload_repository = DocumentUploadRepository(store)
    job_repository = DocumentDiagnosisJobRepository(store)
    await upload_repository.create(
        UploadAsset(
            id="upl-2",
            user_id="other-user",
            filename="piece.jpg",
            content_type="image/jpeg",
            size_bytes=2048,
            object_name="users/other-user/diagnosis/upl-2/piece.jpg",
            status=UploadStatus.UPLOADED,
            created_at=utc_now(),
        )
    )

    usecase = CreateDiagnosisJobUseCase(upload_repository, job_repository)

    with pytest.raises(AuthorizationError):
        await usecase.execute(
            actor=AuthenticatedUser(user_id="user-123"),
            command=CreateDiagnosisJobCommand(upload_ids=["upl-2"]),
        )
