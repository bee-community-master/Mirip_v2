"""Tests for diagnosis job creation."""

from __future__ import annotations

import pytest

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.infrastructure.compute.service import DiagnosisVmLaunchResult
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import (
    DocumentDiagnosisJobRepository,
    DocumentUploadRepository,
)
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import JobStatus, UploadStatus
from mirip_backend.shared.exceptions import AuthorizationError, ValidationError
from mirip_backend.usecases.diagnosis.create_job import (
    CreateDiagnosisJobCommand,
    CreateDiagnosisJobUseCase,
)


class FakeVmLauncher:
    async def launch_for_job(self, *, job, model_uri: str, worker_mode: str):  # type: ignore[no-untyped-def]
        return DiagnosisVmLaunchResult(
            instance_name="mirip-diagnosis-job-123",
            zone="asia-northeast3-b",
            launch_state="launched",
            model_bundle_uri=model_uri,
            target_job_id=job.id,
        )

    async def delete_instance(self, *, instance_name: str, zone: str) -> None:
        return None


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


async def test_create_diagnosis_job_rejects_pending_upload() -> None:
    store = MemoryDocumentStore()
    upload_repository = DocumentUploadRepository(store)
    job_repository = DocumentDiagnosisJobRepository(store)
    await upload_repository.create(
        UploadAsset(
            id="upl-3",
            user_id="user-123",
            filename="piece.jpg",
            content_type="image/jpeg",
            size_bytes=2048,
            object_name="users/user-123/diagnosis/upl-3/piece.jpg",
            status=UploadStatus.PENDING,
            created_at=utc_now(),
        )
    )

    usecase = CreateDiagnosisJobUseCase(upload_repository, job_repository)

    with pytest.raises(ValidationError):
        await usecase.execute(
            actor=AuthenticatedUser(user_id="user-123"),
            command=CreateDiagnosisJobCommand(upload_ids=["upl-3"]),
        )


async def test_create_diagnosis_job_records_vm_launch_metadata() -> None:
    store = MemoryDocumentStore()
    upload_repository = DocumentUploadRepository(store)
    job_repository = DocumentDiagnosisJobRepository(store)
    upload = UploadAsset(
        id="upl-4",
        user_id="user-123",
        filename="piece.jpg",
        content_type="image/jpeg",
        size_bytes=2048,
        object_name="users/user-123/diagnosis/upl-4/piece.jpg",
        status=UploadStatus.UPLOADED,
        created_at=utc_now(),
    )
    await upload_repository.create(upload)

    usecase = CreateDiagnosisJobUseCase(
        upload_repository,
        job_repository,
        vm_launcher=FakeVmLauncher(),
        worker_model_uri="gs://mirip-v2-assets/models/vitl-cpu-bundle",
        worker_mode="cpu_onnx",
    )
    job = await usecase.execute(
        actor=AuthenticatedUser(user_id="user-123"),
        command=CreateDiagnosisJobCommand(upload_ids=["upl-4"], department="fine_art"),
    )

    assert job.status == JobStatus.QUEUED
    assert job.metadata["launch_state"] == "launched"
    assert job.metadata["model_bundle_uri"] == "gs://mirip-v2-assets/models/vitl-cpu-bundle"
    assert job.metadata["target_job_id"] == job.id
