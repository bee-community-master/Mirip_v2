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


def _make_uploaded_asset(
    *,
    upload_id: str,
    user_id: str = "user-123",
    status: UploadStatus = UploadStatus.UPLOADED,
) -> UploadAsset:
    return UploadAsset(
        id=upload_id,
        user_id=user_id,
        filename="piece.jpg",
        content_type="image/jpeg",
        size_bytes=2048,
        object_name=f"users/{user_id}/diagnosis/{upload_id}/piece.jpg",
        status=status,
        created_at=utc_now(),
    )


async def test_create_diagnosis_job_stores_queued_job() -> None:
    store = MemoryDocumentStore()
    upload_repository = DocumentUploadRepository(store)
    job_repository = DocumentDiagnosisJobRepository(store)
    upload = _make_uploaded_asset(upload_id="upl-1")
    await upload_repository.create(upload)

    usecase = CreateDiagnosisJobUseCase(upload_repository, job_repository)
    job = await usecase.execute(
        actor=AuthenticatedUser(user_id="user-123"),
        command=CreateDiagnosisJobCommand(upload_ids=["upl-1"], department="fine_art"),
    )

    assert job.status == JobStatus.QUEUED
    assert job.user_id == "user-123"
    assert job.metadata["input_object_names"] == [upload.object_name]


async def test_create_diagnosis_job_rejects_foreign_upload() -> None:
    store = MemoryDocumentStore()
    upload_repository = DocumentUploadRepository(store)
    job_repository = DocumentDiagnosisJobRepository(store)
    await upload_repository.create(_make_uploaded_asset(upload_id="upl-2", user_id="other-user"))

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
        _make_uploaded_asset(upload_id="upl-3", status=UploadStatus.PENDING)
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
    upload = _make_uploaded_asset(upload_id="upl-4")
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
    assert job.metadata["input_object_names"] == [upload.object_name]


async def test_create_diagnosis_job_fails_fast_when_cpu_onnx_launcher_is_missing() -> None:
    store = MemoryDocumentStore()
    upload_repository = DocumentUploadRepository(store)
    job_repository = DocumentDiagnosisJobRepository(store)
    upload = _make_uploaded_asset(upload_id="upl-5")
    await upload_repository.create(upload)

    usecase = CreateDiagnosisJobUseCase(
        upload_repository,
        job_repository,
        vm_launcher=None,
        worker_model_uri="gs://mirip-v2-assets/models/vitl-cpu-bundle",
        worker_mode="cpu_onnx",
    )
    job = await usecase.execute(
        actor=AuthenticatedUser(user_id="user-123"),
        command=CreateDiagnosisJobCommand(upload_ids=["upl-5"], department="fine_art"),
    )

    assert job.status == JobStatus.FAILED
    assert job.failure_reason == "Worker VM launcher is not configured"
