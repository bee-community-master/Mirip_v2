"""Create diagnosis job usecase."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.diagnosis.entities import DiagnosisJob
from mirip_backend.domain.diagnosis.metadata import build_diagnosis_job_metadata
from mirip_backend.domain.diagnosis.repositories import DiagnosisJobRepository
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.domain.uploads.repositories import UploadRepository
from mirip_backend.infrastructure.compute.service import DiagnosisVmLauncher
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import JobStatus
from mirip_backend.shared.exceptions import DependencyError, ValidationError
from mirip_backend.shared.ids import new_id
from mirip_backend.usecases.uploads.validation import (
    load_owned_upload,
    require_uploaded_asset,
)


@dataclass(slots=True, frozen=True)
class CreateDiagnosisJobCommand:
    upload_ids: list[str] = field(default_factory=list)
    job_type: str = "evaluate"
    department: str = "visual_design"
    include_feedback: bool = True
    theme: str | None = None
    language: str = "ko"


class CreateDiagnosisJobUseCase:
    """Validate uploads and enqueue a diagnosis job."""

    def __init__(
        self,
        upload_repository: UploadRepository,
        job_repository: DiagnosisJobRepository,
        vm_launcher: DiagnosisVmLauncher | None = None,
        worker_model_uri: str | None = None,
        worker_mode: str = "stub",
    ) -> None:
        self._upload_repository = upload_repository
        self._job_repository = job_repository
        self._vm_launcher = vm_launcher
        self._worker_model_uri = worker_model_uri
        self._worker_mode = worker_mode

    async def execute(
        self,
        *,
        actor: AuthenticatedUser,
        command: CreateDiagnosisJobCommand,
    ) -> DiagnosisJob:
        self._validate_command(command)
        uploads = await self._ensure_uploads_ready(actor=actor, upload_ids=command.upload_ids)
        created = await self._job_repository.create(
            self._build_job(actor=actor, command=command, uploads=uploads)
        )
        return await self._launch_worker_vm(created)

    def _validate_command(self, command: CreateDiagnosisJobCommand) -> None:
        if not command.upload_ids:
            raise ValidationError("At least one upload is required")
        if command.job_type == "compare" and len(command.upload_ids) < 2:
            raise ValidationError("Compare jobs require at least two uploads")

    async def _ensure_uploads_ready(
        self,
        *,
        actor: AuthenticatedUser,
        upload_ids: list[str],
    ) -> list[UploadAsset]:
        uploads: list[UploadAsset] = []
        for upload_id in upload_ids:
            upload = await load_owned_upload(
                upload_repository=self._upload_repository,
                actor=actor,
                upload_id=upload_id,
                not_found_message=f"Upload {upload_id} does not exist",
            )
            require_uploaded_asset(upload, message="Diagnosis jobs require uploaded assets")
            uploads.append(upload)
        return uploads

    def _build_job(
        self,
        *,
        actor: AuthenticatedUser,
        command: CreateDiagnosisJobCommand,
        uploads: list[UploadAsset],
    ) -> DiagnosisJob:
        now = utc_now()
        return DiagnosisJob(
            id=new_id("job"),
            user_id=actor.user_id,
            upload_ids=command.upload_ids,
            job_type=command.job_type,
            department=command.department,
            include_feedback=command.include_feedback,
            theme=command.theme,
            language=command.language,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            metadata=build_diagnosis_job_metadata(
                upload_object_names=[upload.object_name for upload in uploads]
            ),
        )

    async def _launch_worker_vm(self, job: DiagnosisJob) -> DiagnosisJob:
        if self._vm_launcher is None:
            if self._worker_mode == "cpu_onnx":
                return await self._fail_job(
                    job,
                    reason="Worker VM launcher is not configured",
                )
            return job
        if self._worker_mode == "stub":
            return await self._launch_vm(job, model_uri=self._worker_model_uri or "")
        if not self._worker_model_uri:
            return await self._fail_job(
                job,
                reason="Worker model bundle URI is not configured",
            )
        return await self._launch_vm(job, model_uri=self._worker_model_uri)

    async def _launch_vm(self, job: DiagnosisJob, *, model_uri: str) -> DiagnosisJob:
        try:
            launch = await self._vm_launcher.launch_for_job(
                job=job,
                model_uri=model_uri,
                worker_mode=self._worker_mode,
            )
        except DependencyError:
            return await self._fail_job(job, reason="Worker VM launch failed")
        metadata = dict(job.metadata)
        metadata.update(launch.metadata_patch())
        return await self._job_repository.update(
            replace(
                job,
                metadata=metadata,
                updated_at=utc_now(),
            )
        )

    async def _fail_job(self, job: DiagnosisJob, *, reason: str) -> DiagnosisJob:
        return await self._job_repository.update(
            replace(
                job,
                status=JobStatus.FAILED,
                failure_reason=reason,
                updated_at=utc_now(),
            )
        )
