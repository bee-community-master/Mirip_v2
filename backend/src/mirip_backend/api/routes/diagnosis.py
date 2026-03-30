"""Diagnosis routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from mirip_backend.api.deps.auth import CurrentUserDep
from mirip_backend.api.deps.services import ContainerDep
from mirip_backend.api.schemas.diagnosis import (
    CreateDiagnosisJobRequest,
    DiagnosisHistoryResponse,
    DiagnosisJobResponse,
    DiagnosisJobStatusResponse,
    DiagnosisResultResponse,
)
from mirip_backend.usecases.diagnosis.create_job import (
    CreateDiagnosisJobCommand,
    CreateDiagnosisJobUseCase,
)
from mirip_backend.usecases.diagnosis.get_job_status import GetDiagnosisJobStatusUseCase
from mirip_backend.usecases.diagnosis.list_history import ListDiagnosisHistoryUseCase

router = APIRouter(prefix="/v1/diagnosis", tags=["diagnosis"])


@router.post("/jobs", response_model=DiagnosisJobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    payload: CreateDiagnosisJobRequest,
    current_user: CurrentUserDep,
    container: ContainerDep,
) -> DiagnosisJobResponse:
    usecase = CreateDiagnosisJobUseCase(
        upload_repository=container.upload_repository,
        job_repository=container.diagnosis_job_repository,
    )
    job = await usecase.execute(
        actor=current_user,
        command=CreateDiagnosisJobCommand(
            upload_ids=payload.upload_ids,
            job_type=payload.job_type,
            department=payload.department,
            include_feedback=payload.include_feedback,
            theme=payload.theme,
            language=payload.language,
        ),
    )
    return DiagnosisJobResponse(
        id=job.id,
        job_type=job.job_type,
        department=job.department,
        status=job.status.value,
        upload_ids=job.upload_ids,
        created_at=job.created_at,
        updated_at=job.updated_at,
        attempts=job.attempts,
        failure_reason=job.failure_reason,
    )


@router.get("/jobs/{job_id}", response_model=DiagnosisJobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: CurrentUserDep,
    container: ContainerDep,
) -> DiagnosisJobStatusResponse:
    usecase = GetDiagnosisJobStatusUseCase(
        job_repository=container.diagnosis_job_repository,
        result_repository=container.diagnosis_result_repository,
    )
    view = await usecase.execute(actor=current_user, job_id=job_id)
    result = None
    if view.result is not None:
        result = DiagnosisResultResponse(
            id=view.result.id,
            job_id=view.result.job_id,
            tier=view.result.tier,
            scores=view.result.scores,
            probabilities=view.result.probabilities,
            feedback=view.result.feedback,
            summary=view.result.summary,
            created_at=view.result.created_at,
        )
    return DiagnosisJobStatusResponse(
        job=DiagnosisJobResponse(
            id=view.job.id,
            job_type=view.job.job_type,
            department=view.job.department,
            status=view.job.status.value,
            upload_ids=view.job.upload_ids,
            created_at=view.job.created_at,
            updated_at=view.job.updated_at,
            attempts=view.job.attempts,
            failure_reason=view.job.failure_reason,
        ),
        result=result,
    )


@router.get("/history", response_model=DiagnosisHistoryResponse)
async def get_history(
    current_user: CurrentUserDep,
    container: ContainerDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> DiagnosisHistoryResponse:
    usecase = ListDiagnosisHistoryUseCase(container.diagnosis_result_repository)
    page = await usecase.execute(actor=current_user, limit=limit, offset=offset)
    return DiagnosisHistoryResponse(
        items=[
            DiagnosisResultResponse(
                id=item.id,
                job_id=item.job_id,
                tier=item.tier,
                scores=item.scores,
                probabilities=item.probabilities,
                feedback=item.feedback,
                summary=item.summary,
                created_at=item.created_at,
            )
            for item in page.items
        ],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )
