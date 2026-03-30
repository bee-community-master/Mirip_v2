"""Competition routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from mirip_backend.api.deps.auth import CurrentUserDep
from mirip_backend.api.deps.services import ContainerDep
from mirip_backend.api.schemas.competitions import (
    CompetitionListResponse,
    CompetitionResponse,
    CompetitionSubmissionResponse,
    CreateCompetitionSubmissionRequest,
)
from mirip_backend.usecases.competitions.create_submission import (
    CreateCompetitionSubmissionCommand,
    CreateCompetitionSubmissionUseCase,
)
from mirip_backend.usecases.competitions.list_competitions import ListCompetitionsUseCase

router = APIRouter(prefix="/v1/competitions", tags=["competitions"])


@router.get("", response_model=CompetitionListResponse)
async def list_competitions(
    container: ContainerDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> CompetitionListResponse:
    usecase = ListCompetitionsUseCase(container.competition_repository)
    page = await usecase.execute(limit=limit, offset=offset)
    return CompetitionListResponse(
        items=[
            CompetitionResponse(
                id=item.id,
                title=item.title,
                description=item.description,
                visibility=item.visibility.value,
                opens_at=item.opens_at,
                closes_at=item.closes_at,
                tags=item.tags,
            )
            for item in page.items
        ],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post(
    "/{competition_id}/submissions",
    response_model=CompetitionSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_submission(
    competition_id: str,
    payload: CreateCompetitionSubmissionRequest,
    current_user: CurrentUserDep,
    container: ContainerDep,
) -> CompetitionSubmissionResponse:
    usecase = CreateCompetitionSubmissionUseCase(
        competition_repository=container.competition_repository,
        submission_repository=container.competition_submission_repository,
        upload_repository=container.upload_repository,
    )
    submission = await usecase.execute(
        actor=current_user,
        command=CreateCompetitionSubmissionCommand(
            competition_id=competition_id,
            upload_id=payload.upload_id,
            statement=payload.statement,
        ),
    )
    return CompetitionSubmissionResponse(
        id=submission.id,
        competition_id=submission.competition_id,
        upload_id=submission.upload_id,
        statement=submission.statement,
        created_at=submission.created_at,
    )
