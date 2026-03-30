"""Create competition submission usecase."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.competitions.entities import CompetitionSubmission
from mirip_backend.domain.competitions.repositories import (
    CompetitionRepository,
    CompetitionSubmissionRepository,
)
from mirip_backend.domain.uploads.repositories import UploadRepository
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.exceptions import AuthorizationError, NotFoundError
from mirip_backend.shared.ids import new_id


@dataclass(slots=True, frozen=True)
class CreateCompetitionSubmissionCommand:
    competition_id: str
    upload_id: str
    statement: str | None = None


class CreateCompetitionSubmissionUseCase:
    def __init__(
        self,
        competition_repository: CompetitionRepository,
        submission_repository: CompetitionSubmissionRepository,
        upload_repository: UploadRepository,
    ) -> None:
        self._competition_repository = competition_repository
        self._submission_repository = submission_repository
        self._upload_repository = upload_repository

    async def execute(
        self,
        *,
        actor: AuthenticatedUser,
        command: CreateCompetitionSubmissionCommand,
    ) -> CompetitionSubmission:
        competition = await self._competition_repository.get(command.competition_id)
        if competition is None:
            raise NotFoundError("Competition not found")

        upload = await self._upload_repository.get(command.upload_id)
        if upload is None:
            raise NotFoundError("Upload not found")
        if upload.user_id != actor.user_id:
            raise AuthorizationError("Upload ownership mismatch")

        submission = CompetitionSubmission(
            id=new_id("sub"),
            competition_id=command.competition_id,
            user_id=actor.user_id,
            upload_id=command.upload_id,
            statement=command.statement,
            created_at=utc_now(),
        )
        return await self._submission_repository.create(submission)
