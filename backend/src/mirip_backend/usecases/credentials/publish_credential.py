"""Publish credential usecase."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.credentials.entities import Credential
from mirip_backend.domain.credentials.repositories import CredentialRepository
from mirip_backend.domain.diagnosis.repositories import DiagnosisResultRepository
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import CredentialStatus, Visibility
from mirip_backend.shared.exceptions import AuthorizationError, NotFoundError
from mirip_backend.shared.ids import new_id


@dataclass(slots=True, frozen=True)
class PublishCredentialCommand:
    result_id: str
    title: str
    visibility: Visibility = Visibility.PUBLIC


class PublishCredentialUseCase:
    def __init__(
        self,
        credential_repository: CredentialRepository,
        result_repository: DiagnosisResultRepository,
    ) -> None:
        self._credential_repository = credential_repository
        self._result_repository = result_repository

    async def execute(
        self,
        *,
        actor: AuthenticatedUser,
        command: PublishCredentialCommand,
    ) -> Credential:
        result = await self._result_repository.get(command.result_id)
        if result is None:
            raise NotFoundError("Diagnosis result not found")
        if result.user_id != actor.user_id:
            raise AuthorizationError()

        credential = Credential(
            id=new_id("cred"),
            user_id=actor.user_id,
            result_id=result.id,
            title=command.title,
            status=CredentialStatus.PUBLISHED,
            visibility=command.visibility,
            created_at=utc_now(),
        )
        return await self._credential_repository.create(credential)
