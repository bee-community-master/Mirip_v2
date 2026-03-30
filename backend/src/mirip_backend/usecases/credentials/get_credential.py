"""Get credential usecase."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.credentials.entities import Credential
from mirip_backend.domain.credentials.repositories import CredentialRepository
from mirip_backend.shared.enums import Visibility
from mirip_backend.shared.exceptions import NotFoundError


@dataclass(slots=True, frozen=True)
class CredentialView:
    credential: Credential


class GetCredentialUseCase:
    """Load a credential while enforcing public/private visibility rules."""

    def __init__(self, credential_repository: CredentialRepository) -> None:
        self._credential_repository = credential_repository

    async def execute(
        self,
        *,
        actor: AuthenticatedUser | None,
        credential_id: str,
    ) -> CredentialView:
        credential = await self._credential_repository.get(credential_id)
        if credential is None:
            raise NotFoundError("Credential not found")
        if credential.visibility != Visibility.PUBLIC and (
            actor is None or credential.user_id != actor.user_id
        ):
            raise NotFoundError("Credential not found")
        return CredentialView(credential=credential)
