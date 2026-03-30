"""Get authenticated profile usecase."""

from __future__ import annotations

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.profiles.entities import Profile
from mirip_backend.domain.profiles.repositories import ProfileRepository
from mirip_backend.shared.exceptions import NotFoundError


class GetMyProfileUseCase:
    def __init__(self, profile_repository: ProfileRepository) -> None:
        self._profile_repository = profile_repository

    async def execute(self, *, actor: AuthenticatedUser) -> Profile:
        profile = await self._profile_repository.get_by_user_id(actor.user_id)
        if profile is None:
            raise NotFoundError("Profile not found")
        return profile
