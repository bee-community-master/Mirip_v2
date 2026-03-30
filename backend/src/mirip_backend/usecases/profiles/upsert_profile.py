"""Upsert profile usecase."""

from __future__ import annotations

from dataclasses import dataclass, field

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.profiles.entities import Profile
from mirip_backend.domain.profiles.repositories import ProfileRepository
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import Visibility


@dataclass(slots=True, frozen=True)
class UpsertProfileCommand:
    handle: str
    display_name: str
    bio: str | None = None
    visibility: Visibility = Visibility.PUBLIC
    portfolio_item_ids: list[str] = field(default_factory=list)


class UpsertProfileUseCase:
    def __init__(self, profile_repository: ProfileRepository) -> None:
        self._profile_repository = profile_repository

    async def execute(self, *, actor: AuthenticatedUser, command: UpsertProfileCommand) -> Profile:
        profile = Profile(
            user_id=actor.user_id,
            handle=command.handle,
            display_name=command.display_name,
            bio=command.bio,
            visibility=command.visibility,
            portfolio_item_ids=command.portfolio_item_ids,
            updated_at=utc_now(),
        )
        return await self._profile_repository.upsert(profile)
