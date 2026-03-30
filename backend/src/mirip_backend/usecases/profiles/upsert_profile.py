"""Upsert profile usecase."""

from __future__ import annotations

from dataclasses import dataclass, field

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.profiles.entities import Profile
from mirip_backend.domain.profiles.repositories import PortfolioRepository, ProfileRepository
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import Visibility
from mirip_backend.shared.exceptions import AuthorizationError, ConflictError, NotFoundError


@dataclass(slots=True, frozen=True)
class UpsertProfileCommand:
    handle: str
    display_name: str
    bio: str | None = None
    visibility: Visibility = Visibility.PUBLIC
    portfolio_item_ids: list[str] = field(default_factory=list)


class UpsertProfileUseCase:
    def __init__(
        self,
        profile_repository: ProfileRepository,
        portfolio_repository: PortfolioRepository,
    ) -> None:
        self._profile_repository = profile_repository
        self._portfolio_repository = portfolio_repository

    async def execute(self, *, actor: AuthenticatedUser, command: UpsertProfileCommand) -> Profile:
        existing_profile = await self._profile_repository.get_by_handle(command.handle)
        if existing_profile is not None and existing_profile.user_id != actor.user_id:
            raise ConflictError("Handle is already in use")

        portfolio_items = await self._portfolio_repository.list_by_ids(command.portfolio_item_ids)
        if len(portfolio_items) != len(command.portfolio_item_ids):
            raise NotFoundError("One or more portfolio items do not exist")
        if any(item.user_id != actor.user_id for item in portfolio_items):
            raise AuthorizationError("Portfolio items must belong to the authenticated user")

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
