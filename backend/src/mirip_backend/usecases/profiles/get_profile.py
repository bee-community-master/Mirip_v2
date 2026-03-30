"""Get public profile usecase."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.profiles.entities import PortfolioItem, Profile
from mirip_backend.domain.profiles.repositories import PortfolioRepository, ProfileRepository
from mirip_backend.shared.enums import Visibility
from mirip_backend.shared.exceptions import NotFoundError


@dataclass(slots=True, frozen=True)
class PublicProfileView:
    profile: Profile
    portfolio_items: list[PortfolioItem]


class GetProfileUseCase:
    def __init__(
        self,
        profile_repository: ProfileRepository,
        portfolio_repository: PortfolioRepository,
    ) -> None:
        self._profile_repository = profile_repository
        self._portfolio_repository = portfolio_repository

    async def execute(self, *, handle: str) -> PublicProfileView:
        profile = await self._profile_repository.get_by_handle(handle)
        if profile is None or profile.visibility != Visibility.PUBLIC:
            raise NotFoundError("Public profile not found")
        items = await self._portfolio_repository.list_by_ids(profile.portfolio_item_ids)
        public_items = [item for item in items if item.visibility == Visibility.PUBLIC]
        return PublicProfileView(profile=profile, portfolio_items=public_items)
