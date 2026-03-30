"""List authenticated portfolio items usecase."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.common.models import Page
from mirip_backend.domain.profiles.entities import PortfolioItem
from mirip_backend.domain.profiles.repositories import PortfolioRepository


@dataclass(slots=True, frozen=True)
class ListPortfolioItemsQuery:
    limit: int = 50
    offset: int = 0


class ListPortfolioItemsUseCase:
    def __init__(self, portfolio_repository: PortfolioRepository) -> None:
        self._portfolio_repository = portfolio_repository

    async def execute(
        self,
        *,
        actor: AuthenticatedUser,
        query: ListPortfolioItemsQuery,
    ) -> Page[PortfolioItem]:
        return await self._portfolio_repository.list_by_user(
            actor.user_id,
            limit=query.limit,
            offset=query.offset,
        )
