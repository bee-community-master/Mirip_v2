"""List competitions usecase."""

from __future__ import annotations

from mirip_backend.domain.common.models import Page
from mirip_backend.domain.competitions.entities import Competition
from mirip_backend.domain.competitions.repositories import CompetitionRepository


class ListCompetitionsUseCase:
    def __init__(self, competition_repository: CompetitionRepository) -> None:
        self._competition_repository = competition_repository

    async def execute(self, *, limit: int, offset: int) -> Page[Competition]:
        return await self._competition_repository.list_public(limit=limit, offset=offset)
