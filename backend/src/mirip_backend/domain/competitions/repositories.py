"""Competition repository contracts."""

from __future__ import annotations

from typing import Protocol

from mirip_backend.domain.common.models import Page
from mirip_backend.domain.competitions.entities import Competition, CompetitionSubmission


class CompetitionRepository(Protocol):
    async def list_public(self, *, limit: int, offset: int) -> Page[Competition]: ...

    async def get(self, competition_id: str) -> Competition | None: ...

    async def create(self, competition: Competition) -> Competition: ...


class CompetitionSubmissionRepository(Protocol):
    async def create(self, submission: CompetitionSubmission) -> CompetitionSubmission: ...

    async def list_by_user(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
    ) -> Page[CompetitionSubmission]: ...
