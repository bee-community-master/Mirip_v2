"""Profile repository contracts."""

from __future__ import annotations

from typing import Protocol

from mirip_backend.domain.common.models import Page
from mirip_backend.domain.profiles.entities import PortfolioItem, Profile


class ProfileRepository(Protocol):
    async def upsert(self, profile: Profile) -> Profile: ...

    async def get_by_user_id(self, user_id: str) -> Profile | None: ...

    async def get_by_handle(self, handle: str) -> Profile | None: ...


class PortfolioRepository(Protocol):
    async def create(self, item: PortfolioItem) -> PortfolioItem: ...

    async def list_by_user(
        self, user_id: str, *, limit: int, offset: int
    ) -> Page[PortfolioItem]: ...

    async def list_by_ids(self, item_ids: list[str]) -> list[PortfolioItem]: ...
