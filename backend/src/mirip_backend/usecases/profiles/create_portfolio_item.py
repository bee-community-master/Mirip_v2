"""Create portfolio item usecase."""

from __future__ import annotations

from dataclasses import dataclass

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.profiles.entities import PortfolioItem
from mirip_backend.domain.profiles.repositories import PortfolioRepository
from mirip_backend.domain.uploads.repositories import UploadRepository
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import Visibility
from mirip_backend.shared.ids import new_id
from mirip_backend.usecases.uploads.validation import (
    load_owned_upload,
    require_uploaded_asset,
)


@dataclass(slots=True, frozen=True)
class CreatePortfolioItemCommand:
    title: str
    description: str | None = None
    asset_upload_id: str = ""
    visibility: Visibility = Visibility.PUBLIC


class CreatePortfolioItemUseCase:
    """Create a portfolio item from an uploaded asset owned by the user."""

    def __init__(
        self,
        portfolio_repository: PortfolioRepository,
        upload_repository: UploadRepository,
    ) -> None:
        self._portfolio_repository = portfolio_repository
        self._upload_repository = upload_repository

    async def execute(
        self,
        *,
        actor: AuthenticatedUser,
        command: CreatePortfolioItemCommand,
    ) -> PortfolioItem:
        upload = await load_owned_upload(
            upload_repository=self._upload_repository,
            actor=actor,
            upload_id=command.asset_upload_id,
        )
        require_uploaded_asset(
            upload,
            message="Portfolio items require an uploaded asset",
        )

        item = PortfolioItem(
            id=new_id("port"),
            user_id=actor.user_id,
            title=command.title,
            description=command.description,
            asset_upload_id=command.asset_upload_id,
            created_at=utc_now(),
            visibility=command.visibility,
        )
        return await self._portfolio_repository.create(item)
