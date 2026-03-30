"""Tests for portfolio item creation rules."""

from __future__ import annotations

import pytest

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import (
    DocumentPortfolioRepository,
    DocumentUploadRepository,
)
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import UploadStatus
from mirip_backend.shared.exceptions import ValidationError
from mirip_backend.usecases.profiles.create_portfolio_item import (
    CreatePortfolioItemCommand,
    CreatePortfolioItemUseCase,
)


async def test_create_portfolio_item_requires_uploaded_asset() -> None:
    store = MemoryDocumentStore()
    portfolio_repository = DocumentPortfolioRepository(store)
    upload_repository = DocumentUploadRepository(store)
    await upload_repository.create(
        UploadAsset(
            id="upl-1",
            user_id="user-1",
            filename="piece.png",
            content_type="image/png",
            size_bytes=128,
            object_name="users/user-1/portfolio/upl-1/piece.png",
            status=UploadStatus.PENDING,
            created_at=utc_now(),
        )
    )

    usecase = CreatePortfolioItemUseCase(portfolio_repository, upload_repository)

    with pytest.raises(ValidationError):
        await usecase.execute(
            actor=AuthenticatedUser(user_id="user-1"),
            command=CreatePortfolioItemCommand(
                title="Poster Study",
                asset_upload_id="upl-1",
            ),
        )
