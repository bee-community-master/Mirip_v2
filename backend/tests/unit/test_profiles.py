"""Tests for profile update rules."""

from __future__ import annotations

import pytest

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.profiles.entities import PortfolioItem, Profile
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import (
    DocumentPortfolioRepository,
    DocumentProfileRepository,
)
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import Visibility
from mirip_backend.shared.exceptions import AuthorizationError, ConflictError
from mirip_backend.usecases.profiles.upsert_profile import (
    UpsertProfileCommand,
    UpsertProfileUseCase,
)


async def test_upsert_profile_rejects_handle_owned_by_another_user() -> None:
    store = MemoryDocumentStore()
    profile_repository = DocumentProfileRepository(store)
    portfolio_repository = DocumentPortfolioRepository(store)
    await profile_repository.upsert(
        Profile(
            user_id="other-user",
            handle="taken-handle",
            display_name="Other",
            bio=None,
            visibility=Visibility.PUBLIC,
            updated_at=utc_now(),
        )
    )
    usecase = UpsertProfileUseCase(profile_repository, portfolio_repository)

    with pytest.raises(ConflictError):
        await usecase.execute(
            actor=AuthenticatedUser(user_id="user-1"),
            command=UpsertProfileCommand(
                handle="taken-handle",
                display_name="Mine",
            ),
        )


async def test_upsert_profile_rejects_foreign_portfolio_items() -> None:
    store = MemoryDocumentStore()
    profile_repository = DocumentProfileRepository(store)
    portfolio_repository = DocumentPortfolioRepository(store)
    await portfolio_repository.create(
        PortfolioItem(
            id="port-1",
            user_id="other-user",
            title="Foreign item",
            description=None,
            asset_upload_id="upl-1",
            created_at=utc_now(),
            visibility=Visibility.PUBLIC,
        )
    )
    usecase = UpsertProfileUseCase(profile_repository, portfolio_repository)

    with pytest.raises(AuthorizationError):
        await usecase.execute(
            actor=AuthenticatedUser(user_id="user-1"),
            command=UpsertProfileCommand(
                handle="my-handle",
                display_name="Mine",
                portfolio_item_ids=["port-1"],
            ),
        )
