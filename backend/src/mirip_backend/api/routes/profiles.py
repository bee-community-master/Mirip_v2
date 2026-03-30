"""Profile routes."""

from __future__ import annotations

from fastapi import APIRouter

from mirip_backend.api.deps.auth import CurrentUserDep
from mirip_backend.api.deps.services import ContainerDep
from mirip_backend.api.schemas.profiles import (
    PortfolioItemResponse,
    ProfileResponse,
    PublicProfileResponse,
    UpsertProfileRequest,
)
from mirip_backend.shared.enums import Visibility
from mirip_backend.usecases.profiles.get_profile import GetProfileUseCase
from mirip_backend.usecases.profiles.upsert_profile import (
    UpsertProfileCommand,
    UpsertProfileUseCase,
)

router = APIRouter(prefix="/v1/profiles", tags=["profiles"])


@router.put("/me", response_model=ProfileResponse)
async def upsert_profile(
    payload: UpsertProfileRequest,
    current_user: CurrentUserDep,
    container: ContainerDep,
) -> ProfileResponse:
    usecase = UpsertProfileUseCase(
        profile_repository=container.profile_repository,
        portfolio_repository=container.portfolio_repository,
    )
    profile = await usecase.execute(
        actor=current_user,
        command=UpsertProfileCommand(
            handle=payload.handle,
            display_name=payload.display_name,
            bio=payload.bio,
            visibility=Visibility(payload.visibility),
            portfolio_item_ids=payload.portfolio_item_ids,
        ),
    )
    return ProfileResponse(
        user_id=profile.user_id,
        handle=profile.handle,
        display_name=profile.display_name,
        bio=profile.bio,
        visibility=profile.visibility.value,
        portfolio_item_ids=profile.portfolio_item_ids,
        updated_at=profile.updated_at,
    )


@router.get("/{handle}", response_model=PublicProfileResponse)
async def get_public_profile(
    handle: str,
    container: ContainerDep,
) -> PublicProfileResponse:
    usecase = GetProfileUseCase(
        profile_repository=container.profile_repository,
        portfolio_repository=container.portfolio_repository,
    )
    view = await usecase.execute(handle=handle)
    return PublicProfileResponse(
        profile=ProfileResponse(
            user_id=view.profile.user_id,
            handle=view.profile.handle,
            display_name=view.profile.display_name,
            bio=view.profile.bio,
            visibility=view.profile.visibility.value,
            portfolio_item_ids=view.profile.portfolio_item_ids,
            updated_at=view.profile.updated_at,
        ),
        portfolio_items=[
            PortfolioItemResponse(
                id=item.id,
                title=item.title,
                description=item.description,
                asset_upload_id=item.asset_upload_id,
                visibility=item.visibility.value,
                created_at=item.created_at,
            )
            for item in view.portfolio_items
        ],
    )
