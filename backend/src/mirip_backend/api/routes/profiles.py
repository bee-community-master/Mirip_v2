"""Profile routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from mirip_backend.api.deps.auth import CurrentUserDep
from mirip_backend.api.deps.services import ContainerDep
from mirip_backend.api.schemas.profiles import (
    CreatePortfolioItemRequest,
    PortfolioItemListResponse,
    PortfolioItemResponse,
    ProfileResponse,
    PublicProfileResponse,
    UpsertProfileRequest,
)
from mirip_backend.domain.profiles.entities import PortfolioItem, Profile
from mirip_backend.shared.enums import Visibility
from mirip_backend.usecases.profiles.create_portfolio_item import (
    CreatePortfolioItemCommand,
    CreatePortfolioItemUseCase,
)
from mirip_backend.usecases.profiles.get_my_profile import GetMyProfileUseCase
from mirip_backend.usecases.profiles.get_profile import GetProfileUseCase
from mirip_backend.usecases.profiles.list_portfolio_items import (
    ListPortfolioItemsQuery,
    ListPortfolioItemsUseCase,
)
from mirip_backend.usecases.profiles.upsert_profile import (
    UpsertProfileCommand,
    UpsertProfileUseCase,
)

router = APIRouter(prefix="/v1/profiles", tags=["profiles"])


def _to_profile_response(profile: Profile) -> ProfileResponse:
    return ProfileResponse(
        user_id=profile.user_id,
        handle=profile.handle,
        display_name=profile.display_name,
        bio=profile.bio,
        visibility=profile.visibility.value,
        portfolio_item_ids=profile.portfolio_item_ids,
        updated_at=profile.updated_at,
    )


def _to_portfolio_item_response(item: PortfolioItem) -> PortfolioItemResponse:
    return PortfolioItemResponse(
        id=item.id,
        title=item.title,
        description=item.description,
        asset_upload_id=item.asset_upload_id,
        visibility=item.visibility.value,
        created_at=item.created_at,
    )


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
    return _to_profile_response(profile)


@router.get("/me", response_model=ProfileResponse)
async def get_my_profile(
    current_user: CurrentUserDep,
    container: ContainerDep,
) -> ProfileResponse:
    usecase = GetMyProfileUseCase(container.profile_repository)
    profile = await usecase.execute(actor=current_user)
    return _to_profile_response(profile)


@router.post(
    "/me/portfolio-items",
    response_model=PortfolioItemResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_portfolio_item(
    payload: CreatePortfolioItemRequest,
    current_user: CurrentUserDep,
    container: ContainerDep,
) -> PortfolioItemResponse:
    usecase = CreatePortfolioItemUseCase(
        portfolio_repository=container.portfolio_repository,
        upload_repository=container.upload_repository,
    )
    item = await usecase.execute(
        actor=current_user,
        command=CreatePortfolioItemCommand(
            title=payload.title,
            description=payload.description,
            asset_upload_id=payload.asset_upload_id,
            visibility=Visibility(payload.visibility),
        ),
    )
    return _to_portfolio_item_response(item)


@router.get("/me/portfolio-items", response_model=PortfolioItemListResponse)
async def list_my_portfolio_items(
    current_user: CurrentUserDep,
    container: ContainerDep,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> PortfolioItemListResponse:
    usecase = ListPortfolioItemsUseCase(container.portfolio_repository)
    page = await usecase.execute(
        actor=current_user,
        query=ListPortfolioItemsQuery(limit=limit, offset=offset),
    )
    return PortfolioItemListResponse(
        items=[_to_portfolio_item_response(item) for item in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
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
        profile=_to_profile_response(view.profile),
        portfolio_items=[_to_portfolio_item_response(item) for item in view.portfolio_items],
    )
