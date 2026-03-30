"""Profile API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UpsertProfileRequest(BaseModel):
    handle: str = Field(min_length=3, max_length=50)
    display_name: str = Field(min_length=1, max_length=80)
    bio: str | None = Field(default=None, max_length=1000)
    visibility: Literal["public", "private"] = "public"
    portfolio_item_ids: list[str] = Field(default_factory=list)


class PortfolioItemResponse(BaseModel):
    id: str
    title: str
    description: str | None = None
    asset_upload_id: str
    visibility: str
    created_at: datetime


class ProfileResponse(BaseModel):
    user_id: str
    handle: str
    display_name: str
    bio: str | None = None
    visibility: str
    portfolio_item_ids: list[str]
    updated_at: datetime | None = None


class PublicProfileResponse(BaseModel):
    profile: ProfileResponse
    portfolio_items: list[PortfolioItemResponse]


class CreatePortfolioItemRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    asset_upload_id: str = Field(min_length=1, max_length=64)
    visibility: Literal["public", "private"] = "public"


class PortfolioItemListResponse(BaseModel):
    items: list[PortfolioItemResponse]
    total: int
    limit: int
    offset: int
