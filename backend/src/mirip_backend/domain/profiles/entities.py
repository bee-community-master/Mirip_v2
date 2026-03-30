"""Profile domain entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from mirip_backend.shared.enums import Visibility


@dataclass(slots=True, frozen=True)
class Profile:
    user_id: str
    handle: str
    display_name: str
    bio: str | None
    visibility: Visibility
    portfolio_item_ids: list[str] = field(default_factory=list)
    updated_at: datetime | None = None


@dataclass(slots=True, frozen=True)
class PortfolioItem:
    id: str
    user_id: str
    title: str
    description: str | None
    asset_upload_id: str
    created_at: datetime
    visibility: Visibility
