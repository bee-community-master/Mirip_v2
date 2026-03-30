"""Competition domain entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from mirip_backend.shared.enums import Visibility


@dataclass(slots=True, frozen=True)
class Competition:
    id: str
    title: str
    description: str
    visibility: Visibility
    opens_at: datetime | None = None
    closes_at: datetime | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class CompetitionSubmission:
    id: str
    competition_id: str
    user_id: str
    upload_id: str
    statement: str | None
    created_at: datetime
