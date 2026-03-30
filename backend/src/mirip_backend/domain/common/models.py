"""Common domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True, frozen=True)
class SignedUploadSession:
    upload_url: str
    method: str
    object_name: str
    headers: dict[str, str] = field(default_factory=dict)
    expires_at: datetime | None = None


@dataclass(slots=True, frozen=True)
class HealthDependency:
    name: str
    status: str
    detail: str | None = None


@dataclass(slots=True, frozen=True)
class Page[T]:
    items: list[T]
    total: int
    limit: int
    offset: int
