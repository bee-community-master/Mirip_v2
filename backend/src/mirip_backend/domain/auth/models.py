"""Authentication domain models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str | None = None
    roles: tuple[str, ...] = field(default_factory=tuple)
    is_service_account: bool = False
