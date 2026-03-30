"""Credential domain entities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mirip_backend.shared.enums import CredentialStatus, Visibility


@dataclass(slots=True, frozen=True)
class Credential:
    id: str
    user_id: str
    result_id: str
    title: str
    status: CredentialStatus
    visibility: Visibility
    created_at: datetime
