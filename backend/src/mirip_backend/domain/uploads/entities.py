"""Upload domain entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from mirip_backend.shared.enums import UploadStatus


@dataclass(slots=True, frozen=True)
class UploadAsset:
    id: str
    user_id: str
    filename: str
    content_type: str
    size_bytes: int
    object_name: str
    status: UploadStatus
    created_at: datetime
    metadata: dict[str, str] = field(default_factory=dict)
