"""Diagnosis domain entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mirip_backend.shared.enums import JobStatus


@dataclass(slots=True, frozen=True)
class DiagnosisJob:
    id: str
    user_id: str
    upload_ids: list[str]
    job_type: str
    department: str
    include_feedback: bool
    theme: str | None
    language: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    attempts: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DiagnosisResult:
    id: str
    job_id: str
    user_id: str
    tier: str
    scores: dict[str, float]
    probabilities: list[dict[str, Any]]
    feedback: dict[str, Any] | None
    created_at: datetime
    summary: str | None = None
