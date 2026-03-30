"""Competition API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CompetitionResponse(BaseModel):
    id: str
    title: str
    description: str
    visibility: str
    opens_at: datetime | None = None
    closes_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)


class CompetitionListResponse(BaseModel):
    items: list[CompetitionResponse]
    total: int
    limit: int
    offset: int


class CreateCompetitionSubmissionRequest(BaseModel):
    upload_id: str
    statement: str | None = Field(default=None, max_length=2000)


class CompetitionSubmissionResponse(BaseModel):
    id: str
    competition_id: str
    upload_id: str
    statement: str | None = None
    created_at: datetime
