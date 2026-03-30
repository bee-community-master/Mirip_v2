"""Diagnosis API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DepartmentType = Literal["visual_design", "industrial_design", "fine_art", "craft"]
LanguageType = Literal["ko", "en"]
JobType = Literal["evaluate", "compare"]


class CreateDiagnosisJobRequest(BaseModel):
    upload_ids: list[str] = Field(min_length=1, max_length=10)
    job_type: JobType = "evaluate"
    department: DepartmentType = "visual_design"
    include_feedback: bool = True
    theme: str | None = Field(default=None, max_length=200)
    language: LanguageType = "ko"


class DiagnosisJobResponse(BaseModel):
    id: str
    job_type: str
    department: str
    status: str
    upload_ids: list[str]
    created_at: datetime
    updated_at: datetime
    attempts: int
    failure_reason: str | None = None


class DiagnosisResultResponse(BaseModel):
    id: str
    job_id: str
    tier: str
    scores: dict[str, float]
    probabilities: list[dict[str, object]]
    feedback: dict[str, object] | None = None
    summary: str | None = None
    created_at: datetime


class DiagnosisJobStatusResponse(BaseModel):
    job: DiagnosisJobResponse
    result: DiagnosisResultResponse | None = None


class DiagnosisHistoryResponse(BaseModel):
    items: list[DiagnosisResultResponse]
    total: int
    limit: int
    offset: int
