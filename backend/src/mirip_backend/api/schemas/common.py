"""Shared API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str
    message: str
    detail: dict[str, object] | None = None


class HealthDependencyResponse(BaseModel):
    name: str
    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dependencies: list[HealthDependencyResponse] = Field(default_factory=list)
