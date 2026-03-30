"""Upload API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CreateUploadSessionRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=256)
    content_type: str = Field(min_length=3, max_length=128)
    size_bytes: int = Field(ge=1, le=25 * 1024 * 1024)
    category: str = Field(default="diagnosis", min_length=1, max_length=64)


class UploadAssetResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    object_name: str
    category: str | None = None
    status: str
    created_at: datetime


class UploadSessionResponse(BaseModel):
    upload_url: str
    method: str
    object_name: str
    headers: dict[str, str]
    expires_at: datetime | None = None


class CreateUploadSessionResponse(BaseModel):
    upload: UploadAssetResponse
    session: UploadSessionResponse


class CompleteUploadResponse(BaseModel):
    upload: UploadAssetResponse


class UploadListResponse(BaseModel):
    items: list[UploadAssetResponse]
    total: int
    limit: int
    offset: int


UploadStatusLiteral = Literal["pending", "uploaded", "consumed"]
