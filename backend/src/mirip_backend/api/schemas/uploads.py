"""Upload API schemas."""

from __future__ import annotations

from datetime import datetime

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
