"""Credential API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PublishCredentialRequest(BaseModel):
    result_id: str
    title: str = Field(min_length=1, max_length=120)
    visibility: Literal["public", "private"] = "public"


class CredentialResponse(BaseModel):
    id: str
    result_id: str
    title: str
    status: str
    visibility: str
    created_at: datetime
