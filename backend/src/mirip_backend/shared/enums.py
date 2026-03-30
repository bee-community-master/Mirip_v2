"""Shared enumerations."""

from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


class UploadStatus(StrEnum):
    PENDING = "pending"
    UPLOADED = "uploaded"
    CONSUMED = "consumed"


class CredentialStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"


class Visibility(StrEnum):
    PRIVATE = "private"
    PUBLIC = "public"
