"""Application settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseModel):
    title: str = "Mirip v2 API"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


class FirebaseSettings(BaseModel):
    project_id: str | None = None
    credentials_path: str | None = None
    allow_insecure_dev_auth: bool = True
    local_dev_token: str = "local-dev-token"


class GCSSettings(BaseModel):
    bucket_name: str | None = None
    upload_url_ttl_minutes: int = 15


class JobSettings(BaseModel):
    lease_seconds: int = 300
    max_attempts: int = 5
    worker_poll_interval_seconds: int = 5


class WorkerSettings(BaseModel):
    mode: Literal["stub", "gpu"] = "stub"
    model_uri: str | None = None
    run_once: bool = False


class ObservabilitySettings(BaseModel):
    log_level: str = "INFO"
    json_logs: bool = True


class Settings(BaseSettings):
    """Top-level settings with nested sections."""

    model_config = SettingsConfigDict(
        env_prefix="MIRIP_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "test", "prod"] = "local"
    data_backend: Literal["memory", "firestore"] = "memory"
    storage_backend: Literal["fake", "gcs"] = "fake"
    api: ApiSettings = Field(default_factory=ApiSettings)
    firebase: FirebaseSettings = Field(default_factory=FirebaseSettings)
    gcs: GCSSettings = Field(default_factory=GCSSettings)
    job: JobSettings = Field(default_factory=JobSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
