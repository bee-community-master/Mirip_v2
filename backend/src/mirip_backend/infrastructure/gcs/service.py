"""Google Cloud Storage services."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from mirip_backend.domain.common.models import HealthDependency, SignedUploadSession
from mirip_backend.infrastructure.config.settings import GCSSettings
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.exceptions import DependencyError


@dataclass(slots=True)
class GCSStorageService:
    settings: GCSSettings
    backend: str

    def _client(self) -> Any:
        from google.cloud import storage  # type: ignore[attr-defined]

        if self.settings.credentials_path is not None:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                self.settings.credentials_path
            )  # type: ignore[no-untyped-call]
            return storage.Client(project=self.settings.project_id, credentials=credentials)
        return storage.Client(project=self.settings.project_id)

    async def create_upload_session(
        self,
        *,
        object_name: str,
        content_type: str,
        metadata: dict[str, str],
    ) -> SignedUploadSession:
        expires_at = utc_now() + timedelta(minutes=self.settings.upload_url_ttl_minutes)
        if self.backend == "fake" or self.settings.bucket_name is None:
            return SignedUploadSession(
                upload_url=f"https://example.invalid/upload/{object_name}",
                method="PUT",
                object_name=object_name,
                headers={"content-type": content_type, "x-mirip-mode": "fake"},
                expires_at=expires_at,
            )

        def _create_signed_upload_url() -> str:
            client = self._client()
            bucket = client.bucket(self.settings.bucket_name)
            blob = bucket.blob(object_name)
            blob.metadata = metadata
            return str(
                blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(minutes=self.settings.upload_url_ttl_minutes),
                    method="PUT",
                    content_type=content_type,
                )
            )

        try:
            upload_url = await asyncio.to_thread(_create_signed_upload_url)
        except Exception as exc:
            raise DependencyError("GCS upload session creation failed") from exc
        return SignedUploadSession(
            upload_url=upload_url,
            method="PUT",
            object_name=object_name,
            headers={"content-type": content_type},
            expires_at=expires_at,
        )

    async def object_exists(self, *, object_name: str) -> bool:
        if self.backend == "fake" or self.settings.bucket_name is None:
            return True

        def _blob_exists() -> bool:
            client = self._client()
            bucket = client.bucket(self.settings.bucket_name)
            blob = bucket.blob(object_name)
            return bool(blob.exists())

        try:
            return await asyncio.to_thread(_blob_exists)
        except Exception as exc:
            raise DependencyError("GCS object lookup failed") from exc

    async def check(self) -> HealthDependency:
        if self.backend == "fake" or self.settings.bucket_name is None:
            return HealthDependency(name="gcs", status="healthy", detail="fake-storage")

        def _bucket_exists() -> bool:
            client = self._client()
            bucket = client.bucket(self.settings.bucket_name)
            return bool(bucket.exists())

        try:
            exists = await asyncio.to_thread(_bucket_exists)
        except Exception as exc:
            return HealthDependency(name="gcs", status="unhealthy", detail=str(exc))
        return HealthDependency(
            name="gcs",
            status="healthy" if exists else "unknown",
            detail=self.settings.bucket_name,
        )
