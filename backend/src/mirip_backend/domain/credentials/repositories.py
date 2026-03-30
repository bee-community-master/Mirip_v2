"""Credential repository contracts."""

from __future__ import annotations

from typing import Protocol

from mirip_backend.domain.credentials.entities import Credential


class CredentialRepository(Protocol):
    async def create(self, credential: Credential) -> Credential: ...

    async def get(self, credential_id: str) -> Credential | None: ...
