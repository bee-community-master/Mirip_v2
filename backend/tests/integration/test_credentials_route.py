"""Integration tests for credential access control."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import AsyncClient

from mirip_backend.domain.credentials.entities import Credential
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import CredentialStatus, Visibility


async def test_public_credential_is_readable_without_auth(
    client: AsyncClient,
    app: FastAPI,
) -> None:
    repository = app.state.container.credential_repository
    credential = Credential(
        id="cred-public",
        user_id="owner-1",
        result_id="res-1",
        title="Public credential",
        status=CredentialStatus.PUBLISHED,
        visibility=Visibility.PUBLIC,
        created_at=utc_now(),
    )
    await repository.create(credential)

    response = await client.get("/v1/credentials/cred-public")

    assert response.status_code == 200
    assert response.json()["id"] == "cred-public"


async def test_private_credential_requires_owner_auth(
    client: AsyncClient,
    app: FastAPI,
    auth_headers: dict[str, str],
) -> None:
    repository = app.state.container.credential_repository
    credential = Credential(
        id="cred-private",
        user_id="local-dev-user",
        result_id="res-2",
        title="Private credential",
        status=CredentialStatus.PUBLISHED,
        visibility=Visibility.PRIVATE,
        created_at=utc_now(),
    )
    await repository.create(credential)

    anonymous_response = await client.get("/v1/credentials/cred-private")
    assert anonymous_response.status_code == 404

    owner_response = await client.get("/v1/credentials/cred-private", headers=auth_headers)
    assert owner_response.status_code == 200
    assert owner_response.json()["visibility"] == "private"
