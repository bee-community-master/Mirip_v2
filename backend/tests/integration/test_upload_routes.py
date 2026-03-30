"""Integration tests for upload routes."""

from __future__ import annotations

from httpx import AsyncClient


async def test_upload_create_complete_and_list_flow_without_auth_header(
    client: AsyncClient,
) -> None:
    create_response = await client.post(
        "/v1/uploads",
        json={
            "filename": "portfolio.png",
            "content_type": "image/png",
            "size_bytes": 1024,
            "category": "portfolio",
        },
    )

    assert create_response.status_code == 201
    upload = create_response.json()["upload"]
    assert upload["status"] == "pending"
    assert upload["category"] == "portfolio"

    complete_response = await client.post(f"/v1/uploads/{upload['id']}/complete")
    assert complete_response.status_code == 200
    assert complete_response.json()["upload"]["status"] == "uploaded"

    list_response = await client.get(
        "/v1/uploads", params={"status": "uploaded", "category": "portfolio"}
    )
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["id"] == upload["id"]
