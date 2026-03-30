"""Integration tests for editable profile routes."""

from __future__ import annotations

from httpx import AsyncClient


async def test_profile_me_and_portfolio_item_routes(client: AsyncClient) -> None:
    upload_response = await client.post(
        "/v1/uploads",
        json={
            "filename": "portfolio.png",
            "content_type": "image/png",
            "size_bytes": 1024,
            "category": "portfolio",
        },
    )
    upload_id = upload_response.json()["upload"]["id"]
    await client.post(f"/v1/uploads/{upload_id}/complete")

    create_item_response = await client.post(
        "/v1/profiles/me/portfolio-items",
        json={
            "title": "Poster Study",
            "description": "Route integration item",
            "asset_upload_id": upload_id,
            "visibility": "public",
        },
    )
    assert create_item_response.status_code == 201
    portfolio_item_id = create_item_response.json()["id"]

    upsert_profile_response = await client.put(
        "/v1/profiles/me",
        json={
            "handle": "route-user",
            "display_name": "Route User",
            "bio": "Profile route integration",
            "visibility": "public",
            "portfolio_item_ids": [portfolio_item_id],
        },
    )
    assert upsert_profile_response.status_code == 200

    my_profile_response = await client.get("/v1/profiles/me")
    assert my_profile_response.status_code == 200
    assert my_profile_response.json()["handle"] == "route-user"

    my_items_response = await client.get("/v1/profiles/me/portfolio-items")
    assert my_items_response.status_code == 200
    assert my_items_response.json()["items"][0]["id"] == portfolio_item_id
