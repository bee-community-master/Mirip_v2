"""Seed local backend data."""

from __future__ import annotations

import asyncio

from mirip_backend.domain.profiles.entities import PortfolioItem, Profile
from mirip_backend.infrastructure.config.container import build_container
from mirip_backend.infrastructure.config.settings import get_settings
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import Visibility


async def main() -> None:
    settings = get_settings()
    container = await build_container(settings)

    profile = Profile(
        user_id="local-dev-user",
        handle="mirip-demo",
        display_name="Mirip Demo Artist",
        bio="Seeded profile for local scaffold testing.",
        visibility=Visibility.PUBLIC,
        portfolio_item_ids=["port-1"],
        updated_at=utc_now(),
    )
    await container.profile_repository.upsert(profile)
    await container.portfolio_repository.create(
        PortfolioItem(
            id="port-1",
            user_id="local-dev-user",
            title="Seeded Poster Study",
            description="Created by the seed script.",
            asset_upload_id="seed-upload-1",
            created_at=utc_now(),
            visibility=Visibility.PUBLIC,
        )
    )
    print("Seeded demo profile and portfolio item.")


if __name__ == "__main__":
    asyncio.run(main())
