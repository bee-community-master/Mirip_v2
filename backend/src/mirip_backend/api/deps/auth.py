"""Authentication dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from mirip_backend.api.deps.services import ContainerDep
from mirip_backend.domain.auth.models import AuthenticatedUser


async def get_current_user(
    request: Request,
    container: ContainerDep,
) -> AuthenticatedUser:
    authorization = request.headers.get("Authorization")
    return await container.auth_service.authenticate(authorization)


CurrentUserDep = Annotated[AuthenticatedUser, Depends(get_current_user)]
