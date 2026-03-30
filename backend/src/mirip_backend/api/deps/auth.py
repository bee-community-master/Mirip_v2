"""Authentication dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from mirip_backend.api.deps.services import ContainerDep
from mirip_backend.domain.auth.models import AuthenticatedUser


def _should_use_insecure_dev_auth(container: ContainerDep) -> bool:
    return (
        container.settings.app_env in {"local", "test"}
        and container.settings.firebase.allow_insecure_dev_auth
    )


async def get_current_user(
    request: Request,
    container: ContainerDep,
) -> AuthenticatedUser:
    authorization = request.headers.get("Authorization")
    if authorization is None and _should_use_insecure_dev_auth(container):
        return container.auth_service.build_insecure_dev_user()
    return await container.auth_service.authenticate(authorization)


async def get_optional_user(
    request: Request,
    container: ContainerDep,
) -> AuthenticatedUser | None:
    authorization = request.headers.get("Authorization")
    if authorization is None:
        return None
    return await container.auth_service.authenticate(authorization)


CurrentUserDep = Annotated[AuthenticatedUser, Depends(get_current_user)]
OptionalCurrentUserDep = Annotated[AuthenticatedUser | None, Depends(get_optional_user)]
