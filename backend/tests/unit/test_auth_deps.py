"""Tests for authentication dependencies."""

from __future__ import annotations

from types import SimpleNamespace

from starlette.requests import Request

from mirip_backend.api.deps.auth import get_current_user, get_optional_user
from mirip_backend.domain.auth.models import AuthenticatedUser


class StubAuthService:
    async def authenticate(self, authorization: str | None) -> AuthenticatedUser:
        if authorization == "Bearer token":
            return AuthenticatedUser(user_id="user-1")
        raise AssertionError("authenticate should not be called for missing insecure auth")

    def build_insecure_dev_user(self) -> AuthenticatedUser:
        return AuthenticatedUser(user_id="local-dev-user", email="dev@local.test")


def _request(headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers or [],
    }
    return Request(scope)


async def test_get_current_user_uses_insecure_dev_user_without_header() -> None:
    container = SimpleNamespace(
        settings=SimpleNamespace(
            app_env="test",
            firebase=SimpleNamespace(
                allow_insecure_dev_auth=True,
                local_dev_token="local-dev-token",
            ),
        ),
        auth_service=StubAuthService(),
    )

    user = await get_current_user(_request(), container)

    assert user.user_id == "local-dev-user"


async def test_get_optional_user_returns_none_without_header() -> None:
    container = SimpleNamespace(
        settings=SimpleNamespace(
            app_env="test",
            firebase=SimpleNamespace(
                allow_insecure_dev_auth=True,
                local_dev_token="local-dev-token",
            ),
        ),
        auth_service=StubAuthService(),
    )

    user = await get_optional_user(_request(), container)

    assert user is None
