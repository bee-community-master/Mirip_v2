"""Firebase authentication provider."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from threading import Lock

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.common.models import HealthDependency
from mirip_backend.infrastructure.config.settings import FirebaseSettings
from mirip_backend.shared.exceptions import AuthenticationError, DependencyError


@dataclass(slots=True)
class FirebaseAuthService:
    settings: FirebaseSettings
    _initialized: bool = False
    _init_lock: Lock = field(default_factory=Lock)

    def _initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            if not self.settings.project_id and not self.settings.credentials_path:
                return

            import firebase_admin
            from firebase_admin import credentials

            try:
                firebase_admin.get_app()
                self._initialized = True
                return
            except ValueError:
                pass

            if self.settings.credentials_path:
                credential = credentials.Certificate(self.settings.credentials_path)
                firebase_admin.initialize_app(credential, {"projectId": self.settings.project_id})
            else:
                firebase_admin.initialize_app(options={"projectId": self.settings.project_id})
            self._initialized = True

    async def authenticate(self, authorization: str | None) -> AuthenticatedUser:
        if authorization is None or not authorization.startswith("Bearer "):
            raise AuthenticationError()

        token = authorization.removeprefix("Bearer ").strip()
        if self.settings.allow_insecure_dev_auth and token == self.settings.local_dev_token:
            return AuthenticatedUser(user_id="local-dev-user", email="dev@local.test")

        await asyncio.to_thread(self._initialize)
        if not self._initialized:
            raise DependencyError("Firebase auth is not configured for token verification")

        from firebase_admin import auth

        decoded = await asyncio.to_thread(auth.verify_id_token, token)
        email = decoded.get("email")
        roles = tuple(str(role) for role in decoded.get("roles", ()))
        return AuthenticatedUser(
            user_id=str(decoded["uid"]),
            email=str(email) if email is not None else None,
            roles=roles,
            is_service_account=bool(decoded.get("admin", False)),
        )

    async def check(self) -> HealthDependency:
        if self.settings.allow_insecure_dev_auth and not self.settings.project_id:
            return HealthDependency(name="firebase_auth", status="healthy", detail="local-dev-auth")
        if not self.settings.project_id and not self.settings.credentials_path:
            return HealthDependency(name="firebase_auth", status="unknown", detail="not-configured")
        await asyncio.to_thread(self._initialize)
        return HealthDependency(name="firebase_auth", status="healthy", detail="initialized")
