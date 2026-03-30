"""Shared application exceptions."""

from __future__ import annotations

from typing import Any


class MiripError(Exception):
    """Base application error."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.detail = detail or {}


class AuthenticationError(MiripError):
    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(message, code="AUTHENTICATION_ERROR", status_code=401)


class AuthorizationError(MiripError):
    def __init__(self, message: str = "Not allowed to access this resource") -> None:
        super().__init__(message, code="AUTHORIZATION_ERROR", status_code=403)


class NotFoundError(MiripError):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, code="NOT_FOUND", status_code=404)


class ConflictError(MiripError):
    def __init__(self, message: str = "Resource conflict") -> None:
        super().__init__(message, code="CONFLICT", status_code=409)


class ValidationError(MiripError):
    def __init__(self, message: str = "Validation failed") -> None:
        super().__init__(message, code="VALIDATION_ERROR", status_code=400)


class DependencyError(MiripError):
    def __init__(self, message: str = "Dependency unavailable") -> None:
        super().__init__(message, code="DEPENDENCY_ERROR", status_code=503)
