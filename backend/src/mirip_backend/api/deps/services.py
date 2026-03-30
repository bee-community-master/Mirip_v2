"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from mirip_backend.infrastructure.config.container import ApplicationContainer


def get_container(request: Request) -> ApplicationContainer:
    return request.app.state.container  # type: ignore[no-any-return]


ContainerDep = Annotated[ApplicationContainer, Depends(get_container)]
