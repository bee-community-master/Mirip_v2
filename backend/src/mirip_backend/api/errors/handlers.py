"""Exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mirip_backend.api.schemas.common import ErrorResponse
from mirip_backend.shared.exceptions import MiripError


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(MiripError)
    async def handle_mirip_error(_: Request, exc: MiripError) -> JSONResponse:
        payload = ErrorResponse(code=exc.code, message=exc.message, detail=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        payload = ErrorResponse(code="INTERNAL_ERROR", message=str(exc))
        return JSONResponse(status_code=500, content=payload.model_dump())
