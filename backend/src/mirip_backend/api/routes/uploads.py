"""Upload routes."""

from __future__ import annotations

from fastapi import APIRouter, status

from mirip_backend.api.deps.auth import CurrentUserDep
from mirip_backend.api.deps.services import ContainerDep
from mirip_backend.api.schemas.uploads import (
    CreateUploadSessionRequest,
    CreateUploadSessionResponse,
    UploadAssetResponse,
    UploadSessionResponse,
)
from mirip_backend.usecases.uploads.create_upload_session import (
    CreateUploadSessionCommand,
    CreateUploadSessionUseCase,
)

router = APIRouter(prefix="/v1/uploads", tags=["uploads"])


@router.post("", response_model=CreateUploadSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_upload_session(
    payload: CreateUploadSessionRequest,
    current_user: CurrentUserDep,
    container: ContainerDep,
) -> CreateUploadSessionResponse:
    usecase = CreateUploadSessionUseCase(container.upload_repository, container.storage_service)
    result = await usecase.execute(
        actor=current_user,
        command=CreateUploadSessionCommand(
            filename=payload.filename,
            content_type=payload.content_type,
            size_bytes=payload.size_bytes,
            category=payload.category,
        ),
    )
    return CreateUploadSessionResponse(
        upload=UploadAssetResponse(
            id=result.upload.id,
            filename=result.upload.filename,
            content_type=result.upload.content_type,
            size_bytes=result.upload.size_bytes,
            object_name=result.upload.object_name,
            status=result.upload.status.value,
            created_at=result.upload.created_at,
        ),
        session=UploadSessionResponse(
            upload_url=result.session.upload_url,
            method=result.session.method,
            object_name=result.session.object_name,
            headers=result.session.headers,
            expires_at=result.session.expires_at,
        ),
    )
