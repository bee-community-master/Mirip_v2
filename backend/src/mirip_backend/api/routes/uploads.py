"""Upload routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from mirip_backend.api.deps.auth import CurrentUserDep
from mirip_backend.api.deps.services import ContainerDep
from mirip_backend.api.schemas.uploads import (
    CompleteUploadResponse,
    CreateUploadSessionRequest,
    CreateUploadSessionResponse,
    UploadAssetResponse,
    UploadListResponse,
    UploadSessionResponse,
    UploadStatusLiteral,
)
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.shared.enums import UploadStatus
from mirip_backend.usecases.uploads.complete_upload import CompleteUploadUseCase
from mirip_backend.usecases.uploads.create_upload_session import (
    CreateUploadSessionCommand,
    CreateUploadSessionUseCase,
)
from mirip_backend.usecases.uploads.list_uploads import ListUploadsQuery, ListUploadsUseCase

router = APIRouter(prefix="/v1/uploads", tags=["uploads"])
UPLOAD_STATUS_QUERY = Query(default=None, alias="status")


def _to_upload_response(upload: UploadAsset) -> UploadAssetResponse:
    return UploadAssetResponse(
        id=upload.id,
        filename=upload.filename,
        content_type=upload.content_type,
        size_bytes=upload.size_bytes,
        object_name=upload.object_name,
        category=upload.metadata.get("category"),
        status=upload.status.value,
        created_at=upload.created_at,
    )


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
        upload=_to_upload_response(result.upload),
        session=UploadSessionResponse(
            upload_url=result.session.upload_url,
            method=result.session.method,
            object_name=result.session.object_name,
            headers=result.session.headers,
            expires_at=result.session.expires_at,
        ),
    )


@router.get("", response_model=UploadListResponse)
async def list_uploads(
    current_user: CurrentUserDep,
    container: ContainerDep,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    category: str | None = Query(default=None, min_length=1, max_length=64),
    status_filter: UploadStatusLiteral | None = UPLOAD_STATUS_QUERY,
) -> UploadListResponse:
    usecase = ListUploadsUseCase(container.upload_repository)
    page = await usecase.execute(
        actor=current_user,
        query=ListUploadsQuery(
            limit=limit,
            offset=offset,
            category=category,
            status=UploadStatus(status_filter) if status_filter is not None else None,
        ),
    )
    return UploadListResponse(
        items=[_to_upload_response(item) for item in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post(
    "/{upload_id}/complete",
    response_model=CompleteUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def complete_upload(
    upload_id: str,
    current_user: CurrentUserDep,
    container: ContainerDep,
) -> CompleteUploadResponse:
    usecase = CompleteUploadUseCase(container.upload_repository)
    upload = await usecase.execute(actor=current_user, upload_id=upload_id)
    return CompleteUploadResponse(upload=_to_upload_response(upload))
