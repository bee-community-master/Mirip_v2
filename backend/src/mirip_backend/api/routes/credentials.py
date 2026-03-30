"""Credential routes."""

from __future__ import annotations

from fastapi import APIRouter

from mirip_backend.api.deps.auth import CurrentUserDep, OptionalCurrentUserDep
from mirip_backend.api.deps.services import ContainerDep
from mirip_backend.api.schemas.credentials import CredentialResponse, PublishCredentialRequest
from mirip_backend.shared.enums import Visibility
from mirip_backend.shared.exceptions import NotFoundError
from mirip_backend.usecases.credentials.publish_credential import (
    PublishCredentialCommand,
    PublishCredentialUseCase,
)

router = APIRouter(prefix="/v1/credentials", tags=["credentials"])


@router.post("", response_model=CredentialResponse)
async def publish_credential(
    payload: PublishCredentialRequest,
    current_user: CurrentUserDep,
    container: ContainerDep,
) -> CredentialResponse:
    usecase = PublishCredentialUseCase(
        credential_repository=container.credential_repository,
        result_repository=container.diagnosis_result_repository,
    )
    credential = await usecase.execute(
        actor=current_user,
        command=PublishCredentialCommand(
            result_id=payload.result_id,
            title=payload.title,
            visibility=Visibility(payload.visibility),
        ),
    )
    return CredentialResponse(
        id=credential.id,
        result_id=credential.result_id,
        title=credential.title,
        status=credential.status.value,
        visibility=credential.visibility.value,
        created_at=credential.created_at,
    )


@router.get("/{credential_id}", response_model=CredentialResponse)
async def get_credential(
    credential_id: str,
    current_user: OptionalCurrentUserDep,
    container: ContainerDep,
) -> CredentialResponse:
    credential = await container.credential_repository.get(credential_id)
    if credential is None:
        raise NotFoundError("Credential not found")
    if credential.visibility != Visibility.PUBLIC and (
        current_user is None or credential.user_id != current_user.user_id
    ):
        raise NotFoundError("Credential not found")
    return CredentialResponse(
        id=credential.id,
        result_id=credential.result_id,
        title=credential.title,
        status=credential.status.value,
        visibility=credential.visibility.value,
        created_at=credential.created_at,
    )
