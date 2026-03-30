"""Tests for credential publishing."""

from __future__ import annotations

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.diagnosis.entities import DiagnosisResult
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import (
    DocumentCredentialRepository,
    DocumentDiagnosisResultRepository,
)
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import Visibility
from mirip_backend.usecases.credentials.publish_credential import (
    PublishCredentialCommand,
    PublishCredentialUseCase,
)


async def test_publish_credential_looks_up_result_by_result_id() -> None:
    store = MemoryDocumentStore()
    credential_repository = DocumentCredentialRepository(store)
    result_repository = DocumentDiagnosisResultRepository(store)
    await result_repository.create(
        DiagnosisResult(
            id="res-123",
            job_id="job-999",
            user_id="user-1",
            tier="A",
            scores={"composition": 91.0},
            probabilities=[],
            feedback=None,
            created_at=utc_now(),
            summary="done",
        )
    )
    usecase = PublishCredentialUseCase(credential_repository, result_repository)

    credential = await usecase.execute(
        actor=AuthenticatedUser(user_id="user-1"),
        command=PublishCredentialCommand(
            result_id="res-123",
            title="My credential",
            visibility=Visibility.PUBLIC,
        ),
    )

    assert credential.result_id == "res-123"
