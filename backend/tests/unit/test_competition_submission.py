"""Tests for competition submission rules."""

from __future__ import annotations

from datetime import timedelta

import pytest

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.competitions.entities import Competition, CompetitionSubmission
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import (
    DocumentCompetitionRepository,
    DocumentCompetitionSubmissionRepository,
    DocumentUploadRepository,
)
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import UploadStatus, Visibility
from mirip_backend.shared.exceptions import ConflictError, ValidationError
from mirip_backend.usecases.competitions.create_submission import (
    CreateCompetitionSubmissionCommand,
    CreateCompetitionSubmissionUseCase,
)


async def test_submission_rejects_duplicate_entry_for_same_competition() -> None:
    store = MemoryDocumentStore()
    competition_repository = DocumentCompetitionRepository(store)
    submission_repository = DocumentCompetitionSubmissionRepository(store)
    upload_repository = DocumentUploadRepository(store)
    now = utc_now()

    await competition_repository.create(
        Competition(
            id="comp-1",
            title="Open competition",
            description="desc",
            visibility=Visibility.PUBLIC,
            opens_at=now - timedelta(days=1),
            closes_at=now + timedelta(days=1),
        )
    )
    await upload_repository.create(
        UploadAsset(
            id="upl-1",
            user_id="user-1",
            filename="piece.png",
            content_type="image/png",
            size_bytes=100,
            object_name="users/user-1/competition/upl-1/piece.png",
            status=UploadStatus.UPLOADED,
            created_at=now,
        )
    )
    await submission_repository.create(
        CompetitionSubmission(
            id="sub-1",
            competition_id="comp-1",
            user_id="user-1",
            upload_id="upl-1",
            statement=None,
            created_at=now,
        )
    )

    usecase = CreateCompetitionSubmissionUseCase(
        competition_repository=competition_repository,
        submission_repository=submission_repository,
        upload_repository=upload_repository,
    )

    with pytest.raises(ConflictError):
        await usecase.execute(
            actor=AuthenticatedUser(user_id="user-1"),
            command=CreateCompetitionSubmissionCommand(
                competition_id="comp-1",
                upload_id="upl-1",
            ),
        )


async def test_submission_rejects_closed_competition() -> None:
    store = MemoryDocumentStore()
    competition_repository = DocumentCompetitionRepository(store)
    submission_repository = DocumentCompetitionSubmissionRepository(store)
    upload_repository = DocumentUploadRepository(store)
    now = utc_now()

    await competition_repository.create(
        Competition(
            id="comp-closed",
            title="Closed competition",
            description="desc",
            visibility=Visibility.PUBLIC,
            opens_at=now - timedelta(days=2),
            closes_at=now - timedelta(minutes=1),
        )
    )
    await upload_repository.create(
        UploadAsset(
            id="upl-closed",
            user_id="user-1",
            filename="piece.png",
            content_type="image/png",
            size_bytes=100,
            object_name="users/user-1/competition/upl-closed/piece.png",
            status=UploadStatus.UPLOADED,
            created_at=now,
        )
    )

    usecase = CreateCompetitionSubmissionUseCase(
        competition_repository=competition_repository,
        submission_repository=submission_repository,
        upload_repository=upload_repository,
    )

    with pytest.raises(ValidationError):
        await usecase.execute(
            actor=AuthenticatedUser(user_id="user-1"),
            command=CreateCompetitionSubmissionCommand(
                competition_id="comp-closed",
                upload_id="upl-closed",
            ),
        )


async def test_submission_rejects_pending_upload() -> None:
    store = MemoryDocumentStore()
    competition_repository = DocumentCompetitionRepository(store)
    submission_repository = DocumentCompetitionSubmissionRepository(store)
    upload_repository = DocumentUploadRepository(store)
    now = utc_now()

    await competition_repository.create(
        Competition(
            id="comp-open",
            title="Open competition",
            description="desc",
            visibility=Visibility.PUBLIC,
            opens_at=now - timedelta(days=1),
            closes_at=now + timedelta(days=1),
        )
    )
    await upload_repository.create(
        UploadAsset(
            id="upl-pending",
            user_id="user-1",
            filename="piece.png",
            content_type="image/png",
            size_bytes=100,
            object_name="users/user-1/competition/upl-pending/piece.png",
            status=UploadStatus.PENDING,
            created_at=now,
        )
    )

    usecase = CreateCompetitionSubmissionUseCase(
        competition_repository=competition_repository,
        submission_repository=submission_repository,
        upload_repository=upload_repository,
    )

    with pytest.raises(ValidationError):
        await usecase.execute(
            actor=AuthenticatedUser(user_id="user-1"),
            command=CreateCompetitionSubmissionCommand(
                competition_id="comp-open",
                upload_id="upl-pending",
            ),
        )
