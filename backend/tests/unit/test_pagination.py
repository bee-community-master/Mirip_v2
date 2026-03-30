"""Tests for repository pagination metadata."""

from __future__ import annotations

from mirip_backend.domain.diagnosis.entities import DiagnosisResult
from mirip_backend.infrastructure.firestore.client import MemoryDocumentStore
from mirip_backend.infrastructure.firestore.repositories import DocumentDiagnosisResultRepository
from mirip_backend.shared.clock import utc_now


async def test_result_pagination_reports_full_total() -> None:
    repository = DocumentDiagnosisResultRepository(MemoryDocumentStore())

    for index in range(3):
        await repository.create(
            DiagnosisResult(
                id=f"res-{index}",
                job_id=f"job-{index}",
                user_id="user-1",
                tier="A",
                scores={"composition": 90.0},
                probabilities=[],
                feedback=None,
                created_at=utc_now(),
                summary=None,
            )
        )

    page = await repository.list_by_user("user-1", limit=1, offset=1)

    assert len(page.items) == 1
    assert page.total == 3
    assert page.limit == 1
    assert page.offset == 1
