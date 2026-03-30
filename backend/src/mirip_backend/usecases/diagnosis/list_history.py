"""List diagnosis history usecase."""

from __future__ import annotations

from mirip_backend.domain.auth.models import AuthenticatedUser
from mirip_backend.domain.common.models import Page
from mirip_backend.domain.diagnosis.entities import DiagnosisResult
from mirip_backend.domain.diagnosis.repositories import DiagnosisResultRepository


class ListDiagnosisHistoryUseCase:
    """List result history for the authenticated user."""

    def __init__(self, result_repository: DiagnosisResultRepository) -> None:
        self._result_repository = result_repository

    async def execute(
        self,
        *,
        actor: AuthenticatedUser,
        limit: int,
        offset: int,
    ) -> Page[DiagnosisResult]:
        return await self._result_repository.list_by_user(actor.user_id, limit=limit, offset=offset)
