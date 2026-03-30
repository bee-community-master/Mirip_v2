"""Dependency container."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mirip_backend.domain.competitions.entities import Competition
from mirip_backend.infrastructure.auth.firebase.provider import FirebaseAuthService
from mirip_backend.infrastructure.config.settings import Settings
from mirip_backend.infrastructure.firestore.client import (
    DocumentStore,
    FirestoreDocumentStore,
    MemoryDocumentStore,
)
from mirip_backend.infrastructure.firestore.repositories import (
    DocumentCompetitionRepository,
    DocumentCompetitionSubmissionRepository,
    DocumentCredentialRepository,
    DocumentDiagnosisJobRepository,
    DocumentDiagnosisResultRepository,
    DocumentPortfolioRepository,
    DocumentProfileRepository,
    DocumentUploadRepository,
)
from mirip_backend.infrastructure.gcs.service import GCSStorageService
from mirip_backend.infrastructure.jobs.queue import JobQueueService
from mirip_backend.infrastructure.observability.health import HealthReporter
from mirip_backend.shared.enums import Visibility


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    auth_service: FirebaseAuthService
    storage_service: GCSStorageService
    upload_repository: DocumentUploadRepository
    diagnosis_job_repository: DocumentDiagnosisJobRepository
    diagnosis_result_repository: DocumentDiagnosisResultRepository
    competition_repository: DocumentCompetitionRepository
    competition_submission_repository: DocumentCompetitionSubmissionRepository
    credential_repository: DocumentCredentialRepository
    profile_repository: DocumentProfileRepository
    portfolio_repository: DocumentPortfolioRepository
    job_queue: JobQueueService
    health_reporter: HealthReporter


async def _seed_memory_competitions(repository: DocumentCompetitionRepository) -> None:
    existing = await repository.list_public(limit=1, offset=0)
    if existing.total > 0:
        return
    seed_items = [
        Competition(
            id="comp-visual-2026",
            title="2026 Visual Design Trial Contest",
            description="Frontend-backed sample competition for scaffold verification.",
            visibility=Visibility.PUBLIC,
            opens_at=datetime(2026, 4, 1, 0, 0, 0),
            closes_at=datetime(2026, 5, 31, 23, 59, 59),
            tags=["design", "trial"],
        ),
        Competition(
            id="comp-fineart-2026",
            title="2026 Fine Art Mock Jury",
            description="Sample public competition used by the in-memory local backend.",
            visibility=Visibility.PUBLIC,
            opens_at=datetime(2026, 6, 1, 0, 0, 0),
            closes_at=datetime(2026, 7, 15, 23, 59, 59),
            tags=["fine-art", "mock"],
        ),
    ]
    for item in seed_items:
        await repository.create(item)


async def build_container(settings: Settings) -> ApplicationContainer:
    store: DocumentStore
    if settings.data_backend == "firestore":
        store = FirestoreDocumentStore(
            project_id=settings.firebase.project_id,
            credentials_path=settings.firebase.credentials_path,
        )
    else:
        store = MemoryDocumentStore()

    upload_repository = DocumentUploadRepository(store)
    diagnosis_job_repository = DocumentDiagnosisJobRepository(store)
    diagnosis_result_repository = DocumentDiagnosisResultRepository(store)
    competition_repository = DocumentCompetitionRepository(store)
    competition_submission_repository = DocumentCompetitionSubmissionRepository(store)
    credential_repository = DocumentCredentialRepository(store)
    profile_repository = DocumentProfileRepository(store)
    portfolio_repository = DocumentPortfolioRepository(store)

    if settings.data_backend == "memory":
        await _seed_memory_competitions(competition_repository)

    auth_service = FirebaseAuthService(settings.firebase)
    storage_service = GCSStorageService(settings.gcs, backend=settings.storage_backend)
    health_reporter = HealthReporter(checks=[auth_service, storage_service])
    job_queue = JobQueueService(settings.job, diagnosis_job_repository)

    return ApplicationContainer(
        settings=settings,
        auth_service=auth_service,
        storage_service=storage_service,
        upload_repository=upload_repository,
        diagnosis_job_repository=diagnosis_job_repository,
        diagnosis_result_repository=diagnosis_result_repository,
        competition_repository=competition_repository,
        competition_submission_repository=competition_submission_repository,
        credential_repository=credential_repository,
        profile_repository=profile_repository,
        portfolio_repository=portfolio_repository,
        job_queue=job_queue,
        health_reporter=health_reporter,
    )
