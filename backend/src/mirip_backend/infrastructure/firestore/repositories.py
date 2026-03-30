"""Document-backed repository implementations."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from mirip_backend.domain.common.models import Page
from mirip_backend.domain.competitions.entities import Competition, CompetitionSubmission
from mirip_backend.domain.credentials.entities import Credential
from mirip_backend.domain.diagnosis.entities import DiagnosisJob, DiagnosisResult
from mirip_backend.domain.profiles.entities import PortfolioItem, Profile
from mirip_backend.domain.uploads.entities import UploadAsset
from mirip_backend.infrastructure.firestore.client import DocumentStore
from mirip_backend.shared.clock import utc_now
from mirip_backend.shared.enums import CredentialStatus, JobStatus, UploadStatus, Visibility


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (JobStatus, UploadStatus, CredentialStatus, Visibility)):
        return value.value
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _deserialize_datetime(value: Any) -> datetime | None:
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    if isinstance(value, datetime):
        return value
    return None


def _page(items: list[Any], *, total: int, limit: int, offset: int) -> Page[Any]:
    return Page(items=items, total=total, limit=limit, offset=offset)


class DocumentUploadRepository:
    COLLECTION = "uploads"

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def create(self, upload: UploadAsset) -> UploadAsset:
        await self._store.put(self.COLLECTION, upload.id, _serialize(asdict(upload)))
        return upload

    async def get(self, upload_id: str) -> UploadAsset | None:
        doc = await self._store.get(self.COLLECTION, upload_id)
        if doc is None:
            return None
        return UploadAsset(
            id=str(doc["id"]),
            user_id=str(doc["user_id"]),
            filename=str(doc["filename"]),
            content_type=str(doc["content_type"]),
            size_bytes=int(doc["size_bytes"]),
            object_name=str(doc["object_name"]),
            status=UploadStatus(str(doc["status"])),
            created_at=_deserialize_datetime(doc["created_at"]) or datetime.min,
            metadata=dict(doc.get("metadata", {})),
        )

    async def list_by_user(self, user_id: str) -> list[UploadAsset]:
        docs = await self._store.query(self.COLLECTION, filters=(("user_id", user_id),))
        return [
            item for item in [await self.get(str(doc["id"])) for doc in docs] if item is not None
        ]

    async def update(self, upload: UploadAsset) -> UploadAsset:
        await self._store.put(self.COLLECTION, upload.id, _serialize(asdict(upload)))
        return upload


class DocumentDiagnosisJobRepository:
    COLLECTION = "diagnosis_jobs"

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def create(self, job: DiagnosisJob) -> DiagnosisJob:
        await self._store.put(self.COLLECTION, job.id, _serialize(asdict(job)))
        return job

    async def get(self, job_id: str) -> DiagnosisJob | None:
        doc = await self._store.get(self.COLLECTION, job_id)
        if doc is None:
            return None
        return DiagnosisJob(
            id=str(doc["id"]),
            user_id=str(doc["user_id"]),
            upload_ids=[str(item) for item in doc["upload_ids"]],
            job_type=str(doc["job_type"]),
            department=str(doc["department"]),
            include_feedback=bool(doc["include_feedback"]),
            theme=doc.get("theme"),
            language=str(doc["language"]),
            status=JobStatus(str(doc["status"])),
            created_at=_deserialize_datetime(doc["created_at"]) or datetime.min,
            updated_at=_deserialize_datetime(doc["updated_at"]) or datetime.min,
            attempts=int(doc.get("attempts", 0)),
            lease_owner=doc.get("lease_owner"),
            lease_expires_at=_deserialize_datetime(doc.get("lease_expires_at")),
            failure_reason=doc.get("failure_reason"),
            metadata=dict(doc.get("metadata", {})),
        )

    async def update(self, job: DiagnosisJob) -> DiagnosisJob:
        await self._store.put(self.COLLECTION, job.id, _serialize(asdict(job)))
        return job

    async def list_by_user(self, user_id: str, *, limit: int, offset: int) -> Page[DiagnosisJob]:
        docs = await self._store.query(
            self.COLLECTION,
            filters=(("user_id", user_id),),
            order_by="created_at",
            descending=True,
        )
        jobs = [
            item for item in [await self.get(str(doc["id"])) for doc in docs] if item is not None
        ]
        return _page(jobs[offset : offset + limit], total=len(jobs), limit=limit, offset=offset)

    async def lease_next_ready_job(
        self,
        *,
        worker_id: str,
        lease_until: datetime,
    ) -> DiagnosisJob | None:
        docs = await self._store.query(self.COLLECTION, order_by="created_at", descending=False)
        now = utc_now()
        for doc in docs:
            job = await self.get(str(doc["id"]))
            if job is None:
                continue
            lease_expired = job.lease_expires_at is not None and job.lease_expires_at <= now
            if job.status not in {JobStatus.QUEUED, JobStatus.EXPIRED} and not (
                job.status in {JobStatus.LEASED, JobStatus.RUNNING} and lease_expired
            ):
                continue
            leased = DiagnosisJob(
                **{
                    **asdict(job),
                    "status": JobStatus.LEASED,
                    "lease_owner": worker_id,
                    "lease_expires_at": lease_until,
                    "attempts": job.attempts + 1,
                    "updated_at": lease_until,
                }
            )
            await self.update(leased)
            return leased
        return None


class DocumentDiagnosisResultRepository:
    COLLECTION = "diagnosis_results"

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def create(self, result: DiagnosisResult) -> DiagnosisResult:
        await self._store.put(self.COLLECTION, result.id, _serialize(asdict(result)))
        return result

    async def get(self, result_id: str) -> DiagnosisResult | None:
        doc = await self._store.get(self.COLLECTION, result_id)
        if doc is None:
            return None
        return DiagnosisResult(
            id=str(doc["id"]),
            job_id=str(doc["job_id"]),
            user_id=str(doc["user_id"]),
            tier=str(doc["tier"]),
            scores={str(key): float(value) for key, value in doc["scores"].items()},
            probabilities=list(doc["probabilities"]),
            feedback=dict(doc["feedback"]) if doc.get("feedback") is not None else None,
            created_at=_deserialize_datetime(doc["created_at"]) or datetime.min,
            summary=doc.get("summary"),
        )

    async def get_by_job_id(self, job_id: str) -> DiagnosisResult | None:
        docs = await self._store.query(self.COLLECTION, filters=(("job_id", job_id),), limit=1)
        if not docs:
            return None
        return await self.get(str(docs[0]["id"]))

    async def list_by_user(self, user_id: str, *, limit: int, offset: int) -> Page[DiagnosisResult]:
        docs = await self._store.query(
            self.COLLECTION,
            filters=(("user_id", user_id),),
            order_by="created_at",
            descending=True,
        )
        results = [
            item for item in [await self.get(str(doc["id"])) for doc in docs] if item is not None
        ]
        return _page(
            results[offset : offset + limit],
            total=len(results),
            limit=limit,
            offset=offset,
        )


class DocumentCompetitionRepository:
    COLLECTION = "competitions"

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def list_public(self, *, limit: int, offset: int) -> Page[Competition]:
        docs = await self._store.query(
            self.COLLECTION,
            filters=(("visibility", Visibility.PUBLIC.value),),
            order_by="title",
        )
        items = [self._to_entity(doc) for doc in docs]
        return _page(items[offset : offset + limit], total=len(items), limit=limit, offset=offset)

    async def get(self, competition_id: str) -> Competition | None:
        doc = await self._store.get(self.COLLECTION, competition_id)
        return self._to_entity(doc) if doc is not None else None

    async def create(self, competition: Competition) -> Competition:
        await self._store.put(self.COLLECTION, competition.id, _serialize(asdict(competition)))
        return competition

    def _to_entity(self, doc: dict[str, Any]) -> Competition:
        return Competition(
            id=str(doc["id"]),
            title=str(doc["title"]),
            description=str(doc["description"]),
            visibility=Visibility(str(doc["visibility"])),
            opens_at=_deserialize_datetime(doc.get("opens_at")),
            closes_at=_deserialize_datetime(doc.get("closes_at")),
            tags=[str(item) for item in doc.get("tags", [])],
        )


class DocumentCompetitionSubmissionRepository:
    COLLECTION = "competition_submissions"

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def create(self, submission: CompetitionSubmission) -> CompetitionSubmission:
        await self._store.put(self.COLLECTION, submission.id, _serialize(asdict(submission)))
        return submission

    async def list_by_user(
        self, user_id: str, *, limit: int, offset: int
    ) -> Page[CompetitionSubmission]:
        docs = await self._store.query(
            self.COLLECTION,
            filters=(("user_id", user_id),),
            order_by="created_at",
            descending=True,
        )
        items = [
            CompetitionSubmission(
                id=str(doc["id"]),
                competition_id=str(doc["competition_id"]),
                user_id=str(doc["user_id"]),
                upload_id=str(doc["upload_id"]),
                statement=doc.get("statement"),
                created_at=_deserialize_datetime(doc["created_at"]) or datetime.min,
            )
            for doc in docs[offset : offset + limit]
        ]
        return _page(items, total=len(docs), limit=limit, offset=offset)


class DocumentCredentialRepository:
    COLLECTION = "credentials"

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def create(self, credential: Credential) -> Credential:
        await self._store.put(self.COLLECTION, credential.id, _serialize(asdict(credential)))
        return credential

    async def get(self, credential_id: str) -> Credential | None:
        doc = await self._store.get(self.COLLECTION, credential_id)
        if doc is None:
            return None
        return Credential(
            id=str(doc["id"]),
            user_id=str(doc["user_id"]),
            result_id=str(doc["result_id"]),
            title=str(doc["title"]),
            status=CredentialStatus(str(doc["status"])),
            visibility=Visibility(str(doc["visibility"])),
            created_at=_deserialize_datetime(doc["created_at"]) or datetime.min,
        )


class DocumentProfileRepository:
    COLLECTION = "profiles"

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def upsert(self, profile: Profile) -> Profile:
        await self._store.put(self.COLLECTION, profile.user_id, _serialize(asdict(profile)))
        return profile

    async def get_by_user_id(self, user_id: str) -> Profile | None:
        doc = await self._store.get(self.COLLECTION, user_id)
        return self._to_entity(doc) if doc is not None else None

    async def get_by_handle(self, handle: str) -> Profile | None:
        docs = await self._store.query(self.COLLECTION, filters=(("handle", handle),), limit=1)
        return self._to_entity(docs[0]) if docs else None

    def _to_entity(self, doc: dict[str, Any]) -> Profile:
        return Profile(
            user_id=str(doc["user_id"]),
            handle=str(doc["handle"]),
            display_name=str(doc["display_name"]),
            bio=doc.get("bio"),
            visibility=Visibility(str(doc["visibility"])),
            portfolio_item_ids=[str(item) for item in doc.get("portfolio_item_ids", [])],
            updated_at=_deserialize_datetime(doc.get("updated_at")),
        )


class DocumentPortfolioRepository:
    COLLECTION = "portfolio_items"

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def create(self, item: PortfolioItem) -> PortfolioItem:
        await self._store.put(self.COLLECTION, item.id, _serialize(asdict(item)))
        return item

    async def list_by_user(self, user_id: str, *, limit: int, offset: int) -> Page[PortfolioItem]:
        docs = await self._store.query(
            self.COLLECTION,
            filters=(("user_id", user_id),),
            order_by="created_at",
            descending=True,
        )
        items = [self._to_entity(doc) for doc in docs[offset : offset + limit]]
        return _page(items, total=len(docs), limit=limit, offset=offset)

    async def list_by_ids(self, item_ids: list[str]) -> list[PortfolioItem]:
        items: list[PortfolioItem] = []
        for item_id in item_ids:
            doc = await self._store.get(self.COLLECTION, item_id)
            if doc is not None:
                items.append(self._to_entity(doc))
        return items

    def _to_entity(self, doc: dict[str, Any]) -> PortfolioItem:
        return PortfolioItem(
            id=str(doc["id"]),
            user_id=str(doc["user_id"]),
            title=str(doc["title"]),
            description=doc.get("description"),
            asset_upload_id=str(doc["asset_upload_id"]),
            created_at=_deserialize_datetime(doc["created_at"]) or datetime.min,
            visibility=Visibility(str(doc["visibility"])),
        )
