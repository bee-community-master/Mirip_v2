"""Document store abstractions for Firestore and tests."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

Filter = tuple[str, Any]


class DocumentStore(Protocol):
    async def put(self, collection: str, doc_id: str, data: dict[str, Any]) -> dict[str, Any]: ...

    async def get(self, collection: str, doc_id: str) -> dict[str, Any] | None: ...

    async def query(
        self,
        collection: str,
        *,
        filters: tuple[Filter, ...] = (),
        limit: int | None = None,
        offset: int = 0,
        order_by: str | None = None,
        descending: bool = False,
    ) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class MemoryDocumentStore:
    """Simple in-memory document store used for tests and local scaffolding."""

    collections: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    async def put(self, collection: str, doc_id: str, data: dict[str, Any]) -> dict[str, Any]:
        self.collections[collection][doc_id] = dict(data)
        return dict(data)

    async def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        record = self.collections[collection].get(doc_id)
        return dict(record) if record is not None else None

    async def query(
        self,
        collection: str,
        *,
        filters: tuple[Filter, ...] = (),
        limit: int | None = None,
        offset: int = 0,
        order_by: str | None = None,
        descending: bool = False,
    ) -> list[dict[str, Any]]:
        items = list(self.collections[collection].values())
        for field_name, expected in filters:
            items = [item for item in items if item.get(field_name) == expected]
        if order_by is not None:
            items.sort(key=lambda item: str(item.get(order_by, "")), reverse=descending)
        sliced = items[offset : None if limit is None else offset + limit]
        return [dict(item) for item in sliced]


@dataclass(slots=True)
class FirestoreDocumentStore:
    """Firestore-backed document store."""

    project_id: str | None = None
    credentials_path: str | None = None

    def _client(self) -> Any:
        from google.cloud import firestore

        if self.credentials_path is not None:
            return firestore.Client.from_service_account_json(
                self.credentials_path, project=self.project_id
            )  # type: ignore[no-untyped-call]
        return firestore.Client(project=self.project_id)

    async def put(self, collection: str, doc_id: str, data: dict[str, Any]) -> dict[str, Any]:
        def _put() -> None:
            client = self._client()
            client.collection(collection).document(doc_id).set(data)

        await asyncio.to_thread(_put)
        return data

    async def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        def _get() -> dict[str, Any] | None:
            client = self._client()
            snapshot = client.collection(collection).document(doc_id).get()
            return snapshot.to_dict() if snapshot.exists else None

        return await asyncio.to_thread(_get)

    async def query(
        self,
        collection: str,
        *,
        filters: tuple[Filter, ...] = (),
        limit: int | None = None,
        offset: int = 0,
        order_by: str | None = None,
        descending: bool = False,
    ) -> list[dict[str, Any]]:
        def _query() -> list[dict[str, Any]]:
            client = self._client()
            query = client.collection(collection)
            for field_name, expected in filters:
                query = query.where(field_name, "==", expected)
            if order_by is not None:
                direction = "DESCENDING" if descending else "ASCENDING"
                query = query.order_by(order_by, direction=direction)
            if offset:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)
            return [snapshot.to_dict() for snapshot in query.stream()]

        return await asyncio.to_thread(_query)
