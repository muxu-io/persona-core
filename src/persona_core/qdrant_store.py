"""Qdrant store wrapper. In-memory mode for tests; HTTP for runtime."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from persona_core.records import EpisodicRecord, RecordType


@dataclass
class ScoredHit:
    record: EpisodicRecord
    similarity: float


class QdrantStore:
    """Multi-tenant store over a single shared collection. Every read and write is
    scoped to `persona_id` (stamped into the payload on write, injected as a `must`
    filter on every read/count/scroll) — the load-bearing isolation invariant.
    `persona_id` is mandatory by construction: no code path can query without it.
    """

    def __init__(self, client: QdrantClient, collection: str, vector_size: int, persona_id: str):
        self._client = client
        self.collection = collection
        self.vector_size = vector_size
        self.persona_id = persona_id
        self.dirty: bool = False

    @classmethod
    def in_memory(cls, collection: str, vector_size: int, persona_id: str) -> QdrantStore:
        return cls(QdrantClient(":memory:"), collection, vector_size, persona_id)

    @classmethod
    def http(
        cls, host: str, port: int, collection: str, vector_size: int, persona_id: str
    ) -> QdrantStore:
        return cls(QdrantClient(host=host, port=port), collection, vector_size, persona_id)

    def _tenant_condition(self) -> qm.FieldCondition:
        return qm.FieldCondition(key="persona_id", match=qm.MatchValue(value=self.persona_id))

    def ensure_collection(self) -> None:
        if self.collection_exists():
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.vector_size, distance=qm.Distance.COSINE),
        )
        # Tenant index: co-locates points per persona and makes the filter cheap.
        # Local/in-memory Qdrant ignores payload indexes (and warns); the filter
        # still enforces isolation, so the warning is noise in tests.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name="persona_id",
                field_schema=qm.KeywordIndexParams(
                    type=qm.KeywordIndexType.KEYWORD, is_tenant=True
                ),
            )

    def write(self, record: EpisodicRecord) -> None:
        if record.embedding is None:
            raise ValueError("record.embedding must be populated before write")
        payload = record.to_qdrant_payload()
        payload["persona_id"] = self.persona_id
        self._client.upsert(
            collection_name=self.collection,
            points=[
                qm.PointStruct(
                    id=record.id,
                    vector=record.embedding,
                    payload=payload,
                )
            ],
        )
        self.dirty = True

    def delete_by_id(self, point_id: str) -> None:
        """Delete a single point by id. No-op if id is absent."""
        self._client.delete(
            collection_name=self.collection,
            points_selector=qm.PointIdsList(points=[point_id]),
        )
        self.dirty = True

    def query(
        self,
        query_vector: list[float],
        limit: int,
        types: list[RecordType] | None = None,
    ) -> list[ScoredHit]:
        """Top-k by cosine similarity for this persona, optionally restricted to
        record types. The persona_id filter is always applied.
        """
        must = [self._tenant_condition()]
        should = (
            [qm.FieldCondition(key="type", match=qm.MatchValue(value=t.value)) for t in types]
            if types
            else None
        )
        result = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            query_filter=qm.Filter(must=must, should=should),
            with_payload=True,
        )
        return [
            ScoredHit(
                record=EpisodicRecord.from_qdrant_payload(str(p.id), p.payload),
                similarity=float(p.score),
            )
            for p in result.points
        ]

    def count(self) -> int:
        flt = qm.Filter(must=[self._tenant_condition()])
        return self._client.count(self.collection, count_filter=flt, exact=True).count

    def count_unconsolidated(self) -> int:
        flt = qm.Filter(
            must=[
                self._tenant_condition(),
                qm.FieldCondition(key="consolidated", match=qm.MatchValue(value=False)),
            ]
        )
        return self._client.count(self.collection, count_filter=flt, exact=True).count

    def list_by_type(self, type: RecordType) -> list[EpisodicRecord]:
        """Return every record of the given type for this persona. No vector query."""
        flt = qm.Filter(
            must=[
                self._tenant_condition(),
                qm.FieldCondition(key="type", match=qm.MatchValue(value=type.value)),
            ]
        )
        points, _ = self._client.scroll(
            collection_name=self.collection,
            scroll_filter=flt,
            limit=10000,
            with_payload=True,
        )
        return [EpisodicRecord.from_qdrant_payload(str(p.id), p.payload) for p in points]

    def collection_exists(self) -> bool:
        existing = {c.name for c in self._client.get_collections().collections}
        return self.collection in existing

    def fetch_session_window(
        self,
        session_id: str,
        before: datetime,
        limit: int,
    ) -> list[EpisodicRecord]:
        """Fetch up to `limit` most-recent records in `session_id` strictly before `before`."""
        flt = qm.Filter(
            must=[
                self._tenant_condition(),
                qm.FieldCondition(key="session_id", match=qm.MatchValue(value=session_id)),
            ]
        )
        records, _ = self._client.scroll(
            collection_name=self.collection,
            scroll_filter=flt,
            limit=10000,
            with_payload=True,
        )
        parsed = [EpisodicRecord.from_qdrant_payload(str(p.id), p.payload) for p in records]
        parsed = [r for r in parsed if r.created_at < before]
        parsed.sort(key=lambda r: r.created_at, reverse=True)
        return parsed[:limit]
