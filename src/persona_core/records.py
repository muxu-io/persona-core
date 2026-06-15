"""Episodic record schema. Polymorphic from day one."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class RecordType(StrEnum):
    TURN_PAIR = "turn_pair"
    SEEDED_NARRATIVE = "seeded_narrative"
    # extracted_fact, cluster_summary added in Stage 2 (design doc §3.4)


class ProcessingDepth(StrEnum):
    PRIMAL = "primal"
    PARTIALLY_REFLECTED = "partially_reflected"
    OWNED = "owned"


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class EpisodicRecord:
    type: RecordType
    session_id: str | None
    content: dict[str, Any]
    initial_salience: float
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)
    emotional_valence: float = 0.0
    emotional_intensity: float = 0.0
    access_count: int = 0
    last_accessed: datetime | None = None
    processing_depth: ProcessingDepth = ProcessingDepth.PRIMAL
    consolidated: bool = False
    embedding: list[float] | None = None
    scenario_id: str | None = None

    @property
    def current_salience(self) -> float:
        # Stage 1: equal to initial_salience. Stage 2's sleep pass introduces
        # decay and access bonuses (design doc §3.3).
        return self.initial_salience

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Serialize all non-vector fields to a JSON-serializable dict."""
        payload = {
            "type": self.type.value,
            "session_id": self.session_id,
            "content": self.content,
            "initial_salience": self.initial_salience,
            "created_at": self.created_at.isoformat(),
            "emotional_valence": self.emotional_valence,
            "emotional_intensity": self.emotional_intensity,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "processing_depth": self.processing_depth.value,
            "consolidated": self.consolidated,
        }
        if self.scenario_id is not None:
            payload["scenario_id"] = self.scenario_id
        return payload

    @classmethod
    def from_qdrant_payload(cls, point_id: str, payload: dict[str, Any]) -> EpisodicRecord:
        return cls(
            id=point_id,
            type=RecordType(payload["type"]),
            session_id=payload["session_id"],
            content=payload["content"],
            initial_salience=payload["initial_salience"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            emotional_valence=payload.get("emotional_valence", 0.0),
            emotional_intensity=payload.get("emotional_intensity", 0.0),
            access_count=payload.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(payload["last_accessed"])
                if payload.get("last_accessed")
                else None
            ),
            processing_depth=ProcessingDepth(payload.get("processing_depth", "primal")),
            consolidated=payload.get("consolidated", False),
            scenario_id=payload.get("scenario_id"),
        )
