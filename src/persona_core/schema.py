"""Persona definition schema. Substrate + self-concept layered model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Presence(StrEnum):
    ALWAYS_ON = "always_on"
    TRIGGERED = "triggered"


@dataclass
class Dimension:
    """One persona dimension — substrate or self-concept."""

    name: str
    presence: Presence
    prose: str
    structured: dict[str, Any]
    voice_anchor: str | None = None
    perception: str | None = None
    triggers: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SeededTrauma:
    """A formative event seeded into the episodic store at persona birth."""

    name: str
    prose: str
    structured: dict[str, Any]
    initial_salience: float
    emotional_valence: float
    emotional_intensity: float

    @property
    def age_at(self) -> int | None:
        return self.structured.get("age_at")

    @property
    def charge(self) -> str | None:
        return self.structured.get("charge")


@dataclass
class Persona:
    persona_id: str
    spec_version: int
    identity: dict[str, Any]
    substrate: dict[str, Dimension]
    self_concept: dict[str, Dimension]
    traumas: list[SeededTrauma] = field(default_factory=list)

    def all_triggered_dimensions(self) -> list[Dimension]:
        return [
            d
            for d in (*self.substrate.values(), *self.self_concept.values())
            if d.presence == Presence.TRIGGERED
        ]

    def all_always_on_dimensions(self) -> list[Dimension]:
        return [
            d
            for d in (*self.substrate.values(), *self.self_concept.values())
            if d.presence == Presence.ALWAYS_ON
        ]
