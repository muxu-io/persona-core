"""Pure mapping between the in-memory `Persona` dataclass and the JSON-native
`definition` document stored in Postgres. The `embedding` vector is intentionally
omitted — it is recomputed at load time, never persisted in the definition.
"""

from __future__ import annotations

from typing import Any

from persona_core.schema import Dimension, Persona, Presence, SeededTrauma


def dimension_to_dict(d: Dimension) -> dict[str, Any]:
    return {
        "presence": d.presence.value,
        "prose": d.prose,
        "structured": d.structured,
        "voice_anchor": d.voice_anchor,
        "perception": d.perception,
        "triggers": d.triggers,
    }


def dimension_from_dict(name: str, data: dict[str, Any]) -> Dimension:
    return Dimension(
        name=name,
        presence=Presence(data["presence"]),
        prose=data.get("prose", ""),
        structured=data.get("structured", {}) or {},
        voice_anchor=data.get("voice_anchor"),
        perception=data.get("perception"),
        triggers=data.get("triggers", {}) or {},
    )


def trauma_to_dict(t: SeededTrauma) -> dict[str, Any]:
    return {
        "name": t.name,
        "prose": t.prose,
        "structured": t.structured,
        "initial_salience": t.initial_salience,
        "emotional_valence": t.emotional_valence,
        "emotional_intensity": t.emotional_intensity,
    }


def trauma_from_dict(data: dict[str, Any]) -> SeededTrauma:
    return SeededTrauma(
        name=data["name"],
        prose=data.get("prose", ""),
        structured=data.get("structured", {}) or {},
        initial_salience=float(data.get("initial_salience", 0.5)),
        emotional_valence=float(data.get("emotional_valence", 0.0)),
        emotional_intensity=float(data.get("emotional_intensity", 0.0)),
    )


def persona_to_definition(p: Persona) -> dict[str, Any]:
    return {
        "identity": p.identity,
        "substrate": {k: dimension_to_dict(v) for k, v in p.substrate.items()},
        "self_concept": {k: dimension_to_dict(v) for k, v in p.self_concept.items()},
        "traumas": [trauma_to_dict(t) for t in p.traumas],
    }


def definition_to_persona(
    persona_id: str, spec_version: int, definition: dict[str, Any]
) -> Persona:
    return Persona(
        persona_id=persona_id,
        spec_version=spec_version,
        identity=definition.get("identity", {}) or {},
        substrate={
            k: dimension_from_dict(k, v) for k, v in (definition.get("substrate") or {}).items()
        },
        self_concept={
            k: dimension_from_dict(k, v) for k, v in (definition.get("self_concept") or {}).items()
        },
        traumas=[trauma_from_dict(t) for t in (definition.get("traumas") or [])],
    )
