"""Deterministic Markdown+YAML renderer for a persona definition. Relocated from
persona_web._render_persona and adapted to persona_core.Persona. The output is an
audit/rollback artifact, NOT the source of truth — it must round-trip through
persona_core.parser.parse_persona_file unchanged.
"""

from __future__ import annotations

from typing import Any

import yaml

from persona_core.schema import Dimension, Persona, SeededTrauma

_HEADING_TO_KEY = {
    "cognitive_profile": "Cognitive profile",
    "temperament": "Temperament",
    "ethics_innate": "Ethics innate",
    "environment": "Environment",
    "instincts": "Instincts",
    "sexuality": "Sexuality",
    "body": "Body",
    "face": "Face",
    "age_and_presentation": "Age and presentation",
    "marks": "Marks",
    "posture": "Posture",
    "voice": "Voice",
    "body_image": "Body image",
    "moral_self_image": "Moral self-image",
    "origin_narrative": "Origin narrative",
    "ethics_articulated": "Ethics articulated",
}


def _key_to_heading(key: str) -> str:
    return _HEADING_TO_KEY.get(key, key.replace("_", " ").capitalize())


def _yaml(block: dict[str, Any]) -> str:
    return yaml.safe_dump(block, sort_keys=False, allow_unicode=True).rstrip()


def _substrate_atoms(d: Dimension) -> dict[str, Any]:
    atoms: dict[str, Any] = {"presence": d.presence.value}
    if d.voice_anchor is not None:
        atoms["voice_anchor"] = d.voice_anchor
    if d.triggers:
        atoms["triggers"] = d.triggers
    atoms.update(d.structured)
    return atoms


def _trauma_atoms(t: SeededTrauma) -> dict[str, Any]:
    atoms = dict(t.structured)
    atoms["initial_salience"] = t.initial_salience
    atoms["emotional_valence"] = t.emotional_valence
    atoms["emotional_intensity"] = t.emotional_intensity
    return atoms


def render_export_md(p: Persona) -> str:
    parts: list[str] = []
    parts.append("---")
    parts.append(f"persona_id: {p.persona_id}")
    parts.append(f"spec_version: {p.spec_version}")
    parts.append("---")
    parts.append("")
    parts.append(f"# {p.identity.get('name', p.persona_id)}")
    parts.append("")
    parts.append("## Identity")
    parts.append("")
    parts.append("```yaml")
    parts.append(_yaml(p.identity))
    parts.append("```")
    parts.append("")

    if p.substrate:
        parts.append("## Substrate")
        parts.append("")
        for key, dim in p.substrate.items():
            parts.append(f"### {_key_to_heading(key)}")
            parts.append("")
            parts.append("```yaml")
            parts.append(_yaml(_substrate_atoms(dim)))
            parts.append("```")
            parts.append("")
            if dim.prose:
                parts.append(dim.prose.strip())
                parts.append("")
            if dim.perception:
                parts.append("#### Perception")
                parts.append("")
                parts.append(dim.perception.strip())
                parts.append("")

    if p.self_concept:
        parts.append("## Self-concept")
        parts.append("")
        for key, dim in p.self_concept.items():
            parts.append(f"### {_key_to_heading(key)}")
            parts.append("")
            atoms: dict[str, Any] = {"presence": dim.presence.value}
            atoms.update(dim.structured)
            parts.append("```yaml")
            parts.append(_yaml(atoms))
            parts.append("```")
            parts.append("")
            if dim.prose:
                parts.append(dim.prose.strip())
                parts.append("")

    if p.traumas:
        parts.append("## Substrate · Traumas")
        parts.append("")
        for t in p.traumas:
            parts.append(f"### {t.name}")
            parts.append("")
            parts.append("```yaml")
            parts.append(_yaml(_trauma_atoms(t)))
            parts.append("```")
            parts.append("")
            if t.prose:
                parts.append(t.prose.strip())
                parts.append("")

    return "\n".join(parts).rstrip() + "\n"
