"""Persona file parser. Markdown body with embedded YAML code fences for atoms."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import frontmatter
import yaml

from persona_core.schema import Dimension, Persona, Presence, SeededTrauma

_YAML_FENCE_RE = re.compile(
    r"^```yaml\s*\n(.*?)\n```\s*$",
    re.DOTALL | re.MULTILINE,
)


def parse_persona_file(path: Path) -> Persona:
    if not path.exists():
        raise FileNotFoundError(path)
    post = frontmatter.load(path)
    persona_id = post.metadata["persona_id"]
    spec_version = int(post.metadata.get("spec_version", 1))

    sections = split_by_h2(post.content)
    identity = _parse_identity(sections.get("Identity", ""))
    substrate = _parse_dimension_block(sections.get("Substrate", ""))
    self_concept = _parse_dimension_block(sections.get("Self-concept", ""))
    traumas = _parse_traumas(sections.get("Substrate · Traumas", ""))

    return Persona(
        persona_id=persona_id,
        spec_version=spec_version,
        identity=identity,
        substrate=substrate,
        self_concept=self_concept,
        traumas=traumas,
    )


def split_by_h2(content: str) -> dict[str, str]:
    """Split body by `## Heading` blocks, returning {heading: body}."""
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in content.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def split_by_h3(content: str) -> list[tuple[str, str]]:
    """Return [(heading, body)] for each `### Heading` block."""
    blocks: list[tuple[str, str]] = []
    current_heading: str | None = None
    buf: list[str] = []
    for line in content.splitlines():
        m = re.match(r"^###\s+(.+?)\s*$", line)
        if m:
            if current_heading is not None:
                blocks.append((current_heading, "\n".join(buf).strip()))
            current_heading = m.group(1)
            buf = []
        elif current_heading is not None:
            buf.append(line)
    if current_heading is not None:
        blocks.append((current_heading, "\n".join(buf).strip()))
    return blocks


def _extract_yaml_block(body: str) -> tuple[dict[str, Any], str]:
    """Pull the first ```yaml code fence; return (parsed, remaining_body)."""
    m = _YAML_FENCE_RE.search(body)
    if not m:
        return {}, body.strip()
    parsed = yaml.safe_load(m.group(1)) or {}
    remaining = (body[: m.start()] + body[m.end() :]).strip()
    return parsed, remaining


def _extract_perception(body: str) -> tuple[str | None, str]:
    """Pull the first `#### Perception` sub-block; return (perception_text, remaining_body)."""
    pattern = re.compile(r"^####\s+Perception\s*$", re.MULTILINE)
    m = pattern.search(body)
    if not m:
        return None, body
    perception = body[m.end() :].strip()
    remaining = body[: m.start()].strip()
    return perception, remaining


def _parse_identity(body: str) -> dict[str, Any]:
    parsed, _ = _extract_yaml_block(body)
    return parsed


def _heading_to_key(heading: str) -> str:
    """`Physical · Body` → `physical.body`. `Cognitive profile` → `cognitive_profile`."""
    if "·" in heading:
        parts = [p.strip().lower().replace(" ", "_").replace("-", "_") for p in heading.split("·")]
        return ".".join(parts)
    return heading.strip().lower().replace(" ", "_").replace("-", "_")


def _parse_dimension_block(body: str) -> dict[str, Dimension]:
    out: dict[str, Dimension] = {}
    for heading, dim_body in split_by_h3(body):
        perception, dim_body_no_perc = _extract_perception(dim_body)
        atoms, prose = _extract_yaml_block(dim_body_no_perc)
        presence_str = atoms.pop("presence", "always_on")
        voice_anchor = atoms.pop("voice_anchor", None)
        triggers = atoms.pop("triggers", {}) or {}
        name = _heading_to_key(heading)
        out[name] = Dimension(
            name=name,
            presence=Presence(presence_str),
            voice_anchor=voice_anchor,
            structured=atoms,
            prose=prose,
            perception=perception,
            triggers=triggers,
        )
    return out


def _parse_traumas(body: str) -> list[SeededTrauma]:
    out: list[SeededTrauma] = []
    for heading, t_body in split_by_h3(body):
        atoms, prose = _extract_yaml_block(t_body)
        # heading like "Trauma — first failure" → name "first failure"
        name = heading.split("—", 1)[-1].strip() if "—" in heading else heading.strip()
        out.append(
            SeededTrauma(
                name=name,
                prose=prose,
                structured={
                    k: v
                    for k, v in atoms.items()
                    if k
                    not in {"type", "initial_salience", "emotional_valence", "emotional_intensity"}
                },
                initial_salience=float(atoms.get("initial_salience", 0.5)),
                emotional_valence=float(atoms.get("emotional_valence", 0.0)),
                emotional_intensity=float(atoms.get("emotional_intensity", 0.0)),
            )
        )
    return out
