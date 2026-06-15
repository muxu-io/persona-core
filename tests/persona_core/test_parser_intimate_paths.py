"""Persona parser — optional intimate_reference_image + self_portrait_image paths."""

from __future__ import annotations

from pathlib import Path

from persona_core.parser import parse_persona_file


def test_parser_accepts_intimate_reference_image_under_sexuality(tmp_path: Path):
    md = tmp_path / "smoke.md"
    md.write_text(
        """---
persona_id: smoke
spec_version: 1
---

## Identity
- name: Ada

## Substrate

### Sexuality

```yaml
presence: always_on
intimate_reference_image: state/smoke/media/portrait-intimate.webp
```

Comfortable with intimacy.
""",
        encoding="utf-8",
    )
    persona = parse_persona_file(md)
    sx = persona.substrate.get("sexuality")
    assert sx is not None
    assert (
        sx.structured.get("intimate_reference_image") == "state/smoke/media/portrait-intimate.webp"
    )


def test_parser_accepts_self_portrait_image_under_body_image(tmp_path: Path):
    md = tmp_path / "smoke.md"
    md.write_text(
        """---
persona_id: smoke
spec_version: 1
---

## Identity
- name: Ada

## Substrate

### Cognitive profile

```yaml
presence: always_on
voice_anchor: precise
```

A wry mind.

## Self-concept

### Body image

```yaml
self_portrait_image: state/smoke/media/portrait-self.webp
```

Sees herself as solid.
""",
        encoding="utf-8",
    )
    persona = parse_persona_file(md)
    bi = persona.self_concept.get("body_image")
    assert bi is not None
    assert bi.structured.get("self_portrait_image") == "state/smoke/media/portrait-self.webp"
