from pathlib import Path

from persona_core.export import render_export_md
from persona_core.parser import parse_persona_file
from persona_core.serialization import definition_to_persona, persona_to_definition

FIXTURE = Path(__file__).parent.parent / "fixtures" / "minimal_persona.md"


def test_export_roundtrips_through_parser(tmp_path):
    original = parse_persona_file(FIXTURE)
    definition = persona_to_definition(original)
    persona = definition_to_persona(original.persona_id, original.spec_version, definition)

    md = render_export_md(persona)
    out = tmp_path / "export.md"
    out.write_text(md)

    reparsed = parse_persona_file(out)
    assert persona_to_definition(reparsed) == definition


def test_export_starts_with_frontmatter():
    persona = parse_persona_file(FIXTURE)
    md = render_export_md(persona)
    assert md.startswith("---\n")
    assert f"persona_id: {persona.persona_id}" in md
