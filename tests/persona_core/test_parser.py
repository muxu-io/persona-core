from pathlib import Path

import pytest

from persona_core.parser import parse_persona_file
from persona_core.schema import Persona, Presence

FIXTURE = Path(__file__).parent.parent / "fixtures" / "minimal_persona.md"


def test_parse_returns_persona():
    p = parse_persona_file(FIXTURE)
    assert isinstance(p, Persona)
    assert p.persona_id == "test-persona"
    assert p.spec_version == 1


def test_parse_reads_identity():
    p = parse_persona_file(FIXTURE)
    assert p.identity["name"] == "Test Persona"
    assert p.identity["age"] == 40


def test_parse_substrate_dimensions():
    p = parse_persona_file(FIXTURE)
    assert "cognitive_profile" in p.substrate
    cog = p.substrate["cognitive_profile"]
    assert cog.presence == Presence.ALWAYS_ON
    assert cog.structured["verbal_register"] == "plain"
    assert "low-hedging" in cog.prose


def test_parse_physical_body_with_voice_anchor_and_perception():
    p = parse_persona_file(FIXTURE)
    body = p.substrate["physical.body"]
    assert body.presence == Presence.TRIGGERED
    assert body.voice_anchor == "compact, careful with movement"
    assert body.structured["height"] == "170 cm"
    assert body.triggers == {"keywords": ["body", "look", "appearance"]}
    assert body.perception is not None
    assert "fine, mostly invisible" in body.perception


def test_parse_self_concept_dimensions():
    p = parse_persona_file(FIXTURE)
    assert "moral_self_image" in p.self_concept
    moral = p.self_concept["moral_self_image"]
    assert moral.structured["core_stance"] == "striving"
    assert moral.structured["current_chapter"] == "rebuilding"


def test_parse_traumas():
    p = parse_persona_file(FIXTURE)
    assert len(p.traumas) == 1
    t = p.traumas[0]
    assert t.name == "first failure"
    assert t.age_at == 22
    assert t.initial_salience == 0.8
    assert "viva at 22" in t.prose


def test_parse_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_persona_file(Path("/nonexistent/persona.md"))


def test_parse_missing_persona_id_raises(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("---\nspec_version: 1\n---\n\n# No ID\n\n## Identity\n\n```yaml\nname: x\n```\n")
    with pytest.raises(KeyError):
        parse_persona_file(bad)


def test_voice_tts_params_round_trips_design_metadata(tmp_path):
    """Voice-design tooling writes engine/design_prose/seed into tts_params.
    The parser must pass these through opaquely so the persona file stays
    a stable record of how the voice was authored."""
    p = tmp_path / "vd-test.md"
    p.write_text(
        "---\n"
        "persona_id: vd-test\n"
        "spec_version: 1\n"
        "---\n"
        "# Voice Design Test\n\n"
        "## Identity\n\n"
        "```yaml\n"
        "name: Voice Design Test\n"
        "age: 35\n"
        "```\n\n"
        "## Substrate\n\n"
        "### Voice\n\n"
        "```yaml\n"
        "presence: always_on\n"
        "voice_anchor: warm Glaswegian alto, slight rasp\n"
        "tts_ref: state/vd-test/media/voice-sample.wav\n"
        "tts_params:\n"
        "  engine: voxcpm2\n"
        "  design_prose: warm Glaswegian alto, slight smoker's rasp\n"
        "  seed: 4242\n"
        "```\n\n"
        "Soft, lived-in voice with character.\n",
        encoding="utf-8",
    )
    persona = parse_persona_file(p)
    voice = persona.substrate["voice"]
    params = voice.structured["tts_params"]
    assert params["engine"] == "voxcpm2"
    assert params["design_prose"] == "warm Glaswegian alto, slight smoker's rasp"
    assert params["seed"] == 4242
