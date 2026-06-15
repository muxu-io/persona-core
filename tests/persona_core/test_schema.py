from persona_core.schema import (
    Dimension,
    Persona,
    Presence,
    SeededTrauma,
)


def test_dimension_defaults_and_required_fields():
    d = Dimension(
        name="cognitive_profile",
        presence=Presence.ALWAYS_ON,
        prose="Quick, concrete thinker who hedges in groups.",
        structured={"verbal_register": "plain"},
    )
    assert d.name == "cognitive_profile"
    assert d.voice_anchor is None
    assert d.perception is None
    assert d.triggers == {}
    assert d.embedding is None


def test_dimension_for_triggered_substrate_field():
    d = Dimension(
        name="physical.body",
        presence=Presence.TRIGGERED,
        voice_anchor="compact, wiry, carries herself slightly hunched",
        prose="Compact and wiry...",
        structured={"build": "compact, wiry", "height": "163 cm"},
        triggers={"keywords": ["body", "look", "appearance"]},
        perception="serviceable but plain",
    )
    assert d.presence == Presence.TRIGGERED
    assert d.voice_anchor.startswith("compact")
    assert "keywords" in d.triggers


def test_persona_construction():
    cog = Dimension(
        name="cognitive_profile",
        presence=Presence.ALWAYS_ON,
        prose="quick, concrete",
        structured={},
    )
    body_image = Dimension(
        name="body_image",
        presence=Presence.ALWAYS_ON,
        prose="serviceable but plain",
        structured={},
    )
    p = Persona(
        persona_id="ada-mcleish",
        spec_version=1,
        identity={"name": "Ada McLeish", "age": 56, "role": "trauma nurse"},
        substrate={"cognitive_profile": cog},
        self_concept={"body_image": body_image},
        traumas=[],
    )
    assert p.persona_id == "ada-mcleish"
    assert p.substrate["cognitive_profile"].prose == "quick, concrete"


def test_seeded_trauma_carries_metadata():
    t = SeededTrauma(
        name="brother-died-age-9",
        prose="When he was nine, he watched his brother step into traffic...",
        structured={"age_at": 9, "charge": "high"},
        initial_salience=0.95,
        emotional_valence=-0.9,
        emotional_intensity=1.0,
    )
    assert t.age_at == 9
    assert t.charge == "high"
    assert t.initial_salience == 0.95
