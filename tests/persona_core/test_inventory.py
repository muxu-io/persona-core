from pathlib import Path

from persona_core.inventory import (
    EntryCategory,
    InventoryEntry,
    InventoryReport,
    compute_inventory,
)
from persona_core.parser import parse_persona_file

_NO_MEDIA = lambda name: False  # noqa: E731 — terse media-existence stub for tests


def _load_blanks_persona():
    fixture = Path(__file__).parent.parent / "fixtures" / "inventory_blanks_persona.md"
    return parse_persona_file(fixture)


def test_inventory_flags_missing_canonical_portrait():
    persona = _load_blanks_persona()

    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=0)

    slots = {e.slot for e in report.entries if e.category == EntryCategory.MISSING}
    assert "portrait_canonical" in slots
    assert "portrait_face" in slots
    assert "voice_sample" in slots
    assert "self_portrait" in slots
    assert "avatar" in slots


def test_inventory_returns_inventory_report_shape():
    persona = _load_blanks_persona()
    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=0)
    assert isinstance(report, InventoryReport)
    assert report.persona_id == "blanks-test"
    assert all(isinstance(e, InventoryEntry) for e in report.entries)
    # Entries are ordered: missing first, partial next.
    cats = [e.category for e in report.entries]
    assert cats == sorted(cats, key=lambda c: 0 if c == EntryCategory.MISSING else 1)


def test_inventory_does_not_flag_artifact_when_path_set_and_media_present():
    persona = _load_blanks_persona()
    # Inject a non-null canonical_image and report its bare key as present in store-svc.
    persona.substrate["physical.body"].structured[
        "canonical_image"
    ] = "state/blanks-test/media/portrait-canonical.webp"

    report = compute_inventory(
        persona=persona,
        media_exists=lambda name: name == "portrait-canonical.webp",
        qdrant_seeded_count=0,
    )

    slots = {e.slot for e in report.entries if e.category == EntryCategory.MISSING}
    assert "portrait_canonical" not in slots


def test_inventory_flags_missing_whole_dimensions():
    persona = _load_blanks_persona()
    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=0)

    missing_dim_slots = {
        e.slot
        for e in report.entries
        if e.category == EntryCategory.MISSING and e.slot.startswith("dimension:")
    }
    # Substrate (expected, missing in fixture):
    assert "dimension:temperament" in missing_dim_slots
    assert "dimension:ethics_innate" in missing_dim_slots
    assert "dimension:environment" in missing_dim_slots
    assert "dimension:instincts" in missing_dim_slots
    assert "dimension:sexuality" in missing_dim_slots
    # Self-concept (expected, missing in fixture):
    assert "dimension:moral_self_image" in missing_dim_slots
    assert "dimension:ethics_articulated" in missing_dim_slots


def test_inventory_does_not_flag_present_dimensions():
    persona = _load_blanks_persona()
    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=0)

    missing_dim_slots = {
        e.slot
        for e in report.entries
        if e.category == EntryCategory.MISSING and e.slot.startswith("dimension:")
    }
    # cognitive_profile, physical.body, physical.face, physical.voice, body_image,
    # origin_narrative are present in the fixture.
    assert "dimension:cognitive_profile" not in missing_dim_slots
    assert "dimension:body_image" not in missing_dim_slots


def test_inventory_flags_partial_dimension_missing_voice_anchor():
    persona = _load_blanks_persona()
    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=0)

    partial_slots = {
        (e.slot, e.detail) for e in report.entries if e.category == EntryCategory.PARTIAL
    }
    # origin_narrative has no voice_anchor (and no triggers, but it's always_on
    # so triggers is moot). Implementation should flag voice_anchor only.
    assert any(
        s.startswith("dimension:origin_narrative") and "voice_anchor" in d for s, d in partial_slots
    )


def test_inventory_flags_partial_dimension_missing_triggers():
    persona = _load_blanks_persona()
    # physical.body is TRIGGERED in the fixture and has triggers populated.
    # Strip the triggers to exercise the "no triggers" partial branch.
    persona.substrate["physical.body"].triggers = {}

    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=0)

    partial = {(e.slot, e.detail) for e in report.entries if e.category == EntryCategory.PARTIAL}
    assert ("dimension:physical.body", "substrate.physical.body: no triggers") in partial


def test_inventory_does_not_flag_no_triggers_on_always_on_dim():
    persona = _load_blanks_persona()
    # cognitive_profile is ALWAYS_ON in the fixture and has no triggers — that's
    # not a partial signal because triggers is moot for always_on dimensions.
    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=0)

    partial = {(e.slot, e.detail) for e in report.entries if e.category == EntryCategory.PARTIAL}
    assert not any(s == "dimension:cognitive_profile" and "no triggers" in d for s, d in partial)


def _load_minimal_persona():
    # tests/fixtures/minimal_persona.md has one trauma named "first failure".
    return parse_persona_file(Path(__file__).parent.parent / "fixtures" / "minimal_persona.md")


def test_inventory_flags_no_traumas_when_file_and_qdrant_both_empty():
    persona = _load_blanks_persona()  # zero traumas in this fixture
    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=0)

    slots = {e.slot for e in report.entries if e.category == EntryCategory.MISSING}
    assert "no_seeded_narratives" in slots


def test_inventory_no_trauma_flag_when_qdrant_has_records():
    persona = _load_blanks_persona()
    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=2)

    slots = {e.slot for e in report.entries if e.category == EntryCategory.MISSING}
    assert "no_seeded_narratives" not in slots


def test_inventory_no_trauma_flag_when_qdrant_unreachable():
    persona = _load_blanks_persona()
    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=None)

    slots = {e.slot for e in report.entries if e.category == EntryCategory.MISSING}
    # Conservative: don't flag when we can't be sure.
    assert "no_seeded_narratives" not in slots


def test_inventory_flags_trauma_image_partial_only_when_at_least_one_image_exists():
    from persona_core.schema import SeededTrauma

    persona = _load_minimal_persona()  # one trauma: "first failure"
    # A second trauma lets one image signal opt-in while the other stays missing —
    # the store-svc model checks each trauma's own image key, not a directory listing.
    persona.traumas.append(
        SeededTrauma(
            name="second wound",
            prose="A later wound, rendered.",
            structured={},
            initial_salience=0.5,
            emotional_valence=0.0,
            emotional_intensity=0.0,
        )
    )

    # Case A: no trauma images at all → no flag (author opted out).
    report = compute_inventory(persona=persona, media_exists=_NO_MEDIA, qdrant_seeded_count=1)
    partial_slots_a = {e.slot for e in report.entries if e.category == EntryCategory.PARTIAL}
    assert not any(s.startswith("trauma_image:") for s in partial_slots_a)

    # Case B: the second trauma's image exists → "first failure" is flagged
    # (the persona uses trauma images and is missing this one).
    report = compute_inventory(
        persona=persona,
        media_exists=lambda name: name == "second wound.webp",
        qdrant_seeded_count=1,
    )
    partial_slots_b = {e.slot for e in report.entries if e.category == EntryCategory.PARTIAL}
    assert "trauma_image:first failure" in partial_slots_b
