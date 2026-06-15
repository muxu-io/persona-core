from datetime import UTC, datetime

from persona_core.records import EpisodicRecord, ProcessingDepth, RecordType


def test_turn_pair_record_defaults():
    rec = EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id="s-1",
        content={"user": "hello", "assistant": "hi there"},
        initial_salience=0.4,
    )
    assert rec.id  # uuid auto-assigned
    assert rec.type == RecordType.TURN_PAIR
    assert rec.consolidated is False
    assert rec.processing_depth == ProcessingDepth.PRIMAL
    assert rec.access_count == 0
    assert rec.last_accessed is None
    assert rec.emotional_valence == 0.0
    assert rec.emotional_intensity == 0.0
    assert isinstance(rec.created_at, datetime)
    assert rec.created_at.tzinfo is not None  # tz-aware


def test_seeded_narrative_record_consolidated_at_birth():
    rec = EpisodicRecord(
        type=RecordType.SEEDED_NARRATIVE,
        session_id=None,
        content={"event": "child witnesses brother's death", "age_at": 9},
        initial_salience=0.95,
        consolidated=True,
        emotional_valence=-0.9,
        emotional_intensity=1.0,
    )
    assert rec.consolidated is True
    assert rec.session_id is None


def test_current_salience_equals_initial_in_stage_1():
    rec = EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id="s-1",
        content={"user": "x", "assistant": "y"},
        initial_salience=0.6,
    )
    assert rec.current_salience == 0.6


def test_payload_round_trip_preserves_fields():
    original = EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id="s-42",
        content={"user": "what's my dog's name?", "assistant": "Bramble."},
        initial_salience=0.55,
        emotional_valence=0.2,
        emotional_intensity=0.3,
        access_count=2,
        last_accessed=datetime(2026, 5, 1, 10, 30, tzinfo=UTC),
    )
    payload = original.to_qdrant_payload()
    restored = EpisodicRecord.from_qdrant_payload(original.id, payload)
    assert restored.id == original.id
    assert restored.type == original.type
    assert restored.session_id == original.session_id
    assert restored.content == original.content
    assert restored.initial_salience == original.initial_salience
    assert restored.access_count == 2
    assert restored.last_accessed == original.last_accessed
    assert restored.processing_depth == ProcessingDepth.PRIMAL
    assert restored.consolidated is False


def test_record_payload_roundtrip_with_scenario_id():
    rec = EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id="s-1",
        content={"user": "hi", "assistant": "hello"},
        initial_salience=0.4,
        scenario_id="shift-cut-corridor",
    )
    payload = rec.to_qdrant_payload()
    assert payload["scenario_id"] == "shift-cut-corridor"

    restored = EpisodicRecord.from_qdrant_payload(rec.id, payload)
    assert restored.scenario_id == "shift-cut-corridor"


def test_record_payload_omits_scenario_id_when_none():
    rec = EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id="s-1",
        content={"user": "hi", "assistant": "hello"},
        initial_salience=0.4,
    )
    payload = rec.to_qdrant_payload()
    assert "scenario_id" not in payload  # additive: don't pollute pre-existing record shape


def test_record_payload_legacy_no_scenario_id_deserializes_to_none():
    legacy_payload = {
        "type": "turn_pair",
        "session_id": "s-old",
        "content": {"user": "u", "assistant": "a"},
        "initial_salience": 0.5,
        "created_at": "2026-04-15T10:00:00+00:00",
        "emotional_valence": 0.0,
        "emotional_intensity": 0.0,
        "access_count": 0,
        "last_accessed": None,
        "processing_depth": "primal",
        "consolidated": False,
    }
    rec = EpisodicRecord.from_qdrant_payload("legacy-id", legacy_payload)
    assert rec.scenario_id is None
