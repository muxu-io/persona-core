from datetime import UTC, datetime, timedelta

import httpx
import pytest

from persona_core.qdrant_store import QdrantStore
from persona_core.records import EpisodicRecord, RecordType


@pytest.fixture
def store():
    s = QdrantStore.in_memory(collection="test", vector_size=4, persona_id="test")
    s.ensure_collection()
    yield s


def _rec(content_user: str, embedding, **kwargs) -> EpisodicRecord:
    return EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id="s-1",
        content={"user": content_user, "assistant": "ok"},
        initial_salience=kwargs.pop("salience", 0.4),
        embedding=embedding,
        **kwargs,
    )


def test_write_and_count(store):
    store.write(_rec("hello", [1.0, 0.0, 0.0, 0.0]))
    store.write(_rec("hi", [0.0, 1.0, 0.0, 0.0]))
    assert store.count() == 2
    assert store.count_unconsolidated() == 2


def test_count_filters_consolidated(store):
    store.write(_rec("hello", [1.0, 0.0, 0.0, 0.0]))
    store.write(_rec("hi", [0.0, 1.0, 0.0, 0.0], consolidated=True))
    assert store.count() == 2
    assert store.count_unconsolidated() == 1


def test_query_returns_top_k_by_similarity(store):
    store.write(_rec("alpha", [1.0, 0.0, 0.0, 0.0]))
    store.write(_rec("beta", [0.0, 1.0, 0.0, 0.0]))
    store.write(_rec("gamma", [0.5, 0.5, 0.0, 0.0]))
    results = store.query(query_vector=[1.0, 0.0, 0.0, 0.0], limit=2)
    assert len(results) == 2
    contents = [r.record.content["user"] for r in results]
    assert contents[0] == "alpha"  # exact match wins
    assert "gamma" in contents  # second-best


def test_query_carries_similarity_score(store):
    store.write(_rec("alpha", [1.0, 0.0, 0.0, 0.0]))
    results = store.query(query_vector=[1.0, 0.0, 0.0, 0.0], limit=1)
    assert results[0].similarity == pytest.approx(1.0, abs=1e-3)


def test_query_filters_by_type(store):
    store.write(_rec("turn-pair-content", [1.0, 0.0, 0.0, 0.0]))
    store.write(
        EpisodicRecord(
            type=RecordType.SEEDED_NARRATIVE,
            session_id=None,
            content={"event": "an early failure"},
            initial_salience=0.8,
            consolidated=True,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
    )
    turn_only = store.query(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        limit=10,
        types=[RecordType.TURN_PAIR],
    )
    assert len(turn_only) == 1
    assert turn_only[0].record.type == RecordType.TURN_PAIR


def test_fetch_session_window(store):
    base = datetime.now(tz=UTC)
    for i in range(5):
        rec = _rec(f"turn-{i}", [0.0, 0.0, 0.0, 1.0])
        rec.created_at = base + timedelta(minutes=i)
        store.write(rec)
    window = store.fetch_session_window(
        session_id="s-1", before=base + timedelta(minutes=3), limit=2
    )
    # window contains the two most-recent turns before t+3min, i.e. turn-1 and turn-2
    contents = sorted([r.content["user"] for r in window])
    assert contents == ["turn-1", "turn-2"]


def _qdrant_up() -> bool:
    try:
        return httpx.get("http://localhost:6333/healthz", timeout=1.0).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _qdrant_up(), reason="Qdrant not running on :6333")
def test_real_qdrant_round_trip():
    s = QdrantStore.http(
        host="localhost",
        port=6333,
        collection="test_round_trip",
        vector_size=4,
        persona_id="test_round_trip",
    )
    # cleanup any leftover collection from a prior failed run
    try:
        s._client.delete_collection("test_round_trip")
    except Exception:
        pass
    try:
        s.ensure_collection()
        rec = EpisodicRecord(
            type=RecordType.TURN_PAIR,
            session_id="s-int",
            content={"user": "ping", "assistant": "pong"},
            initial_salience=0.4,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        s.write(rec)
        hits = s.query([1.0, 0.0, 0.0, 0.0], limit=1)
        assert len(hits) == 1
        assert hits[0].record.content["user"] == "ping"
    finally:
        s._client.delete_collection("test_round_trip")


def test_new_store_is_not_dirty():
    s = QdrantStore.in_memory(collection="fresh", vector_size=4, persona_id="fresh")
    assert s.dirty is False


def test_write_marks_store_dirty(store):
    assert store.dirty is False
    store.write(_rec("hello", [1.0, 0.0, 0.0, 0.0]))
    assert store.dirty is True


def test_ensure_collection_does_not_mark_dirty():
    s = QdrantStore.in_memory(collection="dirty-test", vector_size=4, persona_id="dirty-test")
    s.ensure_collection()
    assert s.dirty is False


def test_collection_exists_true_after_ensure():
    s = QdrantStore.in_memory(collection="exists-true", vector_size=4, persona_id="exists-true")
    s.ensure_collection()
    assert s.collection_exists() is True


def test_collection_exists_false_when_absent():
    s = QdrantStore.in_memory(collection="exists-false", vector_size=4, persona_id="exists-false")
    assert s.collection_exists() is False


def _seeded_record(name: str) -> EpisodicRecord:
    rec = EpisodicRecord(
        type=RecordType.SEEDED_NARRATIVE,
        session_id=None,
        content={"event": f"event for {name}", "name": name},
        initial_salience=0.5,
        consolidated=True,
    )
    rec.embedding = [0.0] * 4
    return rec


def _turn_pair_record() -> EpisodicRecord:
    rec = EpisodicRecord(
        type=RecordType.TURN_PAIR,
        session_id="s1",
        content={"user": "hi", "assistant": "hello"},
        initial_salience=0.4,
    )
    rec.embedding = [0.0] * 4
    return rec


def test_list_by_type_returns_only_matching_records():
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    store.write(_seeded_record("alpha"))
    store.write(_seeded_record("beta"))
    store.write(_turn_pair_record())

    seeded = store.list_by_type(RecordType.SEEDED_NARRATIVE)

    assert {r.content["name"] for r in seeded} == {"alpha", "beta"}
    assert all(r.type == RecordType.SEEDED_NARRATIVE for r in seeded)


def test_list_by_type_returns_empty_when_no_match():
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    store.write(_turn_pair_record())

    assert store.list_by_type(RecordType.SEEDED_NARRATIVE) == []


def test_delete_by_id_removes_record():
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    rec = _seeded_record("alpha")
    store.write(rec)
    assert store.count() == 1

    store.delete_by_id(rec.id)

    assert store.count() == 0


def test_delete_by_id_is_noop_for_missing_id():
    store = QdrantStore.in_memory(collection="t", vector_size=4, persona_id="t")
    store.ensure_collection()
    # No record written. Delete should not raise.
    store.delete_by_id("00000000-0000-0000-0000-000000000000")
    assert store.count() == 0


def _isolation_rec(name, vec):
    from persona_core.records import EpisodicRecord, RecordType

    r = EpisodicRecord(
        type=RecordType.SEEDED_NARRATIVE,
        session_id=None,
        content={"event": name, "name": name},
        initial_salience=0.5,
    )
    r.embedding = vec
    return r


def test_shared_collection_isolates_by_persona_id():
    from qdrant_client import QdrantClient

    from persona_core.qdrant_store import QdrantStore

    client = QdrantClient(":memory:")
    store_a = QdrantStore(client, collection="persona_memory", vector_size=3, persona_id="ada")
    store_b = QdrantStore(client, collection="persona_memory", vector_size=3, persona_id="bo")
    store_a.ensure_collection()  # idempotent; both share one collection
    store_b.ensure_collection()

    store_a.write(_isolation_rec("ada-mem", [1.0, 0.0, 0.0]))
    store_b.write(_isolation_rec("bo-mem", [1.0, 0.0, 0.0]))

    from persona_core.records import RecordType

    assert {r.content["name"] for r in store_a.list_by_type(RecordType.SEEDED_NARRATIVE)} == {
        "ada-mem"
    }
    assert {r.content["name"] for r in store_b.list_by_type(RecordType.SEEDED_NARRATIVE)} == {
        "bo-mem"
    }
    # A query on the shared collection never returns B's point for tenant A.
    hits = store_a.query([1.0, 0.0, 0.0], limit=10)
    assert {h.record.content["name"] for h in hits} == {"ada-mem"}
    assert store_a.count() == 1 and store_b.count() == 1
