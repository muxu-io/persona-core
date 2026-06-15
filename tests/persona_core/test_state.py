import json
from datetime import UTC, datetime

import httpx
import respx

from persona_core.state import RuntimeState
from persona_core.store_client import StoreClient

BASE = "http://store:7600"


@respx.mock
def test_load_reads_runtime_from_store():
    respx.get(f"{BASE}/personas/ada/runtime").mock(
        return_value=httpx.Response(
            200,
            json={
                "session_count": 3,
                "last_session_at": "2026-06-14T00:00:00+00:00",
                "last_sleep_pass_at": None,
                "memory_seeded": True,
            },
        )
    )
    rs = RuntimeState.load(StoreClient(BASE), "ada")
    assert rs.session_count == 3
    assert rs.last_session_at is not None


@respx.mock
def test_load_defaults_when_runtime_absent():
    respx.get(f"{BASE}/personas/ada/runtime").mock(return_value=httpx.Response(404))
    rs = RuntimeState.load(StoreClient(BASE), "ada")
    assert rs.session_count == 0
    assert rs.last_session_at is None


@respx.mock
def test_save_puts_runtime_payload():
    route = respx.put(f"{BASE}/personas/ada/runtime").mock(return_value=httpx.Response(204))
    rs = RuntimeState(
        persona_id="ada",
        client=StoreClient(BASE),
        session_count=5,
        last_session_at=datetime(2026, 6, 14, tzinfo=UTC),
    )
    rs.save()
    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body["session_count"] == 5
    assert body["last_session_at"].startswith("2026-06-14")
