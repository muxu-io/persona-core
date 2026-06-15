import httpx
import respx

from persona_core.store_client import StoreClient

BASE = "http://store:7600"


@respx.mock
def test_get_persona_deserializes_definition():
    respx.get(f"{BASE}/personas/ada").mock(
        return_value=httpx.Response(
            200,
            json={
                "persona_id": "ada",
                "spec_version": 2,
                "definition": {
                    "identity": {"name": "Ada"},
                    "substrate": {},
                    "self_concept": {},
                    "traumas": [],
                },
                "tags": [],
            },
        )
    )
    persona = StoreClient(BASE).get_persona("ada")
    assert persona is not None
    assert persona.identity["name"] == "Ada"
    assert persona.spec_version == 2


@respx.mock
def test_get_persona_missing_returns_none():
    respx.get(f"{BASE}/personas/ghost").mock(return_value=httpx.Response(404))
    assert StoreClient(BASE).get_persona("ghost") is None


@respx.mock
def test_put_runtime_posts_payload():
    route = respx.put(f"{BASE}/personas/ada/runtime").mock(return_value=httpx.Response(204))
    StoreClient(BASE).put_runtime(
        "ada", session_count=5, last_session_at="2026-06-14T00:00:00+00:00"
    )
    assert route.called


@respx.mock
def test_get_media_returns_bytes():
    respx.get(f"{BASE}/personas/ada/media/voice-sample.wav").mock(
        return_value=httpx.Response(200, content=b"RIFFfake")
    )
    assert StoreClient(BASE).get_media("ada", "voice-sample.wav") == b"RIFFfake"


@respx.mock
def test_get_media_missing_returns_none():
    respx.get(f"{BASE}/personas/ada/media/none.wav").mock(return_value=httpx.Response(404))
    assert StoreClient(BASE).get_media("ada", "none.wav") is None
