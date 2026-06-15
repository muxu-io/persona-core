from pathlib import Path

import httpx
import pytest
import respx

from persona_core.bootstrap import LoadedPersona, PersonaNotInStore, load_persona
from persona_core.parser import parse_persona_file
from persona_core.serialization import persona_to_definition
from persona_core.store_client import StoreClient

BASE = "http://store:7600"
FIXTURE = Path(__file__).parent.parent / "fixtures" / "minimal_persona.md"


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1] * 768


def _definition() -> dict:
    return persona_to_definition(parse_persona_file(FIXTURE))


@respx.mock
def test_load_persona_fetches_and_precomputes():
    pid = "test-persona"
    respx.get(f"{BASE}/personas/{pid}").mock(
        return_value=httpx.Response(
            200,
            json={"persona_id": pid, "spec_version": 1, "definition": _definition(), "tags": []},
        )
    )
    respx.get(f"{BASE}/personas/{pid}/runtime").mock(return_value=httpx.Response(404))

    embedder = _FakeEmbedder()
    loaded = load_persona(pid, StoreClient(BASE), embedder)

    assert isinstance(loaded, LoadedPersona)
    assert loaded.persona.persona_id == pid
    assert loaded.runtime_state.session_count == 0
    triggered = loaded.persona.all_triggered_dimensions()
    assert triggered  # the fixture has a triggered dimension
    assert all(d.embedding is not None for d in triggered)
    assert embedder.calls  # precompute embedded the triggered dimension prose


@respx.mock
def test_load_persona_missing_raises():
    respx.get(f"{BASE}/personas/ghost").mock(return_value=httpx.Response(404))
    with pytest.raises(PersonaNotInStore):
        load_persona("ghost", StoreClient(BASE), _FakeEmbedder())
