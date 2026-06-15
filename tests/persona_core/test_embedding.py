import httpx
import pytest

from persona_core.embedding import EmbeddingClient


class _FakeOllama:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def embeddings(self, model, prompt):
        self.calls.append((model, prompt))
        return {"embedding": self.vector}


def test_embed_returns_vector():
    fake = _FakeOllama([0.1] * 768)
    client = EmbeddingClient(model="nomic-embed-text", transport=fake)
    vec = client.embed("she carries herself slightly hunched")
    assert len(vec) == 768
    assert fake.calls == [("nomic-embed-text", "she carries herself slightly hunched")]


def test_embed_many_caches_repeats():
    fake = _FakeOllama([0.2] * 768)
    client = EmbeddingClient(model="nomic-embed-text", transport=fake)
    a = client.embed("alpha")
    b = client.embed("alpha")
    assert a == b
    assert len(fake.calls) == 1, "second call should hit cache"


def test_embed_different_strings_separate_calls():
    fake = _FakeOllama([0.3] * 768)
    client = EmbeddingClient(model="nomic-embed-text", transport=fake)
    client.embed("alpha")
    client.embed("beta")
    assert len(fake.calls) == 2


def _ollama_up() -> bool:
    try:
        httpx.get("http://localhost:11434/", timeout=1.0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running on :11434")
def test_real_ollama_embedding():
    client = EmbeddingClient(model="nomic-embed-text")
    vec = client.embed("she carries herself slightly hunched")
    assert len(vec) == 768
    assert all(isinstance(x, float) for x in vec)
