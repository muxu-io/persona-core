"""Embedding client. Wraps Ollama's embeddings API with an in-memory cache."""

from __future__ import annotations

from typing import Any, Protocol


class _Transport(Protocol):
    def embeddings(self, model: str, prompt: str) -> dict[str, Any]: ...


class EmbeddingClient:
    def __init__(self, model: str, transport: _Transport | None = None, host: str | None = None):
        self.model = model
        if transport is None:
            import ollama

            self._transport = ollama.Client(host=host) if host else ollama.Client()
        else:
            self._transport = transport
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        result = self._transport.embeddings(model=self.model, prompt=text)
        vec = list(result["embedding"])
        self._cache[text] = vec
        return vec

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def cache_size(self) -> int:
        return len(self._cache)
