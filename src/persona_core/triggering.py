"""Triggering pipeline. Selects which triggered dimensions enter context this turn."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from persona_core.embedding import EmbeddingClient
from persona_core.schema import Dimension, Persona


@dataclass(frozen=True)
class TriggerConfig:
    threshold: float = 0.55
    top_k: int = 3


def select_triggered(
    persona: Persona,
    user_message: str,
    embedder: EmbeddingClient,
    config: TriggerConfig,
) -> list[Dimension]:
    triggered_dims = persona.all_triggered_dimensions()
    if not triggered_dims:
        return []
    query_vec = embedder.embed(user_message)

    scored: list[tuple[float, Dimension]] = []
    for dim in triggered_dims:
        sim = _similarity(query_vec, dim.embedding) if dim.embedding else 0.0
        keyword_hit = _keyword_match(user_message, dim.triggers.get("keywords", []))
        # Keyword hit forces the dimension above threshold.
        score = max(sim, 1.0 if keyword_hit else 0.0)
        if score >= config.threshold:
            scored.append((score, dim))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [dim for _, dim in scored[: config.top_k]]


def _similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _keyword_match(message: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)
    return bool(pattern.search(message))


def precompute_dimension_embeddings(persona: Persona, embedder: EmbeddingClient) -> None:
    """Populate `dim.embedding` for every triggered dimension. Call at startup."""
    for dim in persona.all_triggered_dimensions():
        if dim.embedding is None:
            dim.embedding = embedder.embed(dim.prose)
