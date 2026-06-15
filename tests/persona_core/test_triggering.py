from persona_core.embedding import EmbeddingClient
from persona_core.schema import Dimension, Persona, Presence
from persona_core.triggering import TriggerConfig, select_triggered


class _FakeOllama:
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def embeddings(self, model, prompt):
        # default to zero vector if prompt unmapped
        return {"embedding": self.mapping.get(prompt, [0.0] * 4)}


def _persona_with_dims(dims: dict[str, Dimension]) -> Persona:
    return Persona(
        persona_id="p",
        spec_version=1,
        identity={},
        substrate=dims,
        self_concept={},
        traumas=[],
    )


def test_select_triggered_returns_above_threshold():
    body_dim = Dimension(
        name="physical.body",
        presence=Presence.TRIGGERED,
        prose="compact, wiry",
        structured={},
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    env_dim = Dimension(
        name="environment",
        presence=Presence.TRIGGERED,
        prose="working-class Glasgow",
        structured={},
        embedding=[0.0, 1.0, 0.0, 0.0],
    )
    persona = _persona_with_dims({"physical.body": body_dim, "environment": env_dim})
    ec = EmbeddingClient(
        model="nomic",
        transport=_FakeOllama({"how do you look?": [1.0, 0.0, 0.0, 0.0]}),
    )
    config = TriggerConfig(threshold=0.55, top_k=3)
    selected = select_triggered(
        persona=persona,
        user_message="how do you look?",
        embedder=ec,
        config=config,
    )
    names = [d.name for d in selected]
    assert names == ["physical.body"]


def test_select_triggered_respects_top_k():
    dims = {
        f"d{i}": Dimension(
            name=f"d{i}",
            presence=Presence.TRIGGERED,
            prose="",
            structured={},
            embedding=[1.0 - i * 0.01, 0.0, 0.0, 0.0],
        )
        for i in range(5)
    }
    persona = _persona_with_dims(dims)
    ec = EmbeddingClient(
        model="nomic",
        transport=_FakeOllama({"q": [1.0, 0.0, 0.0, 0.0]}),
    )
    config = TriggerConfig(threshold=0.0, top_k=3)
    selected = select_triggered(persona=persona, user_message="q", embedder=ec, config=config)
    assert len(selected) == 3


def test_keyword_trigger_overrides_low_similarity():
    body_dim = Dimension(
        name="physical.body",
        presence=Presence.TRIGGERED,
        prose="compact, wiry",
        structured={},
        embedding=[0.0, 1.0, 0.0, 0.0],  # orthogonal to query
        triggers={"keywords": ["body", "look"]},
    )
    persona = _persona_with_dims({"physical.body": body_dim})
    ec = EmbeddingClient(
        model="nomic",
        transport=_FakeOllama({"the body of work": [1.0, 0.0, 0.0, 0.0]}),
    )
    config = TriggerConfig(threshold=0.55, top_k=3)
    selected = select_triggered(
        persona=persona,
        user_message="the body of work",
        embedder=ec,
        config=config,
    )
    assert any(d.name == "physical.body" for d in selected)


def test_always_on_dims_are_excluded_from_triggering():
    cog = Dimension(
        name="cognitive_profile",
        presence=Presence.ALWAYS_ON,
        prose="x",
        structured={},
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    persona = _persona_with_dims({"cognitive_profile": cog})
    ec = EmbeddingClient(
        model="nomic",
        transport=_FakeOllama({"q": [1.0, 0.0, 0.0, 0.0]}),
    )
    config = TriggerConfig(threshold=0.5, top_k=3)
    selected = select_triggered(persona=persona, user_message="q", embedder=ec, config=config)
    assert selected == []
