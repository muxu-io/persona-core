"""Load an already-born persona for a runtime session: fetch its definition and
runtime state from the store service and precompute dimension embeddings. The
persona's memory already lives in the shared Qdrant collection; there is no
collection creation, no trauma seeding, and no seeded-marker file — trauma seeding
is the store service's projection (persona_store.projection).
"""

from __future__ import annotations

from dataclasses import dataclass

from persona_core.embedding import EmbeddingClient
from persona_core.schema import Persona
from persona_core.state import RuntimeState
from persona_core.store_client import StoreClient
from persona_core.triggering import precompute_dimension_embeddings


@dataclass
class LoadedPersona:
    persona: Persona
    runtime_state: RuntimeState


class PersonaNotInStore(Exception):
    """Raised when load_persona is asked for a persona absent from the store."""


def load_persona(persona_id: str, client: StoreClient, embedder: EmbeddingClient) -> LoadedPersona:
    persona = client.get_persona(persona_id)
    if persona is None:
        raise PersonaNotInStore(persona_id)
    runtime_state = RuntimeState.load(client, persona_id)
    precompute_dimension_embeddings(persona, embedder)
    return LoadedPersona(persona=persona, runtime_state=runtime_state)
