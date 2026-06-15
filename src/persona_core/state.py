"""Persona runtime state, backed by the persona-store cold path (Postgres via
store-svc). Same field surface as before; load/save are HTTP, not yaml.
`unconsolidated_count` is a live count, not persisted — callers populate it from
the Qdrant store (store.count_unconsolidated()).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from persona_core.store_client import StoreClient


@dataclass
class RuntimeState:
    persona_id: str
    client: StoreClient
    session_count: int = 0
    last_session_at: datetime | None = None
    last_sleep_pass_at: datetime | None = None  # remains None through Stage 1
    unconsolidated_count: int = 0

    @classmethod
    def load(cls, client: StoreClient, persona_id: str) -> RuntimeState:
        data = client.get_runtime(persona_id) or {}
        return cls(
            persona_id=persona_id,
            client=client,
            session_count=int(data.get("session_count", 0)),
            last_session_at=_parse_dt(data.get("last_session_at")),
            last_sleep_pass_at=_parse_dt(data.get("last_sleep_pass_at")),
        )

    def save(self) -> None:
        self.client.put_runtime(
            self.persona_id,
            session_count=self.session_count,
            last_session_at=(self.last_session_at.isoformat() if self.last_session_at else None),
            last_sleep_pass_at=(
                self.last_sleep_pass_at.isoformat() if self.last_sleep_pass_at else None
            ),
        )


def _parse_dt(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)
