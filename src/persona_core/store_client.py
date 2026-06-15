"""Synchronous HTTP client for the persona-store cold path. Used by the runtime
(bootstrap/state/cli). Returns persona_core domain objects."""

from __future__ import annotations

from typing import Any

import httpx

from persona_core.schema import Persona
from persona_core.serialization import definition_to_persona


class StoreClient:
    def __init__(self, base_url: str, *, timeout_s: float = 10.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s

    def get_persona(self, persona_id: str) -> Persona | None:
        with httpx.Client(base_url=self._base, timeout=self._timeout) as h:
            resp = h.get(f"/personas/{persona_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return definition_to_persona(data["persona_id"], data["spec_version"], data["definition"])

    def get_runtime(self, persona_id: str) -> dict[str, Any] | None:
        with httpx.Client(base_url=self._base, timeout=self._timeout) as h:
            resp = h.get(f"/personas/{persona_id}/runtime")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def put_runtime(self, persona_id: str, **fields: Any) -> None:
        with httpx.Client(base_url=self._base, timeout=self._timeout) as h:
            resp = h.put(f"/personas/{persona_id}/runtime", json=fields)
        resp.raise_for_status()

    def list_scenarios(self, persona_id: str) -> list[dict[str, Any]]:
        with httpx.Client(base_url=self._base, timeout=self._timeout) as h:
            resp = h.get(f"/personas/{persona_id}/scenarios")
        resp.raise_for_status()
        return resp.json()["items"]

    def get_scenario(self, persona_id: str, scenario_id: str) -> dict[str, Any] | None:
        with httpx.Client(base_url=self._base, timeout=self._timeout) as h:
            resp = h.get(f"/personas/{persona_id}/scenarios/{scenario_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_media(self, persona_id: str, name: str) -> bytes | None:
        with httpx.Client(base_url=self._base, timeout=self._timeout) as h:
            resp = h.get(f"/personas/{persona_id}/media/{name}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content
