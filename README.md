# persona-core

The shared **data-model + storage kernel** for the persona runtime and its
authoring services. It owns the persona schema, episodic-memory records, the
Markdown+YAML parser/serializer, the Qdrant store (multi-tenant by `persona_id`),
embedding, triggering, inventory, the deterministic Markdown exporter, and the
cold-path HTTP `StoreClient`.

`persona-core` depends on nothing in the runtime or authoring layers (no
back-edges) — it is the linchpin every other repo pip-installs.

## Install

Published to a **public** GitLab PyPI registry under muxu-io (no auth to install):

```toml
# pyproject.toml
[[tool.poetry.source]]
name = "gitlab"
url = "https://gitlab.com/api/v4/projects/<project-id>/packages/pypi/simple"
priority = "supplemental"
```

```bash
poetry add persona-core
```

## Frozen public surface

Modules: `schema`, `records`, `parser`, `serialization`, `embedding`,
`qdrant_store`, `triggering`, `inventory`, `store_client`, `export`,
`bootstrap` (slim `load_persona`), `state`.

The cold-path read surface on `StoreClient` is the frozen six: `get_persona`,
`get_runtime`, `put_runtime`, `list_scenarios`, `get_scenario`, `get_media`.

## Develop

```bash
poetry install
poetry run pytest          # unit-level; no live services (qdrant in-memory, respx for HTTP)
poetry run ruff check src tests
poetry run black --check src tests
```

License: Apache-2.0.
