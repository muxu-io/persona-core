"""Inventory: what slots on this persona are blank or partial?

Pure functions; no I/O of their own. Media existence is supplied by the caller as a
`media_exists(name) -> bool` callable (the CLI backs it with store-svc; tests fake it).
The CLI subcommand wraps these into JSON for the re-authoring skill.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from persona_core.schema import Persona, Presence

MediaExists = Callable[[str], bool]


class EntryCategory(StrEnum):
    MISSING = "missing"
    PARTIAL = "partial"


@dataclass
class InventoryEntry:
    slot: str  # canonical slot identifier, e.g. "voice_sample", "dimension:sexuality"
    category: EntryCategory
    label: str  # human-readable for the menu
    detail: str = ""  # optional extra info ("no voice_anchor", etc.)


@dataclass
class InventoryReport:
    persona_id: str
    entries: list[InventoryEntry] = field(default_factory=list)


# Slot definitions: (slot, dimension_key, atom_name, label, parent_path_for_label)
# Note dimension keys: parser converts "Physical · Body" → "physical.body" (dot, not underscore).
# Parser key conventions: "Physical · Body" → "physical.body" (dot-separated), other
# headings → snake_case ("Cognitive profile" → "cognitive_profile").
EXPECTED_SUBSTRATE = (
    "cognitive_profile",
    "temperament",
    "ethics_innate",
    "environment",
    "physical.body",
    "physical.face",
    "physical.voice",
    "instincts",
    "sexuality",
)
EXPECTED_SELF_CONCEPT = (
    "body_image",
    "moral_self_image",
    "origin_narrative",
    "ethics_articulated",
)
# Display labels override the raw key for the menu.
_DIMENSION_LABELS = {
    "physical.body": "Body",
    "physical.face": "Face",
    "physical.voice": "Voice",
    "ethics_innate": "Ethics (innate)",
    "ethics_articulated": "Ethics (articulated)",
    "moral_self_image": "Moral self-image",
    "origin_narrative": "Origin narrative",
    "cognitive_profile": "Cognitive profile",
}


_ARTIFACT_SLOTS: list[tuple[str, str, str, str, str]] = [
    (
        "portrait_canonical",
        "physical.body",
        "canonical_image",
        "Canonical portrait",
        "substrate.physical.body",
    ),
    (
        "portrait_face",
        "physical.face",
        "canonical_image",
        "Face portrait",
        "substrate.physical.face",
    ),
    (
        "voice_sample",
        "physical.voice",
        "tts_ref",
        "Voice sample",
        "substrate.physical.voice",
    ),
    (
        "avatar",
        "physical.body",
        "avatar_artifact",
        "Avatar",
        "substrate.physical.body",
    ),
]


def compute_inventory(
    *,
    persona: Persona,
    media_exists: MediaExists,
    qdrant_seeded_count: int | None,
) -> InventoryReport:
    """Build the inventory report from a parsed Persona, a media-existence check
    (`media_exists(name) -> bool`, backed by store-svc), and (optionally) the count
    of SEEDED_NARRATIVE records in Qdrant.

    `qdrant_seeded_count is None` means Qdrant was unreachable; trauma-related
    flags depending on Qdrant state are skipped.
    """
    entries: list[InventoryEntry] = []
    entries.extend(_artifact_entries(persona, media_exists))
    entries.extend(_self_portrait_entries(persona, media_exists))
    entries.extend(_missing_dimension_entries(persona))

    no_traumas = _no_seeded_narratives_entry(persona, qdrant_seeded_count)
    if no_traumas is not None:
        entries.append(no_traumas)

    trauma_missing, trauma_partial = _trauma_entries(persona, media_exists)
    entries.extend(trauma_missing)
    entries.extend(trauma_partial)

    entries.extend(_partial_dimension_entries(persona))

    # Stable sort: MISSING before PARTIAL. Within a category, append order is preserved.
    entries.sort(key=lambda e: 0 if e.category == EntryCategory.MISSING else 1)
    return InventoryReport(persona_id=persona.persona_id, entries=entries)


def _artifact_entries(persona: Persona, media_exists: MediaExists) -> list[InventoryEntry]:
    out: list[InventoryEntry] = []
    for slot, dim_key, atom, label, parent in _ARTIFACT_SLOTS:
        dim = persona.substrate.get(dim_key)
        if dim is None:
            # Dimension absent — a bigger issue flagged elsewhere.
            # Don't flag the artifact under a non-existent dimension here.
            continue
        path_in_file = dim.structured.get(atom)
        if _is_missing_artifact(path_in_file, media_exists):
            out.append(
                InventoryEntry(
                    slot=slot,
                    category=EntryCategory.MISSING,
                    label=label,
                    detail=parent,
                )
            )
    return out


def _self_portrait_entries(persona: Persona, media_exists: MediaExists) -> list[InventoryEntry]:
    dim = persona.self_concept.get("body_image")
    if dim is None:
        return []
    path_in_file = dim.structured.get("self_portrait_image")
    if _is_missing_artifact(path_in_file, media_exists):
        return [
            InventoryEntry(
                slot="self_portrait",
                category=EntryCategory.MISSING,
                label="Self-portrait",
                detail="self_concept.body_image",
            )
        ]
    return []


def _label_for_dim(key: str) -> str:
    return _DIMENSION_LABELS.get(key, key.replace("_", " ").capitalize())


def _missing_dimension_entries(persona: Persona) -> list[InventoryEntry]:
    out: list[InventoryEntry] = []
    for key in EXPECTED_SUBSTRATE:
        if key not in persona.substrate:
            out.append(
                InventoryEntry(
                    slot=f"dimension:{key}",
                    category=EntryCategory.MISSING,
                    label=f"{_label_for_dim(key)} dimension",
                    detail="substrate",
                )
            )
    for key in EXPECTED_SELF_CONCEPT:
        if key not in persona.self_concept:
            out.append(
                InventoryEntry(
                    slot=f"dimension:{key}",
                    category=EntryCategory.MISSING,
                    label=f"{_label_for_dim(key)} dimension",
                    detail="self_concept",
                )
            )
    return out


def _partial_dimension_entries(persona: Persona) -> list[InventoryEntry]:
    out: list[InventoryEntry] = []
    for parent, dims in (
        ("substrate", persona.substrate),
        ("self_concept", persona.self_concept),
    ):
        for key, dim in dims.items():
            problems: list[str] = []
            if not dim.voice_anchor:
                problems.append("no voice_anchor")
            # `triggers` only matters for triggered dimensions (it's moot on always_on).
            if dim.presence == Presence.TRIGGERED and not dim.triggers:
                problems.append("no triggers")
            for problem in problems:
                out.append(
                    InventoryEntry(
                        slot=f"dimension:{key}",
                        category=EntryCategory.PARTIAL,
                        label=f"{_label_for_dim(key)} — {problem}",
                        detail=f"{parent}.{key}: {problem}",
                    )
                )
    return out


def _trauma_image_key(name: str) -> str:
    return f"{name}.webp"


def _trauma_entries(
    persona: Persona, media_exists: MediaExists
) -> tuple[list[InventoryEntry], list[InventoryEntry]]:
    """Return (missing, partial) trauma-related entries.

    The "trauma image" partial flag only fires for individual traumas when at
    least one *other* persona trauma already has a rendered image in store-svc —
    the heuristic is that the author opted in to rendering trauma images and is
    just missing this one. Authors who haven't rendered any aren't nagged.
    """
    missing: list[InventoryEntry] = []
    partial: list[InventoryEntry] = []

    any_image = any(media_exists(_trauma_image_key(t.name)) for t in persona.traumas)
    if any_image:
        # Author opted in to trauma images — flag any persona trauma without one.
        for t in persona.traumas:
            if not media_exists(_trauma_image_key(t.name)):
                partial.append(
                    InventoryEntry(
                        slot=f"trauma_image:{t.name}",
                        category=EntryCategory.PARTIAL,
                        label=f"Trauma image — {t.name}",
                        detail=f"trauma '{t.name}' has no rendered image",
                    )
                )

    return missing, partial


def _no_seeded_narratives_entry(
    persona: Persona, qdrant_seeded_count: int | None
) -> InventoryEntry | None:
    """Flag when both the file and Qdrant have zero seeded narratives.

    Suppressed when `qdrant_seeded_count is None` — Qdrant unreachable, can't
    be sure the store is actually empty.
    """
    if qdrant_seeded_count is None:
        return None
    if persona.traumas or qdrant_seeded_count > 0:
        return None
    return InventoryEntry(
        slot="no_seeded_narratives",
        category=EntryCategory.MISSING,
        label="No seeded narratives yet",
        detail="add formative events to seed the episodic store",
    )


def _is_missing_artifact(path_in_file: object, media_exists: MediaExists) -> bool:
    """Decide whether an artifact slot is missing.

    - If `path_in_file` is None → missing.
    - Otherwise resolve the media key as the bare filename (`Path(ref).name`) so
      both new bare-name refs and legacy full-path refs map to the same store key,
      and check it via `media_exists`.
    """
    if path_in_file is None:
        return True
    return not media_exists(Path(str(path_in_file)).name)
