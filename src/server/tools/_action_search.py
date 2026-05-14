"""Action catalog search: rank built-ins + plug-ins by relevance to a task query.

Owned by: server/tools — read-path support for km_search_actions. Builds a
unified index over the built-in catalog (km_builtin_actions.json) and every
installed plug-in (.kmactions plist), pre-extracts per-field haystack text,
and ranks via rapidfuzz field-weighted scoring. The semantic-similarity
layer (fastembed BGE-small-en-v1.5) is intentionally NOT loaded here in
Phase 5a; rapidfuzz alone covers the current 16-entry catalog cleanly,
and the embedding layer becomes worthwhile once the captured-template
scrape (Phase 3) inflates the catalog past ~100 entries.

@stable
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rapidfuzz import fuzz

from ._action_templates import load_catalog, load_templates
from .plugin_action_tools import _scan_installed_plugins

logger = logging.getLogger(__name__)

# Weights tuned so that an exact identifier match always outranks a description
# match alone, but keywords + task_hints together can outrank a near-miss
# identifier (so a query like "wait for image" beats "wait_for_button" against
# a curated "wait for image" task_hint).
_FIELD_WEIGHTS: dict[str, float] = {
    "identifier": 3.0,
    "title": 2.5,
    "keywords": 2.0,
    "task_hints": 2.0,
    "description": 1.5,
    "parameters": 1.0,
    "category": 0.5,
}
_FIELD_MATCH_THRESHOLD = 0.6  # fuzz.WRatio/100; below this we don't list the field as "matched"
_MAX_FIELD_TOTAL = sum(_FIELD_WEIGHTS.values())


@dataclass(frozen=True)
class IndexEntry:
    """Pre-computed haystacks for one catalog entry."""

    identifier: str
    macro_action_type: str
    title: str
    description: str
    category: str
    keywords: tuple[str, ...]
    task_hints: tuple[str, ...]
    parameters: tuple[str, ...]
    result_targets: tuple[str, ...]
    builder_supported: bool
    raw: dict[str, Any] = field(repr=False)


@lru_cache(maxsize=1)
def build_index() -> tuple[IndexEntry, ...]:
    """Build the unified search index from built-in catalog + installed plug-ins.

    Captured templates not yet represented by a curated catalog entry get
    auto-surfaced via _from_template so they're discoverable immediately
    after the scrape runs, even before keywords/task_hints are hand-tuned.
    """
    entries: list[IndexEntry] = []
    catalog_macro_types: set[str] = set()
    for spec in load_catalog():
        entry = _from_builtin(spec)
        entries.append(entry)
        if entry.macro_action_type:
            catalog_macro_types.add(entry.macro_action_type)
    for spec in _scan_installed_plugins():
        entries.append(_from_plugin(spec))
    for mat, template in load_templates().items():
        if mat in catalog_macro_types:
            continue
        entries.append(_from_template(mat, template))
    logger.info("action_search index built entries=%d", len(entries))
    return tuple(entries)


def search(
    query: str,
    *,
    category: str | None = None,
    builder_supported: bool | None = None,
    result_target: str | None = None,
    min_score: float = 0.0,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Rank catalog entries by relevance to `query`, return top `limit` matches.

    Empty `query` returns the filtered catalog sorted by category then
    identifier (the discovery-listing mode). `min_score` is in normalized
    [0, 1] units; default 0.0 means no threshold. Filters apply before
    scoring so they never cause the result to be re-padded by lower hits.
    """
    entries = _filtered(build_index(), category, builder_supported, result_target)
    if not query.strip():
        ordered = sorted(entries, key=lambda e: (e.category, e.identifier))
        return [_to_response(e, 0.0, ()) for e in ordered[:limit]]
    scored = []
    for entry in entries:
        total, matched = _score_entry(query, entry)
        normalized = total / _MAX_FIELD_TOTAL
        if normalized >= min_score:
            scored.append((normalized, matched, entry))
    scored.sort(key=lambda triple: -triple[0])
    return [_to_response(e, s, m) for s, m, e in scored[:limit]]


def _from_builtin(spec: dict[str, Any]) -> IndexEntry:
    return IndexEntry(
        identifier=spec["identifier"],
        macro_action_type=spec.get("MacroActionType", ""),
        title=spec.get("title", spec["identifier"]),
        description=spec.get("description", ""),
        category=spec.get("category", ""),
        keywords=tuple(spec.get("keywords", [])),
        task_hints=tuple(spec.get("task_hints", [])),
        parameters=tuple(p.get("label", "") for p in spec.get("parameters", [])),
        result_targets=tuple(spec.get("result_targets", [])),
        builder_supported=spec.get("builder_supported", False),
        raw=spec,
    )


def _pascal_to_snake(pascal: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", pascal)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _from_template(macro_action_type: str, template: dict[str, Any]) -> IndexEntry:
    category_path = template.get("category_path") or []
    category = category_path[0].lower().replace(" ", "_") if category_path else "captured"
    label = category_path[1] if len(category_path) > 1 else macro_action_type
    return IndexEntry(
        identifier=_pascal_to_snake(macro_action_type),
        macro_action_type=macro_action_type,
        title=label,
        description=label,
        category=category,
        keywords=(),
        task_hints=(),
        parameters=(),
        result_targets=(),
        builder_supported=True,
        raw={
            "identifier": _pascal_to_snake(macro_action_type),
            "MacroActionType": macro_action_type,
            "title": label,
            "category": category,
            "captured_only": True,
        },
    )


def _from_plugin(spec: dict[str, Any]) -> IndexEntry:
    return IndexEntry(
        identifier=spec["identifier"],
        macro_action_type="",
        title=spec.get("title", spec["identifier"]),
        description=spec.get("help") or spec.get("title", ""),
        category="plug_in",
        keywords=tuple(spec.get("keywords", [])),
        task_hints=(),
        parameters=tuple(p.get("label", "") for p in spec.get("parameters", [])),
        result_targets=tuple(spec.get("result_targets", [])),
        builder_supported=True,
        raw=spec,
    )


def _score_entry(query: str, entry: IndexEntry) -> tuple[float, tuple[str, ...]]:
    fields: dict[str, str] = {
        "identifier": entry.identifier,
        "title": entry.title,
        "description": entry.description,
        "category": entry.category,
        "keywords": " ".join(entry.keywords),
        "task_hints": " ".join(entry.task_hints),
        "parameters": " ".join(entry.parameters),
    }
    total = 0.0
    matched: list[str] = []
    for name, text in fields.items():
        ratio = fuzz.WRatio(query, text) / 100.0 if text else 0.0
        if ratio >= _FIELD_MATCH_THRESHOLD:
            matched.append(name)
        total += _FIELD_WEIGHTS[name] * ratio
    return total, tuple(matched)


def _filtered(
    entries: tuple[IndexEntry, ...],
    category: str | None,
    builder_supported: bool | None,
    result_target: str | None,
) -> list[IndexEntry]:
    out: list[IndexEntry] = []
    for entry in entries:
        if category is not None and entry.category != category:
            continue
        if builder_supported is not None and entry.builder_supported != builder_supported:
            continue
        if result_target is not None and result_target not in entry.result_targets:
            continue
        out.append(entry)
    return out


def _to_response(
    entry: IndexEntry,
    score: float,
    matched_fields: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "identifier": entry.identifier,
        "category": entry.category,
        "title": entry.title,
        "description": entry.description,
        "keywords": list(entry.keywords),
        "builder_supported": entry.builder_supported,
        "result_targets": list(entry.result_targets),
        "score": round(score, 4),
        "matched_fields": list(matched_fields),
        "macro_action_type": entry.macro_action_type or None,
    }
