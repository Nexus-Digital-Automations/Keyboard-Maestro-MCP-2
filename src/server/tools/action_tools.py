"""Action-type discovery for the MCP server.

Owner: Keyboard-Maestro-MCP-2 server team. Provides the single discovery tool
``km_list_action_types``, which advertises *what the server can actually
build* — not what KM supports in general. Two sources:

- Six built-in action types whose plist emitters in
  ``action_builder_tools._build_action_xml`` are verified against KM's
  "Copy as XML" output (``pause``, ``type_text``, ``paste``,
  ``set_variable``, ``run_applescript``, ``execute_macro``).
- Every installed third-party plug-in returned by
  ``plugin_action_tools._scan_installed_plugins`` — KM uniformly represents
  these as a single ``MacroActionType=ExecutePlugIn`` plist, so the same
  emitter handles all of them.

History: a previous catalog advertised 146 built-in types backed by a
synthetic ElementTree emitter. KM silently rejected that XML and replaced
the action with a "Log ‘Invalid XML From AppleScript’" placeholder,
corrupting the user's macro (F17 in docs/km_mcp_audit_report.md). The
catalog now reflects what we have verified emitters for; expanding it
means adding a new branch to ``_build_action_xml`` and a new entry here in
the same change.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

from ._action_search import search as _search_actions
from .plugin_action_tools import _scan_installed_plugins

logger = logging.getLogger(__name__)

# Built-in action types with verified plist emitters in
# action_builder_tools._build_action_xml. Keep in sync with that emitter:
# adding a branch there without an entry here makes it undiscoverable;
# adding an entry here without an emitter makes km_action_builder reject
# the type at append time. Categories are descriptive only — clients filter
# on them.
_BUILTIN_ACTION_TYPES: tuple[dict[str, Any], ...] = (
    {
        "identifier": "pause",
        "category": "control",
        "description": "Pause execution for N seconds.",
        "required_parameters": [],
        "optional_parameters": ["seconds"],
    },
    {
        "identifier": "type_text",
        "category": "text",
        "description": "Type a string by simulating keystrokes.",
        "required_parameters": ["text"],
        "optional_parameters": [],
    },
    {
        "identifier": "paste",
        "category": "text",
        "description": "Insert text by pasting from the clipboard.",
        "required_parameters": ["text"],
        "optional_parameters": [],
    },
    {
        "identifier": "set_variable",
        "category": "variable",
        "description": "Set a Keyboard Maestro variable to a text value.",
        "required_parameters": ["variable"],
        "optional_parameters": ["text"],
    },
    {
        "identifier": "run_applescript",
        "category": "system",
        "description": "Execute inline AppleScript source.",
        "required_parameters": ["source"],
        "optional_parameters": [],
    },
    {
        "identifier": "execute_macro",
        "category": "control",
        "description": "Trigger another macro by name or UUID.",
        "required_parameters": ["target_macro"],
        "optional_parameters": [],
    },
)


def _plugin_entry(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert one ``_scan_installed_plugins`` dict into the response schema."""
    return {
        "identifier": spec["identifier"],
        "category": "plug_in",
        "description": spec.get("help") or spec["title"],
        "keywords": spec.get("keywords", []),
        "required_parameters": [p["label"] for p in spec["parameters"]],
        "optional_parameters": [],
        "parameter_count": len(spec["parameters"]),
        "plugin_metadata": {
            "title": spec["title"],
            "result_targets": spec["result_targets"],
            "parameter_types": [p["type"] for p in spec["parameters"]],
            "bundle_path": spec["bundle_path"],
            "author": spec.get("author"),
            "help_url": spec.get("help_url"),
        },
    }


def _builtin_entry(spec: dict[str, Any]) -> dict[str, Any]:
    """Attach ``parameter_count`` so built-ins match the plug-in shape."""
    return {
        **spec,
        "parameter_count": len(spec["required_parameters"]) + len(spec["optional_parameters"]),
    }


def _matches(entry: dict[str, Any], search_lower: str) -> bool:
    return (
        search_lower in entry["identifier"].lower()
        or search_lower in entry["description"].lower()
    )


async def km_list_action_types(
    category: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Filter by action category. Built-ins use control / text / variable / "
                "system; installed third-party plug-ins use plug_in."
            ),
            examples=["control", "text", "variable", "system", "plug_in", None],
        ),
    ] = None,
    search: Annotated[
        str | None,
        Field(
            default=None,
            description="Substring match against identifier or description.",
            min_length=1,
            max_length=100,
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(
            default=50,
            description="Maximum number of entries to return.",
            ge=1,
            le=200,
        ),
    ] = 50,
    ctx: Context = None,
) -> dict[str, Any]:
    """List action types this server can append to a macro via km_action_builder.

    Returns the six verified built-in types plus every installed Keyboard
    Maestro plug-in (``category="plug_in"``). Plug-in entries carry an extra
    ``plugin_metadata`` block with the bundle's title, result targets, parameter
    types, and on-disk path.

    Failure modes: the only failure path is an unreadable plug-in folder, which
    is logged and treated as zero plug-ins (built-ins are still returned).
    """
    correlation_id = str(uuid.uuid4())
    if ctx:
        await ctx.info(
            f"Listing action types [category={category}, search={search}] [ID: {correlation_id}]"
        )

    try:
        plugins = await asyncio.to_thread(_scan_installed_plugins)
    except OSError as exc:
        logger.warning("Plug-in scan failed; returning built-ins only: %s", exc)
        plugins = []

    entries: list[dict[str, Any]] = [_builtin_entry(b) for b in _BUILTIN_ACTION_TYPES]
    entries.extend(_plugin_entry(p) for p in plugins)

    if category:
        entries = [e for e in entries if e["category"] == category.lower()]
    if search:
        search_lower = search.lower()
        entries = [e for e in entries if _matches(e, search_lower)]

    total_found = len(entries)
    entries = entries[:limit]

    category_counts: dict[str, int] = {}
    for builtin in _BUILTIN_ACTION_TYPES:
        category_counts[builtin["category"]] = category_counts.get(builtin["category"], 0) + 1
    if plugins:
        category_counts["plug_in"] = len(plugins)

    return {
        "success": True,
        "data": {
            "actions": entries,
            "summary": {
                "total_available": len(_BUILTIN_ACTION_TYPES) + len(plugins),
                "total_found": total_found,
                "returned": len(entries),
                "filtered_by_category": category,
                "filtered_by_search": search,
                "limit_applied": limit,
            },
            "categories": category_counts,
        },
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "correlation_id": correlation_id,
            "registry_version": "2.0.0",
        },
    }


async def km_search_actions(
    query: Annotated[
        str,
        Field(
            default="",
            description=(
                "Natural-language description of the task you want to perform "
                "(e.g. 'click a button on screen', 'wait for image', 'speak text'). "
                "Empty string returns the full catalog ordered by category then identifier."
            ),
            max_length=200,
        ),
    ] = "",
    category: Annotated[
        str | None,
        Field(
            default=None,
            description="Restrict to one category: control / text / variable / system / mouse / plug_in.",
        ),
    ] = None,
    builder_supported: Annotated[
        bool | None,
        Field(
            default=None,
            description=(
                "True = only actions km_action_builder can emit XML for. "
                "False = only actions present for discovery but not yet buildable."
            ),
        ),
    ] = None,
    result_target: Annotated[
        str | None,
        Field(
            default=None,
            description="Filter to actions that can write their result to a given target (e.g. 'Variable', 'Clipboard').",
        ),
    ] = None,
    min_score: Annotated[
        float,
        Field(
            default=0.0,
            description="Drop matches below this normalized [0, 1] score. Default 0.0 disables threshold.",
            ge=0.0,
            le=1.0,
        ),
    ] = 0.0,
    limit: Annotated[
        int,
        Field(default=10, description="Maximum results to return.", ge=1, le=100),
    ] = 10,
    ctx: Context = None,
) -> dict[str, Any]:
    """Rank built-in and plug-in action types by relevance to a task query.

    Fuzzy multi-field search across identifier, title, description, keywords,
    task_hints, parameter labels, category, and result_targets. Empty `query`
    returns the filtered catalog as a listing. Each result carries a
    normalized score, the fields that matched, and the canonical
    MacroActionType (for built-ins) — the latter is what
    `km_action_builder(action_type=...)` consumes for snake_case identifiers.

    Failure modes: any plug-in folder read failure is logged and the search
    falls back to the built-in catalog (logged once, never raised).
    """
    correlation_id = str(uuid.uuid4())
    if ctx:
        await ctx.info(
            f"km_search_actions q={query!r} cat={category} bs={builder_supported} "
            f"rt={result_target} min={min_score} limit={limit} [ID: {correlation_id}]"
        )
    results = await asyncio.to_thread(
        _search_actions,
        query,
        category=category,
        builder_supported=builder_supported,
        result_target=result_target,
        min_score=min_score,
        limit=limit,
    )
    return {
        "success": True,
        "data": {
            "actions": results,
            "summary": {
                "query": query,
                "returned": len(results),
                "filters": {
                    "category": category,
                    "builder_supported": builder_supported,
                    "result_target": result_target,
                    "min_score": min_score,
                },
                "limit": limit,
                "semantic": False,
            },
        },
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "correlation_id": correlation_id,
            "ranker": "rapidfuzz-WRatio-field-weighted",
        },
    }
