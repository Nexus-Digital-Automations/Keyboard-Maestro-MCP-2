"""MCP tool: ``km_set_macro_triggers`` — replace a macro's full trigger list.

@stable Wholesale-replaces the ``Triggers`` array on a macro by going
through the export-edit-reimport pipeline in
``src.integration.km_macro_rebuild``. UID rotation is unavoidable
(Phase 0 probe verdict — KM 11 rejects ``set xml of macro``); the
response carries ``old_macro_id`` and ``new_macro_id`` so callers can
fix cross-macro references.

This tool is the canonical primitive for trigger-level mutation that
KM 11 AppleScript otherwise blocks. ``km_trigger_crud`` continues to
own HotKey-only operations that don't need a UID rotation.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context  # noqa: TC002 — pydantic Field reads annotation at runtime
from pydantic import Field

from src.core.logging import get_logger
from src.integration.km_macro_rebuild import (
    fetch_macro_snapshot,
    rebuild_macro_via_reimport,
)
from src.server.initialization import get_km_client

logger = get_logger(__name__)

# KM 11 trigger plist types this tool will pass through to the pipeline.
# Extending the set is cheap — KM does its own validation on import, so the
# guard's role is rejecting obvious typos early rather than enumerating KM's
# full surface.
_KNOWN_TRIGGER_TYPES: frozenset[str] = frozenset(
    {
        "HotKey",
        "Application",
        "Login",
        "EngineLaunch",
        "WakeTrigger",
    },
)


async def km_set_macro_triggers(
    macro_id: Annotated[
        str,
        Field(description="Target macro UUID or name.", min_length=1, max_length=255),
    ],
    triggers: Annotated[
        list[dict[str, Any]],
        Field(
            description=(
                "Full replacement trigger list. Each item is a plist dict with at "
                "minimum a 'MacroTriggerType' key set to one of: HotKey, "
                "Application, Login, EngineLaunch, WakeTrigger. Pass an empty list "
                "to strip all triggers (the macro then only fires via ExecuteMacro)."
            ),
        ),
    ],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Replace the macro's Triggers array. UID rotates; see new_macro_id."""
    invalid = _reject_unknown_types(triggers)
    if invalid is not None:
        return _fail(
            "VALIDATION_ERROR",
            invalid,
            f"Pass 'MacroTriggerType' as one of {sorted(_KNOWN_TRIGGER_TYPES)}.",
        )

    km = get_km_client()
    snap_result = await fetch_macro_snapshot(km, macro_id)
    if snap_result.is_left():
        err = snap_result.get_left()
        return _fail(
            "NOT_FOUND" if err.code == "NOT_FOUND_ERROR" else "EXPORT_FAILED",
            err.message,
            "Verify the macro UID/name via km_list_macros.",
        )
    snapshot = snap_result.get_right()

    new_plist = dict(snapshot.plist)
    new_plist["Triggers"] = [dict(t) for t in triggers]

    rebuild_result = await rebuild_macro_via_reimport(km, snapshot, new_plist)
    if rebuild_result.is_left():
        err = rebuild_result.get_left()
        return _fail("REBUILD_FAILED", err.message, "Inspect KM Editor for import dialogs or naming conflicts.")
    rebuilt = rebuild_result.get_right()

    if ctx:
        await ctx.info(
            f"km_set_macro_triggers: macro={macro_id} count={len(triggers)} "
            f"old_uid={rebuilt.old_uid} new_uid={rebuilt.new_uid}",
        )
    return {
        "success": True,
        "data": {
            "old_macro_id": rebuilt.old_uid,
            "new_macro_id": rebuilt.new_uid,
            "group_name": rebuilt.group_name,
            "trigger_count": len(triggers),
            "uuid_changed": True,
        },
        "warning": (
            "The macro's UUID changed. Any ExecuteMacro action in other "
            "macros that referenced the old UID must be rewritten to use "
            "the new UID."
        ),
    }


def _reject_unknown_types(triggers: list[dict[str, Any]]) -> str | None:
    for i, trig in enumerate(triggers):
        kind = trig.get("MacroTriggerType")
        if not isinstance(kind, str) or not kind:
            return f"triggers[{i}] missing 'MacroTriggerType' string"
        if kind not in _KNOWN_TRIGGER_TYPES:
            return f"triggers[{i}].MacroTriggerType {kind!r} is not in the known-types allowlist"
    return None


def _fail(code: str, message: str, suggestion: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, "recovery_suggestion": suggestion},
    }
