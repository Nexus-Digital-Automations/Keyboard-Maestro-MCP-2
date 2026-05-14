"""Attach a system-event trigger to a macro via .kmmacros re-import.

@stable km_add_system_trigger appends one system trigger (Login,
EngineLaunch, WakeTrigger) to a macro. KM 11 AppleScript rejects
``make new trigger with properties {xml:...}`` for any
MacroTriggerType other than HotKey, so the only path is the
export-edit-reimport pipeline owned by ``src.integration.km_macro_rebuild``.

This tool is now a thin wrapper over that pipeline; ``km_set_macro_triggers``
covers wholesale replacement. The UUID-rotation trade-off remains:
the macro's UID changes on every call, and the response surfaces
``old_macro_id`` → ``new_macro_id`` so callers can fix cross-macro
references.

Failure modes:
- VALIDATION_ERROR: unknown trigger_kind.
- NOT_FOUND: source macro doesn't exist.
- EXPORT_FAILED / IMPORT_FAILED: KM rejected the fetch / rebuild.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from fastmcp import Context  # noqa: TC002 — pydantic Field reads annotation at runtime
from pydantic import Field

from src.core.logging import get_logger
from src.integration.km_macro_rebuild import (
    fetch_macro_snapshot,
    rebuild_macro_via_reimport,
)
from src.server.initialization import get_km_client

logger = get_logger(__name__)

# Trigger plist values: confirmed against KM 11 Engine binary strings.
_TRIGGER_PLIST: dict[str, dict[str, str]] = {
    "login": {"MacroTriggerType": "Login"},
    "engine_launch": {"MacroTriggerType": "EngineLaunch"},
    "system_wake": {"MacroTriggerType": "WakeTrigger"},
}


async def km_add_system_trigger(
    macro_id: Annotated[
        str,
        Field(description="Target macro UUID or name.", min_length=1, max_length=255),
    ],
    trigger_kind: Annotated[
        Literal["login", "engine_launch", "system_wake"],
        Field(
            description=(
                "'login' fires when the user logs in; 'engine_launch' fires "
                "when the KM Engine starts; 'system_wake' fires when the Mac "
                "wakes from sleep."
            ),
        ),
    ],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Append a system-event trigger; UUID rotates (see new_macro_id)."""
    if trigger_kind not in _TRIGGER_PLIST:
        return _fail(
            "VALIDATION_ERROR",
            f"trigger_kind {trigger_kind!r} not supported. "
            f"Pass one of {sorted(_TRIGGER_PLIST)}.",
            "See km_add_system_trigger docstring.",
        )

    km = get_km_client()
    snap_result = await fetch_macro_snapshot(km, macro_id)
    if snap_result.is_left():
        err = snap_result.get_left()
        code = "NOT_FOUND" if err.code == "NOT_FOUND_ERROR" else "EXPORT_FAILED"
        return _fail(code, err.message, "Verify the UID/name via km_list_macros.")
    snapshot = snap_result.get_right()

    new_plist = dict(snapshot.plist)
    new_triggers = list(new_plist.get("Triggers", []))
    new_triggers.append(dict(_TRIGGER_PLIST[trigger_kind]))
    new_plist["Triggers"] = new_triggers

    rebuild_result = await rebuild_macro_via_reimport(km, snapshot, new_plist)
    if rebuild_result.is_left():
        err = rebuild_result.get_left()
        return _fail("IMPORT_FAILED", err.message, "Inspect KM Editor for import dialogs or naming conflicts.")
    rebuilt = rebuild_result.get_right()

    if ctx:
        await ctx.info(
            f"km_add_system_trigger: macro={macro_id} kind={trigger_kind} "
            f"old_uid={rebuilt.old_uid} new_uid={rebuilt.new_uid}",
        )
    return {
        "success": True,
        "data": {
            "old_macro_id": rebuilt.old_uid,
            "new_macro_id": rebuilt.new_uid,
            "group_name": rebuilt.group_name,
            "trigger_kind": trigger_kind,
            "uuid_changed": True,
        },
        "warning": (
            "The macro's UUID changed. Any ExecuteMacro action in other "
            "macros that referenced the old UID must be rewritten to use "
            "the new UID."
        ),
    }


def _fail(code: str, message: str, suggestion: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, "recovery_suggestion": suggestion},
    }
