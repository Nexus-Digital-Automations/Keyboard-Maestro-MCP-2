"""System-event trigger tool — login, engine-launch, system-wake.

Covers the trigger types that fire on macOS / Keyboard Maestro lifecycle
events (no hotkey, no time schedule, no file watch). KM doesn't expose a
literal "login trigger" — instead, an `engine launch trigger` fires when
the KM Engine starts, which on a default install is at login. The tool
accepts a friendly ``trigger_kind`` (``login``, ``engine_launch``,
``system_wake``) and maps to KM's AppleScript trigger class.

Failure modes:
- VALIDATION_ERROR: missing/invalid macro_id or trigger_kind
- KM_CONNECTION_FAILED: KM Engine unreachable
- ATTACH_FAILED: AppleScript reported an error (macro not found, etc.)
"""

import asyncio
import logging
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from ..initialization import get_km_client

logger = logging.getLogger(__name__)

# friendly name → KM AppleScript trigger class name
TRIGGER_CLASS_BY_KIND: dict[str, str] = {
    "login": "engine launch trigger",
    "engine_launch": "engine launch trigger",
    "system_wake": "system wake trigger",
}


def _failure(code: str, message: str, suggestion: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, "recovery_suggestion": suggestion},
    }


async def _check_kme_alive() -> dict[str, Any] | None:
    km = get_km_client()
    connection = await asyncio.get_event_loop().run_in_executor(
        None, km.check_connection,
    )
    if connection.is_left() or not connection.get_right():
        return _failure(
            "KM_CONNECTION_FAILED",
            "Cannot connect to Keyboard Maestro Engine.",
            "Start Keyboard Maestro and ensure the Engine is running.",
        )
    return None


async def km_add_system_trigger(
    macro_id: Annotated[
        str,
        Field(description="Target macro UUID or name.", min_length=1, max_length=255),
    ],
    trigger_kind: Annotated[
        Literal["login", "engine_launch", "system_wake"],
        Field(
            description=(
                "'login' and 'engine_launch' both create an engine-launch trigger "
                "(KM Engine starts at login on default installs). 'system_wake' "
                "fires when the Mac wakes from sleep."
            ),
        ),
    ],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Attach a system-event trigger (login, engine-launch, system-wake) to a macro.

    Returns ``{"success": True, "data": {...}}`` on success or
    ``{"success": False, "error": {...}}`` with a structured error.
    """
    logger.warning(
        "km_add_system_trigger duplicates the trigger surface; "
        "calls still work but this name will fold into "
        "km_trigger_lifecycle(kind='system', operation='add') in a future release.",
    )
    if ctx:
        await ctx.info(
            f"km_add_system_trigger macro={macro_id!r} kind={trigger_kind}",
        )

    connection_error = await _check_kme_alive()
    if connection_error is not None:
        return connection_error

    km = get_km_client()
    escaped_id = km._escape_applescript_string(macro_id.strip())  # noqa: SLF001
    trigger_class = TRIGGER_CLASS_BY_KIND[trigger_kind]
    script = f'''
    tell application "Keyboard Maestro"
        try
            set targetMacro to first macro whose name is "{escaped_id}"
            make new {trigger_class} at end of triggers of targetMacro
            return "attached"
        on error errMsg
            return "ERROR: " & errMsg
        end try
    end tell
    '''
    result = await km.execute_applescript_async(script)
    if result.is_left():
        return _failure("ATTACH_FAILED", result.get_left().message, "Check macro exists.")
    output = result.get_right().strip()
    if output.startswith("ERROR:"):
        return _failure(
            "ATTACH_FAILED", output[6:].strip(),
            "Verify the macro exists and trigger type is supported by your KM version.",
        )
    return {
        "success": True,
        "data": {
            "macro_id": macro_id,
            "trigger_kind": trigger_kind,
            "trigger_class": trigger_class,
            "attached": True,
        },
    }
