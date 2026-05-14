"""System-event trigger tool — login, engine-launch, system-wake.

@deprecated KM 11 AppleScript only honours ``make new trigger with
properties {xml:...}`` for ``HotKey``-type plists. Smoke-tested all three
``MacroTriggerType`` strings KM's Engine binary exposes (``EngineLaunch``,
``WakeTrigger``, ``Login``) plus their AppleScript class-name variants —
each returns ``Connection is invalid`` (errAEEventNotHandled, -609) on
construction. The previous class-name approach (``make new engine launch
trigger``) was a separate bug: those classes don't exist in KM's sdef,
which only declares the single ``trigger`` class.

This tool now short-circuits with ``UNSUPPORTED_OPERATION`` until KM
exposes a writable surface (a future Editor.sdef revision, or the
existing kmmacros-import path used by ``km_create_macro``).

Failure modes:
- UNSUPPORTED_OPERATION: every call, until upstream support lands
"""

from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from src.core.logging import get_logger

logger = get_logger(__name__)


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
    """Attach a system-event trigger to a macro.

    Returns ``{"success": False, "error": {"code": "UNSUPPORTED_OPERATION", ...}}``
    because KM 11 AppleScript rejects non-HotKey trigger creation.
    """
    logger.warning(
        "km_add_system_trigger called for macro=%s kind=%s; KM AppleScript "
        "does not currently support creation of this trigger type.",
        macro_id,
        trigger_kind,
    )
    if ctx:
        await ctx.info(
            f"km_add_system_trigger rejected: macro={macro_id!r} kind={trigger_kind}",
        )
    return {
        "success": False,
        "error": {
            "code": "UNSUPPORTED_OPERATION",
            "message": (
                "KM 11 AppleScript only accepts HotKey-type triggers via "
                "'make new trigger with properties {xml:...}'. EngineLaunch, "
                "WakeTrigger, and Login plists are rejected with "
                "'Connection is invalid' on construction."
            ),
            "recovery_suggestion": (
                "Create the macro via km_create_macro using a .kmmacros import "
                "that already carries the desired system trigger, or attach the "
                "trigger by hand in the KM Editor."
            ),
        },
        "metadata": {
            "macro_id": macro_id,
            "trigger_kind": trigger_kind,
        },
    }
