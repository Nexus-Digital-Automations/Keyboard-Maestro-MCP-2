"""Trigger management tool — attach, list, remove, clear, enable/disable.

Adapter over `KMClient` trigger primitives. Operation dispatch only;
AppleScript and KM-engine logic live in `src/integration/km_client.py`.

v1 trigger types: ``hotkey`` (config: ``key``, ``modifiers``) and
``application`` (config: ``application``, ``event``). Other KM trigger
types (time, file, system event) are deferred to a later round.
"""

import asyncio
import logging
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from ...core.types import MacroId
from ..initialization import get_km_client

logger = logging.getLogger(__name__)


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


async def _do_list(macro_id: str | None) -> dict[str, Any]:
    if not macro_id or not macro_id.strip():
        return _failure("VALIDATION_ERROR", "macro_id is required.", "Pass macro_id.")
    result = await get_km_client().list_macro_triggers_async(MacroId(macro_id.strip()))
    if result.is_left():
        return _failure("LIST_FAILED", result.get_left().message, "Verify the macro exists.")
    return {"success": True, "data": {"macro_id": macro_id, "triggers": result.get_right()}}


async def _do_add(
    macro_id: str | None,
    trigger_type: str | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not macro_id or not macro_id.strip() or not trigger_type:
        return _failure(
            "VALIDATION_ERROR",
            "macro_id and trigger_type are required for operation='add'.",
            "Pass both, plus a config dict appropriate for the trigger_type.",
        )
    result = await get_km_client().attach_trigger_async(
        MacroId(macro_id.strip()), trigger_type, config or {},
    )
    if result.is_left():
        return _failure("ADD_FAILED", result.get_left().message, "Check config shape.")
    return {
        "success": True,
        "data": {"macro_id": macro_id, "trigger_type": trigger_type, "attached": True},
    }


async def _do_remove(macro_id: str | None, trigger_index: int | None) -> dict[str, Any]:
    if not macro_id or not macro_id.strip() or trigger_index is None:
        return _failure(
            "VALIDATION_ERROR",
            "macro_id and trigger_index are required.",
            "Use operation='list' first to find the 1-indexed position.",
        )
    result = await get_km_client().remove_macro_trigger_async(
        MacroId(macro_id.strip()), trigger_index,
    )
    if result.is_left():
        return _failure("REMOVE_FAILED", result.get_left().message, "Verify the index is valid.")
    return {"success": True, "data": {"macro_id": macro_id, "trigger_index": trigger_index}}


async def _do_clear(macro_id: str | None) -> dict[str, Any]:
    if not macro_id or not macro_id.strip():
        return _failure("VALIDATION_ERROR", "macro_id is required.", "Pass macro_id.")
    result = await get_km_client().clear_macro_triggers_async(MacroId(macro_id.strip()))
    if result.is_left():
        return _failure("CLEAR_FAILED", result.get_left().message, "Verify the macro exists.")
    return {"success": True, "data": {"macro_id": macro_id, "cleared": True}}


async def _do_set_enabled(
    macro_id: str | None,
    trigger_index: int | None,
    enabled: bool | None,
) -> dict[str, Any]:
    if not macro_id or not macro_id.strip() or trigger_index is None or enabled is None:
        return _failure(
            "VALIDATION_ERROR",
            "macro_id, trigger_index, and enabled are all required.",
            "Pass all three.",
        )
    result = await get_km_client().set_trigger_enabled_async(
        MacroId(macro_id.strip()), trigger_index, enabled,
    )
    if result.is_left():
        return _failure("TOGGLE_FAILED", result.get_left().message, "Verify the index is valid.")
    return {
        "success": True,
        "data": {"macro_id": macro_id, "trigger_index": trigger_index, "enabled": enabled},
    }


async def km_trigger_manager(
    operation: Annotated[
        Literal["list", "add", "remove", "clear", "set_enabled"],
        Field(description="Trigger operation."),
    ],
    macro_id: Annotated[
        str | None,
        Field(default=None, description="Target macro name or UUID.", max_length=255),
    ] = None,
    trigger_type: Annotated[
        str | None,
        Field(
            default=None,
            description="For 'add': 'hotkey' or 'application'.",
            max_length=64,
        ),
    ] = None,
    config: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description=(
                "For 'add'. hotkey: {'key': 'f', 'modifiers': ['command','option']}. "
                "application: {'application': 'Safari', 'event': 'launches'}."
            ),
        ),
    ] = None,
    trigger_index: Annotated[
        int | None,
        Field(default=None, description="1-indexed trigger position.", ge=1),
    ] = None,
    enabled: Annotated[
        bool | None,
        Field(default=None, description="Target enabled state for set_enabled."),
    ] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Manage triggers on Keyboard Maestro macros.

    Failure modes:
    - VALIDATION_ERROR: missing or invalid argument for the operation
    - KM_CONNECTION_FAILED: Keyboard Maestro Engine is not reachable
    - ADD_FAILED / LIST_FAILED / REMOVE_FAILED / CLEAR_FAILED / TOGGLE_FAILED:
      AppleScript reported an error (macro not found, invalid config,
      unsupported trigger type, index out of range)
    """
    if ctx:
        await ctx.info(f"km_trigger_manager op={operation} macro={macro_id!r}")

    connection_error = await _check_kme_alive()
    if connection_error is not None:
        return connection_error

    if operation == "list":
        return await _do_list(macro_id)
    if operation == "add":
        return await _do_add(macro_id, trigger_type, config)
    if operation == "remove":
        return await _do_remove(macro_id, trigger_index)
    if operation == "clear":
        return await _do_clear(macro_id)
    return await _do_set_enabled(macro_id, trigger_index, enabled)
