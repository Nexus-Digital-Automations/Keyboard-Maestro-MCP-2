"""Macro editor tool — create, delete, rename, duplicate, enable/disable macros.

Adapter over `KMClient` CRUD primitives. Operation dispatch only — all
AppleScript / KM engine logic lives in `src/integration/km_client.py`.
"""

import asyncio
import logging
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from ...core.types import MacroId
from ...integration.kmmacros_import import create_empty_macro
from ..initialization import get_km_client

logger = logging.getLogger(__name__)


def _failure(code: str, message: str, suggestion: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, "recovery_suggestion": suggestion},
    }


async def _check_kme_alive() -> dict[str, Any] | None:
    """Return a failure dict if the KM engine is unreachable, else None."""
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


async def _do_create(group_id: str | None, new_name: str | None) -> dict[str, Any]:
    if not new_name or not new_name.strip():
        return _failure(
            "VALIDATION_ERROR",
            "new_name is required for operation='create'.",
            "Pass new_name with the macro name to create.",
        )
    if not group_id or not group_id.strip():
        return _failure(
            "VALIDATION_ERROR",
            "group_id is required for operation='create'.",
            "Pass group_id with the parent group's name or UUID.",
        )
    result = await create_empty_macro(
        get_km_client(), group_id.strip(), new_name.strip(),
    )
    if result.is_left():
        err = result.get_left()
        message = err.message.lower()
        if "not found" in message:
            return _failure(
                "GROUP_NOT_FOUND",
                err.message,
                "Check the group name/UUID; list with km_macro_group_manager.",
            )
        return _failure(
            "IMPORT_FAILED",
            err.message,
            "KM may have prompted for permission on first import; check the Editor and retry.",
        )
    return {"success": True, "data": result.get_right()}


async def _do_delete(macro_id: str | None) -> dict[str, Any]:
    if not macro_id or not macro_id.strip():
        return _failure("VALIDATION_ERROR", "macro_id is required.", "Pass macro_id.")
    result = await get_km_client().delete_macro_async(MacroId(macro_id.strip()))
    if result.is_left():
        err = result.get_left()
        return _failure("DELETE_FAILED", err.message, "Verify the macro exists.")
    return {"success": True, "data": {"macro_id": macro_id, "deleted": True}}


async def _do_rename(macro_id: str | None, new_name: str | None) -> dict[str, Any]:
    if not macro_id or not new_name or not new_name.strip():
        return _failure(
            "VALIDATION_ERROR",
            "macro_id and new_name are both required.",
            "Pass both fields.",
        )
    result = await get_km_client().rename_macro_async(
        MacroId(macro_id.strip()), new_name.strip(),
    )
    if result.is_left():
        err = result.get_left()
        return _failure("RENAME_FAILED", err.message, "Check macro exists and name is available.")
    return {"success": True, "data": {"macro_id": macro_id, "new_name": new_name}}


async def _do_duplicate(macro_id: str | None, new_name: str | None) -> dict[str, Any]:
    if not macro_id or not macro_id.strip():
        return _failure("VALIDATION_ERROR", "macro_id is required.", "Pass macro_id.")
    result = await get_km_client().duplicate_macro_async(
        MacroId(macro_id.strip()), new_name.strip() if new_name else None,
    )
    if result.is_left():
        err = result.get_left()
        return _failure("DUPLICATE_FAILED", err.message, "Verify the source macro exists.")
    return {"success": True, "data": result.get_right()}


async def _do_set_enabled(macro_id: str | None, enabled: bool | None) -> dict[str, Any]:
    if not macro_id or not macro_id.strip() or enabled is None:
        return _failure(
            "VALIDATION_ERROR",
            "macro_id and enabled are both required.",
            "Pass both fields.",
        )
    result = await get_km_client().set_macro_enabled_async(
        MacroId(macro_id.strip()), enabled,
    )
    if result.is_left():
        err = result.get_left()
        return _failure("TOGGLE_FAILED", err.message, "Verify the macro exists.")
    return {"success": True, "data": {"macro_id": macro_id, "enabled": enabled}}


async def km_macro_editor(
    operation: Annotated[
        Literal["create", "delete", "rename", "duplicate", "set_enabled"],
        Field(description="Macro CRUD operation."),
    ],
    macro_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Macro name or UUID. Required for delete/rename/duplicate/set_enabled.",
            max_length=255,
        ),
    ] = None,
    group_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Parent group name or UUID. Required for create.",
            max_length=255,
        ),
    ] = None,
    new_name: Annotated[
        str | None,
        Field(
            default=None,
            description="New name for create/rename, optional name for duplicate.",
            max_length=255,
        ),
    ] = None,
    enabled: Annotated[
        bool | None,
        Field(
            default=None,
            description="Target enabled state for set_enabled.",
        ),
    ] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Create, delete, rename, duplicate, or toggle Keyboard Maestro macros.

    Failure modes:
    - VALIDATION_ERROR: missing or empty required argument for the operation
    - KM_CONNECTION_FAILED: Keyboard Maestro Engine is not reachable
    - CREATE_FAILED / DELETE_FAILED / RENAME_FAILED / DUPLICATE_FAILED / TOGGLE_FAILED:
      AppleScript reported an error (macro not found, name conflict, permission denied)
    """
    if ctx:
        await ctx.info(f"km_macro_editor op={operation} macro={macro_id!r}")

    connection_error = await _check_kme_alive()
    if connection_error is not None:
        return connection_error

    if operation == "create":
        return await _do_create(group_id, new_name)
    if operation == "delete":
        return await _do_delete(macro_id)
    if operation == "rename":
        return await _do_rename(macro_id, new_name)
    if operation == "duplicate":
        return await _do_duplicate(macro_id, new_name)
    return await _do_set_enabled(macro_id, enabled)
