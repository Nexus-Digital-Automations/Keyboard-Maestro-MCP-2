"""Macro group tool — list, create, delete, rename, enable/disable groups.

Adapter over `KMClient` group primitives. Operation dispatch only — all
AppleScript / KM engine logic lives in `src/integration/km_client.py`.
"""

import asyncio
import logging
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from ...core.types import GroupId
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


async def _do_list() -> dict[str, Any]:
    result = await get_km_client().list_groups_async()
    if result.is_left():
        err = result.get_left()
        return _failure("LIST_FAILED", err.message, "Check accessibility permissions.")
    return {"success": True, "data": {"groups": result.get_right()}}


async def _do_create(new_name: str | None) -> dict[str, Any]:
    if not new_name or not new_name.strip():
        return _failure(
            "VALIDATION_ERROR",
            "new_name is required for operation='create'.",
            "Pass new_name with the group name to create.",
        )
    result = await get_km_client().create_group_async(new_name.strip())
    if result.is_left():
        err = result.get_left()
        return _failure("CREATE_FAILED", err.message, "Check name is unique.")
    return {"success": True, "data": result.get_right()}


async def _do_delete(group_id: str | None) -> dict[str, Any]:
    if not group_id or not group_id.strip():
        return _failure("VALIDATION_ERROR", "group_id is required.", "Pass group_id.")
    result = await get_km_client().delete_group_async(GroupId(group_id.strip()))
    if result.is_left():
        err = result.get_left()
        return _failure("DELETE_FAILED", err.message, "Verify the group exists.")
    return {"success": True, "data": {"group_id": group_id, "deleted": True}}


async def _do_rename(group_id: str | None, new_name: str | None) -> dict[str, Any]:
    if not group_id or not new_name or not new_name.strip():
        return _failure(
            "VALIDATION_ERROR",
            "group_id and new_name are both required.",
            "Pass both fields.",
        )
    result = await get_km_client().rename_group_async(
        GroupId(group_id.strip()), new_name.strip(),
    )
    if result.is_left():
        err = result.get_left()
        return _failure("RENAME_FAILED", err.message, "Verify the group exists.")
    return {"success": True, "data": {"group_id": group_id, "new_name": new_name}}


async def _do_set_enabled(group_id: str | None, enabled: bool | None) -> dict[str, Any]:
    if not group_id or not group_id.strip() or enabled is None:
        return _failure(
            "VALIDATION_ERROR",
            "group_id and enabled are both required.",
            "Pass both fields.",
        )
    result = await get_km_client().set_group_enabled_async(
        GroupId(group_id.strip()), enabled,
    )
    if result.is_left():
        err = result.get_left()
        return _failure("TOGGLE_FAILED", err.message, "Verify the group exists.")
    return {"success": True, "data": {"group_id": group_id, "enabled": enabled}}


async def km_macro_group_manager(
    operation: Annotated[
        Literal["list", "create", "delete", "rename", "set_enabled"],
        Field(description="Macro group operation."),
    ],
    group_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Group name or UUID. Required for delete/rename/set_enabled.",
            max_length=255,
        ),
    ] = None,
    new_name: Annotated[
        str | None,
        Field(
            default=None,
            description="New name for create/rename.",
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
    """List, create, delete, rename, or toggle Keyboard Maestro macro groups.

    Failure modes:
    - VALIDATION_ERROR: missing or empty required argument for the operation
    - KM_CONNECTION_FAILED: Keyboard Maestro Engine is not reachable
    - LIST_FAILED / CREATE_FAILED / DELETE_FAILED / RENAME_FAILED / TOGGLE_FAILED:
      AppleScript reported an error (group not found, name conflict, permission denied)
    """
    if ctx:
        await ctx.info(f"km_macro_group_manager op={operation} group={group_id!r}")

    connection_error = await _check_kme_alive()
    if connection_error is not None:
        return connection_error

    if operation == "list":
        return await _do_list()
    if operation == "create":
        return await _do_create(new_name)
    if operation == "delete":
        return await _do_delete(group_id)
    if operation == "rename":
        return await _do_rename(group_id, new_name)
    return await _do_set_enabled(group_id, enabled)
