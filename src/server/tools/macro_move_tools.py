"""Macro movement tool — move a macro to another group.

Thin adapter over ``KMClient.move_macro_to_group_async``. Accepts both
macro name and macro UUID (the client AppleScript uses
``whose name is X or uid is X``). Group "Global Macro Group" is a normal
user-facing group in KM, NOT a system group, and is permitted.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from ...core.types import GroupId, MacroId
from ..initialization import get_km_client

if TYPE_CHECKING:
    from fastmcp import Context

logger = logging.getLogger(__name__)


async def km_move_macro_to_group(
    macro_identifier: Annotated[
        str,
        Field(
            description="Macro name or UUID to move",
            min_length=1,
            max_length=255,
            pattern=r"^[a-zA-Z0-9_\s\-\.\+\(\)\[\]\{\}\|\&\~\!\@\#\$\%\^\*\=\?\:\;\,\/\\]+$|^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        ),
    ],
    target_group: Annotated[
        str,
        Field(
            description="Target group name or UUID",
            min_length=1,
            max_length=255,
            pattern=r"^[a-zA-Z0-9_\s\-\.\+\(\)\[\]\{\}\|\&\~\!\@\#\$\%\^\*\=\?\:\;\,\/\\]+$",
        ),
    ],
    create_group_if_missing: Annotated[
        bool,
        Field(default=False, description="Create target group if it doesn't exist"),
    ] = False,
    preserve_group_settings: Annotated[
        bool,
        Field(default=True, description="Maintain group-specific activation settings"),
    ] = True,
    timeout_seconds: Annotated[
        int,
        Field(default=30, ge=5, le=120, description="Operation timeout in seconds"),
    ] = 30,
    ctx: Context = None,
) -> dict[str, Any]:
    """Move a macro from its current group to ``target_group``.

    Accepts either the macro's name or UUID, and likewise for the target
    group. Delegates all AppleScript/KM logic to ``KMClient``.

    Failure modes:
    - ``MACRO_NOT_FOUND`` / ``NOT_FOUND_ERROR``: macro doesn't exist
    - ``GROUP_NOT_FOUND``: target group doesn't exist and ``create_group_if_missing`` is False
    - ``VALIDATION_ERROR``: macro is already in the target group
    - ``EXECUTION_ERROR``: AppleScript or KM rejected the move
    - ``KM_CONNECTION_FAILED``: the engine is not reachable
    """
    # preserve_group_settings is reserved for future per-group activation logic;
    # the underlying client move is already settings-preserving.
    del preserve_group_settings
    correlation_id = str(uuid.uuid4())
    start_time = datetime.now()

    if ctx:
        await ctx.info(
            f"Moving macro '{macro_identifier}' to '{target_group}'",
        )

    try:
        result = await get_km_client().move_macro_to_group_async(
            MacroId(macro_identifier.strip()),
            GroupId(target_group.strip()),
            create_missing=create_group_if_missing,
        )
    except Exception:
        logger.exception(
            "Unexpected error moving macro '%s' to '%s' [correlation_id=%s]",
            macro_identifier,
            target_group,
            correlation_id,
        )
        raise

    execution_time = (datetime.now() - start_time).total_seconds()

    if result.is_left():
        err = result.get_left()
        logger.warning(
            "Macro move failed: '%s' -> '%s' [%s]: %s [correlation_id=%s]",
            macro_identifier,
            target_group,
            err.code,
            err.message,
            correlation_id,
        )
        return {
            "success": False,
            "error": {
                "code": err.code,
                "message": err.message,
                "details": err.message,
                "recovery_suggestion": "Verify the macro exists and the target group is accessible.",
            },
            "metadata": {
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "operation": "move_macro_to_group",
            },
        }

    move = result.get_right()
    if ctx:
        await ctx.info(f"Moved macro to '{target_group}'")
    return {
        "success": True,
        "data": {
            "macro_identifier": macro_identifier,
            "source_group": str(move.source_group),
            "target_group": str(move.target_group),
            "operation_time": execution_time,
            "conflicts_resolved": list(move.conflicts_resolved),
            "group_created": create_group_if_missing,
        },
        "metadata": {
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "operation": "move_macro_to_group",
        },
    }
