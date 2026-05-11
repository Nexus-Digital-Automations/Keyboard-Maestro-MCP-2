"""Keyboard simulation tool via System Events AppleScript.

v1 is keyboard-only: ``type_text``, ``press_key``, ``press_key_code``.
Mouse simulation (click, scroll, move) deferred — System Events
``click at {x, y}`` only works on UI elements, not arbitrary coordinates,
and reliable mouse control needs a helper like ``cliclick``. Users
needing mouse automation should compose a KM "Move and Click" action
via ``km_action_builder`` instead.

Requires macOS Accessibility permission for the host process
(typically Claude Code or whatever runs the MCP server). System
Settings → Privacy & Security → Accessibility.
"""

import logging
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from ..initialization import get_km_client

logger = logging.getLogger(__name__)

ALLOWED_MODIFIERS = {"command", "option", "control", "shift"}


def _failure(code: str, message: str, suggestion: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, "recovery_suggestion": suggestion},
    }


def _modifiers_clause(modifiers: list[str] | None) -> str:
    if not modifiers:
        return ""
    safe = [m for m in modifiers if m in ALLOWED_MODIFIERS]
    if not safe:
        return ""
    parts = ", ".join(f"{m} down" for m in safe)
    return f" using {{{parts}}}"


async def _run(script: str, operation: str) -> dict[str, Any]:
    km = get_km_client()
    result = await km.execute_applescript_async(script)
    if result.is_left():
        err = result.get_left()
        msg = err.message
        # Accessibility-permission errors surface as osascript -1719 / -25211.
        if "-1719" in msg or "-25211" in msg or "not authorized" in msg.lower():
            return _failure(
                "ACCESSIBILITY_PERMISSION_REQUIRED",
                "macOS Accessibility permission required for keyboard simulation.",
                "Open System Settings → Privacy & Security → Accessibility and enable "
                "the host process running this MCP server.",
            )
        return _failure(f"{operation.upper()}_FAILED", msg, "Check osascript output.")
    return {"success": True, "data": {"operation": operation}}


async def _do_type_text(text: str | None) -> dict[str, Any]:
    if text is None:
        return _failure(
            "VALIDATION_ERROR", "text is required for operation='type_text'.",
            "Pass text with the string to type.",
        )
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    script = f'tell application "System Events" to keystroke "{escaped}"'
    return await _run(script, "type_text")


async def _do_press_key(key: str | None, modifiers: list[str] | None) -> dict[str, Any]:
    if not key or len(key) != 1:
        return _failure(
            "VALIDATION_ERROR",
            "key must be a single character (e.g., 'f'). Use press_key_code for special keys.",
            "Pass key=<single char>.",
        )
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f'tell application "System Events" to keystroke "{escaped}"{_modifiers_clause(modifiers)}'
    )
    return await _run(script, "press_key")


async def _do_press_key_code(
    key_code: int | None,
    modifiers: list[str] | None,
) -> dict[str, Any]:
    if key_code is None or key_code < 0 or key_code > 127:
        return _failure(
            "VALIDATION_ERROR",
            "key_code must be an int in [0, 127] (e.g., 36=Return, 48=Tab, 122=F1).",
            "Pass key_code with the macOS virtual keycode.",
        )
    script = (
        f'tell application "System Events" to key code {key_code}{_modifiers_clause(modifiers)}'
    )
    return await _run(script, "press_key_code")


async def km_input_simulator(
    operation: Annotated[
        Literal["type_text", "press_key", "press_key_code"],
        Field(description="Keyboard input operation."),
    ],
    text: Annotated[
        str | None,
        Field(default=None, description="For type_text: string to type.", max_length=10000),
    ] = None,
    key: Annotated[
        str | None,
        Field(default=None, description="For press_key: single character.", max_length=1),
    ] = None,
    key_code: Annotated[
        int | None,
        Field(default=None, description="For press_key_code: macOS virtual keycode.", ge=0, le=127),
    ] = None,
    modifiers: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Modifier keys: any of 'command','option','control','shift'.",
        ),
    ] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Simulate keyboard input via macOS System Events.

    Failure modes:
    - VALIDATION_ERROR: missing or malformed argument for the operation
    - ACCESSIBILITY_PERMISSION_REQUIRED: host process lacks Accessibility
      permission; grant it in System Settings → Privacy & Security
    - TYPE_TEXT_FAILED / PRESS_KEY_FAILED / PRESS_KEY_CODE_FAILED:
      osascript reported an unexpected error
    """
    if ctx:
        await ctx.info(f"km_input_simulator op={operation}")

    if operation == "type_text":
        return await _do_type_text(text)
    if operation == "press_key":
        return await _do_press_key(key, modifiers)
    return await _do_press_key_code(key_code, modifiers)
