"""Append an IfThenElse action to a macro from a flat condition spec.

@stable km_add_condition wraps the shared IfThenElse XML emitter
(src/integration/km_if_then_else_xml.py) so callers can attach a single
conditional branch without hand-building plist. For multi-condition
matching, nested control flow, or arbitrary inner actions, prefer
km_control_flow (if_then_else mode) or km_action_builder paste_xml.

Decision record: the previous implementation emitted a top-level
``<condition>`` element on the macro that KM never accepted ("variable
condition is not defined" error). KM has no per-macro condition; the
only writable surface is the IfThenElse action's ConditionList. This
rewrite generates that action XML instead.

Failure modes:
- VALIDATION_ERROR: missing macro_id / operand, or unmappable
  condition_type/operator.
- APPEND_FAILED: KM rejected the action (missing macro, engine down).
"""

from typing import Any

try:
    from fastmcp import Server  # type: ignore[attr-defined]
except ImportError:
    Server = None  # type: ignore[assignment,misc]

from src.core.logging import get_logger
from src.core.types import MacroId
from src.integration.km_if_then_else_xml import (
    UnsupportedConditionType,
    UnsupportedOperator,
    build_condition_dict,
    build_execute_macro_action,
    build_if_then_else_xml,
)
from src.server.initialization import get_km_client

logger = get_logger(__name__)


def _failure(code: str, message: str, suggestion: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, "recovery_suggestion": suggestion},
    }


async def km_add_condition(
    macro_identifier: str,
    condition_type: str,
    operator: str,
    operand: str,
    case_sensitive: bool = True,
    negate: bool = False,
    action_on_true: str | None = None,
    action_on_false: str | None = None,
    timeout_seconds: int = 10,
    ctx: Any = None,
) -> dict[str, Any]:
    """Append a single-condition IfThenElse action to ``macro_identifier``.

    Args:
        macro_identifier: Target macro name or UUID.
        condition_type: One of ``variable``, ``text``, ``application``,
            ``calculation``. The legacy values ``system`` and ``logic`` are
            rejected (no direct KM mapping).
        operator: ``equals``, ``contains``, ``greater``, ``less``,
            ``regex``, ``exists`` etc. — mapped per condition_type.
        operand: For ``variable`` / ``text``: ``"VarName=compare"`` (or
            just ``"VarName"`` for exists/empty checks). For
            ``application``: bundle id or app name. For ``calculation``:
            the KM expression.
        case_sensitive: Honored for text conditions only.
        negate: Invert the condition.
        action_on_true: Optional macro name/UID; emitted as one
            ExecuteMacro inner action inside ThenActions.
        action_on_false: Same, for ElseActions.
        timeout_seconds: Carried into ``TimeOutAbortsMacro`` (kept true
            by default; the seconds value is informational here because
            KM stores it on the surrounding action).
        ctx: MCP context (unused).

    Returns:
        ``{"success": True, "data": {"macro_id", "macro_action_type"}}``
        on success; failure envelope otherwise.
    """
    del ctx, timeout_seconds
    if not macro_identifier or not macro_identifier.strip():
        return _failure(
            "VALIDATION_ERROR",
            "macro_identifier is required.",
            "Pass the macro UUID or unique name.",
        )

    try:
        condition_xml = build_condition_dict(
            condition_type,
            operator,
            operand,
            case_sensitive=case_sensitive,
            negate=negate,
        )
    except (UnsupportedConditionType, UnsupportedOperator) as exc:
        logger.warning(
            "km_add_condition rejected unsupported spec: type=%s op=%s err=%s",
            condition_type,
            operator,
            exc,
        )
        return _failure(
            "UNSUPPORTED_OPERATION" if isinstance(exc, UnsupportedConditionType)
            else "VALIDATION_ERROR",
            str(exc),
            "See km_add_condition docstring for supported condition_type / operator combos.",
        )

    then_xml = build_execute_macro_action(action_on_true) if action_on_true else ""
    else_xml = build_execute_macro_action(action_on_false) if action_on_false else ""
    action_xml = build_if_then_else_xml(condition_xml, then_xml, else_xml)

    result = await get_km_client().append_macro_action_async(
        MacroId(macro_identifier.strip()),
        action_xml,
    )
    if result.is_left():
        err = result.get_left()
        logger.warning("km_add_condition append failed: macro=%s err=%s", macro_identifier, err.message)
        return _failure(
            "APPEND_FAILED",
            err.message,
            "Verify the macro exists and KM engine is running.",
        )
    return {
        "success": True,
        "data": {
            "macro_id": macro_identifier,
            "macro_action_type": "IfThenElse",
            "condition_type": condition_type,
            "operator": operator,
            "then_inner_actions": 1 if action_on_true else 0,
            "else_inner_actions": 1 if action_on_false else 0,
        },
    }


def register_condition_tools(server: Server) -> None:
    """Register condition-related tools with the MCP server."""
    server.add_tool(km_add_condition)
