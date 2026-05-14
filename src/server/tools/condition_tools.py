"""Condition tools for adding conditional logic to Keyboard Maestro macros.

This module implements the km_add_condition MCP tool that enables intelligent automation
through comprehensive conditional logic, supporting text, application, system, and variable
conditions with advanced security validation and functional programming patterns.
"""

from typing import Any

try:
    from fastmcp import Server
except ImportError:
    Server = None

from src.core.logging import get_logger

logger = get_logger(__name__)


async def km_add_condition(
    macro_identifier: str,  # Target macro (name or UUID)
    condition_type: str,  # text|application|system|variable|logic
    operator: str,  # contains|equals|greater|less|regex|exists
    operand: str,  # Comparison value with validation
    case_sensitive: bool = True,  # Text comparison sensitivity
    negate: bool = False,  # Invert condition result
    action_on_true: str | None = None,  # Action for true condition
    action_on_false: str | None = None,  # Action for false condition
    timeout_seconds: int = 10,  # Condition evaluation timeout
    ctx: Any = None,
) -> dict[str, Any]:
    """Add conditional logic to a Keyboard Maestro macro for intelligent automation.

    @deprecated KM AppleScript has no top-level `make new condition` on a macro;
    conditions only exist inside If Then Else / While / Pause Until action XML.
    The previous implementation generated a `<condition>` element and was always
    rejected by KM with "The variable condition is not defined". This tool now
    short-circuits with UNSUPPORTED_OPERATION mirroring km_control_flow.

    Args:
        macro_identifier: Target macro name or UUID
        condition_type: Type of condition (text, application, system, variable, logic)
        operator: Comparison operator (contains, equals, greater, less, regex, exists)
        operand: Value to compare against
        case_sensitive: Whether text comparisons are case sensitive
        negate: Whether to invert the condition result
        action_on_true: Optional action to execute when condition is true
        action_on_false: Optional action to execute when condition is false
        timeout_seconds: Maximum time to evaluate condition

    Returns:
        Dict containing condition ID, validation status, and integration details

    Raises:
        ValidationError: If condition parameters are invalid
        SecurityError: If condition contains security risks
        PermissionDeniedError: If insufficient permissions for condition type

    """
    logger.warning(
        "km_add_condition called for macro=%s type=%s op=%s; "
        "feature is not yet implemented (see deprecated docstring).",
        macro_identifier,
        condition_type,
        operator,
    )
    return {
        "success": False,
        "error": {
            "code": "UNSUPPORTED_OPERATION",
            "message": (
                "km_add_condition is not yet implemented. KM has no top-level "
                "'condition' element on a macro; conditions live inside If Then Else, "
                "While, and Pause Until actions."
            ),
            "recovery_suggestion": (
                "Build the surrounding If Then Else action XML by hand and append it "
                "with km_action_builder(operation='append', action_type='paste_xml')."
            ),
        },
        "metadata": {
            "macro_identifier": macro_identifier,
            "condition_type": condition_type,
            "operator": operator,
        },
    }


def register_condition_tools(server: Server) -> None:
    """Register condition-related tools with the MCP server."""
    server.add_tool(km_add_condition)
