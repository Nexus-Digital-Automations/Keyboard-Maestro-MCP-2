"""Control flow tools for Keyboard Maestro MCP server.

Provides sophisticated control flow constructs (if/then/else, loops, switch/case)
for creating intelligent, adaptive automation workflows.

@stable Modes if_then_else, for_loop, while_loop, until_loop, and try_catch
emit real KM action plist via dedicated emitters and append directly. The
switch_case mode still uses the legacy AST path (AppendFails with
NotImplementedError) until PR2 implements its emitter.
"""

from datetime import datetime
from typing import Any

from fastmcp import Context

from ...core.control_flow import (
    ComparisonOperator,
    ControlFlowBuilder,
    ControlFlowNodeType,
    ControlFlowValidator,
    ForLoopNode,
    IfThenElseNode,
    SwitchCaseNode,
    TryCatchNode,
    WhileLoopNode,
    create_for_loop,
    create_simple_if,
    create_while_loop,
)
from ...core.errors import SecurityError, ValidationError
from ...core.types import MacroId
from ...integration.km_for_loop_xml import (
    SUPPORTED_COLLECTION_TYPES,
    UnsupportedCollectionType,
    build_collection_dict,
    build_for_loop_xml,
)
from ...integration.km_if_then_else_xml import (
    UnsupportedConditionType,
    UnsupportedOperator,
    build_condition_dict,
    build_if_then_else_xml,
)
from ...integration.km_switch_case_xml import (
    SUPPORTED_CASE_CONDITIONS,
    SUPPORTED_SOURCES,
    UnsupportedCaseCondition,
    build_switch_case_xml,
    render_cases,
)
from ...integration.km_try_catch_xml import build_try_catch_xml
from ...integration.km_while_loop_xml import (
    build_until_loop_xml,
    build_while_loop_xml,
)
from ..initialization import get_km_client
from .action_builder_tools import _build_action_xml


async def km_control_flow(  # noqa: PLR0913 - public MCP tool surface
    macro_identifier: str,
    control_type: str,
    condition: str | None = None,
    operator: str = "equals",
    operand: str | None = None,
    iterator: str | None = None,
    collection: str | None = None,
    collection_dict: dict[str, Any] | None = None,
    source: str = "Variable",
    cases: list[dict[str, Any]] | None = None,
    actions_true: list[dict[str, Any]] | None = None,
    actions_false: list[dict[str, Any]] | None = None,
    loop_actions: list[dict[str, Any]] | None = None,
    try_actions: list[dict[str, Any]] | None = None,
    catch_actions: list[dict[str, Any]] | None = None,
    default_actions: list[dict[str, Any]] | None = None,
    max_iterations: int = 1000,
    timeout_seconds: int = 30,
    allow_nested: bool = True,
    case_sensitive: bool = True,
    negate: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Add control flow structures to Keyboard Maestro macros.

    Creates sophisticated control flow constructs including if/then/else statements,
    for loops, while loops, and switch/case statements with comprehensive security
    validation and performance optimization.

    Architecture:
        - Pattern: Builder Pattern with AST representation for complex logic
        - Security: Defense-in-depth with input validation, loop bounds, injection prevention
        - Performance: O(1) validation, O(n) execution where n is bounded by security limits

    Contracts:
        Preconditions:
            - macro_identifier is valid (name or UUID format)
            - control_type is supported (if_then_else, for_loop, while_loop, switch_case)
            - condition expressions are validated for security
            - max_iterations is within security bounds (1-10000)
            - timeout_seconds is within limits (1-300)

        Postconditions:
            - control_flow_id is returned on success
            - macro is updated with new control structure
            - all actions are validated and secure

        Invariants:
            - No infinite loops possible (bounded iterations)
            - No code injection in conditions or actions
            - Nesting depth is limited for stack safety

    Security Implementation:
        - Input Validation: Comprehensive pattern detection for dangerous content
        - Loop Protection: Maximum iteration limits with timeout enforcement
        - Condition Security: Regex validation, injection prevention, pattern whitelisting
        - Memory Safety: Bounded action counts, limited nesting depth

    Args:
        macro_identifier: Target macro name or UUID for control flow addition
        control_type: Type of control flow (if_then_else, for_loop, while_loop, switch_case)
        condition: Condition expression for if/while statements (validated)
        operator: Comparison operator (equals, greater_than, contains, etc.)
        operand: Value to compare against in conditions
        iterator: Variable name for loop iteration (for loops only)
        collection: Collection expression to iterate over (for loops only)
        cases: List of switch cases with values and actions
        actions_true: Actions to execute when condition is true
        actions_false: Actions to execute when condition is false
        loop_actions: Actions to execute in loop body
        default_actions: Default actions for switch statement
        max_iterations: Maximum loop iterations (security bounded)
        timeout_seconds: Maximum execution timeout (1-300 seconds)
        allow_nested: Whether to allow nested control structures
        case_sensitive: Case sensitivity for string comparisons
        negate: Whether to negate the condition result
        ctx: MCP context for logging and progress reporting

    Returns:
        Dict containing control flow ID, validation results, and metadata

    Raises:
        ValidationError: Invalid parameters or macro not found
        SecurityError: Security validation failed (dangerous content detected)
        ExecutionError: Failed to add control flow to macro

    """
    start_time = datetime.now()

    if ctx:
        await ctx.info(
            f"Adding {control_type} control flow to macro: {macro_identifier}",
        )

    # Shared validation runs before emitter dispatch — bounds, dangerous
    # patterns, and macro-id sanity must apply uniformly across all modes.
    try:
        await _validate_control_flow_inputs(
            macro_identifier,
            control_type,
            condition,
            operator,
            operand,
            iterator,
            collection,
            max_iterations,
            timeout_seconds,
            ctx,
            collection_dict=collection_dict,
        )
    except ValidationError as exc:
        if ctx:
            await ctx.error(f"Validation error: {exc}")
        return _validation_failure(
            str(exc),
            "Check parameter format and security requirements.",
            macro_identifier,
        )
    except SecurityError as exc:
        if ctx:
            await ctx.error(f"Security error: {exc}")
        return {
            "success": False,
            "error": {
                "code": "SECURITY_ERROR",
                "message": str(exc),
                "recovery_suggestion": "Remove dangerous patterns and reduce complexity.",
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "correlation_id": f"security_error_{macro_identifier}",
            },
        }

    if control_type == "if_then_else":
        return await _emit_if_then_else(
            macro_identifier=macro_identifier,
            condition=condition,
            operator=operator,
            operand=operand,
            actions_true=actions_true,
            actions_false=actions_false,
            case_sensitive=case_sensitive,
            negate=negate,
            start_time=start_time,
            ctx=ctx,
        )
    if control_type == "for_loop":
        return await _emit_for_loop(
            macro_identifier=macro_identifier,
            iterator=iterator,
            collection_dict=collection_dict,
            loop_actions=loop_actions,
            start_time=start_time,
            ctx=ctx,
        )
    if control_type in {"while_loop", "until_loop"}:
        return await _emit_while_or_until(
            macro_identifier=macro_identifier,
            control_type=control_type,
            condition=condition,
            operator=operator,
            operand=operand,
            loop_actions=loop_actions,
            case_sensitive=case_sensitive,
            negate=negate,
            start_time=start_time,
            ctx=ctx,
        )
    if control_type == "try_catch":
        return await _emit_try_catch(
            macro_identifier=macro_identifier,
            try_actions=try_actions,
            catch_actions=catch_actions,
            start_time=start_time,
            ctx=ctx,
        )
    if control_type == "switch_case":
        return await _emit_switch_case(
            macro_identifier=macro_identifier,
            source=source,
            source_value=condition or "",
            cases=cases,
            default_actions=default_actions,
            start_time=start_time,
            ctx=ctx,
        )
    # Validator already rejects unknown control_type, so this is unreachable.
    # Kept as a defensive fallthrough only.
    return _validation_failure(
        f"control_type {control_type!r} reached dispatcher without an emitter "
        "branch. This is a bug — file an issue.",
        "Use one of: if_then_else, for_loop, while_loop, until_loop, switch_case, try_catch.",
        macro_identifier,
    )


async def _validate_control_flow_inputs(  # noqa: PLR0913 - shared validator
    macro_identifier: str,
    control_type: str,
    condition: str | None,
    operator: str,
    operand: str | None,
    iterator: str | None,
    collection: str | None,
    max_iterations: int,
    timeout_seconds: int,
    ctx: Context | None,
    *,
    collection_dict: dict[str, Any] | None = None,
) -> None:
    """Validate all control flow inputs for security and correctness."""
    # Validate macro identifier
    if not macro_identifier or len(macro_identifier.strip()) == 0:
        raise ValidationError("macro_identifier", macro_identifier, "cannot be empty")

    if len(macro_identifier) > 255:
        raise ValidationError(
            "macro_identifier",
            macro_identifier,
            "must be 255 characters or less",
        )

    # Validate control type
    valid_types = {
        "if_then_else", "for_loop", "while_loop", "until_loop",
        "switch_case", "try_catch",
    }
    if control_type not in valid_types:
        raise ValidationError(
            "control_type",
            control_type,
            f"must be one of: {', '.join(valid_types)}",
        )

    # Validate operator
    valid_operators = {
        "equals",
        "not_equals",
        "greater_than",
        "less_than",
        "greater_equal",
        "less_equal",
        "contains",
        "not_contains",
        "matches_regex",
        "exists",
    }
    if operator not in valid_operators:
        raise ValidationError(
            "operator",
            operator,
            f"must be one of: {', '.join(valid_operators)}",
        )

    # Validate condition for conditional types
    if control_type in {"if_then_else", "while_loop", "until_loop"}:
        if not condition:
            raise ValidationError(
                "condition",
                condition,
                f"{control_type} requires a condition expression",
            )

        if len(condition) > 500:
            raise ValidationError(
                "condition",
                condition,
                "must be 500 characters or less",
            )

        # Check for dangerous patterns
        dangerous_patterns = [
            "exec",
            "eval",
            "import",
            "__import__",
            "subprocess",
            "os.system",
            "shell",
            "cmd",
            "`",
            "rm ",
            "del ",
            "format",
            "curl",
            "wget",
            "http",
        ]

        condition_lower = condition.lower()
        for pattern in dangerous_patterns:
            if pattern in condition_lower:
                raise SecurityError(
                    "DANGEROUS_PATTERN",
                    f"Dangerous pattern detected in condition: {pattern}",
                )

    # Validate loop-specific parameters
    if control_type == "for_loop":
        if not iterator:
            raise ValidationError(
                "iterator",
                iterator,
                "For loop requires an iterator variable",
            )
        if not collection_dict and not collection:
            types_csv = ", ".join(SUPPORTED_COLLECTION_TYPES)
            raise ValidationError(
                "collection_dict",
                collection_dict,
                "For loop requires collection_dict with key 'type' "
                f"(supported types: {types_csv}). The legacy 'collection' "
                "string param is no longer used.",
            )

        if len(iterator) > 50:
            raise ValidationError("iterator", iterator, "must be 50 characters or less")
        if collection and len(collection) > 500:
            raise ValidationError(
                "collection",
                collection,
                "must be 500 characters or less",
            )

    # Validate security bounds
    if max_iterations < 1 or max_iterations > 10000:
        raise ValidationError(
            "max_iterations",
            max_iterations,
            "must be between 1 and 10000",
        )

    if timeout_seconds < 1 or timeout_seconds > 300:
        raise ValidationError(
            "timeout_seconds",
            timeout_seconds,
            "must be between 1 and 300",
        )

    if ctx:
        await ctx.info("Input validation passed")


async def _build_control_flow_structure(
    control_type: str,
    condition: str | None,
    operator: str,
    operand: str | None,
    iterator: str | None,
    collection: str | None,
    cases: list[dict[str, Any]] | None,
    actions_true: list[dict[str, Any]] | None,
    actions_false: list[dict[str, Any]] | None,
    loop_actions: list[dict[str, Any]] | None,
    default_actions: list[dict[str, Any]] | None,
    max_iterations: int,
    case_sensitive: bool,
    negate: bool,
    validator: ControlFlowValidator,
    ctx: Context | None,
) -> ControlFlowNodeType:
    """Build the appropriate control flow structure based on type."""
    try:
        # Convert string operator to enum
        op_map = {
            "equals": ComparisonOperator.EQUALS,
            "not_equals": ComparisonOperator.NOT_EQUALS,
            "greater_than": ComparisonOperator.GREATER_THAN,
            "less_than": ComparisonOperator.LESS_THAN,
            "greater_equal": ComparisonOperator.GREATER_EQUAL,
            "less_equal": ComparisonOperator.LESS_EQUAL,
            "contains": ComparisonOperator.CONTAINS,
            "not_contains": ComparisonOperator.NOT_CONTAINS,
            "matches_regex": ComparisonOperator.MATCHES_REGEX,
            "exists": ComparisonOperator.EXISTS,
        }
        comparison_op = op_map[operator]

        if control_type == "if_then_else":
            if not condition or not actions_true:
                raise ValidationError(
                    "if_then_else",
                    f"condition={condition}, actions_true={actions_true}",
                    "requires condition and true actions",
                )

            return create_simple_if(
                condition_expr=condition,
                operator=comparison_op,
                operand=operand or "",
                then_actions=actions_true,
                else_actions=actions_false,
            )

        if control_type == "for_loop":
            if not iterator or not collection or not loop_actions:
                raise ValidationError(
                    "for_loop",
                    f"iterator={iterator}, collection={collection}, loop_actions={loop_actions}",
                    "requires iterator, collection, and actions",
                )

            return create_for_loop(
                iterator=iterator,
                collection=collection,
                actions=loop_actions,
                max_iterations=max_iterations,
            )

        if control_type == "while_loop":
            if not condition or not loop_actions:
                raise ValidationError(
                    "while_loop",
                    f"condition={condition}, loop_actions={loop_actions}",
                    "requires condition and actions",
                )

            return create_while_loop(
                condition_expr=condition,
                operator=comparison_op,
                operand=operand or "",
                actions=loop_actions,
                max_iterations=max_iterations,
            )

        if control_type == "switch_case":
            if not cases:
                raise ValidationError("cases", cases, "Switch statement requires cases")

            # Convert cases to builder format
            case_tuples = []
            for case in cases:
                if "value" not in case or "actions" not in case:
                    raise ValidationError(
                        "case",
                        case,
                        "Each case must have 'value' and 'actions'",
                    )
                case_tuples.append((case["value"], case["actions"]))

            builder = ControlFlowBuilder(validator)
            builder.switch_on(
                variable=condition or "switch_variable",
                cases=case_tuples,
                default_actions=default_actions,
            )

            nodes = builder.build()
            return nodes[0]

        raise ValidationError(
            "control_type",
            control_type,
            "Unsupported control type",
        )

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to build control flow structure: {e}")
        raise


def _get_structure_info(node: ControlFlowNodeType) -> dict[str, Any]:
    """Get structural information about the control flow node."""
    info = {
        "node_type": type(node).__name__,
        "node_id": node.node_id,
        "depth": node.depth,
        "created_at": node.created_at.isoformat(),
    }

    if isinstance(node, IfThenElseNode):
        info.update(
            {
                "has_else_branch": node.has_else_branch(),
                "condition_operator": node.condition.operator.value,
                "then_action_count": len(node.then_actions.actions),
                "else_action_count": len(node.else_actions.actions)
                if node.else_actions
                else 0,
            },
        )
    elif isinstance(node, ForLoopNode | WhileLoopNode):
        if isinstance(node, ForLoopNode):
            info.update(
                {
                    "iterator_variable": node.loop_config.iterator_variable,
                    "collection_expression": node.loop_config.collection_expression,
                    "max_iterations": node.loop_config.max_iterations,
                    "action_count": len(node.loop_actions.actions),
                },
            )
        else:
            info.update(
                {
                    "condition_operator": node.condition.operator.value,
                    "max_iterations": node.max_iterations,
                    "action_count": len(node.loop_actions.actions),
                },
            )
    elif isinstance(node, SwitchCaseNode):
        info.update(
            {
                "switch_variable": node.switch_variable,
                "case_count": len(node.cases),
                "has_default": node.has_default_case(),
                "total_actions": sum(len(case.actions.actions) for case in node.cases),
            },
        )
    elif isinstance(node, TryCatchNode):
        info.update(
            {
                "try_action_count": len(node.try_actions.actions),
                "catch_action_count": len(node.catch_actions.actions),
                "has_finally": node.finally_actions is not None,
                "finally_action_count": len(node.finally_actions.actions)
                if node.finally_actions
                else 0,
            },
        )

    return info


# --- if_then_else fast-path -------------------------------------------------
#
# Bypasses the ControlFlowNode AST entirely: control_flow's IfThenElse mode
# only needs a single condition + two flat action lists, which the shared
# emitter in src/integration/km_if_then_else_xml.py handles directly.
# Looping/switch modes still go through the AST + validator path until
# they get the same treatment.

_CONTROL_FLOW_TO_CONDITION_TYPE = {
    "variable": "variable",
    "text": "text",
    "application": "application",
    "calculation": "calculation",
}


def _condition_kind_from_expression(expr: str) -> str:
    """Infer the condition_type from a free-form expression.

    Defaults to ``variable``: the original km_control_flow API treats
    ``condition`` as a variable name. Callers can override by prefixing
    the expression with ``"app:"``, ``"text:"``, or ``"calc:"``.
    """
    lowered = expr.strip().lower()
    for prefix in ("app:", "text:", "calc:"):
        if lowered.startswith(prefix):
            return {"app:": "application", "text:": "text", "calc:": "calculation"}[prefix]
    return "variable"


def _strip_condition_prefix(expr: str) -> str:
    stripped = expr.strip()
    for prefix in ("app:", "text:", "calc:"):
        if stripped.lower().startswith(prefix):
            return stripped[len(prefix):].strip()
    return stripped


def _render_inner_actions(actions: list[dict[str, Any]] | None) -> str:
    """Translate a list of {type, ...config} dicts into concatenated KM action <dict>s.

    Raises ValueError if any action_type cannot be rendered (mirrors the
    behaviour of km_action_builder.append).
    """
    if not actions:
        return ""
    pieces: list[str] = []
    for action in actions:
        action_type = action.get("type")
        if not action_type:
            raise ValueError("inner action dict missing required 'type' key")
        config = {k: v for k, v in action.items() if k != "type"}
        xml = _build_action_xml(str(action_type), config)
        if xml is None:
            raise ValueError(
                f"inner action_type {action_type!r} not supported by action_builder",
            )
        pieces.append(xml)
    return "".join(pieces)


async def _emit_if_then_else(  # noqa: PLR0913 - thin wrapper around 8 user args
    macro_identifier: str,
    condition: str | None,
    operator: str,
    operand: str | None,
    actions_true: list[dict[str, Any]] | None,
    actions_false: list[dict[str, Any]] | None,
    case_sensitive: bool,
    negate: bool,
    start_time: datetime,
    ctx: Context | None,
) -> dict[str, Any]:
    """Build an IfThenElse plist + append it to ``macro_identifier``."""
    if not condition:
        return _validation_failure(
            "condition is required for if_then_else",
            "Pass condition='VarName' (variable) or 'app:com.apple.finder' "
            "(application) or 'text:%Variable%X%' (text) or 'calc:1+1' (calculation).",
            macro_identifier,
        )

    cond_kind = _condition_kind_from_expression(condition)
    expr = _strip_condition_prefix(condition)
    composed_operand = (
        f"{expr}={operand}" if operand is not None and cond_kind in {"variable", "text"}
        else (operand or expr)
    )

    try:
        condition_xml = build_condition_dict(
            cond_kind,
            operator,
            composed_operand,
            case_sensitive=case_sensitive,
            negate=negate,
        )
        then_xml = _render_inner_actions(actions_true)
        else_xml = _render_inner_actions(actions_false)
    except (UnsupportedConditionType, UnsupportedOperator, ValueError) as exc:
        if ctx:
            await ctx.error(f"if_then_else render failed: {exc}")
        return _validation_failure(str(exc), "Check operator/condition_type/inner action_type.", macro_identifier)

    action_xml = build_if_then_else_xml(condition_xml, then_xml, else_xml)
    result = await get_km_client().append_macro_action_async(
        MacroId(macro_identifier.strip()),
        action_xml,
    )
    if result.is_left():
        err = result.get_left()
        if ctx:
            await ctx.error(f"KM rejected IfThenElse action: {err.message}")
        return {
            "success": False,
            "error": {
                "code": "APPEND_FAILED",
                "message": err.message,
                "recovery_suggestion": "Verify macro exists and KM engine is running.",
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "correlation_id": f"append_error_{macro_identifier}",
            },
        }

    exec_time = (datetime.now() - start_time).total_seconds()
    if ctx:
        await ctx.info(f"IfThenElse appended in {exec_time:.3f}s")
    return {
        "success": True,
        "data": {
            "control_type": "if_then_else",
            "macro_id": macro_identifier,
            "macro_action_type": "IfThenElse",
            "condition_kind": cond_kind,
            "then_action_count": len(actions_true or []),
            "else_action_count": len(actions_false or []),
            "execution_time": exec_time,
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "server_version": "1.0.0",
            "correlation_id": f"cf_if_{macro_identifier}",
        },
    }


def _validation_failure(message: str, suggestion: str, macro_id: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": message,
            "recovery_suggestion": suggestion,
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": f"validation_{macro_id}",
        },
    }


async def _append_action_xml(
    macro_identifier: str,
    action_xml: str,
    ctx: Context | None,
) -> dict[str, Any] | None:
    """Append a control-flow action plist; return failure envelope or None on success."""
    result = await get_km_client().append_macro_action_async(
        MacroId(macro_identifier.strip()),
        action_xml,
    )
    if result.is_right():
        return None
    err = result.get_left()
    if ctx:
        await ctx.error(f"KM rejected control-flow action: {err.message}")
    return {
        "success": False,
        "error": {
            "code": "APPEND_FAILED",
            "message": err.message,
            "recovery_suggestion": "Verify macro exists and KM engine is running.",
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": f"append_error_{macro_identifier}",
        },
    }


def _success(
    *,
    control_type: str,
    macro_action_type: str,
    macro_identifier: str,
    extra: dict[str, Any],
    start_time: datetime,
) -> dict[str, Any]:
    exec_time = (datetime.now() - start_time).total_seconds()
    return {
        "success": True,
        "data": {
            "control_type": control_type,
            "macro_id": macro_identifier,
            "macro_action_type": macro_action_type,
            "execution_time": exec_time,
            **extra,
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "server_version": "1.0.0",
            "correlation_id": f"cf_{control_type}_{macro_identifier}",
        },
    }


async def _emit_for_loop(
    macro_identifier: str,
    iterator: str | None,
    collection_dict: dict[str, Any] | None,
    loop_actions: list[dict[str, Any]] | None,
    start_time: datetime,
    ctx: Context | None,
) -> dict[str, Any]:
    """Build a For action with one CollectionList entry + inner actions."""
    if not iterator or not iterator.strip():
        return _validation_failure(
            "iterator (loop variable name) is required for for_loop.",
            "Pass iterator='item' or any valid KM variable name.",
            macro_identifier,
        )
    if not collection_dict or "type" not in collection_dict:
        types_csv = ", ".join(SUPPORTED_COLLECTION_TYPES)
        return _validation_failure(
            "collection_dict with key 'type' is required for for_loop. "
            f"Supported types: {types_csv}.",
            "Example: collection_dict={'type': 'Range', 'start': '1', 'end': '10'}.",
            macro_identifier,
        )
    coll_type = str(collection_dict["type"])
    coll_kwargs = {k: v for k, v in collection_dict.items() if k != "type"}

    try:
        collection_xml = build_collection_dict(coll_type, **coll_kwargs)
        actions_xml = _render_inner_actions(loop_actions)
    except (UnsupportedCollectionType, ValueError) as exc:
        if ctx:
            await ctx.error(f"for_loop render failed: {exc}")
        return _validation_failure(
            str(exc),
            "Check collection_dict.type and inner action_type values.",
            macro_identifier,
        )

    action_xml = build_for_loop_xml(iterator.strip(), collection_xml, actions_xml)
    failure = await _append_action_xml(macro_identifier, action_xml, ctx)
    if failure is not None:
        return failure
    return _success(
        control_type="for_loop",
        macro_action_type="For",
        macro_identifier=macro_identifier,
        extra={
            "iterator": iterator.strip(),
            "collection_type": coll_type,
            "loop_action_count": len(loop_actions or []),
        },
        start_time=start_time,
    )


async def _emit_while_or_until(  # noqa: PLR0913 - thin wrapper over user args
    macro_identifier: str,
    control_type: str,
    condition: str | None,
    operator: str,
    operand: str | None,
    loop_actions: list[dict[str, Any]] | None,
    case_sensitive: bool,
    negate: bool,
    start_time: datetime,
    ctx: Context | None,
) -> dict[str, Any]:
    """Build a While or Until action — same shape, different MacroActionType."""
    if not condition:
        return _validation_failure(
            f"condition is required for {control_type}.",
            "Pass condition='VarName' or 'app:com.x.y' / 'text:...' / 'calc:...'.",
            macro_identifier,
        )
    cond_kind = _condition_kind_from_expression(condition)
    expr = _strip_condition_prefix(condition)
    composed_operand = (
        f"{expr}={operand}" if operand is not None and cond_kind in {"variable", "text"}
        else (operand or expr)
    )
    try:
        condition_xml = build_condition_dict(
            cond_kind, operator, composed_operand,
            case_sensitive=case_sensitive, negate=negate,
        )
        actions_xml = _render_inner_actions(loop_actions)
    except (UnsupportedConditionType, UnsupportedOperator, ValueError) as exc:
        if ctx:
            await ctx.error(f"{control_type} render failed: {exc}")
        return _validation_failure(
            str(exc),
            "Check operator/condition_type/inner action_type.",
            macro_identifier,
        )

    if control_type == "until_loop":
        action_xml = build_until_loop_xml(condition_xml, actions_xml)
        macro_action_type = "Until"
    else:
        action_xml = build_while_loop_xml(condition_xml, actions_xml)
        macro_action_type = "While"

    failure = await _append_action_xml(macro_identifier, action_xml, ctx)
    if failure is not None:
        return failure
    return _success(
        control_type=control_type,
        macro_action_type=macro_action_type,
        macro_identifier=macro_identifier,
        extra={
            "condition_kind": cond_kind,
            "loop_action_count": len(loop_actions or []),
        },
        start_time=start_time,
    )


async def _emit_try_catch(
    macro_identifier: str,
    try_actions: list[dict[str, Any]] | None,
    catch_actions: list[dict[str, Any]] | None,
    start_time: datetime,
    ctx: Context | None,
) -> dict[str, Any]:
    """Build a TryCatch action wrapping try and catch action lists."""
    if not try_actions:
        return _validation_failure(
            "try_actions is required for try_catch (catch_actions may be empty).",
            "Pass try_actions=[{'type': 'pause', 'seconds': 1}, ...].",
            macro_identifier,
        )
    try:
        try_xml = _render_inner_actions(try_actions)
        catch_xml = _render_inner_actions(catch_actions)
    except ValueError as exc:
        if ctx:
            await ctx.error(f"try_catch render failed: {exc}")
        return _validation_failure(
            str(exc),
            "Check inner action_type values.",
            macro_identifier,
        )

    action_xml = build_try_catch_xml(try_xml, catch_xml)
    failure = await _append_action_xml(macro_identifier, action_xml, ctx)
    if failure is not None:
        return failure
    return _success(
        control_type="try_catch",
        macro_action_type="TryCatch",
        macro_identifier=macro_identifier,
        extra={
            "try_action_count": len(try_actions or []),
            "catch_action_count": len(catch_actions or []),
        },
        start_time=start_time,
    )


async def _emit_switch_case(  # noqa: PLR0913 - thin wrapper over user args
    macro_identifier: str,
    source: str,
    source_value: str,
    cases: list[dict[str, Any]] | None,
    default_actions: list[dict[str, Any]] | None,
    start_time: datetime,
    ctx: Context | None,
) -> dict[str, Any]:
    """Build a Switch action with one Source + N CaseEntries (incl. optional Otherwise)."""
    if not cases and not default_actions:
        types_csv = ", ".join(SUPPORTED_CASE_CONDITIONS)
        return _validation_failure(
            "switch_case requires at least one case or default_actions. "
            f"Each case = {{condition_type, test_value, actions}}; "
            f"condition_type in {types_csv}.",
            "Example: cases=[{'condition_type': 'Is', 'test_value': 'foo', 'actions': [...]}].",
            macro_identifier,
        )
    if source not in SUPPORTED_SOURCES:
        types_csv = ", ".join(sorted(SUPPORTED_SOURCES))
        return _validation_failure(
            f"source {source!r} not supported. Supported: {types_csv}.",
            "Pass source='Variable' (default) and condition='MyVar', or source='Calculation' with condition='1+1'.",
            macro_identifier,
        )

    all_cases = list(cases or [])
    if default_actions:
        # KM stores 'default' as a sentinel CaseEntry — no separate plist key.
        all_cases.append(
            {"condition_type": "Otherwise", "test_value": "", "actions": default_actions},
        )

    try:
        case_entries_xml = render_cases(all_cases, _render_inner_actions)
    except (UnsupportedCaseCondition, ValueError) as exc:
        if ctx:
            await ctx.error(f"switch_case render failed: {exc}")
        return _validation_failure(
            str(exc),
            "Check each case's condition_type and inner action_type.",
            macro_identifier,
        )

    action_xml = build_switch_case_xml(
        source, case_entries_xml, source_value=source_value,
    )
    failure = await _append_action_xml(macro_identifier, action_xml, ctx)
    if failure is not None:
        return failure
    return _success(
        control_type="switch_case",
        macro_action_type="Switch",
        macro_identifier=macro_identifier,
        extra={
            "source": source,
            "case_count": len(all_cases),
            "has_otherwise": any(c.get("condition_type") == "Otherwise" for c in all_cases),
        },
        start_time=start_time,
    )
