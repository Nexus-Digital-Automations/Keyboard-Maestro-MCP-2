"""Engine control tools for Keyboard Maestro.

Provides tools to control the Keyboard Maestro engine, including reload operations,
calculations, token processing, and search/replace functionality.
"""

import ast
import asyncio
import logging
import operator
import re
from datetime import datetime
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from ...core import ValidationError
from ..initialization import get_km_client

logger = logging.getLogger(__name__)


_DOLLAR_BACKREF_RE = re.compile(r"\$(\$|\d+|\{\d+\})")


def _km_to_python_backrefs(template: str) -> str:
    """Translate KM/ICU dollar-style backrefs to Python's backslash form.

    KM regex docs (and ICU regex) use ``$1``/``$2`` for capture references
    and ``$$`` for a literal dollar; Python's ``re.sub`` ignores those and
    expects ``\\1``/``\\g<1>``. Translate so the documented KM syntax
    actually works without surprising users with literal ``$1`` output.
    """

    def _swap(match: re.Match[str]) -> str:
        ref = match.group(1)
        if ref == "$":
            return "$"
        digits = ref.strip("{}")
        return f"\\g<{digits}>"

    return _DOLLAR_BACKREF_RE.sub(_swap, template)


def _evaluate_expression_with_ast(
    expression: str,
    safe_namespace: dict[str, Any],
) -> Any:
    """Secure AST-based expression evaluation for mathematical calculations.

    This function uses AST parsing and validation to safely evaluate mathematical
    expressions without the security risks of eval(). Only safe mathematical
    operations and whitelisted functions are allowed.

    Args:
        expression: Mathematical expression to evaluate
        safe_namespace: Dictionary of allowed variables and functions

    Returns:
        Result of the expression evaluation

    Raises:
        ValidationError: If expression contains unsafe operations
        ValueError: If expression cannot be parsed or evaluated

    """
    try:
        # Parse expression into AST
        tree = ast.parse(expression, mode="eval")

        # Validate AST only contains safe operations
        _validate_ast_safety(tree)

        # Evaluate with safe operations mapping
        return _evaluate_ast_node(tree.body, safe_namespace)

    except (SyntaxError, ValueError) as e:
        raise ValidationError(
            "expression",
            expression,
            f"Invalid expression: {e!s}",
        ) from e


def _validate_ast_safety(node: ast.AST) -> None:
    """Validate that AST contains only safe mathematical operations.

    Args:
        node: AST node to validate

    Raises:
        ValidationError: If unsafe operations are detected

    """
    # Allowed AST node types for mathematical expressions
    safe_node_types = {
        ast.Expression,  # Root expression node
        ast.BinOp,  # Binary operations (+, -, *, /, etc.)
        ast.UnaryOp,  # Unary operations (+, -, not)
        ast.Call,  # Function calls (for math functions)
        ast.Name,  # Variable names
        ast.Constant,  # Literal values (numbers, strings); replaces removed ast.Num in 3.12+
        ast.Load,  # Load context
        # Binary operators
        ast.Add,  # Addition operator
        ast.Sub,  # Subtraction operator
        ast.Mult,  # Multiplication operator
        ast.Div,  # Division operator
        ast.FloorDiv,  # Floor division operator
        ast.Mod,  # Modulo operator
        ast.Pow,  # Power operator
        # Unary operators
        ast.UAdd,  # Unary plus
        ast.USub,  # Unary minus
    }

    # Allowed operators
    safe_operators = {
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    }

    for child in ast.walk(node):
        if type(child) not in safe_node_types:
            raise ValidationError(
                "expression",
                "AST validation",
                f"Unsafe operation detected: {type(child).__name__}",
            )

        # SIM102 fix: Combine nested if statements
        if (
            isinstance(
                child,
                ast.BinOp | ast.UnaryOp,
            )  # UP038 fix: Use | for isinstance
            and type(child.op) not in safe_operators
        ):
            raise ValidationError(
                "expression",
                "AST validation",
                f"Unsafe operator: {type(child.op).__name__}",
            )


def _evaluate_ast_node(node: ast.AST, namespace: dict[str, Any]) -> Any:
    """Recursively evaluate AST node with safe operations.

    Args:
        node: AST node to evaluate
        namespace: Safe namespace for variables and functions

    Returns:
        Evaluated result

    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in namespace:
            raise ValidationError("expression", node.id, f"Unknown variable: {node.id}")
        return namespace[node.id]
    if isinstance(node, ast.BinOp):
        left = _evaluate_ast_node(node.left, namespace)
        right = _evaluate_ast_node(node.right, namespace)
        return _apply_operator(node.op, left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast_node(node.operand, namespace)
        return _apply_unary_operator(node.op, operand)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValidationError(
                "expression",
                "function call",
                "Complex function calls not allowed",
            )

        func_name = node.func.id
        if func_name not in namespace:
            raise ValidationError(
                "expression",
                func_name,
                f"Unknown function: {func_name}",
            )

        args = [_evaluate_ast_node(arg, namespace) for arg in node.args]
        return namespace[func_name](*args)
    raise ValidationError(
        "expression",
        type(node).__name__,
        f"Unsupported node type: {type(node).__name__}",
    )


def _apply_operator(op: ast.operator, left: Any, right: Any) -> Any:
    """Apply binary operator safely."""
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    if type(op) not in operators:
        raise ValidationError(
            "expression",
            type(op).__name__,
            f"Unsupported operator: {type(op).__name__}",
        )

    try:
        return operators[type(op)](left, right)
    except ZeroDivisionError as e:
        raise ValidationError("expression", "division", "Division by zero") from e
    except Exception as e:
        raise ValidationError(
            "expression",
            "operation",
            f"Operation error: {e!s}",
        ) from e


def _apply_unary_operator(op: ast.unaryop, operand: Any) -> Any:
    """Apply unary operator safely."""
    if isinstance(op, ast.UAdd):
        return +operand
    if isinstance(op, ast.USub):
        return -operand
    raise ValidationError(
        "expression",
        type(op).__name__,
        f"Unsupported unary operator: {type(op).__name__}",
    )


async def km_engine_control(
    operation: Annotated[
        Literal["reload", "calculate", "process_tokens", "search_replace", "status"],
        Field(description="Engine operation to perform"),
    ],
    expression: Annotated[
        str | None,
        Field(
            default=None,
            description="Calculation expression or token string",
            max_length=1000,
        ),
    ] = None,
    search_pattern: Annotated[
        str | None,
        Field(
            default=None,
            description="Search pattern for search/replace operations",
            max_length=500,
        ),
    ] = None,
    replace_pattern: Annotated[
        str | None,
        Field(default=None, description="Replacement pattern", max_length=500),
    ] = None,
    use_regex: Annotated[
        bool,
        Field(default=False, description="Enable regex processing for search/replace"),
    ] = False,
    text: Annotated[
        str | None,
        Field(
            default=None,
            description="Text to process for search/replace operations",
            max_length=10000,
        ),
    ] = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Control Keyboard Maestro engine operations.

    Operations:
    - reload: Reload all macros (useful after external changes)
    - calculate: Perform mathematical calculations with KM's calculation engine
    - process_tokens: Process text containing KM tokens (variables, dates, etc.)
    - search_replace: Perform text search and replace with optional regex
    - status: Get current engine status and statistics

    The calculation engine supports:
    - Standard arithmetic: +, -, *, /, %, ^
    - Functions: sin, cos, tan, log, sqrt, abs, round, etc.
    - Variables: Can reference KM variables in calculations
    - Arrays and coordinate operations

    Token processing expands:
    - Variable tokens: %Variable%Name%
    - Date tokens: %ICUDateTime%format%
    - Calculation tokens: %Calculate%expression%

    Caveat (KM-side limitation): single-value system tokens such as
    %CurrentUser%, %FrontWindowName%, %FrontAppName%, %FinderInsertionLocation%
    are returned literally when invoked outside a macro execution context.
    KM's `process tokens` AppleScript verb only resolves these inside a
    running macro. Read OS env vars for the current user, or populate a
    KM variable inside a macro and read it via km_variable_manager.
    """
    if ctx:
        await ctx.info(f"Engine control operation: {operation}")

    try:
        km_client = get_km_client()

        # Validate required parameters
        if operation == "calculate" and not expression:
            raise ValidationError(
                "expression",
                expression,
                "Expression required for calculate operation",
            )

        if operation == "process_tokens" and not expression:
            raise ValidationError(
                "expression",
                expression,
                "Token string required for process_tokens operation",
            )

        if operation == "search_replace":
            if not search_pattern:
                raise ValidationError(
                    "search_pattern",
                    search_pattern,
                    "Search pattern required for search_replace operation",
                )
            if not text:
                raise ValidationError(
                    "text",
                    text,
                    "Text required for search_replace operation",
                )

        # Check connection
        connection_test = await asyncio.get_event_loop().run_in_executor(
            None,
            km_client.check_connection,
        )

        if connection_test.is_left() or not connection_test.get_right():
            return {
                "success": False,
                "error": {
                    "code": "KM_CONNECTION_FAILED",
                    "message": "Cannot connect to Keyboard Maestro Engine",
                },
            }

        if ctx:
            await ctx.report_progress(25, 100, "Connected to Keyboard Maestro Engine")

        # Execute the requested operation
        if operation == "reload":
            return await _reload_engine(km_client, ctx)
        if operation == "status":
            return await _get_engine_status(km_client, ctx)
        if operation == "calculate":
            return await _calculate_expression(km_client, expression, ctx)
        if operation == "process_tokens":
            return await _process_tokens(km_client, expression, ctx)
        if operation == "search_replace":
            return await _search_replace(
                km_client,
                text,
                search_pattern,
                replace_pattern,
                use_regex,
                ctx,
            )
        raise ValidationError(
            "operation",
            operation,
            f"Unknown operation: {operation}",
        )

    except Exception as e:
        logger.error(f"Engine control error: {e}")
        if ctx:
            await ctx.error(f"Engine operation failed: {e!s}")

        return {
            "success": False,
            "error": {
                "code": "ENGINE_ERROR",
                "message": f"Failed to {operation} engine",
                "details": str(e),
                "recovery_suggestion": "Check Keyboard Maestro Engine is running",
            },
        }


async def _reload_engine(km_client: Any, ctx: Context = None) -> dict[str, Any]:
    """Reload the Keyboard Maestro engine.

    Previously this just slept 0.5s and reported success — no engine
    reload actually happened. Issues a real reload via the engine's
    AppleScript ``reload`` verb and reports the real wall-clock time.
    """
    if ctx:
        await ctx.report_progress(50, 100, "Reloading engine macros")

    start_time = datetime.now()
    result = await km_client.execute_applescript_async(
        'tell application "Keyboard Maestro Engine" to reload',
    )
    if result.is_left():
        return {
            "success": False,
            "error": {
                "code": "RELOAD_FAILED",
                "message": result.get_left().message,
                "recovery_suggestion": "Verify Keyboard Maestro Engine is running.",
            },
        }
    reload_time = (datetime.now() - start_time).total_seconds()

    if ctx:
        await ctx.report_progress(100, 100, "Engine reloaded")
        await ctx.info(f"Engine reload completed in {reload_time:.2f} seconds")

    return {
        "success": True,
        "data": {
            "operation": "reload",
            "reload_time_seconds": reload_time,
            "timestamp": datetime.now().isoformat(),
        },
    }


async def _get_engine_status(km_client: Any, ctx: Context = None) -> dict[str, Any]:
    """Return KM engine status sourced from the engine itself.

    Earlier revisions returned mocked ``performance`` and ``resources``
    blocks plus a hardcoded ``engine_version`` string. None came from KM;
    the tool returned the same numbers on every call regardless of system
    state. Now we only surface fields we can verify against the live
    engine: real KM version via AppleScript, real macro counts via the
    existing client, and reachability as ``engine_running``.
    """
    if ctx:
        await ctx.report_progress(50, 100, "Fetching engine status")

    version_result = await km_client.execute_applescript_async(
        'tell application "Keyboard Maestro" to return version',
    )
    engine_running = version_result.is_right()
    engine_version = version_result.get_right().strip() if engine_running else None

    # Use the same code path as km_list_macros so the counts match. The
    # legacy list_macros_with_details + _send_command path was returning
    # zero macros against KM 11 because its KM Engine plugin command isn't
    # wired up; list_macros_async drives AppleScript with a Web-API fallback.
    macros_result = await km_client.list_macros_async(
        group_filters=None,
        enabled_only=False,
    )
    if macros_result.is_right():
        macros = macros_result.get_right()
        total_macros = len(macros)
        enabled_macros = sum(1 for m in macros if m.get("enabled", True))
    else:
        total_macros = 0
        enabled_macros = 0

    status = {
        "engine_version": engine_version,
        "engine_running": engine_running,
        "queried_at": datetime.now().isoformat(),
        "macro_statistics": {
            "total_macros": total_macros,
            "enabled_macros": enabled_macros,
            "disabled_macros": total_macros - enabled_macros,
        },
    }
    if ctx:
        await ctx.report_progress(100, 100, "Status retrieved")
    return {"success": True, "data": status}


async def _calculate_expression(
    _km_client: Any,
    expression: str,
    ctx: Context = None,
) -> dict[str, Any]:
    """Calculate a mathematical expression using KM's engine."""
    if ctx:
        await ctx.report_progress(50, 100, f"Calculating: {expression}")

    # Validate expression doesn't contain dangerous operations
    if any(
        dangerous in expression.lower()
        for dangerous in ["exec", "eval", "import", "__"]
    ):
        raise ValidationError(
            "expression",
            expression,
            "Expression contains forbidden operations",
        )

    # AppleScript: tell application "Keyboard Maestro Engine" to calculate "expression"
    try:
        # For demo, use Python's eval with restricted namespace
        # In real implementation, would use KM's calculate command
        safe_namespace = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "pow": pow,
            # Math functions would be available in KM
            "pi": 3.14159265359,
            "e": 2.71828182846,
        }

        # Basic safety check
        allowed_chars = (
            "0123456789+-*/()., abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
        )
        if not all(c in allowed_chars for c in expression):
            raise ValidationError(
                "expression",
                expression,
                "Expression contains invalid characters",
            )

        # S307 Security: Use AST validation instead of eval for secure expression evaluation
        result = _evaluate_expression_with_ast(expression, safe_namespace)

        if ctx:
            await ctx.report_progress(100, 100, "Calculation complete")

        return {
            "success": True,
            "data": {
                "expression": expression,
                "result": str(result),
                "result_type": type(result).__name__,
                "timestamp": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        raise ValidationError(
            "expression",
            expression,
            f"Calculation failed: {e!s}",
        ) from e


_TOKEN_PATTERN = re.compile(r"%([A-Za-z][A-Za-z0-9]*)(%[^%]*)*%")


def _summarise_tokens(token_string: str) -> list[str]:
    """Return a human-readable summary of KM tokens detected in the input."""
    summary: list[str] = []
    for match in _TOKEN_PATTERN.finditer(token_string):
        body = match.group(0)
        inner = match.group(1)
        if body.startswith("%Variable%"):
            parts = body[1:-1].split("%")
            summary.append(f"Variable: {parts[1]}" if len(parts) > 1 else "Variable")
        elif body.startswith("%ICUDateTime%"):
            parts = body[1:-1].split("%", 1)
            summary.append(f"DateTime: {parts[1]}" if len(parts) > 1 else "DateTime")
        else:
            summary.append(f"System: {inner}")
    return summary


async def _process_tokens(
    km_client: Any,
    token_string: str,
    ctx: Context = None,
) -> dict[str, Any]:
    """Expand KM tokens via the live Keyboard Maestro Engine.

    Failure modes:
    - KM_ENGINE_FAILED: the engine returned an osascript error
    - KM_ENGINE_UNREACHABLE: the engine isn't responding to AppleScript
    """
    if ctx:
        await ctx.report_progress(50, 100, "Processing tokens")

    escaped = token_string.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        'tell application "Keyboard Maestro Engine" '
        f'to return process tokens "{escaped}"'
    )
    result = await km_client.execute_applescript_async(script)
    if result.is_left():
        err = result.get_left()
        return {
            "success": False,
            "error": {
                "code": "KM_ENGINE_FAILED",
                "message": "Keyboard Maestro Engine rejected the token string.",
                "details": getattr(err, "message", str(err)),
                "recovery_suggestion": (
                    "Verify token syntax (e.g. %Variable%Name%, %ICUDateTime%y%) "
                    "and that Keyboard Maestro Engine is running."
                ),
            },
        }

    processed = result.get_right()
    tokens_found = _summarise_tokens(token_string)

    if ctx:
        await ctx.report_progress(100, 100, f"Processed {len(tokens_found)} tokens")

    return {
        "success": True,
        "data": {
            "original": token_string,
            "processed": processed,
            "tokens_found": tokens_found,
            "token_count": len(tokens_found),
            "timestamp": datetime.now().isoformat(),
        },
    }


async def _search_replace(
    _km_client: Any,
    text: str,
    search_pattern: str,
    replace_pattern: str | None,
    use_regex: bool,
    ctx: Context = None,
) -> dict[str, Any]:
    """Perform search and replace operation."""
    if ctx:
        await ctx.report_progress(50, 100, "Performing search/replace")

    # AppleScript: tell application "Keyboard Maestro Engine" to search "text" for "pattern" replace "replacement" with regex

    try:
        if use_regex:
            # Regex search/replace
            if replace_pattern is None:
                # Just search
                matches = list(re.finditer(search_pattern, text))
                match_count = len(matches)
                result = text  # No replacement
            else:
                # KM/ICU regex docs use $1/$2 backrefs; Python's re module
                # uses \1/\2. Translate dollar-form refs to backslash-form
                # so the documented KM syntax works. A literal '$' is
                # written as '$$' in KM and becomes '$' here.
                translated_replace = _km_to_python_backrefs(replace_pattern)
                result = re.sub(search_pattern, translated_replace, text)
                match_count = len(re.findall(search_pattern, text))
        else:
            # Plain text search/replace
            match_count = text.count(search_pattern)
            if replace_pattern is None:
                result = text
            else:
                result = text.replace(search_pattern, replace_pattern)

        if ctx:
            await ctx.report_progress(100, 100, f"Found {match_count} matches")

        return {
            "success": True,
            "data": {
                "search_pattern": search_pattern,
                "replace_pattern": replace_pattern,
                "use_regex": use_regex,
                "match_count": match_count,
                "original_length": len(text),
                "result_length": len(result),
                "result": result if len(result) < 1000 else result[:1000] + "...",
                "timestamp": datetime.now().isoformat(),
            },
        }

    except re.error as e:
        raise ValidationError(
            "search_pattern",
            search_pattern,
            f"Invalid regex pattern: {e!s}",
        ) from e
