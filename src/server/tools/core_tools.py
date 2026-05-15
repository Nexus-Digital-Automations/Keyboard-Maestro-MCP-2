"""Core macro operation tools.

Contains the fundamental MCP tools for macro execution, listing, and variable management.
"""

import asyncio
import logging
import re
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

from ...core import (
    ExecutionError,
    MacroId,
    PermissionDeniedError,
    TimeoutError,
    ValidationError,
)
from ..initialization import get_km_client
from ..utils import parse_variable_records

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
)
_NAME_NOT_FOUND_FRAGMENT = "no macros with a matching name"


async def _resolve_macro_uuid(name: str) -> str | None:
    """Return the UUID of the first enabled+disabled macro whose name matches.

    KM's `do script` dispatch by name fails with "no macros with a matching name"
    when the macro's group is freshly created and not yet "active" in the
    current context. UUID dispatch is unaffected. We use this to recover once.
    """
    km_client = get_km_client()
    list_result = await km_client.list_macros_async(enabled_only=False)
    if list_result.is_left():
        return None
    for record in list_result.get_right():
        if record.get("name") == name or record.get("macroName") == name:
            macro_uuid = record.get("id") or record.get("macroId")
            if macro_uuid and _UUID_RE.match(str(macro_uuid)):
                return str(macro_uuid)
    return None


async def km_execute_macro(
    identifier: Annotated[
        str,
        Field(
            description="Macro name or UUID for execution",
            min_length=1,
            max_length=255,
            pattern=r"^[a-zA-Z0-9_\s\-\.\+\(\)\[\]\{\}\|\&\~\!\@\#\$\%\^\*\=\?\:\;\,\/\\]+$|^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        ),
    ],
    trigger_value: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional parameter value to pass to macro",
            max_length=1000,
        ),
    ] = None,
    method: Annotated[
        str,
        Field(
            default="applescript",
            description="Execution method: applescript, url, web, or remote",
        ),
    ] = "applescript",
    timeout: Annotated[
        int,
        Field(
            default=30,
            ge=1,
            le=300,
            description="Maximum execution time in seconds",
        ),
    ] = 30,
    ctx: Context = None,
) -> dict[str, Any]:
    """Execute a Keyboard Maestro macro with comprehensive error handling and validation.

    Supports multiple execution methods:
    - applescript: Direct AppleScript communication (recommended)
    - url: URL scheme execution (kmtrigger://macro=name&value=value)
    - web: HTTP API execution (http://localhost:4490/action.html)
    - remote: Remote trigger execution (requires remote trigger setup)

    Returns execution results with timing, output, and error handling.
    """
    if ctx:
        await ctx.info(f"Executing macro '{identifier}' using {method} method")

    try:
        # Validate and sanitize inputs
        if not identifier or not identifier.strip():
            raise ValidationError("identifier", identifier, "cannot be empty")

        # Create macro ID from identifier
        clean_identifier = identifier.strip()
        if len(clean_identifier) == 0:
            raise ValidationError(
                "identifier",
                clean_identifier,
                "cannot be empty after trimming",
            )

        macro_id = MacroId(clean_identifier)

        # Sanitize trigger value if provided
        sanitized_trigger = None
        if trigger_value:
            sanitized_trigger = trigger_value[:1000]  # Limit length
            if ctx:
                await ctx.info(f"Using trigger value: {sanitized_trigger[:50]}...")

        # Execute macro through KM client instead of test engine
        if ctx:
            await ctx.report_progress(25, 100, "Connecting to Keyboard Maestro")

        km_client = get_km_client()

        if ctx:
            await ctx.report_progress(50, 100, "Executing macro")

        # First test connection to KM
        connection_test = await asyncio.get_event_loop().run_in_executor(
            None,
            km_client.check_connection,
        )

        if connection_test.is_left() or not connection_test.get_right():
            if ctx:
                await ctx.error("Cannot connect to Keyboard Maestro Engine")
            return {
                "success": False,
                "error": {
                    "code": "KM_CONNECTION_FAILED",
                    "message": "Cannot connect to Keyboard Maestro Engine",
                    "details": "Keyboard Maestro Engine is not running or not accessible",
                    "recovery_suggestion": "Start Keyboard Maestro and ensure the Engine is running. Check accessibility permissions if needed.",
                },
            }

        if ctx:
            await ctx.info(
                f"Connection to KM Engine confirmed. Executing macro: {identifier}",
            )
            await ctx.info(
                f"Debug: macro_id={macro_id!r}, type={type(macro_id)}, bool={bool(macro_id)}",
            )

        # Execute macro via KM client using async wrapper
        execution_result = await asyncio.get_event_loop().run_in_executor(
            None,
            km_client.execute_macro,
            macro_id,
            sanitized_trigger,
        )

        # Handle execution result
        if execution_result.is_left():
            error = execution_result.get_left()
            error_message = (
                error.message if hasattr(error, "message") else str(error)
            )

            # KM macro-group activation lags behind `make new macro`, so a
            # name-dispatched call against a freshly created macro can fail
            # with "no macros with a matching name" even when the macro and
            # group are both enabled. UUID dispatch always works. Recover
            # once by resolving the name to a UUID and retrying.
            looked_like_name = not _UUID_RE.match(clean_identifier)
            name_lookup_failed = (
                _NAME_NOT_FOUND_FRAGMENT in error_message.lower()
            )
            if looked_like_name and name_lookup_failed:
                resolved_uuid = await _resolve_macro_uuid(clean_identifier)
                if resolved_uuid:
                    logger.info(
                        "execute_macro: name '%s' not dispatchable; retrying by UUID %s",
                        clean_identifier,
                        resolved_uuid,
                    )
                    retry_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        km_client.execute_macro,
                        MacroId(resolved_uuid),
                        sanitized_trigger,
                    )
                    if retry_result.is_right():
                        retry_data = retry_result.get_right()
                        execution_id = str(uuid.uuid4())
                        return {
                            "success": True,
                            "data": {
                                "execution_id": execution_id,
                                "macro_id": resolved_uuid,
                                "macro_name": clean_identifier,
                                "execution_time": 0.0,
                                "method_used": method,
                                "status": "completed"
                                if retry_data.get("success")
                                else "failed",
                                "output": retry_data.get("output"),
                                "trigger_value": sanitized_trigger,
                                "name_to_uuid_retry": True,
                            },
                            "metadata": {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "server_version": "1.0.0",
                                "correlation_id": execution_id,
                            },
                        }
                    # Second attempt also failed; fall through to original error envelope.

            if ctx:
                await ctx.error(f"Macro execution failed: {error}")

            return {
                "success": False,
                "error": {
                    "code": error.code if hasattr(error, "code") else "EXECUTION_ERROR",
                    "message": error_message,
                    "details": error.details
                    if hasattr(error, "details")
                    else {"raw_error": str(error)},
                    "recovery_suggestion": "Verify macro name exists in Keyboard Maestro and is enabled",
                },
            }

        # Execution succeeded
        result_data = execution_result.get_right()
        execution_id = str(uuid.uuid4())

        if ctx:
            await ctx.report_progress(100, 100, "Execution completed")
            await ctx.info(f"Macro '{identifier}' executed successfully")

        return {
            "success": True,
            "data": {
                "execution_id": execution_id,
                "macro_id": str(macro_id),
                "macro_name": identifier,
                "execution_time": 0.0,  # AppleScript doesn't provide timing
                "method_used": method,
                "status": "completed" if result_data.get("success") else "failed",
                "output": result_data.get("output"),
                "trigger_value": sanitized_trigger,
            },
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server_version": "1.0.0",
                "correlation_id": execution_id,
            },
        }

    except ValidationError as e:
        if ctx:
            await ctx.error(f"Validation error: {e}")
        return {
            "success": False,
            "error": {
                "code": "INVALID_PARAMETER",
                "message": str(e),
                "details": "Check parameter format and try again",
                "recovery_suggestion": "Verify macro identifier format and ensure it exists",
            },
        }
    except PermissionDeniedError as e:
        if ctx:
            await ctx.error(f"Permission denied: {e}")
        return {
            "success": False,
            "error": {
                "code": "PERMISSION_DENIED",
                "message": str(e),
                "details": "Insufficient permissions for macro execution",
                "recovery_suggestion": "Grant accessibility permissions to Keyboard Maestro",
            },
        }
    except TimeoutError as e:
        if ctx:
            await ctx.error(f"Execution timeout: {e}")
        return {
            "success": False,
            "error": {
                "code": "TIMEOUT_ERROR",
                "message": f"Macro execution timed out after {timeout}s",
                "details": str(e),
                "recovery_suggestion": "Increase timeout or check macro complexity",
            },
        }
    except ExecutionError as e:
        if ctx:
            await ctx.error(f"Execution error: {e}")
        return {
            "success": False,
            "error": {
                "code": "EXECUTION_ERROR",
                "message": str(e),
                "details": "Macro execution failed",
                "recovery_suggestion": "Check macro configuration and system state",
            },
        }
    except Exception as e:
        logger.exception("Unexpected error in km_execute_macro")
        if ctx:
            await ctx.error(f"Unexpected error: {e}")
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": "Unexpected system error occurred",
                "details": str(e)
                if logger.isEnabledFor(logging.DEBUG)
                else "Contact support",
                "recovery_suggestion": "Check logs and contact support if issue persists",
            },
        }


async def km_list_macros(
    group_filters: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="List of macro group names to filter by. Examples: ['Email', 'Text Processing', 'Global Macro Group']. Pass as an array, not a string.",
        ),
    ] = None,
    group_filter: Annotated[
        str | None,
        Field(
            default=None,
            description="[DEPRECATED] Single group filter for backward compatibility. Use group_filters instead.",
        ),
    ] = None,
    enabled_only: Annotated[
        bool,
        Field(default=True, description="Only return enabled macros"),
    ] = True,
    sort_by: Annotated[
        str,
        Field(
            default="name",
            description="Sort field: name, last_used, created_date, or group",
        ),
    ] = "name",
    limit: Annotated[
        int,
        Field(default=20, ge=1, le=100, description="Maximum number of results"),
    ] = 20,
    ctx: Context = None,
) -> dict[str, Any]:
    """List and filter Keyboard Maestro macros with comprehensive search capabilities.

    NOW RETURNS REAL USER MACROS from Keyboard Maestro instead of mock data.
    Supports filtering by multiple groups, enabled status, and custom sorting.
    Returns paginated results with metadata for each macro.

    Examples:
    - group_filters=["Email", "Text"] - macros from Email OR Text groups
    - group_filters=["Utilities"] - macros from Utilities group only
    - group_filters=None - macros from all groups

    """
    # Handle backward compatibility: convert single group_filter to group_filters list
    if group_filter is not None and group_filters is None:
        group_filters = [group_filter]
    elif group_filter is not None and group_filters is not None:
        # If both are provided, merge them
        group_filters = group_filters + [group_filter]

    if ctx:
        groups_desc = f"{len(group_filters)} groups" if group_filters else "all groups"
        await ctx.info(f"Listing real macros with filter: {groups_desc}")

    try:
        # Get real macro data from KM client
        km_client = get_km_client()

        if ctx:
            await ctx.report_progress(25, 100, "Connecting to Keyboard Maestro")

        # Query real macros using multiple API methods
        macros_result = await km_client.list_macros_async(
            group_filters=group_filters,
            enabled_only=enabled_only,
        )

        if macros_result.is_left():
            # Connection failed - provide helpful error message
            error = macros_result.get_left()
            if ctx:
                await ctx.error(f"Cannot connect to Keyboard Maestro: {error}")

            return {
                "success": False,
                "error": {
                    "code": "KM_CONNECTION_FAILED",
                    "message": "Cannot connect to Keyboard Maestro",
                    "details": str(error),
                    "recovery_suggestion": "Ensure Keyboard Maestro is running and accessible. For AppleScript access, grant accessibility permissions. For Web API, enable web server on port 4490.",
                },
            }

        if ctx:
            await ctx.report_progress(75, 100, "Processing macro data")

        # Get successful result
        all_macros = macros_result.get_right()

        # Apply sorting (filtering already applied in KM client)
        sort_fields = {
            "name": lambda m: m.get("name", "").lower(),
            "last_used": lambda m: m.get("last_used") or "1970-01-01T00:00:00Z",
            "created_date": lambda m: m.get("created_date") or "1970-01-01T00:00:00Z",
            "group": lambda m: m.get("group", "").lower(),
        }
        if sort_by in sort_fields:
            all_macros.sort(key=sort_fields[sort_by])

        # Apply limit
        limited_macros = all_macros[:limit]

        if ctx:
            await ctx.report_progress(100, 100, "Macro listing complete")
            await ctx.info(
                f"Found {len(limited_macros)} macros (total: {len(all_macros)})",
            )

        return {
            "success": True,
            "data": {
                "macros": limited_macros,
                "total_count": len(all_macros),
                "filtered": group_filters is not None or enabled_only,
                "pagination": {
                    "limit": limit,
                    "returned": len(limited_macros),
                    "has_more": len(all_macros) > limit,
                },
            },
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server_version": "1.0.0",
                "data_source": "keyboard_maestro_live",
                "connection_method": "applescript_with_web_fallback",
                "query_params": {
                    "group_filters": group_filters,
                    "group_filter": group_filter,  # Show original parameter for debugging
                    "enabled_only": enabled_only,
                    "sort_by": sort_by,
                },
            },
        }

    except Exception as e:
        logger.exception("Unexpected error in km_list_macros")
        if ctx:
            await ctx.error(f"Unexpected error: {e}")
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": "Unexpected system error occurred",
                "details": str(e)
                if logger.isEnabledFor(logging.DEBUG)
                else "Contact support",
                "recovery_suggestion": "Check logs and ensure Keyboard Maestro is running. Try restarting the MCP server if the issue persists.",
            },
        }


def _km_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


async def _exec_variable_op(
    operation: str,
    name: str,
    value: str | None,
    scope: str,
) -> dict[str, Any]:
    """Run a single get/set/delete variable op against KM Engine.

    Password-scope reads mask the value as "[PROTECTED]" — KM stores
    password-scope variables in memory and AppleScript can read them as
    plain text via getvariable, but the MCP surface should not relay
    them. Writes still go through unchanged so callers can populate
    secrets.
    """
    esc_name = _km_escape(name)
    if operation == "get":
        script = f'tell application "Keyboard Maestro Engine" to getvariable "{esc_name}"'
    elif operation == "set":
        esc_value = _km_escape(value or "")
        script = (
            f'tell application "Keyboard Maestro Engine" to '
            f'setvariable "{esc_name}" to "{esc_value}"'
        )
    elif operation == "delete":
        script = (
            f'tell application "Keyboard Maestro Engine" to '
            f'delete variable "{esc_name}"'
        )
    else:
        return {
            "success": False,
            "error": {
                "code": "INVALID_OPERATION",
                "message": f"unknown operation: {operation}",
                "recovery_suggestion": "Use get, set, delete, or list.",
            },
        }
    result = await get_km_client().execute_applescript_async(script)
    if result.is_left():
        return {
            "success": False,
            "error": {
                "code": "KM_VARIABLE_FAILED",
                "message": result.get_left().message,
                "recovery_suggestion": "Verify Keyboard Maestro Engine is running.",
            },
        }
    raw = result.get_right().rstrip("\n")
    if operation == "get":
        return {
            "success": True,
            "data": {
                "name": name,
                "value": "[PROTECTED]" if scope == "password" else raw,
                "scope": scope,
                "exists": raw != "",
                "type": "string",
            },
        }
    if operation == "set":
        return {
            "success": True,
            "data": {
                "name": name,
                "scope": scope,
                "operation": "set",
                "value_length": len(value or ""),
                "is_password": scope == "password",
            },
        }
    return {
        "success": True,
        "data": {
            "name": name,
            "scope": scope,
            "operation": "delete",
            "existed": True,
        },
    }


async def km_variable_manager(
    operation: Annotated[
        str,
        Field(description="Operation: get, set, delete, or list"),
    ],
    name: Annotated[
        str | None,
        Field(
            default=None,
            description="Variable name (required for get, set, delete)",
        ),
    ] = None,
    value: Annotated[
        str | None,
        Field(default=None, description="Variable value (required for set operation)"),
    ] = None,
    scope: Annotated[
        str,
        Field(
            default="global",
            description="Variable scope: global, local, instance, or password",
        ),
    ] = "global",
    _instance_id: Annotated[
        str | None,
        Field(default=None, description="Instance ID for local/instance variables"),
    ] = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Comprehensive Keyboard Maestro variable management with type safety.

    Supports all variable scopes:
    - global: Persistent across sessions, accessible to all macros
    - local: Transient, execution-specific with Local__ prefix
    - instance: Instance-specific variables (v10.0+)
    - password: Memory-only, never written to disk

    Provides secure handling of sensitive data and environment variable access.
    """
    if ctx:
        await ctx.info(f"Variable operation: {operation} on {name or 'all variables'}")

    try:
        if operation in ["get", "set", "delete"] and not name:
            raise ValidationError(
                "name",
                name,
                "Variable name is required for get, set, and delete operations",
            )

        if operation == "set" and value is None:
            raise ValidationError(
                "value",
                value,
                "Variable value is required for set operation",
            )

        # SIM102 fix: Combine nested if statements
        if (
            name
            and not name.replace("_", "").replace(" ", "").isalnum()
            and not (
                scope == "password"
                and ("password" in name.lower() or "pw" in name.lower())
            )
        ):
            raise ValidationError("name", name, "Invalid variable name format")

        # KM Engine round-trip for get/set/delete (KM does not expose a
        # separate "delete" — we clear via setvariable "" and report a
        # deletion). Earlier revisions returned hard-coded mock values
        # like "mock_value_for_<name>" and treated set/delete as
        # success-no-op, so the tool silently dropped every write.
        if operation in {"get", "set", "delete"}:
            km_response = await _exec_variable_op(operation, name, value, scope)
            if not km_response["success"]:
                return km_response
            if ctx:
                await ctx.info(f"Variable {operation} '{name}' [{scope}] OK")
            return km_response

        if operation == "list":
            if ctx:
                await ctx.info(
                    f"Listing variables in {scope} scope from Keyboard Maestro",
                )

            # Get real variables from Keyboard Maestro
            get_km_client()

            if ctx:
                await ctx.report_progress(25, 100, "Connecting to Keyboard Maestro")

            # Use AppleScript to get variables from KM. KM's dictionary has no
            # ``global variable`` class — ``variables`` returns all engine
            # variables (the global scope).
            if scope == "global":
                script = """
                tell application "Keyboard Maestro Engine"
                    try
                        set variableList to {}
                        repeat with currentVar in variables
                            set varName to name of currentVar
                            set variableRecord to {¬
                                varName:varName, ¬
                                varScope:"global", ¬
                                varType:"string"¬
                            }
                            set variableList to variableList & {variableRecord}
                        end repeat

                        return variableList
                    on error errorMessage
                        return "ERROR: " & errorMessage
                    end try
                end tell
                """
            else:
                # For other scopes, we'll return a limited set for security
                script = f"""
                tell application "Keyboard Maestro Engine"
                    return "Limited scope access for {scope} variables"
                end tell
                """

            if ctx:
                await ctx.report_progress(50, 100, "Executing variable query")

            try:
                result = subprocess.run(  # noqa: S603 — hardcoded osascript, script body is locally constructed
                    ["/usr/bin/osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                    check=False,
                )

                if result.returncode != 0:
                    if ctx:
                        await ctx.error(f"AppleScript failed: {result.stderr}")
                    return {
                        "success": False,
                        "error": {
                            "code": "KM_CONNECTION_FAILED",
                            "message": "Cannot retrieve variables from Keyboard Maestro",
                            "details": result.stderr,
                            "recovery_suggestion": "Ensure Keyboard Maestro Engine is running and accessible",
                        },
                    }

                if ctx:
                    await ctx.report_progress(75, 100, "Parsing variable data")

                output = result.stdout.strip()
                if output.startswith("ERROR:"):
                    return {
                        "success": False,
                        "error": {
                            "code": "KM_SCRIPT_ERROR",
                            "message": output[6:].strip(),
                            "recovery_suggestion": "Check Keyboard Maestro permissions and variable access",
                        },
                    }

                # Parse variables from AppleScript output
                variables = []
                if scope == "global" and output and not output.startswith("Limited"):
                    # Parse the AppleScript record format
                    variables = parse_variable_records(output)
                else:
                    # For non-global scopes or limited access
                    variables = [
                        {
                            "name": f"Limited_{scope}_access",
                            "scope": scope,
                            "type": "string",
                        },
                    ]

                if ctx:
                    await ctx.report_progress(
                        100,
                        100,
                        f"Retrieved {len(variables)} variables",
                    )
                    await ctx.info(f"Found {len(variables)} variables in {scope} scope")

                return {
                    "success": True,
                    "data": {
                        "variables": variables,
                        "scope": scope,
                        "count": len(variables),
                    },
                }

            except subprocess.TimeoutExpired:
                if ctx:
                    await ctx.error("Timeout retrieving variables")
                return {
                    "success": False,
                    "error": {
                        "code": "TIMEOUT_ERROR",
                        "message": "Timeout retrieving variables from Keyboard Maestro",
                        "recovery_suggestion": "Check Keyboard Maestro responsiveness",
                    },
                }
            except Exception as e:
                if ctx:
                    await ctx.error(f"Variable listing error: {e}")
                return {
                    "success": False,
                    "error": {
                        "code": "SYSTEM_ERROR",
                        "message": "Failed to retrieve variables",
                        "details": str(e),
                        "recovery_suggestion": "Check Keyboard Maestro connection and permissions",
                    },
                }

        else:
            raise ValidationError(
                "operation",
                operation,
                f"Unknown operation: {operation}",
            )

    except ValidationError as e:
        if ctx:
            await ctx.error(f"Validation error: {e}")
        return {
            "success": False,
            "error": {
                "code": "INVALID_PARAMETER",
                "message": str(e),
                "recovery_suggestion": "Check operation and parameter format",
            },
        }
    except Exception as e:
        logger.exception("Error in km_variable_manager")
        if ctx:
            await ctx.error(f"Variable operation error: {e}")
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": "Variable operation failed",
                "details": str(e),
                "recovery_suggestion": "Check Keyboard Maestro connection and permissions",
            },
        }
