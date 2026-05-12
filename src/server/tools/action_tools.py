"""Action Building MCP Tools.

Provides comprehensive action building functionality for programmatic
macro construction with security validation and XML generation.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

from ...actions import ActionBuilder, ActionCategory, ActionRegistry
from ...core import (
    Duration,
    PermissionDeniedError,
    ValidationError,
)
from ..initialization import get_km_client

logger = logging.getLogger(__name__)


async def km_add_action(
    macro_id: Annotated[
        str,
        Field(
            description="Target macro UUID or name for adding the action",
            min_length=1,
            max_length=255,
            examples=["My Macro", "550e8400-e29b-41d4-a716-446655440000"],
        ),
    ],
    action_type: Annotated[
        str,
        Field(
            description="Keyboard Maestro action type identifier",
            min_length=1,
            max_length=100,
            pattern=r"^[a-zA-Z0-9_\s\-\.]+$",
            examples=[
                "Type a String",
                "Pause",
                "If Then Else",
                "Activate a Specific Application",
            ],
        ),
    ],
    action_config: Annotated[
        dict[str, Any],
        Field(
            description="Action-specific configuration parameters as key-value pairs",
            examples=[
                {"text": "Hello World", "by_typing": True},
                {"duration": 2.5},
                {"application": "Safari", "bring_all_windows": False},
            ],
        ),
    ],
    position: Annotated[
        int | None,
        Field(
            default=None,
            description="Position in action list (0-based index, None to append at end)",
            ge=0,
            le=1000,
            examples=[0, 5, None],
        ),
    ] = None,
    timeout: Annotated[
        int | None,
        Field(
            default=None,
            description="Action timeout in seconds (None for no timeout)",
            ge=1,
            le=3600,
            examples=[30, 60, None],
        ),
    ] = None,
    enabled: Annotated[
        bool,
        Field(
            default=True,
            description="Whether the action should be enabled when added",
            examples=[True, False],
        ),
    ] = True,
    abort_on_failure: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to abort the entire macro if this action fails",
            examples=[False, True],
        ),
    ] = False,
    ctx: Context = None,
) -> dict[str, Any]:
    """Add action to existing Keyboard Maestro macro with comprehensive validation.

    This tool enables programmatic construction of complex macros by adding individual
    actions with proper XML generation, security validation, and parameter checking.

    Supports 80+ Keyboard Maestro action types across categories:

    **Text Actions**: Type text, search/replace, format text, change case
    **Application Actions**: Launch apps, select menus, control windows, quit applications
    **System Actions**: Pause, beep, play sounds, execute scripts, display dialogs
    **Variable Actions**: Set/get variables, calculations, dictionary operations
    **Control Flow**: If/then/else, loops, switch statements, macro execution
    **Interface Actions**: Click images/coordinates, type keystrokes, OCR, mouse control
    **File Actions**: Copy/move/delete files, read/write files, folder operations
    **Web Actions**: HTTP requests, download files, form submission, screenshots
    **Clipboard Actions**: Copy/paste, clipboard history, named clipboards
    **Window Actions**: Move/resize windows, window arrangement, get window info
    **Sound Actions**: Text-to-speech, volume control, audio playback/recording
    **Calculation Actions**: Math expressions, date/time operations, unit conversion

    Features comprehensive security validation including:
    - XML injection prevention with pattern detection
    - Parameter sanitization and validation
    - Action type whitelist enforcement
    - Input size limits and format checking
    - Safe XML generation with proper escaping

    Returns detailed results including XML preview, validation status, and error handling.
    """
    correlation_id = str(uuid.uuid4())
    start_time = datetime.now(UTC)

    if ctx:
        await ctx.info(
            f"Adding action '{action_type}' to macro '{macro_id}' [ID: {correlation_id}]",
        )

    try:
        # Initialize action registry and builder
        action_registry = ActionRegistry()
        action_builder = ActionBuilder(action_registry)

        if ctx:
            await ctx.report_progress(10, 100, "Validating action type and parameters")

        # Validate action type exists
        action_def = action_registry.get_action_type(action_type)
        if not action_def:
            available_actions = action_registry.list_action_names()[
                :10
            ]  # Show first 10
            total_actions = action_registry.get_action_count()

            raise ValidationError(
                field_name="action_type",
                value=action_type,
                constraint=f"Must be one of the available action types. Available: {', '.join(available_actions)}... ({total_actions} total actions)",
            )

        # Validate action parameters
        param_validation = action_registry.validate_action_parameters(
            action_type,
            action_config,
        )
        if not param_validation["valid"]:
            error_details = []
            if param_validation["missing_required"]:
                error_details.append(
                    f"Missing required parameters: {param_validation['missing_required']}",
                )
            if param_validation["unknown_params"]:
                error_details.append(
                    f"Unknown parameters: {param_validation['unknown_params']}",
                )

            raise ValidationError(
                field_name="action_config",
                value=action_config,
                constraint=f"Required parameters: {action_def.required_params}; {'; '.join(error_details)}",
            )

        if ctx:
            await ctx.report_progress(30, 100, "Building action configuration")

        # Create action timeout if specified
        action_timeout = Duration.from_seconds(timeout) if timeout else None

        # Add action to builder with validation
        try:
            action_builder.add_action(
                action_type=action_type,
                parameters=action_config,
                position=position,
                enabled=enabled,
                timeout=action_timeout,
                abort_on_failure=abort_on_failure,
            )
        except Exception as e:
            raise ValidationError(
                field_name="action_configuration",
                value=action_config,
                constraint=f"Action builder configuration must be valid: {e!s}",
            ) from e

        if ctx:
            await ctx.report_progress(50, 100, "Generating and validating XML")

        # Generate XML with security validation
        xml_result = action_builder.build_xml()
        if not xml_result["success"]:
            raise ValidationError(
                field_name="xml_generation",
                value=xml_result,
                constraint=f"XML generation must succeed: {xml_result['error']}",
            )

        action_xml = xml_result["xml"]

        if ctx:
            await ctx.report_progress(70, 100, "Integrating with Keyboard Maestro")

        km_client = get_km_client()
        from ...core.types import MacroId

        append_result = await km_client.append_macro_action_async(
            MacroId(macro_id),
            action_xml,
        )
        if append_result.is_left():
            km_error = append_result.get_left()
            logger.warning(
                "km_add_action append failed",
                extra={
                    "correlation_id": correlation_id,
                    "macro_id": macro_id,
                    "action_type": action_type,
                    "km_error": km_error.message,
                },
            )
            return {
                "success": False,
                "error": {
                    "code": "APPEND_FAILED",
                    "message": km_error.message,
                    "details": {
                        "macro_id": macro_id,
                        "action_type": action_type,
                    },
                    "recovery_suggestion": (
                        "Verify the macro exists. Note: km_add_action's 146-type "
                        "catalog ships KM-incompatible XML for most types; use "
                        "km_action_builder for v1 types (pause, type_text, paste, "
                        "set_variable, run_applescript, execute_macro)."
                    ),
                    "error_id": str(uuid.uuid4()),
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "correlation_id": correlation_id,
                },
            }

        if ctx:
            await ctx.report_progress(100, 100, "Action added successfully")

        execution_time = (datetime.now(UTC) - start_time).total_seconds()

        return {
            "success": True,
            "data": {
                "action_added": {
                    "action_type": action_type,
                    "category": action_def.category.value,
                    "macro_id": macro_id,
                    "position": position
                    if position is not None
                    else len(action_builder.actions) - 1,
                    "enabled": enabled,
                    "timeout": timeout,
                    "abort_on_failure": abort_on_failure,
                    "parameter_count": len(action_config),
                    "required_params": action_def.required_params,
                    "optional_params": action_def.optional_params,
                },
                "xml_preview": _truncate_xml_for_preview(action_xml),
                "validation": {
                    "xml_validated": True,
                    "parameters_validated": True,
                    "security_passed": True,
                    "action_count_in_builder": action_builder.get_action_count(),
                },
                "integration": {
                    "km_client_status": "connected",
                    "macro_exists": True,
                    "action_inserted": True,
                },
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "correlation_id": correlation_id,
                "execution_time_seconds": execution_time,
                "server_version": "1.0.0",
                "action_registry_size": action_registry.get_action_count(),
            },
        }

    except ValidationError as e:
        logger.warning(
            f"Validation error in km_add_action: {e!s} [ID: {correlation_id}]",
        )
        return {
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": str(e),
                "details": {
                    "action_type": action_type,
                    "macro_id": macro_id,
                    "position": position,
                    "validation_stage": "parameter_validation",
                },
                "recovery_suggestion": "Check action type spelling and ensure all required parameters are provided",
                "error_id": str(uuid.uuid4()),
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "correlation_id": correlation_id,
            },
        }

    except PermissionDeniedError as e:
        logger.error(
            f"Permission denied in km_add_action: {e!s} [ID: {correlation_id}]",
        )
        return {
            "success": False,
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "Insufficient permissions for action addition",
                "details": {
                    "required_permissions": ["macro_modification"],
                    "action_type": action_type,
                    "macro_id": macro_id,
                },
                "recovery_suggestion": "Grant Keyboard Maestro accessibility permissions",
                "error_id": str(uuid.uuid4()),
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "correlation_id": correlation_id,
            },
        }

    except Exception as e:
        logger.error(
            f"Unexpected error in km_add_action: {e!s} [ID: {correlation_id}]",
        )
        return {
            "success": False,
            "error": {
                "code": "ACTION_ADDITION_ERROR",
                "message": f"Failed to add action to macro: {e!s}",
                "details": {
                    "action_type": action_type,
                    "macro_id": macro_id,
                    "position": position,
                    "error_type": type(e).__name__,
                },
                "recovery_suggestion": "Check macro exists and Keyboard Maestro is running",
                "error_id": str(uuid.uuid4()),
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "correlation_id": correlation_id,
            },
        }


async def km_list_action_types(
    category: Annotated[
        str | None,
        Field(
            default=None,
            description="Filter by action category (text, application, system, etc.)",
            examples=["text", "application", "system", "control", None],
        ),
    ] = None,
    search: Annotated[
        str | None,
        Field(
            default=None,
            description="Search term to filter action types by name or description",
            min_length=1,
            max_length=100,
            examples=["type", "pause", "application", None],
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(
            default=50,
            description="Maximum number of action types to return",
            ge=1,
            le=200,
            examples=[10, 50, 100],
        ),
    ] = 50,
    ctx: Context = None,
) -> dict[str, Any]:
    """List available Keyboard Maestro action types with filtering and search.

    Returns comprehensive information about supported action types including
    categories, required/optional parameters, and descriptions. Useful for
    discovering available actions before using km_add_action.

    Provides detailed metadata for each action type to help with parameter
    configuration and understanding action capabilities.
    """
    correlation_id = str(uuid.uuid4())

    if ctx:
        await ctx.info(
            f"Listing action types [category: {category}, search: {search}] [ID: {correlation_id}]",
        )

    try:
        action_registry = ActionRegistry()

        # Get actions based on filters
        if category:
            try:
                category_enum = ActionCategory(category.lower())
                actions = action_registry.get_actions_by_category(category_enum)
            except ValueError as e:
                valid_categories = [cat.value for cat in ActionCategory]
                raise ValidationError(
                    field_name="category",
                    value=category,
                    constraint=f"Must be one of: {valid_categories}",
                ) from e
        else:
            actions = action_registry.list_all_actions()

        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            actions = [
                action
                for action in actions
                if (
                    search_lower in action.identifier.lower()
                    or search_lower in action.description.lower()
                )
            ]

        # Apply limit
        total_found = len(actions)
        actions = actions[:limit]

        # Format results
        action_list = []
        for action in actions:
            action_info = {
                "identifier": action.identifier,
                "category": action.category.value,
                "description": action.description,
                "required_parameters": action.required_params,
                "optional_parameters": action.optional_params,
                "parameter_count": len(action.required_params)
                + len(action.optional_params),
            }
            action_list.append(action_info)

        # Get category statistics
        category_counts = action_registry.get_category_counts()
        category_stats = {cat.value: count for cat, count in category_counts.items()}

        return {
            "success": True,
            "data": {
                "actions": action_list,
                "summary": {
                    "total_available": action_registry.get_action_count(),
                    "total_found": total_found,
                    "returned": len(action_list),
                    "filtered_by_category": category,
                    "filtered_by_search": search,
                    "limit_applied": limit,
                },
                "categories": category_stats,
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "correlation_id": correlation_id,
                "registry_version": "1.0.0",
            },
        }

    except Exception as e:
        logger.error(f"Error listing action types: {e!s} [ID: {correlation_id}]")
        return {
            "success": False,
            "error": {
                "code": "ACTION_LIST_ERROR",
                "message": f"Failed to list action types: {e!s}",
                "details": {"category": category, "search": search, "limit": limit},
                "error_id": str(uuid.uuid4()),
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "correlation_id": correlation_id,
            },
        }


def _truncate_xml_for_preview(xml: str, max_length: int = 500) -> str:
    """Truncate XML for preview display."""
    if len(xml) <= max_length:
        return xml
    return xml[:max_length] + "..."
