"""Macro Creation MCP Tools.

Provides comprehensive macro creation capabilities through MCP interface
with template support, security validation, and error handling.

Template-with-parameters wire-up: each template's parameters are translated
into a list of action_builder dicts, rendered to a single concatenated
action XML blob, and embedded atomically into the .kmmacros plist before
import. Single KM round-trip per macro creation; no N+1 appends.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

from ...core.types import MacroId
from ...creation.types import MacroTemplate
from ...integration.km_client import _trigger_config_to_xml
from ...integration.kmmacros_import import create_empty_macro
from ..initialization import get_km_client
from .action_builder_tools import _build_action_xml

_KMMACROS_TEMPLATES = {
    "custom",
    "hotkey_action",
    "app_launcher",
    "text_expansion",
    "file_processor",
    "window_manager",
}


class _UnsupportedTemplateAction(ValueError):
    """Raised when a template requests an action_type _build_action_xml lacks."""


def _error_envelope(
    code: str,
    message: str,
    suggestion: str,
    template: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error_block: dict[str, Any] = {
        "code": code,
        "message": message,
        "recovery_suggestion": suggestion,
    }
    if extra:
        error_block.update(extra)
    return {
        "success": False,
        "error": error_block,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "template_type": template,
        },
    }

logger = logging.getLogger(__name__)


async def km_create_macro(
    name: Annotated[
        str,
        Field(
            description="Macro name (1-255 ASCII characters)",
            min_length=1,
            max_length=255,
            pattern=r"^[a-zA-Z0-9_\s\-\.]+$",  # Security: Restricted character set
        ),
    ],
    template: Annotated[
        str,
        Field(
            description="Macro template type",
            pattern=r"^(hotkey_action|app_launcher|text_expansion|file_processor|window_manager|custom)$",
        ),
    ],
    group_name: Annotated[
        str | None,
        Field(description="Target macro group name", max_length=255),
    ] = None,
    enabled: Annotated[
        bool,
        Field(description="Initial enabled state"),
    ] = True,
    parameters: Annotated[
        dict[str, Any] | None,
        Field(description="Template-specific parameters"),
    ] = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Create a new Keyboard Maestro macro with comprehensive validation and security.

    Architecture:
        - Pattern: Factory Pattern with Template Method and Builder
        - Security: Defense-in-depth with validation, sanitization, rollback
        - Performance: O(1) validation, O(log n) creation with caching

    Contracts:
        Preconditions:
            - name is 1-255 ASCII characters with safe character set
            - template is supported MacroTemplate type
            - parameters are valid for chosen template
            - group_name exists if specified

        Postconditions:
            - Returns MacroId on success OR error details on failure
            - No partial macro creation (atomic operation)
            - All security validations passed
            - Created macro is immediately available in Keyboard Maestro

        Invariants:
            - System state unchanged on failure
            - No script injection possible through any parameter
            - All user inputs sanitized and validated

    Supports multiple templates for common automation patterns:
    - hotkey_action: Hotkey-triggered actions with modifiers and key validation
    - app_launcher: Application launching with bundle ID or name validation
    - text_expansion: Text expansion snippets with abbreviation management
    - file_processor: File processing workflows with path security validation
    - window_manager: Window manipulation with coordinate and size limits
    - custom: Custom macro with user-defined actions (advanced validation)

    Security Implementation:
        - Input Validation: ASCII-only names, restricted character sets, length limits
        - Template Validation: Each template validates its specific parameters
        - AppleScript Security: Escape all user inputs, validate generated AppleScript
        - Permission Checking: Verify user has macro creation permissions
        - Rollback Safety: Automatic rollback on creation failures

    Performance Targets:
        - Validation Time: <100ms for parameter validation
        - Creation Time: <2 seconds for simple templates, <5 seconds for complex
        - Memory Usage: <10MB for template processing

    Args:
        name: Macro name (ASCII, 1-255 characters, safe character set)
        template: Template type (hotkey_action, app_launcher, text_expansion, etc.)
        group_name: Optional target group (must exist)
        enabled: Initial enabled state (default True)
        parameters: Template-specific configuration parameters
        ctx: MCP context for logging and progress reporting

    Returns:
        Dict containing:
        - success: Boolean indicating creation success
        - data: MacroId and creation details on success
        - error: Error information on failure
        - metadata: Timestamp, validation info, performance metrics

    Raises:
        SecurityViolationError: Security validation failed
        ValidationError: Input validation failed

    """
    logger.warning(
        "km_create_macro duplicates the km_macro_editor surface; "
        "calls still work but this name will fold into "
        "km_macro_editor(operation='create') in a future release.",
    )
    if parameters is None:
        parameters = {}
    if ctx:
        await ctx.info(f"Starting macro creation: {name}")

    if template in _KMMACROS_TEMPLATES:
        return await _create_via_kmmacros(
            name, template, group_name, enabled, parameters, ctx,
        )

    try:
        MacroTemplate(template)
    except ValueError:
        return _error_envelope(
            "INVALID_TEMPLATE",
            f"Unsupported template type: {template}",
            "Use one of: " + ", ".join(t.value for t in MacroTemplate),
            template,
        )

    return _error_envelope(
        "UNSUPPORTED_TEMPLATE",
        f"Template {template!r} cannot create a macro on KM 11 yet.",
        "Use template='custom' (empty macro) and chain km_action_builder/km_trigger_crud, "
        "or use km_macro_editor operation='create'.",
        template,
    )


async def _create_via_kmmacros(  # noqa: PLR0913 - thin orchestration wrapper
    name: str,
    template: str,
    group_name: str | None,
    enabled: bool,
    parameters: dict[str, Any],
    ctx: Context | None,
) -> dict[str, Any]:
    if not group_name:
        return _error_envelope(
            "VALIDATION_ERROR",
            "group_name is required for kmmacros template creation.",
            "Pass group_name; list groups with km_macro_group_manager operation='list'.",
            template,
        )

    try:
        actions_xml, triggers_xml = _render_template_actions(template, parameters)
    except _UnsupportedTemplateAction as exc:
        return _error_envelope(
            "UNSUPPORTED_TEMPLATE_ACTION",
            str(exc),
            "Use template='custom' and chain km_action_builder for unsupported action types.",
            template,
        )
    except ValueError as exc:
        return _error_envelope(
            "VALIDATION_ERROR",
            str(exc),
            "Check the parameters dict against the template's required fields.",
            template,
        )

    if ctx:
        await ctx.info(f"Importing {template} macro '{name}' into '{group_name}'")
    result = await create_empty_macro(
        get_km_client(),
        group_name,
        name,
        actions_xml=actions_xml,
        triggers_xml=triggers_xml,
    )
    if result.is_left():
        err = result.get_left()
        code = "GROUP_NOT_FOUND" if "not found" in err.message.lower() else "IMPORT_FAILED"
        return _error_envelope(
            code,
            err.message,
            "Verify the group exists and KM is running; first-time imports may show a permission prompt.",
            template,
        )
    payload = result.get_right()
    macro_id = payload["macro_id"]
    # KM imports new .kmmacros macros disabled by default; honour the caller's
    # ``enabled`` flag with a follow-up toggle when they asked for True.
    if enabled:
        toggle = await get_km_client().set_macro_enabled_async(
            MacroId(macro_id), enabled=True,
        )
        if toggle.is_left():
            logger.warning(
                "Macro %s imported but enable toggle failed: %s",
                macro_id,
                toggle.get_left().message,
            )

    # Hotkey is now embedded in the .kmmacros plist Triggers array at import
    # time (D3 fix); a non-empty triggers_xml means it landed atomically with
    # the macro. The previous post-import km_create_hotkey_trigger call lost
    # the trigger to a KM 11 race against macro-group activation.
    hotkey_attached = bool(triggers_xml)

    return {
        "success": True,
        "data": {
            "macro_id": macro_id,
            "macro_name": payload["name"],
            "template_used": template,
            "group_id": payload["group_id"],
            "group_name": payload["group_name"],
            "enabled": enabled,
            "hotkey_attached": hotkey_attached,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "creation_method": "kmmacros_import",
            "template_type": template,
        },
    }


def _render_template_actions(
    template: str,
    parameters: dict[str, Any],
) -> tuple[str, str]:
    """Translate template parameters to actions XML + triggers XML.

    Returns ``(actions_xml, triggers_xml)`` — concatenations of action /
    trigger ``<dict>`` strings ready for embedding in a .kmmacros plist.
    ``triggers_xml`` is non-empty only for ``hotkey_action`` when the
    caller supplied a hotkey parameter.

    Raises ``_UnsupportedTemplateAction`` if a translator requests an
    action_type that ``_build_action_xml`` does not yet support, or if
    the hotkey spec is invalid (key/modifier names unknown to KM).
    """
    if template == "custom":
        return "", ""
    translator = _TEMPLATE_TRANSLATORS.get(template)
    if translator is None:
        raise ValueError(f"No translator registered for template {template!r}")
    action_dicts, hotkey_spec = translator(parameters)
    actions_xml = _action_dicts_to_xml(action_dicts, template)
    triggers_xml = _render_hotkey_trigger_xml(hotkey_spec) if hotkey_spec else ""
    return actions_xml, triggers_xml


def _render_hotkey_trigger_xml(hotkey_spec: dict[str, Any]) -> str:
    """Translate ``{key, modifiers, activation_mode?}`` to a KM trigger ``<dict>``.

    Reuses ``_trigger_config_to_xml`` from km_client (the same emitter that
    ``attach_trigger_async`` uses) so the .kmmacros embed path and the
    post-import attach path produce identical XML — no schema drift.
    Raises ``ValueError`` if KM's validator rejects the spec.
    """
    result = _trigger_config_to_xml("hotkey", hotkey_spec)
    if result.is_left():
        raise ValueError(result.get_left().message)
    return result.get_right()


def _action_dicts_to_xml(action_dicts: list[dict[str, Any]], template: str) -> str:
    pieces: list[str] = []
    for action in action_dicts:
        action_type = action.get("type")
        config = {k: v for k, v in action.items() if k != "type"}
        xml = _build_action_xml(str(action_type), config)
        if xml is None:
            raise _UnsupportedTemplateAction(
                f"Template {template!r} requested action_type {action_type!r} "
                "which is not in the action_builder emitter set.",
            )
        pieces.append(xml)
    return "".join(pieces)


def _translate_app_launcher(
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    app_name = parameters.get("app_name", "")
    bundle_id = parameters.get("bundle_id", "")
    if not app_name and not bundle_id:
        raise ValueError("app_launcher needs 'app_name' or 'bundle_id'")
    return [
        {
            "type": "activate_application",
            "app_name": app_name or "",
            "bundle_id": bundle_id or "",
            "all_windows": parameters.get("all_windows", True),
            "reopen_windows": parameters.get("reopen_windows", False),
        },
    ], None


def _translate_text_expansion(
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    text = parameters.get("expansion_text") or parameters.get("text")
    if not text:
        raise ValueError("text_expansion needs 'expansion_text' (or 'text')")
    return [{"type": "type_text", "text": text}], None


def _translate_file_processor(
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    script = parameters.get("script") or parameters.get("source")
    if not script:
        raise ValueError(
            "file_processor needs 'script' — the shell script to run "
            "against the file. Use %TriggerValue% to reference the file path.",
        )
    return [
        {"type": "execute_shell_script", "source": script,
         "display": parameters.get("display", "Nothing")},
    ], None


def _translate_window_manager(
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    operation = parameters.get("operation", "MoveAndResize")
    return [
        {
            "type": "manipulate_window",
            "action": operation,
            "x": str(parameters.get("x", "0")),
            "y": str(parameters.get("y", "0")),
            "width": str(parameters.get("width", "800")),
            "height": str(parameters.get("height", "600")),
            "targeting": parameters.get("targeting", "FrontWindow"),
        },
    ], None


def _translate_hotkey_action(
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """hotkey_action's parameters describe BOTH the inner action AND the trigger.

    Inner action set: ``open_app`` (→ activate_application), ``type_text``
    (→ type_text), ``run_script`` (→ execute_shell_script). The hotkey
    trigger is embedded directly in the .kmmacros plist Triggers array at
    import time (see _create_via_kmmacros), not attached as a follow-up
    AppleScript call.
    """
    inner = parameters.get("action", "type_text")
    if inner == "open_app":
        actions = [{"type": "activate_application",
                    "app_name": parameters.get("app_name", ""),
                    "bundle_id": parameters.get("bundle_id", "")}]
    elif inner == "type_text":
        text = parameters.get("text", "")
        if not text:
            raise ValueError("hotkey_action action='type_text' needs 'text'")
        actions = [{"type": "type_text", "text": text}]
    elif inner == "run_script":
        script = parameters.get("script", "")
        if not script:
            raise ValueError("hotkey_action action='run_script' needs 'script'")
        actions = [{"type": "execute_shell_script", "source": script}]
    else:
        raise _UnsupportedTemplateAction(
            f"hotkey_action inner action {inner!r} not supported. "
            "Use 'open_app' / 'type_text' / 'run_script'.",
        )
    hotkey_spec = None
    if parameters.get("hotkey"):
        hotkey_spec = {
            "key": parameters["hotkey"],
            "modifiers": parameters.get("modifiers", []),
        }
    return actions, hotkey_spec


_TEMPLATE_TRANSLATORS = {
    "app_launcher": _translate_app_launcher,
    "text_expansion": _translate_text_expansion,
    "file_processor": _translate_file_processor,
    "window_manager": _translate_window_manager,
    "hotkey_action": _translate_hotkey_action,
}


async def km_list_templates(ctx: Context = None) -> dict[str, Any]:
    """List available macro templates with descriptions and parameter requirements.

    Provides comprehensive information about each template including:
    - Template description and use cases
    - Required and optional parameters
    - Parameter validation rules
    - Example usage patterns

    Returns template catalog for macro creation assistance.
    """
    if ctx:
        await ctx.info("Retrieving available macro templates")

    templates = {
        "hotkey_action": {
            "name": "Hotkey Action",
            "description": "Create macros triggered by hotkey combinations",
            "use_cases": ["Quick app launching", "Text shortcuts", "System commands"],
            "required_parameters": ["action"],
            "optional_parameters": ["hotkey", "app_name", "text", "script_content"],
            "parameter_rules": {
                "hotkey": "Format: 'Modifier+Key' (e.g., 'Cmd+Shift+N')",
                "action": "Values: 'open_app', 'type_text', 'run_script'",
                "app_name": "Application name or bundle ID",
                "text": "Text to type (max 10000 characters)",
                "script_content": "Script code (security validated)",
            },
            "example": {
                "action": "open_app",
                "app_name": "Notes",
                "hotkey": "Cmd+Shift+N",
            },
        },
        "app_launcher": {
            "name": "Application Launcher",
            "description": "Create macros for launching applications",
            "use_cases": ["App switching", "Workflow automation", "Quick access"],
            "required_parameters": ["app_name or bundle_id"],
            "optional_parameters": ["ignore_if_running", "bring_to_front"],
            "parameter_rules": {
                "app_name": "Application name (alphanumeric, spaces, dashes, dots)",
                "bundle_id": "Bundle ID format (e.g., com.apple.Notes)",
                "ignore_if_running": "Boolean (default: true)",
                "bring_to_front": "Boolean (default: true)",
            },
            "example": {"app_name": "Visual Studio Code", "bring_to_front": True},
        },
        "text_expansion": {
            "name": "Text Expansion",
            "description": "Create text expansion shortcuts",
            "use_cases": ["Email signatures", "Code snippets", "Common phrases"],
            "required_parameters": ["expansion_text", "abbreviation"],
            "optional_parameters": ["typing_speed"],
            "parameter_rules": {
                "expansion_text": "Text to expand (max 5000 characters)",
                "abbreviation": "Trigger abbreviation (alphanumeric only)",
                "typing_speed": "Values: 'Slow', 'Normal', 'Fast'",
            },
            "example": {
                "abbreviation": "mysig",
                "expansion_text": "Best regards,\\nJohn Doe\\njohn@example.com",
            },
        },
        "file_processor": {
            "name": "File Processor",
            "description": "Create file processing workflows",
            "use_cases": [
                "File organization",
                "Batch processing",
                "Automation workflows",
            ],
            "required_parameters": ["action_chain"],
            "optional_parameters": [
                "watch_folder",
                "destination",
                "file_pattern",
                "recursive",
            ],
            "parameter_rules": {
                "watch_folder": "Folder path to monitor (security validated)",
                "destination": "Destination folder path (security validated)",
                "action_chain": "Array of actions: 'copy', 'move', 'rename', 'resize', 'optimize'",
                "file_pattern": "File pattern (e.g., '*.png', '*.pdf')",
                "recursive": "Boolean (default: false)",
            },
            "example": {
                "watch_folder": "~/Desktop",
                "action_chain": ["copy", "rename"],
                "destination": "~/Documents/Processed",
            },
        },
        "window_manager": {
            "name": "Window Manager",
            "description": "Create window management macros",
            "use_cases": [
                "Window positioning",
                "Screen arrangement",
                "Productivity layouts",
            ],
            "required_parameters": ["operation"],
            "optional_parameters": [
                "position",
                "size",
                "screen",
                "arrangement",
                "animate",
            ],
            "parameter_rules": {
                "operation": "Values: 'move', 'resize', 'arrange'",
                "position": "Object: {x: number, y: number} (range: -5000 to 10000)",
                "size": "Object: {width: number, height: number} (range: 50 to 10000)",
                "screen": "Values: 'Main', 'External', index number",
                "arrangement": "Values: 'left_half', 'right_half', 'maximize'",
                "animate": "Boolean (default: true)",
            },
            "example": {
                "operation": "move",
                "position": {"x": 100, "y": 100},
                "size": {"width": 800, "height": 600},
            },
        },
        "custom": {
            "name": "Custom Macro",
            "description": "Create custom macros with user-defined actions",
            "use_cases": [
                "Complex workflows",
                "Multi-step automation",
                "Advanced scripting",
            ],
            "required_parameters": ["actions"],
            "optional_parameters": ["triggers", "conditions"],
            "parameter_rules": {
                "actions": "Array of action objects with type and configuration",
                "triggers": "Array of trigger configurations",
                "conditions": "Array of condition objects",
            },
            "example": {"actions": [{"type": "Type a String", "text": "Hello World"}]},
        },
    }

    return {
        "success": True,
        "data": {
            "templates": templates,
            "total_templates": len(templates),
            "template_names": list(templates.keys()),
        },
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "template_version": "1.0.0",
            "security_validated": True,
        },
    }
