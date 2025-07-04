#!/usr/bin/env python3
"""
Keyboard Maestro MCP Server - Main Entry Point

Advanced macOS automation through Model Context Protocol using FastMCP framework.
Provides 13 production-ready tools for comprehensive Keyboard Maestro integration.

Security: All operations include input validation, permission checking, and audit logging.
Performance: Sub-second response times with connection pooling and intelligent caching.
Type Safety: Complete branded type system with contract-driven development.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastmcp import FastMCP
from fastmcp.prompts import Message
from pydantic import Field
from typing_extensions import Annotated

# Import modular components
from .server.initialization import initialize_components, get_km_client
from .server.resources import get_server_status, get_tool_help, create_macro_prompt

# Configure logging to stderr to avoid corrupting MCP communications
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('logs/km-mcp-server.log') if Path('logs').exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server with comprehensive configuration
mcp = FastMCP(
    name="KeyboardMaestroMCP",
    instructions="""
This server provides comprehensive Keyboard Maestro automation capabilities through 46+ production-ready MCP tools.

CAPABILITIES:
- Macro execution and management with enterprise-grade error handling
- Variable and dictionary operations with type safety
- Advanced macro search and metadata analysis
- Real-time synchronization and monitoring
- Macro group management and organization
- Macro creation and template-based generation
- Comprehensive clipboard operations with history, named clipboards, and security validation
- Application control and window management
- Secure file operations with path validation, transaction safety, and rollback capability
- Macro movement between groups with validation and rollback
- Mathematical calculation engine with security validation and token support  
- Comprehensive token processing with variable substitution and security validation
- Trigger and condition management with functional programming patterns
- OCR and image recognition automation
- Complete plugin ecosystem with custom action creation, security sandboxing, and third-party integration
- Advanced audio/speech control with TTS and voice recognition
- Interface automation with mouse/keyboard simulation and UI interaction  
- Email/SMS integration with communication automation
- Web automation with HTTP/API integration and webhook support
- Remote triggers with URL schemes and external system integration
- Advanced window management with multi-monitor support and grid positioning
- Dictionary management with JSON processing and data transformation
- Enterprise audit logging with compliance reporting and regulatory support
- IoT device control and automation with multi-protocol support (MQTT, HTTP, Zigbee, Z-Wave)
- Smart home automation with scene management and scheduling
- Sensor monitoring with real-time alerts and data aggregation
- IoT workflow coordination with dependency management and fault tolerance

SECURITY: All operations include input validation, permission checking, and comprehensive audit logging.
PERFORMANCE: Sub-second response times with connection pooling and intelligent caching.
TYPE SAFETY: Complete branded type system with contract-driven development.

Use these tools to automate any macOS task that Keyboard Maestro can perform.
    """.strip()
)

# Register all MCP tools
@mcp.tool()
async def km_execute_macro(
    identifier: Annotated[str, Field(description="Macro name or UUID")],
    trigger_value: Annotated[str, Field(default="", description="Optional parameter to pass to macro")] = "",
    method: Annotated[str, Field(default="applescript", description="Execution method")] = "applescript",
    timeout: Annotated[int, Field(default=30, description="Maximum execution time in seconds")] = 30,
    ctx = None
) -> Dict[str, Any]:
    """Execute a Keyboard Maestro macro with comprehensive error handling."""
    from .server.tools.core_tools import km_execute_macro as _km_execute_macro
    return await _km_execute_macro(identifier, trigger_value, method, timeout, ctx)

@mcp.tool()
async def km_list_macros(
    group_filter: Annotated[str, Field(default="", description="Filter by macro group name")] = "",
    enabled_only: Annotated[bool, Field(default=True, description="Only show enabled macros")] = True,
    sort_by: Annotated[str, Field(default="name", description="Sort field")] = "name",
    limit: Annotated[int, Field(default=20, description="Maximum results")] = 20,
    ctx = None
) -> Dict[str, Any]:
    """List and filter Keyboard Maestro macros."""
    from .server.tools.core_tools import km_list_macros as _km_list_macros
    return await _km_list_macros(group_filter, enabled_only, sort_by, limit, ctx)

@mcp.tool()
async def km_variable_manager(
    operation: Annotated[str, Field(description="Operation: get, set, delete, or list")],
    name: Annotated[str, Field(default="", description="Variable name")] = "",
    value: Annotated[str, Field(default="", description="Variable value for set operation")] = "",
    scope: Annotated[str, Field(default="global", description="Variable scope")] = "global",
    instance_id: Annotated[str, Field(default="", description="Instance ID for local variables")] = "",
    ctx = None
) -> Dict[str, Any]:
    """Manage Keyboard Maestro variables across all scopes."""
    from .server.tools.core_tools import km_variable_manager as _km_variable_manager
    return await _km_variable_manager(operation, name, value, scope, instance_id, ctx)

@mcp.tool()
async def km_search_macros_advanced(
    query: Annotated[str, Field(description="Search query")],
    scope: Annotated[str, Field(default="name_and_group", description="Search scope")] = "name_and_group",
    action_categories: Annotated[str, Field(default="", description="Filter by action categories")] = "",
    complexity_levels: Annotated[str, Field(default="", description="Filter by complexity levels")] = "",
    min_usage_count: Annotated[int, Field(default=0, description="Minimum usage count")] = 0,
    sort_by: Annotated[str, Field(default="name", description="Sort criteria")] = "name",
    ctx = None
) -> Dict[str, Any]:
    """Advanced macro search with comprehensive filtering and metadata analysis."""
    from .server.tools.advanced_tools import km_search_macros_advanced as _km_search_macros_advanced
    return await _km_search_macros_advanced(query, scope, action_categories, complexity_levels, min_usage_count, sort_by, ctx)

@mcp.tool()
async def km_analyze_macro_metadata(
    macro_id: Annotated[str, Field(description="Macro ID or name to analyze")],
    include_relationships: Annotated[bool, Field(default=True, description="Include relationship analysis")] = True,
    ctx = None
) -> Dict[str, Any]:
    """Deep analysis of individual macro metadata and patterns."""
    from .server.tools.advanced_tools import km_analyze_macro_metadata as _km_analyze_macro_metadata
    return await _km_analyze_macro_metadata(macro_id, include_relationships, ctx)

@mcp.tool()
async def km_start_realtime_sync(
    enable_file_monitoring: Annotated[bool, Field(default=True, description="Enable file system monitoring")] = True,
    poll_interval_seconds: Annotated[int, Field(default=30, description="Base polling interval")] = 30,
    ctx = None
) -> Dict[str, Any]:
    """Start real-time macro library synchronization and monitoring."""
    from .server.tools.sync_tools import km_start_realtime_sync as _km_start_realtime_sync
    return await _km_start_realtime_sync(enable_file_monitoring, poll_interval_seconds, ctx)

@mcp.tool()
async def km_stop_realtime_sync(ctx = None) -> Dict[str, Any]:
    """Stop real-time macro library synchronization and monitoring."""
    from .server.tools.sync_tools import km_stop_realtime_sync as _km_stop_realtime_sync
    return await _km_stop_realtime_sync(ctx)

@mcp.tool()
async def km_sync_status(
    include_performance_metrics: Annotated[bool, Field(default=True, description="Include performance metrics")] = True,
    ctx = None
) -> Dict[str, Any]:
    """Get current status of real-time synchronization with performance metrics."""
    from .server.tools.sync_tools import km_sync_status as _km_sync_status
    return await _km_sync_status(include_performance_metrics, ctx)

@mcp.tool()
async def km_force_sync(
    full_resync: Annotated[bool, Field(default=False, description="Force complete resynchronization")] = False,
    ctx = None
) -> Dict[str, Any]:
    """Force immediate synchronization of macro library state."""
    from .server.tools.sync_tools import km_force_sync as _km_force_sync
    return await _km_force_sync(full_resync, ctx)

@mcp.tool()
async def km_list_macro_groups(
    include_macro_count: Annotated[bool, Field(default=True, description="Include macro counts")] = True,
    include_enabled_count: Annotated[bool, Field(default=True, description="Include enabled macro counts")] = True,
    sort_by: Annotated[str, Field(default="name", description="Sort criteria")] = "name",
    ctx = None
) -> Dict[str, Any]:
    """List all macro groups from Keyboard Maestro with comprehensive statistics."""
    from .server.tools.group_tools import km_list_macro_groups as _km_list_macro_groups
    return await _km_list_macro_groups(include_macro_count, include_enabled_count, sort_by, ctx)

@mcp.tool()
async def km_create_macro(
    name: Annotated[str, Field(
        description="Macro name (1-255 ASCII characters)",
        min_length=1,
        max_length=255,
        pattern=r"^[a-zA-Z0-9_\s\-\.]+$"
    )],
    template: Annotated[str, Field(
        description="Macro template type",
        pattern=r"^(hotkey_action|app_launcher|text_expansion|file_processor|window_manager|custom)$"
    )],
    group_name: Annotated[Optional[str], Field(
        default=None,
        description="Target macro group name",
        max_length=255
    )] = None,
    enabled: Annotated[bool, Field(
        default=True,
        description="Initial enabled state"
    )] = True,
    parameters: Annotated[Dict[str, Any], Field(
        default_factory=dict,
        description="Template-specific parameters"
    )] = {},
    ctx = None
) -> Dict[str, Any]:
    """Create a new Keyboard Maestro macro with comprehensive validation and security."""
    from .server.tools.creation_tools import km_create_macro as _km_create_macro
    return await _km_create_macro(name, template, group_name, enabled, parameters, ctx)

@mcp.tool()
async def km_list_templates(ctx = None) -> Dict[str, Any]:
    """List available macro templates with descriptions and parameter requirements."""
    from .server.tools.creation_tools import km_list_templates as _km_list_templates
    return await _km_list_templates(ctx)

@mcp.tool()
async def km_app_control(
    operation: Annotated[str, Field(
        description="Application control operation",
        pattern=r"^(launch|quit|activate|menu_select|get_state)$"
    )],
    app_identifier: Annotated[str, Field(
        description="Application bundle ID or name",
        min_length=1,
        max_length=255,
        pattern=r"^[a-zA-Z0-9\.\-\s]+$"
    )],
    menu_path: Annotated[Optional[List[str]], Field(
        default=None,
        description="Menu path for menu_select operation (max 10 items)",
        max_length=10
    )] = None,
    force_quit: Annotated[bool, Field(
        default=False,
        description="Force termination option for quit operation"
    )] = False,
    wait_for_completion: Annotated[bool, Field(
        default=True,
        description="Wait for operation to complete"
    )] = True,
    timeout_seconds: Annotated[int, Field(
        default=30,
        ge=1,
        le=120,
        description="Operation timeout in seconds"
    )] = 30,
    hide_on_launch: Annotated[bool, Field(
        default=False,
        description="Hide application after launch"
    )] = False,
    ctx = None
) -> Dict[str, Any]:
    """Comprehensive application control with security validation and error handling."""
    from .server.tools.app_control_tools import km_app_control as _km_app_control
    return await _km_app_control(operation, app_identifier, menu_path, force_quit, wait_for_completion, timeout_seconds, hide_on_launch, ctx)

@mcp.tool()
async def km_clipboard_manager(
    operation: Annotated[str, Field(
        description="Clipboard operation type",
        pattern=r"^(get|set|get_history|list_history|manage_named|search_named|stats)$"
    )],
    clipboard_name: Annotated[Optional[str], Field(
        default=None,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-\s]*$"
    )] = None,
    history_index: Annotated[Optional[int], Field(
        default=None,
        ge=0,
        le=199
    )] = None,
    content: Annotated[Optional[str], Field(
        default=None,
        max_length=1_000_000
    )] = None,
    format: Annotated[str, Field(
        default="text",
        pattern=r"^(text|image|file|url)$"
    )] = "text",
    include_sensitive: Annotated[bool, Field(
        default=False
    )] = False,
    tags: Annotated[Optional[List[str]], Field(
        default=None,
        max_length=20
    )] = None,
    description: Annotated[Optional[str], Field(
        default=None,
        max_length=500
    )] = None,
    search_query: Annotated[Optional[str], Field(
        default=None,
        max_length=200
    )] = None,
    overwrite: Annotated[bool, Field(
        default=False
    )] = False,
    ctx = None
) -> Dict[str, Any]:
    """Comprehensive clipboard management with security validation and named clipboard support."""
    from .server.tools.clipboard_tools import km_clipboard_manager as _km_clipboard_manager
    return await _km_clipboard_manager(operation, clipboard_name, history_index, content, format, include_sensitive, tags, description, search_query, overwrite, ctx)

@mcp.tool()
async def km_file_operations(
    operation: Annotated[str, Field(
        description="File operation type",
        pattern=r"^(copy|move|delete|rename|create_folder|get_info)$"
    )],
    source_path: Annotated[str, Field(
        description="Source file or folder path",
        min_length=1,
        max_length=1000
    )],
    destination_path: Annotated[Optional[str], Field(
        default=None,
        description="Destination path for copy/move/rename operations",
        max_length=1000
    )] = None,
    overwrite: Annotated[bool, Field(
        default=False,
        description="Allow overwriting existing files"
    )] = False,
    create_intermediate: Annotated[bool, Field(
        default=False,
        description="Create missing intermediate directories"
    )] = False,
    backup_existing: Annotated[bool, Field(
        default=False,
        description="Create backup of existing files before overwrite"
    )] = False,
    secure_delete: Annotated[bool, Field(
        default=False,
        description="Use secure deletion (multiple overwrite passes)"
    )] = False,
    ctx = None
) -> Dict[str, Any]:
    """Secure file system operations with comprehensive validation and safety features."""
    from .server.tools.file_operation_tools import km_file_operations as _km_file_operations
    return await _km_file_operations(operation, source_path, destination_path, overwrite, create_intermediate, backup_existing, secure_delete, ctx)

@mcp.tool()
async def km_create_hotkey_trigger(
    macro_id: Annotated[str, Field(
        description="Target macro UUID or name",
        min_length=1,
        max_length=255,
        examples=["Quick Notes", "550e8400-e29b-41d4-a716-446655440000"]
    )],
    key: Annotated[str, Field(
        description="Key identifier (letter, number, or special key)",
        min_length=1,
        max_length=20,
        pattern=r"^[a-zA-Z0-9]$|^(space|tab|enter|return|escape|delete|backspace|f[1-9]|f1[0-2]|home|end|pageup|pagedown|up|down|left|right|clear|help|insert)$",
        examples=["n", "space", "f1", "escape"]
    )],
    modifiers: Annotated[List[str], Field(
        description="Modifier keys (cmd, opt, shift, ctrl, fn)",
        default_factory=list,
        examples=[["cmd", "shift"], ["ctrl", "opt"]]
    )] = [],
    activation_mode: Annotated[str, Field(
        default="pressed",
        description="Activation mode for the hotkey",
        pattern=r"^(pressed|released|tapped|held)$",
        examples=["pressed", "tapped", "held"]
    )] = "pressed",
    tap_count: Annotated[int, Field(
        default=1,
        description="Number of taps required (1-4)",
        ge=1,
        le=4,
        examples=[1, 2, 3]
    )] = 1,
    allow_repeat: Annotated[bool, Field(
        default=False,
        description="Allow key repeat for continuous execution"
    )] = False,
    check_conflicts: Annotated[bool, Field(
        default=True,
        description="Check for hotkey conflicts before creation"
    )] = True,
    suggest_alternatives: Annotated[bool, Field(
        default=True,
        description="Provide alternative suggestions if conflicts are found"
    )] = True,
    ctx = None
) -> Dict[str, Any]:
    """Create hotkey trigger for macro with comprehensive validation and conflict detection."""
    from .server.tools.hotkey_tools import km_create_hotkey_trigger as _km_create_hotkey_trigger
    return await _km_create_hotkey_trigger(macro_id, key, modifiers, activation_mode, tap_count, allow_repeat, check_conflicts, suggest_alternatives, ctx)

@mcp.tool()
async def km_list_hotkey_triggers(
    macro_id: Annotated[Optional[str], Field(
        default=None,
        description="Filter by specific macro ID (optional)",
        max_length=255
    )] = None,
    include_conflicts: Annotated[bool, Field(
        default=False,
        description="Include conflict information for each hotkey"
    )] = False,
    ctx = None
) -> Dict[str, Any]:
    """List all registered hotkey triggers with optional filtering and conflict information."""
    from .server.tools.hotkey_tools import km_list_hotkey_triggers as _km_list_hotkey_triggers
    return await _km_list_hotkey_triggers(macro_id, include_conflicts, ctx)

@mcp.tool()
async def km_move_macro_to_group(
    macro_identifier: Annotated[str, Field(
        description="Macro name or UUID to move",
        min_length=1,
        max_length=255,
        pattern=r"^[a-zA-Z0-9_\s\-\.\+\(\)\[\]\{\}\|\&\~\!\@\#\$\%\^\*\=\?\:\;\,\/\\]+$|^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )],
    target_group: Annotated[str, Field(
        description="Target group name",
        min_length=1,
        max_length=255,
        pattern=r"^[a-zA-Z0-9_\s\-\.\+\(\)\[\]\{\}\|\&\~\!\@\#\$\%\^\*\=\?\:\;\,\/\\]+$"
    )],
    create_group_if_missing: Annotated[bool, Field(
        default=False,
        description="Create target group if it doesn't exist"
    )] = False,
    preserve_group_settings: Annotated[bool, Field(
        default=True,
        description="Maintain group-specific activation settings"
    )] = True,
    timeout_seconds: Annotated[int, Field(
        default=30,
        ge=5,
        le=120,
        description="Operation timeout in seconds"
    )] = 30,
    ctx = None
) -> Dict[str, Any]:
    """Move a macro from one group to another with comprehensive validation and conflict resolution."""
    from .server.tools.macro_move_tools import km_move_macro_to_group as _km_move_macro_to_group
    return await _km_move_macro_to_group(macro_identifier, target_group, create_group_if_missing, preserve_group_settings, timeout_seconds, ctx)

@mcp.tool()
async def km_notifications(
    notification_type: Annotated[str, Field(
        description="Notification type: notification, alert, hud, sound",
        pattern=r"^(notification|alert|hud|sound)$"
    )],
    title: Annotated[str, Field(
        description="Notification title (1-100 characters)",
        min_length=1,
        max_length=100
    )],
    message: Annotated[str, Field(
        description="Notification message content (1-500 characters)",
        min_length=1,
        max_length=500
    )],
    sound: Annotated[Optional[str], Field(
        default=None,
        description="Sound name (system sound) or file path",
        max_length=255
    )] = None,
    duration: Annotated[Optional[float], Field(
        default=None,
        description="Display duration in seconds (0.1-60.0)",
        ge=0.1,
        le=60.0
    )] = None,
    buttons: Annotated[List[str], Field(
        default_factory=list,
        description="Button labels for alert dialogs (max 3)",
        max_length=3
    )] = [],
    position: Annotated[str, Field(
        default="center",
        description="HUD position: center, top, bottom, left, right, top_left, top_right, bottom_left, bottom_right",
        pattern=r"^(center|top|bottom|left|right|top_left|top_right|bottom_left|bottom_right)$"
    )] = "center",
    priority: Annotated[str, Field(
        default="normal",
        description="Notification priority: low, normal, high, urgent",
        pattern=r"^(low|normal|high|urgent)$"
    )] = "normal",
    dismissible: Annotated[bool, Field(
        default=True,
        description="Whether notification can be dismissed by user"
    )] = True,
    ctx = None
) -> Dict[str, Any]:
    """Display user notifications with comprehensive formatting and interaction support."""
    from .server.tools.notification_tools import km_notifications as _km_notifications
    return await _km_notifications(notification_type, title, message, sound, duration, buttons, position, priority, dismissible, ctx)

@mcp.tool()
async def km_notification_status(
    notification_id: Annotated[Optional[str], Field(
        default=None,
        description="Notification ID to check status for (optional)"
    )] = None,
    ctx = None
) -> Dict[str, Any]:
    """Get status of active notifications with detailed information."""
    from .server.tools.notification_tools import km_notification_status as _km_notification_status
    return await _km_notification_status(notification_id, ctx)

@mcp.tool()
async def km_dismiss_notifications(
    notification_id: Annotated[Optional[str], Field(
        default=None,
        description="Specific notification ID to dismiss (optional - dismisses all if not provided)"
    )] = None,
    ctx = None
) -> Dict[str, Any]:
    """Dismiss active notifications with optional ID filtering."""
    from .server.tools.notification_tools import km_dismiss_notifications as _km_dismiss_notifications
    return await _km_dismiss_notifications(notification_id, ctx)

@mcp.tool()
async def km_calculator(
    expression: Annotated[str, Field(
        description="Mathematical expression to evaluate",
        min_length=1,
        max_length=1000
    )],
    variables: Annotated[Dict[str, float], Field(
        default_factory=dict,
        description="Variable values for expression evaluation"
    )] = {},
    format_result: Annotated[str, Field(
        default="auto",
        description="Result format: auto, decimal, scientific, percentage, currency",
        pattern=r"^(auto|decimal|scientific|percentage|currency)$"
    )] = "auto",
    precision: Annotated[int, Field(
        default=10,
        description="Decimal precision for results (0-15)",
        ge=0,
        le=15
    )] = 10,
    use_km_engine: Annotated[bool, Field(
        default=True,
        description="Use Keyboard Maestro's calculation engine"
    )] = True,
    validate_only: Annotated[bool, Field(
        default=False,
        description="Validate expression without evaluation"
    )] = False,
    ctx = None
) -> Dict[str, Any]:
    """Evaluate mathematical expressions with comprehensive security and token support."""
    from .server.tools.calculator_tools import km_calculator as _km_calculator
    return await _km_calculator(expression, variables, format_result, precision, use_km_engine, validate_only, ctx)

@mcp.tool()
async def km_token_processor(
    text: Annotated[str, Field(
        description="Text containing Keyboard Maestro tokens",
        min_length=1,
        max_length=10000
    )],
    context: Annotated[str, Field(
        default="text",
        description="Processing context for token evaluation",
        pattern=r"^(text|calculation|regex|filename|url)$"
    )] = "text",
    variables: Annotated[Dict[str, str], Field(
        default_factory=dict,
        description="Variable values for token substitution"
    )] = {},
    use_km_engine: Annotated[bool, Field(
        default=True,
        description="Use Keyboard Maestro's token processing engine"
    )] = True,
    preview_only: Annotated[bool, Field(
        default=False,
        description="Preview tokens without processing"
    )] = False,
    security_level: Annotated[str, Field(
        default="standard",
        description="Security validation level",
        pattern=r"^(minimal|standard|strict)$"
    )] = "standard",
    ctx = None
) -> Dict[str, Any]:
    """Process Keyboard Maestro tokens with comprehensive security and context support."""
    from .server.tools.token_tools import km_token_processor as _km_token_processor
    return await _km_token_processor(text, context, variables, use_km_engine, preview_only, security_level, ctx)

@mcp.tool()
async def km_token_stats(
    ctx = None
) -> Dict[str, Any]:
    """Get token processing statistics and system status."""
    from .server.tools.token_tools import km_token_stats as _km_token_stats
    return await _km_token_stats(ctx)


@mcp.tool()
async def km_window_manager(
    operation: Annotated[str, Field(
        description="Window management operation",
        pattern=r"^(move|resize|minimize|maximize|restore|arrange|get_info|get_screens)$"
    )],
    window_identifier: Annotated[str, Field(
        description="Application name, bundle ID, or window title",
        min_length=1,
        max_length=255
    )],
    position: Annotated[Optional[Dict[str, int]], Field(
        default=None,
        description="Target position {x, y} for move operation"
    )] = None,
    size: Annotated[Optional[Dict[str, int]], Field(
        default=None,
        description="Target size {width, height} for resize operation"
    )] = None,
    screen: Annotated[str, Field(
        default="main",
        description="Target screen (main, external, or index)",
        pattern=r"^(main|external|\d+)$"
    )] = "main",
    window_index: Annotated[int, Field(
        default=0,
        description="Window index for multi-window applications",
        ge=0,
        le=20
    )] = 0,
    arrangement: Annotated[Optional[str], Field(
        default=None,
        description="Predefined arrangement for arrange operation",
        pattern=r"^(left_half|right_half|top_half|bottom_half|top_left_quarter|top_right_quarter|bottom_left_quarter|bottom_right_quarter|center|maximize)$"
    )] = None,
    state: Annotated[Optional[str], Field(
        default=None,
        description="Target window state for minimize/maximize/restore operations",
        pattern=r"^(normal|minimized|maximized|fullscreen)$"
    )] = None,
    ctx = None
) -> Dict[str, Any]:
    """Comprehensive window management with multi-monitor support and coordinate validation."""
    from .server.tools.window_tools import km_window_manager as _km_window_manager
    return await _km_window_manager(operation, window_identifier, position, size, screen, window_index, arrangement, state, ctx)

@mcp.tool()
async def km_window_manager_advanced(
    operation: Annotated[str, Field(
        description="Advanced window management operation",
        pattern=r"^(grid_layout|cross_monitor_move|smart_arrange|workspace_setup)$"
    )],
    window_targets: Annotated[List[str], Field(
        description="List of application names, bundle IDs, or window titles",
        min_items=1,
        max_items=20
    )],
    layout_pattern: Annotated[Optional[str], Field(
        default=None,
        description="Grid layout pattern for arrangement operations",
        pattern=r"^(2x2|3x3|4x2|2x3|1x2|2x1|sidebar_main|main_sidebar|thirds_horizontal|thirds_vertical|quarters|custom)$"
    )] = None,
    target_displays: Annotated[Optional[List[int]], Field(
        default=None,
        description="Target display indices for multi-monitor operations",
        max_items=5
    )] = None,
    workspace_name: Annotated[Optional[str], Field(
        default=None,
        description="Named workspace configuration for save/restore operations",
        min_length=1,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$"
    )] = None,
    positioning_rules: Annotated[Optional[Dict[str, Any]], Field(
        default=None,
        description="Custom positioning rules for smart arrangement"
    )] = None,
    preserve_ratios: Annotated[bool, Field(
        default=True,
        description="Preserve aspect ratios and relative positioning"
    )] = True,
    animate_transitions: Annotated[bool, Field(
        default=False,
        description="Enable smooth window transition animations"
    )] = False,
    save_layout: Annotated[bool, Field(
        default=False,
        description="Save current layout as named workspace"
    )] = False,
    ctx = None
) -> Dict[str, Any]:
    """Advanced multi-monitor window management with grid layouts, intelligent positioning, and workspace management."""
    from .server.tools.advanced_window_tools import AdvancedWindowProcessor
    processor = AdvancedWindowProcessor()
    result = await processor.process_advanced_window_request(
        operation, window_targets, layout_pattern, target_displays, workspace_name,
        positioning_rules, preserve_ratios, animate_transitions, save_layout, ctx
    )
    return result

@mcp.tool()
async def km_dictionary_manager(
    operation: Annotated[str, Field(
        description="Dictionary operation",
        pattern=r"^(create|read|update|delete|query|merge|transform|validate|export|import)$"
    )],
    dictionary_name: Annotated[str, Field(
        description="Dictionary identifier/name",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_.-]+$"
    )],
    key_path: Annotated[Optional[str], Field(
        default=None,
        description="Dot-separated key path for nested access (e.g., 'user.profile.name')",
        max_length=1000
    )] = None,
    value: Annotated[Optional[Any], Field(
        default=None,
        description="Value for set operations or initial data for create"
    )] = None,
    schema: Annotated[Optional[Dict[str, Any]], Field(
        default=None,
        description="JSON schema for validation"
    )] = None,
    query: Annotated[Optional[str], Field(
        default=None,
        description="JSONPath query for data extraction",
        max_length=500
    )] = None,
    merge_strategy: Annotated[str, Field(
        default="deep",
        description="Merge strategy for merge operations",
        pattern=r"^(deep|shallow|replace|append|union)$"
    )] = "deep",
    format_output: Annotated[str, Field(
        default="json",
        description="Output format for read/export operations",
        pattern=r"^(json|yaml|csv|xml)$"
    )] = "json",
    validate_schema: Annotated[bool, Field(
        default=True,
        description="Enable schema validation"
    )] = True,
    timeout_seconds: Annotated[int, Field(
        default=30,
        description="Operation timeout in seconds",
        ge=1,
        le=300
    )] = 30,
    ctx = None
) -> Dict[str, Any]:
    """Advanced dictionary and JSON data management with schema validation, querying, and transformation capabilities."""
    from .server.tools.dictionary_manager_tools import dictionary_manager_tools
    return await dictionary_manager_tools.km_dictionary_manager(
        operation, dictionary_name, key_path, value, schema, query,
        merge_strategy, format_output, validate_schema, timeout_seconds, ctx
    )

@mcp.tool()
async def km_add_condition(
    macro_identifier: Annotated[str, Field(
        description="Target macro name or UUID",
        min_length=1,
        max_length=255
    )],
    condition_type: Annotated[str, Field(
        description="Type of condition: text, app, system, variable, logic",
        pattern=r"^(text|app|application|system|variable|logic)$"
    )],
    operator: Annotated[str, Field(
        description="Comparison operator: contains, equals, greater, less, regex, exists",
        pattern=r"^(contains|equals|greater|less|regex|exists|not_equals|starts_with|ends_with)$"
    )],
    operand: Annotated[str, Field(
        description="Comparison value (max 1000 characters)",
        max_length=1000
    )],
    case_sensitive: Annotated[bool, Field(
        default=True,
        description="Whether text comparisons are case sensitive"
    )] = True,
    negate: Annotated[bool, Field(
        default=False,
        description="Whether to invert the condition result"
    )] = False,
    action_on_true: Annotated[Optional[str], Field(
        default=None,
        description="Optional action to execute when condition is true",
        max_length=255
    )] = None,
    action_on_false: Annotated[Optional[str], Field(
        default=None,
        description="Optional action to execute when condition is false",
        max_length=255
    )] = None,
    timeout_seconds: Annotated[int, Field(
        default=10,
        description="Maximum time to evaluate condition (1-60 seconds)",
        ge=1,
        le=60
    )] = 10,
    ctx = None
) -> Dict[str, Any]:
    """Add conditional logic to Keyboard Maestro macro for intelligent automation workflows."""
    from .server.tools.condition_tools import km_add_condition as _km_add_condition
    return await _km_add_condition(macro_identifier, condition_type, operator, operand, case_sensitive, negate, action_on_true, action_on_false, timeout_seconds, ctx)

@mcp.tool()
async def km_control_flow(
    macro_identifier: Annotated[str, Field(
        description="Target macro name or UUID for control flow addition",
        min_length=1,
        max_length=255
    )],
    control_type: Annotated[str, Field(
        description="Type of control flow: if_then_else, for_loop, while_loop, switch_case",
        pattern=r"^(if_then_else|for_loop|while_loop|switch_case|try_catch)$"
    )],
    condition: Annotated[Optional[str], Field(
        default=None,
        description="Condition expression for if/while statements (max 500 characters)",
        max_length=500
    )] = None,
    operator: Annotated[str, Field(
        default="equals",
        description="Comparison operator: equals, greater_than, contains, matches_regex, etc.",
        pattern=r"^(equals|not_equals|greater_than|less_than|greater_equal|less_equal|contains|not_contains|matches_regex|exists)$"
    )] = "equals",
    operand: Annotated[Optional[str], Field(
        default=None,
        description="Value to compare against in conditions (max 1000 characters)",
        max_length=1000
    )] = None,
    iterator: Annotated[Optional[str], Field(
        default=None,
        description="Variable name for loop iteration (for loops only, max 50 characters)",
        max_length=50
    )] = None,
    collection: Annotated[Optional[str], Field(
        default=None,
        description="Collection expression to iterate over (for loops only, max 500 characters)",
        max_length=500
    )] = None,
    cases: Annotated[Optional[List[Dict[str, Any]]], Field(
        default=None,
        description="List of switch cases with 'value' and 'actions' keys"
    )] = None,
    actions_true: Annotated[Optional[List[Dict[str, Any]]], Field(
        default=None,
        description="Actions to execute when condition is true"
    )] = None,
    actions_false: Annotated[Optional[List[Dict[str, Any]]], Field(
        default=None,
        description="Actions to execute when condition is false"
    )] = None,
    loop_actions: Annotated[Optional[List[Dict[str, Any]]], Field(
        default=None,
        description="Actions to execute in loop body"
    )] = None,
    default_actions: Annotated[Optional[List[Dict[str, Any]]], Field(
        default=None,
        description="Default actions for switch statement"
    )] = None,
    max_iterations: Annotated[int, Field(
        default=1000,
        description="Maximum loop iterations (security bounded 1-10000)",
        ge=1,
        le=10000
    )] = 1000,
    timeout_seconds: Annotated[int, Field(
        default=30,
        description="Maximum execution timeout (1-300 seconds)",
        ge=1,
        le=300
    )] = 30,
    allow_nested: Annotated[bool, Field(
        default=True,
        description="Whether to allow nested control structures"
    )] = True,
    case_sensitive: Annotated[bool, Field(
        default=True,
        description="Case sensitivity for string comparisons"
    )] = True,
    negate: Annotated[bool, Field(
        default=False,
        description="Whether to negate the condition result"
    )] = False,
    ctx = None
) -> Dict[str, Any]:
    """Add sophisticated control flow structures (if/then/else, loops, switch/case) to Keyboard Maestro macros for intelligent automation workflows."""
    from .server.tools.control_flow_tools import km_control_flow as _km_control_flow
    return await _km_control_flow(macro_identifier, control_type, condition, operator, operand, iterator, collection, cases, actions_true, actions_false, loop_actions, default_actions, max_iterations, timeout_seconds, allow_nested, case_sensitive, negate, ctx)

@mcp.tool()
async def km_create_trigger_advanced(
    macro_identifier: Annotated[str, Field(
        description="Target macro name or UUID to add trigger to",
        max_length=255
    )],
    trigger_type: Annotated[str, Field(
        description="Type of trigger: time|file|system|user (or specific subtypes like time_scheduled, file_modified)",
        pattern=r"^(time|file|system|user|time_scheduled|time_recurring|file_created|file_modified|file_deleted|app_launched|app_quit|user_idle|network_connected)$"
    )],
    trigger_config: Annotated[Dict[str, Any], Field(
        description="Trigger-specific configuration object with parameters like schedule_time, watch_path, app_bundle_id, etc."
    )],
    conditions: Annotated[Optional[List[Dict[str, Any]]], Field(
        default=None,
        description="Optional conditional logic to make triggers intelligent (integrates with km_add_condition)"
    )] = None,
    enabled: Annotated[bool, Field(
        default=True,
        description="Whether trigger starts in enabled state"
    )] = True,
    priority: Annotated[int, Field(
        default=0,
        description="Execution priority for trigger (-10 to 10, higher = more priority)",
        ge=-10,
        le=10
    )] = 0,
    timeout_seconds: Annotated[int, Field(
        default=30,
        description="Maximum trigger execution timeout (1-300 seconds)",
        ge=1,
        le=300
    )] = 30,
    max_executions: Annotated[Optional[int], Field(
        default=None,
        description="Optional limit on total trigger executions (1-10000)",
        ge=1,
        le=10000
    )] = None,
    replace_existing: Annotated[bool, Field(
        default=False,
        description="Whether to replace existing triggers on the macro"
    )] = False,
    ctx = None
) -> Dict[str, Any]:
    """Create advanced event-driven triggers for intelligent automation that responds automatically to environmental changes."""
    from .server.tools.advanced_trigger_tools import km_create_trigger_advanced as _km_create_trigger_advanced
    return await _km_create_trigger_advanced(macro_identifier, trigger_type, trigger_config, conditions, enabled, priority, timeout_seconds, max_executions, replace_existing, ctx)

# Resources
@mcp.resource("km://server/status")
def get_server_status_resource() -> Dict[str, Any]:
    """Get current server status and configuration."""
    return get_server_status()

@mcp.resource("km://help/tools")
def get_tool_help_resource() -> str:
    """Get comprehensive help for all available tools."""
    return get_tool_help()

# Prompts
@mcp.prompt()
def create_macro_prompt(
    task_description: Annotated[str, Field(
        description="Description of the automation task to create a macro for"
    )],
    app_context: Annotated[str, Field(
        default="",
        description="Specific application or context for the automation"
    )] = ""
) -> list[Message]:
    """Generate a structured prompt for creating Keyboard Maestro macros."""
    return create_macro_prompt(task_description, app_context)


@mcp.tool()
async def km_plugin_ecosystem(
    operation: Annotated[str, Field(
        description="Plugin operation",
        pattern=r"^(install|uninstall|list|activate|deactivate|execute|configure|status|actions|marketplace)$"
    )],
    plugin_identifier: Annotated[Optional[str], Field(
        default=None,
        description="Plugin ID for operations",
        max_length=255
    )] = None,
    plugin_source: Annotated[Optional[str], Field(
        default=None,
        description="Plugin source (file path, URL, or marketplace ID)",
        max_length=1000
    )] = None,
    action_name: Annotated[Optional[str], Field(
        default=None,
        description="Custom action name to execute",
        max_length=255
    )] = None,
    parameters: Annotated[Optional[Dict], Field(
        default=None,
        description="Action parameters or operation parameters"
    )] = None,
    plugin_config: Annotated[Optional[Dict], Field(
        default=None,
        description="Plugin configuration settings"
    )] = None,
    security_profile: Annotated[str, Field(
        default="standard",
        description="Security profile for plugin execution",
        pattern=r"^(none|standard|strict|sandbox)$"
    )] = "standard",
    api_version: Annotated[str, Field(
        default="1.0",
        description="Plugin API version",
        pattern=r"^\d+\.\d+$"
    )] = "1.0",
    auto_update: Annotated[bool, Field(
        default=False,
        description="Enable automatic plugin updates"
    )] = False,
    dependency_resolution: Annotated[bool, Field(
        default=True,
        description="Automatically resolve plugin dependencies"
    )] = True,
    validation_level: Annotated[str, Field(
        default="strict",
        description="Plugin validation level",
        pattern=r"^(none|basic|standard|strict)$"
    )] = "strict",
    timeout: Annotated[int, Field(
        default=60,
        description="Operation timeout in seconds",
        ge=5,
        le=300
    )] = 60,
    ctx = None
) -> Dict[str, Any]:
    """Comprehensive plugin ecosystem management for custom action creation, installation, and execution."""
    from .server.tools.plugin_ecosystem_tools import PluginEcosystemTools
    
    tools = PluginEcosystemTools()
    return await tools.km_plugin_ecosystem(
        operation, plugin_identifier, plugin_source, action_name, parameters,
        plugin_config, security_profile, api_version, auto_update,
        dependency_resolution, validation_level, timeout, ctx
    )


@mcp.tool()
async def km_audit_system(
    operation: Annotated[str, Field(
        description="Audit operation",
        pattern=r"^(log|query|report|monitor|configure|status)$"
    )],
    event_type: Annotated[Optional[str], Field(
        default=None,
        description="Event type for logging operation",
        max_length=100
    )] = None,
    user_id: Annotated[Optional[str], Field(
        default=None,
        description="User identifier for audit tracking",
        max_length=100
    )] = None,
    resource_id: Annotated[Optional[str], Field(
        default=None,
        description="Resource being accessed or modified",
        max_length=255
    )] = None,
    action_details: Annotated[Optional[Dict], Field(
        default=None,
        description="Detailed action information"
    )] = None,
    compliance_standard: Annotated[str, Field(
        default="general",
        description="Compliance standard for monitoring/reporting",
        pattern=r"^(SOC2|HIPAA|GDPR|PCI_DSS|ISO_27001|NIST|general)$"
    )] = "general",
    time_range: Annotated[Optional[Dict], Field(
        default=None,
        description="Time range for queries and reports"
    )] = None,
    report_format: Annotated[str, Field(
        default="json",
        description="Output format for reports",
        pattern=r"^(json|csv|pdf|html|summary)$"
    )] = "json",
    include_sensitive: Annotated[bool, Field(
        default=False,
        description="Include sensitive data in reports"
    )] = False,
    audit_level: Annotated[str, Field(
        default="standard",
        description="Level of audit detail to capture",
        pattern=r"^(minimal|standard|detailed|comprehensive)$"
    )] = "standard",
    retention_period: Annotated[int, Field(
        default=365,
        description="Audit log retention in days",
        ge=1,
        le=2555
    )] = 365,
    encrypt_logs: Annotated[bool, Field(
        default=True,
        description="Enable audit log encryption"
    )] = True,
    ctx = None
) -> Dict[str, Any]:
    """Advanced audit logging and compliance reporting system for enterprise security and regulatory compliance."""
    from .server.tools.audit_system_tools import km_audit_system as _km_audit_system
    return await _km_audit_system(operation, event_type, user_id, resource_id, action_details, compliance_standard, time_range, report_format, include_sensitive, audit_level, retention_period, encrypt_logs, ctx)

@mcp.tool()
async def km_analytics_engine(
    operation: Annotated[str, Field(
        description="Analytics operation (collect|analyze|report|predict|dashboard|optimize)",
        pattern=r"^(collect|analyze|report|predict|dashboard|optimize)$"
    )],
    analytics_scope: Annotated[str, Field(
        default="ecosystem",
        description="Analysis scope (tool|category|ecosystem|enterprise)",
        pattern=r"^(tool|category|ecosystem|enterprise)$"
    )] = "ecosystem",
    time_range: Annotated[str, Field(
        default="24h", 
        description="Time range for analysis (1h|24h|7d|30d|90d|1y|all)",
        pattern=r"^(1h|24h|7d|30d|90d|1y|all)$"
    )] = "24h",
    metrics_types: Annotated[List[str], Field(
        default=["performance"],
        description="Types of metrics to collect (performance|usage|roi|efficiency|quality|security)"
    )] = ["performance"],
    analysis_depth: Annotated[str, Field(
        default="comprehensive",
        description="Depth of analysis (basic|standard|detailed|comprehensive|ml_enhanced)",
        pattern=r"^(basic|standard|detailed|comprehensive|ml_enhanced)$"
    )] = "comprehensive",
    visualization_format: Annotated[str, Field(
        default="dashboard",
        description="Output format (raw|table|chart|dashboard|report|executive_summary)",
        pattern=r"^(raw|table|chart|dashboard|report|executive_summary)$"
    )] = "dashboard",
    ml_insights: Annotated[bool, Field(
        default=True,
        description="Enable machine learning insights"
    )] = True,
    real_time_monitoring: Annotated[bool, Field(
        default=True,
        description="Enable real-time metrics collection"
    )] = True,
    anomaly_detection: Annotated[bool, Field(
        default=True,
        description="Enable anomaly detection"
    )] = True,
    predictive_analytics: Annotated[bool, Field(
        default=True,
        description="Enable predictive modeling"
    )] = True,
    roi_calculation: Annotated[bool, Field(
        default=True,
        description="Enable ROI and cost-benefit analysis"
    )] = True,
    privacy_mode: Annotated[str, Field(
        default="compliant",
        description="Privacy protection level (none|basic|compliant|strict)",
        pattern=r"^(none|basic|compliant|strict)$"
    )] = "compliant",
    export_format: Annotated[str, Field(
        default="json",
        description="Export format (json|csv|pdf|xlsx|api)",
        pattern=r"^(json|csv|pdf|xlsx|api)$"
    )] = "json",
    enterprise_integration: Annotated[bool, Field(
        default=True,
        description="Enable enterprise system integration"
    )] = True,
    ctx = None
) -> Dict[str, Any]:
    """
    Comprehensive automation analytics engine for deep insights and business intelligence.
    
    Provides advanced analytics capabilities including metrics collection, ML insights,
    ROI analysis, performance monitoring, and executive dashboards across the complete
    48-tool enterprise automation ecosystem.
    
    Key Features:
    - Real-time performance monitoring and metrics collection
    - ML-powered insights and pattern recognition  
    - ROI analysis and cost-benefit calculations
    - Executive dashboards and automated reporting
    - Predictive analytics and optimization recommendations
    - Enterprise-grade privacy compliance and security
    """
    from .server.tools.analytics_engine_tools import km_analytics_engine as _km_analytics_engine
    return await _km_analytics_engine(operation, analytics_scope, time_range, metrics_types, analysis_depth, visualization_format, ml_insights, real_time_monitoring, anomaly_detection, predictive_analytics, roi_calculation, privacy_mode, export_format, None, enterprise_integration, ctx)

@mcp.tool()
async def km_analyze_workflow_intelligence(
    workflow_source: Annotated[str, Field(description="Workflow source (description|existing|template)")],
    workflow_data: Annotated[Union[str, Dict], Field(description="Natural language description or workflow data")],
    analysis_depth: Annotated[str, Field(description="Analysis depth (basic|comprehensive|ai_enhanced)")] = "comprehensive",
    optimization_focus: Annotated[List[str], Field(description="Optimization areas (performance|efficiency|reliability|cost)")] = ["efficiency"],
    include_predictions: Annotated[bool, Field(description="Include predictive performance analysis")] = True,
    generate_alternatives: Annotated[bool, Field(description="Generate alternative workflow designs")] = True,
    cross_tool_optimization: Annotated[bool, Field(description="Enable cross-tool optimization analysis")] = True,
    ctx = None
) -> Dict[str, Any]:
    """
    Analyze workflow intelligence with AI-powered insights and optimization recommendations.
    
    Provides comprehensive workflow analysis including:
    - Natural language processing for workflow descriptions
    - Pattern recognition and inefficiency detection
    - Performance prediction and optimization recommendations
    - Cross-tool optimization opportunities
    - Alternative workflow design generation
    - Quality scoring and improvement suggestions
    """
    from .server.tools.workflow_intelligence_tools import km_analyze_workflow_intelligence as _km_analyze_workflow_intelligence
    return await _km_analyze_workflow_intelligence(workflow_source, workflow_data, analysis_depth, optimization_focus, include_predictions, generate_alternatives, cross_tool_optimization, ctx)

@mcp.tool()
async def km_create_workflow_from_description(
    description: Annotated[str, Field(description="Natural language workflow description", min_length=10)],
    target_complexity: Annotated[str, Field(description="Target complexity (simple|intermediate|advanced|expert)")] = "intermediate",
    preferred_tools: Annotated[Optional[List[str]], Field(description="Preferred tools to use")] = None,
    optimization_goals: Annotated[List[str], Field(description="Optimization goals (performance|efficiency|reliability|cost)")] = ["efficiency"],
    include_error_handling: Annotated[bool, Field(description="Include error handling and validation")] = True,
    generate_visual_design: Annotated[bool, Field(description="Generate visual workflow design")] = True,
    ctx = None
) -> Dict[str, Any]:
    """
    Create intelligent workflow from natural language description.
    
    Uses advanced NLP and AI to:
    - Parse user descriptions and extract workflow intent
    - Generate complete, optimized workflows with appropriate actions and conditions
    - Suggest appropriate tools and components
    - Include error handling and validation
    - Generate visual workflow designs
    - Provide implementation guidance and best practices
    """
    from .server.tools.workflow_intelligence_tools import km_create_workflow_from_description as _km_create_workflow_from_description
    return await _km_create_workflow_from_description(description, target_complexity, preferred_tools, optimization_goals, include_error_handling, generate_visual_design, ctx)


# IoT Integration Tools (TASK_65)

@mcp.tool()
async def km_control_iot_devices(
    device_identifier: Annotated[str, Field(description="Device ID, name, or address")],
    action: Annotated[str, Field(description="Action to perform (on|off|set|get|toggle)")],
    device_type: Annotated[Optional[str], Field(description="Device type (light|sensor|thermostat|switch|camera)")] = None,
    parameters: Annotated[Optional[Dict[str, Any]], Field(description="Action-specific parameters")] = None,
    protocol: Annotated[Optional[str], Field(description="Communication protocol (mqtt|http|zigbee|zwave)")] = None,
    timeout: Annotated[int, Field(description="Operation timeout in seconds", ge=1, le=300)] = 30,
    retry_attempts: Annotated[int, Field(description="Number of retry attempts", ge=0, le=5)] = 2,
    verify_action: Annotated[bool, Field(description="Verify action completion")] = True,
    ctx = None
) -> Dict[str, Any]:
    """Control IoT devices with support for multiple protocols and device types."""
    from .server.tools.iot_integration_tools import km_control_iot_devices as _km_control_iot_devices
    return await _km_control_iot_devices(device_identifier, action, device_type, parameters, protocol, timeout, retry_attempts, verify_action, ctx)

@mcp.tool()
async def km_monitor_sensors(
    sensor_identifiers: Annotated[List[str], Field(description="List of sensor IDs or names to monitor")],
    monitoring_duration: Annotated[int, Field(description="Monitoring duration in seconds", ge=10, le=86400)] = 300,
    sampling_interval: Annotated[int, Field(description="Data sampling interval in seconds", ge=1, le=3600)] = 30,
    trigger_conditions: Annotated[Optional[List[Dict[str, Any]]], Field(description="Automation trigger conditions")] = None,
    data_aggregation: Annotated[Optional[str], Field(description="Data aggregation method (avg|min|max|sum)")] = None,
    alert_thresholds: Annotated[Optional[Dict[str, float]], Field(description="Alert threshold values")] = None,
    export_data: Annotated[bool, Field(description="Export sensor data for analysis")] = False,
    real_time_alerts: Annotated[bool, Field(description="Enable real-time alerting")] = True,
    ctx = None
) -> Dict[str, Any]:
    """Monitor sensor data and trigger automation workflows based on readings and conditions."""
    from .server.tools.iot_integration_tools import km_monitor_sensors as _km_monitor_sensors
    return await _km_monitor_sensors(sensor_identifiers, monitoring_duration, sampling_interval, trigger_conditions, data_aggregation, alert_thresholds, export_data, real_time_alerts, ctx)

@mcp.tool()
async def km_manage_smart_home(
    operation: Annotated[str, Field(description="Operation (create_scene|activate_scene|schedule|status)")],
    scene_name: Annotated[Optional[str], Field(description="Scene name for scene operations")] = None,
    device_settings: Annotated[Optional[Dict[str, Any]], Field(description="Device settings for scene creation")] = None,
    schedule_config: Annotated[Optional[Dict[str, Any]], Field(description="Scheduling configuration")] = None,
    location_context: Annotated[Optional[str], Field(description="Location or room context")] = None,
    user_preferences: Annotated[Optional[Dict[str, Any]], Field(description="User preferences and customization")] = None,
    energy_optimization: Annotated[bool, Field(description="Enable energy optimization")] = True,
    adaptive_automation: Annotated[bool, Field(description="Enable adaptive automation based on usage patterns")] = False,
    ctx = None
) -> Dict[str, Any]:
    """Manage smart home automation with scenes, scheduling, and adaptive optimization."""
    from .server.tools.iot_integration_tools import km_manage_smart_home as _km_manage_smart_home
    return await _km_manage_smart_home(operation, scene_name, device_settings, schedule_config, location_context, user_preferences, energy_optimization, adaptive_automation, ctx)

@mcp.tool()
async def km_coordinate_iot_workflows(
    workflow_name: Annotated[str, Field(description="IoT workflow name")],
    device_sequence: Annotated[List[Dict[str, Any]], Field(description="Sequence of IoT device actions")],
    trigger_conditions: Annotated[List[Dict[str, Any]], Field(description="Workflow trigger conditions")],
    coordination_type: Annotated[str, Field(description="Coordination type (sequential|parallel|conditional)")] = "sequential",
    dependency_management: Annotated[bool, Field(description="Enable device dependency management")] = True,
    fault_tolerance: Annotated[bool, Field(description="Enable fault tolerance and error recovery")] = True,
    performance_optimization: Annotated[bool, Field(description="Enable performance optimization")] = True,
    learning_mode: Annotated[bool, Field(description="Enable learning from workflow execution")] = False,
    ctx = None
) -> Dict[str, Any]:
    """Coordinate complex IoT automation workflows with device dependencies and optimization."""
    from .server.tools.iot_integration_tools import km_coordinate_iot_workflows as _km_coordinate_iot_workflows
    return await _km_coordinate_iot_workflows(workflow_name, device_sequence, trigger_conditions, coordination_type, dependency_management, fault_tolerance, performance_optimization, learning_mode, ctx)


def main():
    """Main entry point for the Keyboard Maestro MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Keyboard Maestro MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport method (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE transport (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    logger.info(f"Starting Keyboard Maestro MCP Server v1.0.0 (Modular)")
    logger.info(f"Transport: {args.transport}")
    
    # Initialize components
    try:
        initialize_components()
        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return 1
    
    if args.transport == "stdio":
        logger.info("Running with stdio transport for local clients")
        mcp.run(transport="stdio")
    else:
        logger.info(f"Running SSE server on {args.host}:{args.port}")
        mcp.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()