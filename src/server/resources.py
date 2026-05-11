"""MCP Resources and prompts for the Keyboard Maestro server.

Contains server status resource, help documentation, and prompt definitions.
"""

import asyncio
import logging
from typing import Annotated, Any

from fastmcp.prompts import Message
from pydantic import Field

from .initialization import get_km_client

logger = logging.getLogger(__name__)


def get_server_status() -> dict[str, Any]:
    """Get current server status and configuration."""
    # Test KM connection status
    km_client = get_km_client()
    try:
        # Quick sync test - we'll make this async in the future
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connection_test = loop.run_until_complete(
            km_client.list_macros_async(enabled_only=True),
        )
        loop.close()

        if connection_test.is_right():
            km_status = "connected"
            macro_count = len(connection_test.get_right())
        else:
            km_status = "disconnected"
            macro_count = 0
    except Exception:
        km_status = "error"
        macro_count = 0

    return {
        "server_name": "KeyboardMaestroMCP",
        "version": "1.0.0",
        "status": "running",
        "engine_status": "initialized",
        "km_connection": km_status,
        "km_macro_count": macro_count,
        "tools_available": 18,  # km_* tools registered in tool_config.py
        "tools_planned": 18,
        "integration_methods": ["applescript", "web_api", "url_scheme"],
        "features": {
            "macro_execution": True,
            "macro_listing": True,
            "macro_editing": True,
            "macro_group_management": True,
            "application_control": True,
            "variable_management": True,
            "trigger_management": True,
            "action_building": True,
            "input_simulation": True,
            "real_time_sync": False,
            "enhanced_metadata": True,
            "plugin_system": False,
            "ocr_integration": False,
        },
    }


def create_macro_prompt(
    task_description: Annotated[
        str,
        Field(description="Description of the automation task to create a macro for"),
    ],
    app_context: Annotated[
        str | None,
        Field(
            default=None,
            description="Specific application or context for the automation",
        ),
    ] = None,
) -> list[Message]:
    """Generate a structured prompt for creating Keyboard Maestro macros."""
    system_prompt = """You are an expert Keyboard Maestro macro developer. Help create efficient, reliable macros for macOS automation tasks."""

    user_prompt = f"""
Task: {task_description}

{f"Context: {app_context}" if app_context else ""}

Please provide a detailed macro design including:

1. **Macro Name and Purpose**
2. **Triggers** (hotkeys, application events, etc.)
3. **Actions Sequence** (step-by-step automation)
4. **Variables** (if needed for data storage)
5. **Error Handling** (conditions and fallbacks)
6. **Testing Strategy** (how to verify it works)

Focus on reliability, user experience, and maintainability.
    """

    # MCP prompt messages support only "user"/"assistant"; embed system context as assistant priming.
    return [
        Message(role="assistant", content=system_prompt),
        Message(role="user", content=user_prompt.strip()),
    ]


def get_tool_help(tool_name: str | None = None) -> str:
    """Get help for a specific tool or all tools."""
    if tool_name:
        # Return help for specific tool
        tool_helps = {
            "km_execute_macro": "Execute a Keyboard Maestro macro with comprehensive error handling.",
            "km_list_macros": "List and filter your actual Keyboard Maestro macros.",
            "km_variable_manager": "Manage Keyboard Maestro variables across all scopes.",
        }
        return tool_helps.get(
            tool_name,
            f"Tool '{tool_name}' not found or unknown tool.",
        )
    return """Keyboard Maestro MCP Tools — 18 production tools across core, macro editing, group management, application control, trigger management, action building, input simulation, file, window, clipboard, token, notification, conditional, and control-flow categories."""


def get_system_info() -> dict:
    """Get system information."""
    import platform

    return {
        "platform": platform.system(),
        "version": platform.release(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
    }


def get_tool_count() -> int:
    """Get current registered tool count from tool_config.py."""
    return 18
