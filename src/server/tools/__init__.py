"""MCP Tools for Keyboard Maestro server.

This package contains all the modularized MCP tools organized by functionality.
"""

from .action_tools import km_add_action, km_list_action_types
from .autonomous_agent_tools import km_autonomous_agent
from .clipboard_tools import km_clipboard_manager
from .core_tools import km_execute_macro, km_list_macros, km_variable_manager
from .engine_tools import km_engine_control
from .file_operation_tools import km_file_operations
from .hotkey_tools import km_create_hotkey_trigger, km_list_hotkey_triggers
from .interface_tools import km_interface_automation
from .notification_tools import km_notifications

__all__ = [
    # Action tools
    "km_add_action",
    # Autonomous agent tools
    "km_autonomous_agent",
    "km_clipboard_manager",
    # Hotkey tools
    "km_create_hotkey_trigger",
    # Engine tools
    "km_engine_control",
    # Core tools
    "km_execute_macro",
    # File operation tools
    "km_file_operations",
    # Interface tools
    "km_interface_automation",
    "km_list_action_types",
    "km_list_hotkey_triggers",
    "km_list_macros",
    # Notification tools
    "km_notifications",
    "km_variable_manager",
]
