"""MCP tools that drive Keyboard Maestro through its Engine.

Tool modules in this package are auto-discovered by
``src.server.tool_registry.ToolDiscovery``; this file only re-exports the
canonical entry points for callers that import them by name.
"""

from .action_tools import km_add_action, km_list_action_types
from .core_tools import km_execute_macro, km_list_macros, km_variable_manager
from .engine_tools import km_engine_control
from .hotkey_tools import km_create_hotkey_trigger, km_list_hotkey_triggers
from .interface_tools import km_interface_automation
from .notification_tools import km_notifications

__all__ = [
    "km_add_action",
    "km_create_hotkey_trigger",
    "km_engine_control",
    "km_execute_macro",
    "km_interface_automation",
    "km_list_action_types",
    "km_list_hotkey_triggers",
    "km_list_macros",
    "km_notifications",
    "km_variable_manager",
]
