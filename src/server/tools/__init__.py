"""MCP tools that drive Keyboard Maestro through its Engine.

Tool modules in this package are auto-discovered by
``src.server.tool_registry.ToolDiscovery``; this file only re-exports the
canonical entry points for callers that import them by name.
"""

from .action_tools import km_add_action, km_list_action_types
from .application_tools import km_application_control
from .core_tools import km_execute_macro, km_list_macros, km_variable_manager
from .engine_tools import km_engine_control
from .hotkey_tools import km_create_hotkey_trigger, km_list_hotkey_triggers
from .interface_tools import km_interface_automation
from .macro_editor_tools import km_macro_editor
from .macro_group_tools import km_macro_group_manager
from .notification_tools import (
    km_dismiss_notifications,
    km_notification_status,
    km_notifications,
)
from .token_tools import km_token_processor, km_token_stats
from .trigger_tools import km_trigger_manager
from .window_tools import km_window_manager

__all__ = [
    "km_add_action",
    "km_application_control",
    "km_create_hotkey_trigger",
    "km_dismiss_notifications",
    "km_engine_control",
    "km_execute_macro",
    "km_interface_automation",
    "km_list_action_types",
    "km_list_hotkey_triggers",
    "km_list_macros",
    "km_macro_editor",
    "km_macro_group_manager",
    "km_notification_status",
    "km_notifications",
    "km_token_processor",
    "km_token_stats",
    "km_trigger_manager",
    "km_variable_manager",
    "km_window_manager",
]
