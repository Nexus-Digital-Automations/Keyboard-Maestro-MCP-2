"""MCP tools that drive Keyboard Maestro through its Engine.

Tool modules in this package are auto-discovered by
``src.server.tool_registry.ToolDiscovery``; this file only re-exports the
canonical entry points for callers that import them by name.

Scope discipline: this server only drives Keyboard Maestro. Raw mouse /
keyboard / screen tools belong in the computer-use MCP, not here.
"""

from .action_builder_tools import km_action_builder
from .action_tools import km_list_action_types
from .application_tools import km_application_control
from .condition_tools import km_add_condition
from .control_flow_tools import km_control_flow
from .core_tools import km_execute_macro, km_list_macros, km_variable_manager
from .creation_tools import km_create_macro, km_list_templates
from .engine_tools import km_engine_control
from .hotkey_tools import km_create_hotkey_trigger, km_list_hotkey_triggers
from .macro_editor_tools import km_macro_editor
from .macro_group_tools import km_macro_group_manager
from .macro_move_tools import km_move_macro_to_group
from .macro_rebuild_tools import km_set_macro_triggers
from .notification_tools import (
    km_dismiss_notifications,
    km_notification_status,
    km_notifications,
)
from .plugin_action_tools import km_build_plugin_action
from .system_trigger_tools import km_add_system_trigger
from .token_tools import km_token_processor, km_token_stats
from .trigger_crud_tools import km_trigger_crud
from .trigger_tools import km_trigger_manager
from .window_tools import km_window_manager

__all__ = [
    "km_action_builder",
    "km_add_condition",
    "km_add_system_trigger",
    "km_application_control",
    "km_build_plugin_action",
    "km_control_flow",
    "km_create_hotkey_trigger",
    "km_create_macro",
    "km_dismiss_notifications",
    "km_engine_control",
    "km_execute_macro",
    "km_list_action_types",
    "km_list_hotkey_triggers",
    "km_list_macros",
    "km_list_templates",
    "km_macro_editor",
    "km_macro_group_manager",
    "km_move_macro_to_group",
    "km_notification_status",
    "km_notifications",
    "km_set_macro_triggers",
    "km_token_processor",
    "km_token_stats",
    "km_trigger_crud",
    "km_trigger_manager",
    "km_variable_manager",
    "km_window_manager",
]
