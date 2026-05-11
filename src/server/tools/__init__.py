"""MCP Tools for Keyboard Maestro server.

This package contains the modular MCP tools organized by functionality.
"""

from .clipboard_tools import km_clipboard_manager
from .condition_tools import km_add_condition
from .control_flow_tools import km_control_flow
from .core_tools import km_execute_macro, km_list_macros, km_variable_manager
from .file_operation_tools import km_file_operations
from .iot_integration_tools import (
    km_control_iot_devices,
    km_coordinate_iot_workflows,
    km_manage_smart_home,
    km_monitor_sensors,
)
from .notification_tools import (
    km_dismiss_notifications,
    km_notification_status,
    km_notifications,
)
from .token_tools import km_token_processor, km_token_stats
from .window_tools import km_window_manager

__all__ = [
    "km_add_condition",
    "km_clipboard_manager",
    "km_control_flow",
    "km_control_iot_devices",
    "km_coordinate_iot_workflows",
    "km_dismiss_notifications",
    "km_execute_macro",
    "km_file_operations",
    "km_list_macros",
    "km_manage_smart_home",
    "km_monitor_sensors",
    "km_notification_status",
    "km_notifications",
    "km_token_processor",
    "km_token_stats",
    "km_variable_manager",
    "km_window_manager",
]
