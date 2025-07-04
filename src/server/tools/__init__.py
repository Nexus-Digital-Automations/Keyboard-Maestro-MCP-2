"""
MCP Tools for Keyboard Maestro server.

This package contains all the modularized MCP tools organized by functionality:
- core_tools: Basic macro operations (execute, list, variables)
- advanced_tools: Enhanced search and metadata analysis (TASK_6)
- sync_tools: Real-time synchronization (TASK_7)
- group_tools: Macro group management
- search_tools: Action and macro search capabilities
- property_tools: Macro property management
- dictionary_tools: Dictionary management operations
- engine_tools: Engine control and calculations
- interface_tools: Mouse and keyboard automation
"""

from .core_tools import km_execute_macro, km_list_macros, km_variable_manager
from .advanced_tools import km_search_macros_advanced, km_analyze_macro_metadata
from .sync_tools import km_start_realtime_sync, km_stop_realtime_sync, km_sync_status, km_force_sync
from .group_tools import km_list_macro_groups
from .clipboard_tools import km_clipboard_manager
from .file_operation_tools import km_file_operations
from .hotkey_tools import km_create_hotkey_trigger, km_list_hotkey_triggers
from .action_tools import km_add_action, km_list_action_types
from .search_tools import km_search_actions
from .property_tools import km_manage_macro_properties
from .dictionary_tools import km_dictionary_manager
from .engine_tools import km_engine_control
from .interface_tools import km_interface_automation
from .notification_tools import km_notifications
from .autonomous_agent_tools import km_autonomous_agent

__all__ = [
    # Core tools
    "km_execute_macro",
    "km_list_macros", 
    "km_variable_manager",
    "km_clipboard_manager",
    
    # Advanced tools
    "km_search_macros_advanced",
    "km_analyze_macro_metadata",
    
    # Sync tools
    "km_start_realtime_sync",
    "km_stop_realtime_sync", 
    "km_sync_status",
    "km_force_sync",
    
    # Group tools
    "km_list_macro_groups",
    
    # File operation tools
    "km_file_operations",
    
    # Hotkey tools
    "km_create_hotkey_trigger",
    "km_list_hotkey_triggers",
    
    # Action tools
    "km_add_action",
    "km_list_action_types",
    
    # Search tools
    "km_search_actions",
    
    # Property tools
    "km_manage_macro_properties",
    
    # Dictionary tools
    "km_dictionary_manager",
    
    # Engine tools
    "km_engine_control",
    
    # Interface tools
    "km_interface_automation",
    
    # Notification tools
    "km_notifications",
    
    # Autonomous agent tools
    "km_autonomous_agent"
]