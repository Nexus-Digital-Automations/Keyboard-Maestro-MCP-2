"""
Keyboard Maestro Integration Layer

This module provides the integration layer connecting the core macro engine
to Keyboard Maestro automation system with event-driven architecture and
functional programming patterns.
"""

from .events import KMEvent, EventHandler, compose_event_handlers, get_default_event_pipeline, get_security_focused_pipeline, DEFAULT_EVENT_PIPELINE, SECURITY_FOCUSED_PIPELINE
from .km_client import KMClient, ConnectionConfig, ConnectionMethod
from .triggers import TriggerState, TriggerInfo, update_trigger_state
from .protocol import MCPProtocolHandler
from .security import validate_km_input, sanitize_trigger_data
from .macro_metadata import MacroMetadataExtractor, EnhancedMacroMetadata, ActionCategory, TriggerCategory, ComplexityLevel
from .smart_filtering import SmartMacroFilter, SearchQuery, SearchScope, SortCriteria, FilterResult
from .sync_manager import MacroSyncManager, SyncConfiguration, MacroChange, ChangeType, SyncStatus
from .file_monitor import KMFileMonitor, FileChangeEvent, WATCHDOG_AVAILABLE

__all__ = [
    "KMEvent",
    "EventHandler", 
    "compose_event_handlers",
    "get_default_event_pipeline",
    "get_security_focused_pipeline",
    "DEFAULT_EVENT_PIPELINE",
    "SECURITY_FOCUSED_PIPELINE",
    "KMClient",
    "ConnectionConfig",
    "ConnectionMethod",
    "TriggerState",
    "TriggerInfo",
    "update_trigger_state",
    "MCPProtocolHandler",
    "validate_km_input",
    "sanitize_trigger_data",
    "MacroMetadataExtractor",
    "EnhancedMacroMetadata",
    "ActionCategory",
    "TriggerCategory",
    "ComplexityLevel",
    "SmartMacroFilter",
    "SearchQuery",
    "SearchScope",
    "SortCriteria",
    "FilterResult",
    "MacroSyncManager",
    "SyncConfiguration",
    "MacroChange",
    "ChangeType",
    "SyncStatus",
    "KMFileMonitor",
    "FileChangeEvent",
    "WATCHDOG_AVAILABLE",
]