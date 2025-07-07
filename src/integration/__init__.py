"""Keyboard Maestro Integration Layer.

This module provides the integration layer connecting the core macro engine
to Keyboard Maestro automation system with event-driven architecture and
functional programming patterns.
"""

from .events import (
    DEFAULT_EVENT_PIPELINE,
    SECURITY_FOCUSED_PIPELINE,
    EventHandler,
    KMEvent,
    compose_event_handlers,
    get_default_event_pipeline,
    get_security_focused_pipeline,
)
from .file_monitor import WATCHDOG_AVAILABLE, FileChangeEvent, KMFileMonitor
from .km_client import ConnectionConfig, ConnectionMethod, KMClient
from .macro_metadata import (
    ActionCategory,
    ComplexityLevel,
    EnhancedMacroMetadata,
    MacroMetadataExtractor,
    TriggerCategory,
)
from .protocol import MCPProtocolHandler
from .security import sanitize_trigger_data, validate_km_input
from .smart_filtering import (
    FilterResult,
    SearchQuery,
    SearchScope,
    SmartMacroFilter,
    SortCriteria,
)
from .sync_manager import (
    ChangeType,
    MacroChange,
    MacroSyncManager,
    SyncConfiguration,
    SyncStatus,
)
from .triggers import TriggerInfo, TriggerState, update_trigger_state

__all__ = [
    "DEFAULT_EVENT_PIPELINE",
    "SECURITY_FOCUSED_PIPELINE",
    "WATCHDOG_AVAILABLE",
    "ActionCategory",
    "ChangeType",
    "ComplexityLevel",
    "ConnectionConfig",
    "ConnectionMethod",
    "EnhancedMacroMetadata",
    "EventHandler",
    "FileChangeEvent",
    "FilterResult",
    "KMClient",
    "KMEvent",
    "KMFileMonitor",
    "MCPProtocolHandler",
    "MacroChange",
    "MacroMetadataExtractor",
    "MacroSyncManager",
    "SearchQuery",
    "SearchScope",
    "SmartMacroFilter",
    "SortCriteria",
    "SyncConfiguration",
    "SyncStatus",
    "TriggerCategory",
    "TriggerInfo",
    "TriggerState",
    "compose_event_handlers",
    "get_default_event_pipeline",
    "get_security_focused_pipeline",
    "sanitize_trigger_data",
    "update_trigger_state",
    "validate_km_input",
]
