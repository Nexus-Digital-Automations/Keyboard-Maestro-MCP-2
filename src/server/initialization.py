"""
Component initialization and dependency management.

Handles initialization of core components like KM client, metadata extractor,
sync manager, and file monitor with proper dependency injection.
"""

import asyncio
import logging
from typing import Optional

from ..core.types import Duration
from ..integration import (
    KMClient,
    ConnectionConfig,
    ConnectionMethod,
    MacroSyncManager,
    SyncConfiguration,
    KMFileMonitor,
    FileChangeEvent,
    SyncStatus,
)
from ..integration.macro_metadata import MacroMetadataExtractor
from ..integration.smart_filtering import SmartMacroFilter

logger = logging.getLogger(__name__)

# Global component instances
_km_client: Optional[KMClient] = None
_metadata_extractor: Optional[MacroMetadataExtractor] = None
_smart_filter: Optional[SmartMacroFilter] = None
_sync_manager: Optional[MacroSyncManager] = None
_file_monitor: Optional[KMFileMonitor] = None


def initialize_components() -> None:
    """Initialize all server components."""
    global _smart_filter
    _smart_filter = SmartMacroFilter()
    logger.info("Server components initialized")


def get_km_client() -> KMClient:
    """Get or initialize Keyboard Maestro client."""
    global _km_client
    if _km_client is None:
        config = ConnectionConfig(
            method=ConnectionMethod.APPLESCRIPT,  # Primary communication method
            web_api_host="localhost",
            web_api_port=4490,  # Default KM web server port
            timeout=Duration.from_seconds(30),
            max_retries=3
        )
        _km_client = KMClient(config)
        logger.info("KM client initialized")
    return _km_client


def get_metadata_extractor() -> MacroMetadataExtractor:
    """Get or initialize metadata extractor."""
    global _metadata_extractor
    if _metadata_extractor is None:
        _metadata_extractor = MacroMetadataExtractor(get_km_client())
        logger.info("Metadata extractor initialized")
    return _metadata_extractor


def get_smart_filter() -> SmartMacroFilter:
    """Get the smart filter instance."""
    global _smart_filter
    if _smart_filter is None:
        _smart_filter = SmartMacroFilter()
    return _smart_filter


def get_sync_manager() -> MacroSyncManager:
    """Get or initialize synchronization manager."""
    global _sync_manager
    if _sync_manager is None:
        config = SyncConfiguration(
            base_poll_interval=Duration.from_seconds(30),
            fast_poll_interval=Duration.from_seconds(5),
            slow_poll_interval=Duration.from_seconds(120),
            cache_ttl=Duration.from_minutes(10)
        )
        _sync_manager = MacroSyncManager(
            get_km_client(),
            get_metadata_extractor(),
            config
        )
        logger.info("Sync manager initialized")
    return _sync_manager


def get_file_monitor() -> KMFileMonitor:
    """Get or initialize file monitor."""
    global _file_monitor
    if _file_monitor is None:
        def on_file_change(event: FileChangeEvent):
            """Handle file system changes."""
            logger.info(f"File change detected: {event.event_type} {event.file_path}")
            # Trigger sync manager to check for changes
            sync_mgr = get_sync_manager()
            if sync_mgr.sync_state.status == SyncStatus.ACTIVE:
                # Force a quick sync on file changes
                asyncio.create_task(sync_mgr.force_sync())
        
        _file_monitor = KMFileMonitor(on_file_change)
        logger.info("File monitor initialized")
    return _file_monitor