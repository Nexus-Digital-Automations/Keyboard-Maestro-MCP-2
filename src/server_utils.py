"""
Server Utilities and Shared Components

Shared utilities and singleton instances for the MCP server.
"""

import asyncio
import logging
from typing import Optional

from .core import get_default_engine
from .integration import MCPProtocolHandler, KMClient, ConnectionConfig, ConnectionMethod
from .integration.macro_metadata import MacroMetadataExtractor
from .integration.smart_filtering import SmartMacroFilter
from .integration.sync_manager import MacroSyncManager, SyncConfiguration
from .integration.file_monitor import KMFileMonitor, FileChangeEvent
from .core.types import Duration

logger = logging.getLogger(__name__)

# Global instances
engine = get_default_engine()
protocol_handler = MCPProtocolHandler()
km_client: Optional[KMClient] = None
metadata_extractor: Optional[MacroMetadataExtractor] = None
smart_filter = SmartMacroFilter()
sync_manager: Optional[MacroSyncManager] = None
file_monitor: Optional[KMFileMonitor] = None


def get_km_client() -> KMClient:
    """Get or initialize Keyboard Maestro client."""
    global km_client
    if km_client is None:
        config = ConnectionConfig(
            method=ConnectionMethod.APPLESCRIPT,  # Primary communication method
            web_api_host="localhost",
            web_api_port=4490,  # Default KM web server port
            timeout=Duration.from_seconds(30),
            max_retries=3
        )
        km_client = KMClient(config)
    return km_client


def get_metadata_extractor() -> MacroMetadataExtractor:
    """Get or initialize metadata extractor."""
    global metadata_extractor
    if metadata_extractor is None:
        metadata_extractor = MacroMetadataExtractor(get_km_client())
    return metadata_extractor


def get_sync_manager() -> MacroSyncManager:
    """Get or initialize synchronization manager."""
    global sync_manager
    if sync_manager is None:
        config = SyncConfiguration(
            base_poll_interval=Duration.from_seconds(30),
            fast_poll_interval=Duration.from_seconds(5),
            slow_poll_interval=Duration.from_seconds(120),
            cache_ttl=Duration.from_minutes(10)
        )
        sync_manager = MacroSyncManager(
            get_km_client(),
            get_metadata_extractor(),
            config
        )
    return sync_manager


def get_file_monitor() -> KMFileMonitor:
    """Get or initialize file monitor."""
    global file_monitor
    if file_monitor is None:
        def on_file_change(event: FileChangeEvent):
            """Handle file system changes."""
            logger.info(f"File change detected: {event.event_type} {event.file_path}")
            # Trigger sync manager to check for changes
            sync_mgr = get_sync_manager()
            from .integration.sync_manager import SyncStatus
            if sync_mgr.sync_state.status == SyncStatus.ACTIVE:
                # Force a quick sync on file changes
                asyncio.create_task(sync_mgr.force_sync())
        
        file_monitor = KMFileMonitor(on_file_change)
    return file_monitor