"""Modular Keyboard Maestro MCP Server.

This package contains the modularized components of the KM MCP server,
organized by functionality for better maintainability and testing.
"""

from .config import ServerConfig, get_server_config
from .initialization import (
    get_file_monitor,
    get_km_client,
    get_metadata_extractor,
    get_sync_manager,
    initialize_components,
)

__all__ = [
    "ServerConfig",
    "get_file_monitor",
    "get_km_client",
    "get_metadata_extractor",
    "get_server_config",
    "get_sync_manager",
    "initialize_components",
]
