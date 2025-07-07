"""Base utilities for tools modules.

Provides common imports and utilities to avoid circular dependencies.
"""
from __future__ import annotations

from typing import Any, Optional



def get_server_utils() -> None:
    """Get server utilities with proper import handling."""
    try:
        from ..server_utils import (
            get_file_monitor,
            get_km_client,
            get_metadata_extractor,
            get_sync_manager,
            smart_filter,
        )

        return {
            "get_km_client": get_km_client,
            "get_metadata_extractor": get_metadata_extractor,
            "get_sync_manager": get_sync_manager,
            "get_file_monitor": get_file_monitor,
            "smart_filter": smart_filter,
        }
    except ImportError:
        import os
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(__file__)))

        from server_utils import (
            get_file_monitor,
            get_km_client,
            get_metadata_extractor,
            get_sync_manager,
            smart_filter,
        )

        return {
            "get_km_client": get_km_client,
            "get_metadata_extractor": get_metadata_extractor,
            "get_sync_manager": get_sync_manager,
            "get_file_monitor": get_file_monitor,
            "smart_filter": smart_filter,
        }
