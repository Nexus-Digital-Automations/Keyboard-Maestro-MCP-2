"""
Base utilities for tools modules.

Provides common imports and utilities to avoid circular dependencies.
"""

# Re-export commonly used components
try:
    from ..core import (
        ExecutionError,
        ExecutionResult,
        MacroId,
        PermissionDeniedError,
        SecurityViolationError,
        TimeoutError,
        ValidationError,
        create_simple_macro,
        get_default_engine,
    )
    from ..core.types import Duration
    from ..integration.macro_metadata import (
        ActionCategory,
        ComplexityLevel,
        EnhancedMacroMetadata,
        TriggerCategory,
    )
    from ..integration.smart_filtering import (
        SearchQuery,
        SearchScope,
        SortCriteria,
    )
except ImportError:
    # Fallback for direct execution
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def get_server_utils():
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
