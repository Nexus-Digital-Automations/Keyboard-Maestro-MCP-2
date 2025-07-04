"""
Base utilities for tools modules.

Provides common imports and utilities to avoid circular dependencies.
"""

# Re-export commonly used components
try:
    from ..core import (
        MacroId,
        ExecutionResult,
        ValidationError,
        SecurityViolationError,
        PermissionDeniedError,
        ExecutionError,
        TimeoutError,
        get_default_engine,
        create_simple_macro,
    )
    from ..core.types import Duration
    from ..integration.macro_metadata import (
        EnhancedMacroMetadata,
        ActionCategory,
        TriggerCategory,
        ComplexityLevel,
    )
    from ..integration.smart_filtering import (
        SearchQuery,
        SearchScope,
        SortCriteria,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from core import (
        MacroId,
        ExecutionResult,
        ValidationError,
        SecurityViolationError,
        PermissionDeniedError,
        ExecutionError,
        TimeoutError,
        get_default_engine,
        create_simple_macro,
    )
    from core.types import Duration
    from integration.macro_metadata import (
        EnhancedMacroMetadata,
        ActionCategory,
        TriggerCategory,
        ComplexityLevel,
    )
    from integration.smart_filtering import (
        SearchQuery,
        SearchScope,
        SortCriteria,
    )


def get_server_utils():
    """Get server utilities with proper import handling."""
    try:
        from ..server_utils import (
            get_km_client,
            get_metadata_extractor,
            get_sync_manager,
            get_file_monitor,
            smart_filter
        )
        return {
            'get_km_client': get_km_client,
            'get_metadata_extractor': get_metadata_extractor,
            'get_sync_manager': get_sync_manager,
            'get_file_monitor': get_file_monitor,
            'smart_filter': smart_filter
        }
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        
        from server_utils import (
            get_km_client,
            get_metadata_extractor,
            get_sync_manager,
            get_file_monitor,
            smart_filter
        )
        return {
            'get_km_client': get_km_client,
            'get_metadata_extractor': get_metadata_extractor,
            'get_sync_manager': get_sync_manager,
            'get_file_monitor': get_file_monitor,
            'smart_filter': smart_filter
        }