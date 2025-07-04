"""
Window Management MCP Tool

Comprehensive window control with multi-monitor support, coordinate validation,
and security boundaries for AI-assisted workspace automation.
"""

from typing import Dict, Any, Optional, List
from typing_extensions import Annotated
from pydantic import Field
import logging

from ...windows.window_manager import (
    WindowManager, Position, Size, WindowState, WindowArrangement
)
from ...applications.app_controller import AppIdentifier
from ...core.types import Duration
from ...integration.km_client import KMError

# MCP context import
from mcp import Context

# Initialize logger
logger = logging.getLogger(__name__)

# Global window manager instance
_window_manager = WindowManager()


async def km_window_manager(
    operation: Annotated[str, Field(
        description="Window management operation",
        pattern=r"^(move|resize|minimize|maximize|restore|arrange|get_info|get_screens)$"
    )],
    window_identifier: Annotated[str, Field(
        description="Application name, bundle ID, or window title",
        min_length=1,
        max_length=255
    )],
    position: Annotated[Optional[Dict[str, int]], Field(
        default=None,
        description="Target position {x, y} for move operation"
    )] = None,
    size: Annotated[Optional[Dict[str, int]], Field(
        default=None,
        description="Target size {width, height} for resize operation"
    )] = None,
    screen: Annotated[str, Field(
        default="main",
        description="Target screen (main, external, or index)",
        pattern=r"^(main|external|\d+)$"
    )] = "main",
    window_index: Annotated[int, Field(
        default=0,
        description="Window index for multi-window applications",
        ge=0,
        le=20
    )] = 0,
    arrangement: Annotated[Optional[str], Field(
        default=None,
        description="Predefined arrangement for arrange operation",
        pattern=r"^(left_half|right_half|top_half|bottom_half|top_left_quarter|top_right_quarter|bottom_left_quarter|bottom_right_quarter|center|maximize)$"
    )] = None,
    state: Annotated[Optional[str], Field(
        default=None,
        description="Target window state for minimize/maximize/restore operations",
        pattern=r"^(normal|minimized|maximized|fullscreen)$"
    )] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Comprehensive window management with multi-monitor support and validation.
    
    Operations:
    - move: Position window at specific coordinates with screen targeting
    - resize: Change window size with bounds checking and validation
    - minimize: Minimize window to dock (equivalent to state=minimized)
    - maximize: Maximize window to screen bounds (equivalent to state=maximized)  
    - restore: Restore window to normal state (equivalent to state=normal)
    - arrange: Apply predefined window arrangements (left_half, right_half, etc.)
    - get_info: Query window position, size, and state information
    - get_screens: List available screens with resolution and positioning data
    
    Features:
    - Multi-monitor support with screen targeting (main, external, or index)
    - Coordinate validation and bounds checking for safe positioning
    - Smart positioning with predefined arrangements and layouts
    - Window state management with proper transition handling
    - Security validation for application identifiers and parameters
    - Comprehensive error handling with detailed failure information
    
    Security:
    - Application identifier validation with pattern matching
    - Coordinate bounds checking to prevent off-screen placement
    - Size validation with minimum and maximum constraints
    - AppleScript injection prevention with parameter escaping
    - Permission validation for window control operations
    
    Performance:
    - Screen information caching with intelligent invalidation
    - Window state caching for reduced AppleScript calls
    - Sub-3-second response times for most operations
    - Efficient multi-step operations (move + resize for arrangements)
    
    Returns comprehensive window operation results with position, size, and metadata.
    """
    
    # Input validation and sanitization
    try:
        # Security: Validate application identifier format
        if not _is_valid_app_identifier(window_identifier):
            return {
                "success": False,
                "error": "INVALID_IDENTIFIER",
                "message": f"Invalid application identifier format: {window_identifier}",
                "timestamp": _get_timestamp()
            }
        
        # Create application identifier
        app_id = _create_app_identifier(window_identifier)
        
        # Route to appropriate operation handler
        if operation == "move":
            return await _handle_move_operation(app_id, position, window_index, screen, ctx)
        elif operation == "resize":
            return await _handle_resize_operation(app_id, size, window_index, ctx)
        elif operation in ["minimize", "maximize", "restore"]:
            target_state = _map_operation_to_state(operation, state)
            return await _handle_state_operation(app_id, target_state, window_index, ctx)
        elif operation == "arrange":
            return await _handle_arrange_operation(app_id, arrangement, window_index, screen, ctx)
        elif operation == "get_info":
            return await _handle_get_info_operation(app_id, window_index, ctx)
        elif operation == "get_screens":
            return await _handle_get_screens_operation(ctx)
        else:
            return {
                "success": False,
                "error": "INVALID_OPERATION",
                "message": f"Unsupported operation: {operation}",
                "timestamp": _get_timestamp()
            }
            
    except Exception as e:
        logger.error(f"Window manager operation failed: {str(e)}")
        return {
            "success": False,
            "error": "EXECUTION_ERROR",
            "message": f"Operation failed: {str(e)}",
            "timestamp": _get_timestamp()
        }


async def _handle_move_operation(
    app_id: AppIdentifier, 
    position_dict: Optional[Dict[str, int]], 
    window_index: int, 
    screen: str,
    ctx: Optional[Context]
) -> Dict[str, Any]:
    """Handle window move operation with validation."""
    
    if not position_dict or "x" not in position_dict or "y" not in position_dict:
        return {
            "success": False,
            "error": "MISSING_POSITION",
            "message": "Position coordinates {x, y} required for move operation",
            "timestamp": _get_timestamp()
        }
    
    try:
        # Create position with validation
        position = Position(position_dict["x"], position_dict["y"])
        
        # Execute move operation
        result = await _window_manager.move_window(
            app_id.primary_identifier(),
            position,
            window_index,
            screen
        )
        
        if result.is_right():
            operation_result = result.get_right()
            return {
                "success": True,
                "operation": "move",
                "window": _format_window_info(operation_result.window_info),
                "execution_time_ms": int(operation_result.operation_time.total_seconds() * 1000),
                "details": operation_result.details,
                "timestamp": _get_timestamp()
            }
        else:
            error = result.get_left()
            return {
                "success": False,
                "error": error.code,
                "message": error.message,
                "timestamp": _get_timestamp()
            }
            
    except ValueError as e:
        return {
            "success": False,
            "error": "INVALID_POSITION",
            "message": f"Invalid position coordinates: {str(e)}",
            "timestamp": _get_timestamp()
        }


async def _handle_resize_operation(
    app_id: AppIdentifier,
    size_dict: Optional[Dict[str, int]],
    window_index: int,
    ctx: Optional[Context]
) -> Dict[str, Any]:
    """Handle window resize operation with validation."""
    
    if not size_dict or "width" not in size_dict or "height" not in size_dict:
        return {
            "success": False,
            "error": "MISSING_SIZE",
            "message": "Size dimensions {width, height} required for resize operation",
            "timestamp": _get_timestamp()
        }
    
    try:
        # Create size with validation
        size = Size(size_dict["width"], size_dict["height"])
        
        # Execute resize operation
        result = await _window_manager.resize_window(
            app_id.primary_identifier(),
            size,
            window_index
        )
        
        if result.is_right():
            operation_result = result.get_right()
            return {
                "success": True,
                "operation": "resize",
                "window": _format_window_info(operation_result.window_info),
                "execution_time_ms": int(operation_result.operation_time.total_seconds() * 1000),
                "details": operation_result.details,
                "timestamp": _get_timestamp()
            }
        else:
            error = result.get_left()
            return {
                "success": False,
                "error": error.code,
                "message": error.message,
                "timestamp": _get_timestamp()
            }
            
    except ValueError as e:
        return {
            "success": False,
            "error": "INVALID_SIZE",
            "message": f"Invalid size dimensions: {str(e)}",
            "timestamp": _get_timestamp()
        }


async def _handle_state_operation(
    app_id: AppIdentifier,
    target_state: WindowState,
    window_index: int,
    ctx: Optional[Context]
) -> Dict[str, Any]:
    """Handle window state change operation."""
    
    try:
        # Execute state change operation
        result = await _window_manager.set_window_state(
            app_id.primary_identifier(),
            target_state,
            window_index
        )
        
        if result.is_right():
            operation_result = result.get_right()
            return {
                "success": True,
                "operation": f"set_state_{target_state.value}",
                "window": _format_window_info(operation_result.window_info),
                "execution_time_ms": int(operation_result.operation_time.total_seconds() * 1000),
                "details": operation_result.details,
                "timestamp": _get_timestamp()
            }
        else:
            error = result.get_left()
            return {
                "success": False,
                "error": error.code,
                "message": error.message,
                "timestamp": _get_timestamp()
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": "STATE_CHANGE_ERROR",
            "message": f"State change failed: {str(e)}",
            "timestamp": _get_timestamp()
        }


async def _handle_arrange_operation(
    app_id: AppIdentifier,
    arrangement_str: Optional[str],
    window_index: int,
    screen: str,
    ctx: Optional[Context]
) -> Dict[str, Any]:
    """Handle window arrangement operation."""
    
    if not arrangement_str:
        return {
            "success": False,
            "error": "MISSING_ARRANGEMENT",
            "message": "Arrangement type required for arrange operation",
            "timestamp": _get_timestamp()
        }
    
    try:
        # Map string to arrangement enum
        arrangement = WindowArrangement(arrangement_str)
        
        # Execute arrangement operation
        result = await _window_manager.arrange_window(
            app_id.primary_identifier(),
            arrangement,
            window_index,
            screen
        )
        
        if result.is_right():
            operation_result = result.get_right()
            return {
                "success": True,
                "operation": f"arrange_{arrangement_str}",
                "window": _format_window_info(operation_result.window_info),
                "execution_time_ms": int(operation_result.operation_time.total_seconds() * 1000),
                "details": operation_result.details,
                "screen": screen,
                "timestamp": _get_timestamp()
            }
        else:
            error = result.get_left()
            return {
                "success": False,
                "error": error.code,
                "message": error.message,
                "timestamp": _get_timestamp()
            }
            
    except ValueError as e:
        return {
            "success": False,
            "error": "INVALID_ARRANGEMENT",
            "message": f"Invalid arrangement type: {arrangement_str}",
            "timestamp": _get_timestamp()
        }


async def _handle_get_info_operation(
    app_id: AppIdentifier,
    window_index: int,
    ctx: Optional[Context]
) -> Dict[str, Any]:
    """Handle get window info operation."""
    
    try:
        # Get window information
        result = await _window_manager.get_window_info(
            app_id.primary_identifier(),
            window_index
        )
        
        if result.is_right():
            window_info = result.get_right()
            return {
                "success": True,
                "operation": "get_info",
                "window": _format_window_info(window_info),
                "timestamp": _get_timestamp()
            }
        else:
            error = result.get_left()
            return {
                "success": False,
                "error": error.code,
                "message": error.message,
                "timestamp": _get_timestamp()
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": "INFO_QUERY_ERROR",
            "message": f"Window info query failed: {str(e)}",
            "timestamp": _get_timestamp()
        }


async def _handle_get_screens_operation(ctx: Optional[Context]) -> Dict[str, Any]:
    """Handle get screens operation."""
    
    try:
        # Get screen information
        screens = await _window_manager.get_screen_info()
        
        return {
            "success": True,
            "operation": "get_screens",
            "screens": [_format_screen_info(screen) for screen in screens],
            "screen_count": len(screens),
            "timestamp": _get_timestamp()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": "SCREEN_QUERY_ERROR",
            "message": f"Screen info query failed: {str(e)}",
            "timestamp": _get_timestamp()
        }


# Utility functions

def _is_valid_app_identifier(identifier: str) -> bool:
    """Validate application identifier format."""
    if not identifier or len(identifier) == 0 or len(identifier) > 255:
        return False
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'[<>"|*?]',  # Invalid filename characters
        r'^\s*$',     # Only whitespace
        r'\.\./',     # Path traversal
        r'[;&|`$]',   # Shell injection characters
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, identifier):
            return False
    
    return True


def _create_app_identifier(identifier: str) -> AppIdentifier:
    """Create AppIdentifier from string."""
    # Determine if it's a bundle ID or app name
    if '.' in identifier and not identifier.startswith('.') and not identifier.endswith('.'):
        # Likely a bundle ID
        return AppIdentifier(bundle_id=identifier)
    else:
        # Treat as app name
        return AppIdentifier(app_name=identifier)


def _map_operation_to_state(operation: str, state_override: Optional[str]) -> WindowState:
    """Map operation string to WindowState enum."""
    if state_override:
        return WindowState(state_override)
    
    if operation == "minimize":
        return WindowState.MINIMIZED
    elif operation == "maximize":
        return WindowState.MAXIMIZED
    elif operation == "restore":
        return WindowState.NORMAL
    else:
        return WindowState.NORMAL


def _format_window_info(window_info) -> Dict[str, Any]:
    """Format window information for response."""
    if not window_info:
        return {}
    
    return {
        "app_identifier": window_info.app_identifier,
        "window_index": window_info.window_index,
        "position": {
            "x": window_info.position.x,
            "y": window_info.position.y
        },
        "size": {
            "width": window_info.size.width,
            "height": window_info.size.height
        },
        "state": window_info.state.value,
        "title": window_info.title,
        "bounds": {
            "x": window_info.position.x,
            "y": window_info.position.y,
            "width": window_info.size.width,
            "height": window_info.size.height
        }
    }


def _format_screen_info(screen_info) -> Dict[str, Any]:
    """Format screen information for response."""
    return {
        "screen_id": screen_info.screen_id,
        "name": screen_info.name,
        "is_main": screen_info.is_main,
        "origin": {
            "x": screen_info.origin.x,
            "y": screen_info.origin.y
        },
        "size": {
            "width": screen_info.size.width,
            "height": screen_info.size.height
        },
        "bounds": {
            "x": screen_info.origin.x,
            "y": screen_info.origin.y,
            "width": screen_info.size.width,
            "height": screen_info.size.height
        },
        "center": {
            "x": screen_info.center_position().x,
            "y": screen_info.center_position().y
        }
    }


def _get_timestamp() -> str:
    """Get current timestamp for responses."""
    from datetime import datetime
    return datetime.now().isoformat()


# Add missing import for regex
import re