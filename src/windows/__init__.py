"""Window management module for macOS window control and multi-monitor support.

This module provides comprehensive window management capabilities including:
- Window positioning and resizing with coordinate validation
- Multi-monitor support with screen detection and targeting
- Window state management (minimize, maximize, restore, fullscreen)
- AppleScript integration for reliable window operations
- Security validation and bounds checking
"""

from .window_manager import (
    Position,
    ScreenInfo,
    Size,
    WindowManager,
    WindowOperationResult,
    WindowState,
)

__all__ = [
    "Position",
    "ScreenInfo",
    "Size",
    "WindowManager",
    "WindowOperationResult",
    "WindowState",
]
