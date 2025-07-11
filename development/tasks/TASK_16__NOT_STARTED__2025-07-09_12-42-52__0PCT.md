# TASK_16: km_window_manager - Window Control

**Created By**: Agent_ADDER+ (High-Impact Tool Implementation) | **Priority**: MEDIUM | **Duration**: 3 hours
**Technique Focus**: Coordinate Validation + Screen Calculations + State Tracking + Multi-Monitor Support
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: TASK_12 (app control for window identification)
**Blocking**: None (standalone window management functionality)

## 📖 Required Reading (Complete before starting)
- [x] **development/protocols/KM_MCP.md**: km_window_manager specification (lines 753-767)
- [x] **src/applications/**: Application control patterns from TASK_12
- [x] **macOS Window Management**: Understanding window APIs and screen coordinates
- [x] **src/core/types.py**: Coordinate types and validation
- [x] **tests/TESTING.md**: Screen calculation and validation testing

## 🎯 Implementation Overview
Create a comprehensive window management system that enables AI assistants to control window positions, sizes, states, and arrangements across multiple monitors with precise coordinate calculations and validation.

<thinking>
Window management is essential for workspace automation:
1. Multi-Monitor Support: Handle multiple displays with different resolutions
2. Coordinate Systems: Validate screen coordinates and window bounds
3. State Management: Track window states (minimized, maximized, fullscreen)
4. Application Integration: Work with app control for window identification
5. Safety Validation: Prevent off-screen placement and invalid sizes
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Core Window Management Infrastructure
- [x] **Window types**: Define WindowSpec, ScreenInfo, WindowState, Position types
- [x] **Coordinate validation**: Screen bounds checking, multi-monitor support
- [x] **State tracking**: Monitor window states and position changes
- [x] **Screen detection**: Query available displays and their properties

### Phase 2: Window Operations & AppleScript Integration
- [x] **Position control**: Move windows to specific coordinates with validation
- [x] **Size control**: Resize windows with minimum/maximum size constraints
- [x] **State control**: Minimize, maximize, restore, fullscreen operations
- [x] **Multi-window support**: Handle applications with multiple windows

### Phase 3: Advanced Features & Calculations
- [x] **Smart positioning**: Snap to screen edges, center on screen, tile arrangements
- [x] **Multi-monitor logic**: Move windows between displays, monitor-specific positioning
- [x] **Window arrangement**: Predefined layouts and custom arrangements
- [x] **Collision detection**: Prevent window overlaps with smart positioning

### Phase 4: MCP Tool Integration
- [x] **Tool implementation**: km_window_manager MCP tool with operation modes
- [x] **Operation types**: move, resize, minimize, maximize, arrange, get_info
- [x] **Response formatting**: Window operation results with position metadata
- [x] **Testing integration**: Screen calculation and window operation tests

## 🔧 Implementation Files & Specifications

### New Files to Create:

#### src/windows/window_manager.py - Core Window Management
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

@dataclass(frozen=True)
class Position:
    """Type-safe screen position."""
    x: int
    y: int
    
    @require(lambda self: -10000 <= self.x <= 10000)
    @require(lambda self: -10000 <= self.y <= 10000)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class Size:
    """Type-safe window size."""
    width: int
    height: int
    
    @require(lambda self: self.width > 0 and self.height > 0)
    @require(lambda self: self.width <= 8192 and self.height <= 8192)
    def __post_init__(self):
        pass

class WindowState(Enum):
    """Window states."""
    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    FULLSCREEN = "fullscreen"

class WindowManager:
    """Secure window management with multi-monitor support."""
    
    @require(lambda app_id: app_id != "")
    @ensure(lambda result: result.is_right() or result.get_left().code in ["WINDOW_NOT_FOUND", "INVALID_POSITION"])
    async def move_window(
        self,
        app_identifier: str,
        position: Position,
        window_index: int = 0
    ) -> Either[KMError, Position]:
        """Move window to specific position with validation."""
        pass
    
    @require(lambda size: size.width > 0 and size.height > 0)
    async def resize_window(
        self,
        app_identifier: str,
        size: Size,
        window_index: int = 0
    ) -> Either[KMError, Size]:
        """Resize window with bounds checking."""
        pass
    
    def get_screen_info(self) -> List[Dict[str, Any]]:
        """Get information about available screens."""
        pass
    
    def validate_position(self, position: Position) -> bool:
        """Validate position is within screen bounds."""
        pass
```

#### src/server/tools/window_tools.py - MCP Tool Implementation
```python
async def km_window_manager(
    operation: Annotated[str, Field(
        description="Window management operation",
        pattern=r"^(move|resize|minimize|maximize|restore|arrange|get_info)$"
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
        ge=0
    )] = 0,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Comprehensive window management with multi-monitor support and validation.
    
    Operations:
    - move: Position window at specific coordinates
    - resize: Change window size with bounds checking
    - minimize: Minimize window to dock
    - maximize: Maximize window to screen bounds
    - restore: Restore window to normal state
    - arrange: Apply predefined window arrangements
    - get_info: Query window position, size, and state
    
    Features:
    - Multi-monitor support with screen targeting
    - Coordinate validation and bounds checking
    - Smart positioning with edge snapping
    - Window state management and tracking
    
    Returns window operation results with position and size metadata.
    """
    # Implementation details...
    pass
```

## ✅ Success Criteria
- [x] Complete coordinate validation with multi-monitor support
- [x] Support for move, resize, minimize, maximize, and arrangement operations
- [x] Real window management with macOS window APIs
- [x] Comprehensive error handling with position validation
- [x] Property-based testing for coordinate calculations and edge cases
- [x] Performance meets sub-3-second targets for window operations
- [x] Integration with application control for window identification
- [x] TESTING.md updated with window management and screen calculation tests
- [x] Documentation with multi-monitor setup and coordinate systems

## 🎨 Usage Examples

### Basic Window Operations
```python
# Move window to specific position
result = await client.call_tool("km_window_manager", {
    "operation": "move",
    "window_identifier": "com.apple.TextEdit",
    "position": {"x": 100, "y": 200},
    "screen": "main"
})

# Resize window
result = await client.call_tool("km_window_manager", {
    "operation": "resize",
    "window_identifier": "Safari",
    "size": {"width": 1200, "height": 800}
})
```