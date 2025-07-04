# TASK_25: km_window_manager_advanced - Multi-Monitor Window Positioning

**Created By**: Agent_ADDER+ (Protocol Gap Analysis) | **Priority**: MEDIUM | **Duration**: 4 hours
**Technique Focus**: Coordinate Mathematics + Multi-Monitor API + Functional Programming
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: IN_PROGRESS
**Assigned**: Agent_ADDER+
**Dependencies**: TASK_12 (app control) for application window management
**Blocking**: Advanced desktop organization and multi-monitor workflows

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Advanced window management
- [ ] **Current Implementation**: src/main.py lines 487-530 - Existing km_window_manager tool
- [ ] **Foundation**: src/core/types.py - Coordinate and screen type definitions
- [ ] **Multi-Monitor APIs**: macOS NSScreen APIs and coordinate system documentation
- [ ] **Testing Framework**: tests/TESTING.md - Window management testing requirements

## 🎯 Problem Analysis
**Classification**: Enhancement of Existing Functionality
**Gap Identified**: Basic window manager exists but lacks advanced multi-monitor features
**Impact**: Limited desktop organization capabilities for complex multi-monitor setups

<thinking>
Root Cause Analysis:
1. Current window manager (TASK_16) provides basic operations but limited multi-monitor support
2. Missing advanced window arrangement patterns for productivity workflows
3. No intelligent screen detection or cross-display positioning algorithms
4. Limited support for complex window grid systems and workspace management
5. Need enhanced coordinate calculations for multi-monitor math
6. Advanced users need sophisticated window organization tools
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Multi-Monitor Infrastructure
- [ ] **Screen detection**: Advanced screen enumeration and properties analysis
- [ ] **Coordinate systems**: Complex multi-monitor coordinate mathematics
- [ ] **Display topology**: Screen relationship mapping and boundary detection
- [ ] **Resolution handling**: DPI scaling and resolution-aware positioning

### Phase 2: Advanced Window Arrangements
- [ ] **Grid systems**: Flexible window grid layouts (3x3, 4x2, custom patterns)
- [ ] **Workspace layouts**: Predefined workspace arrangements for different tasks
- [ ] **Window chains**: Sequential window positioning workflows
- [ ] **Smart positioning**: Intelligent window placement based on content and usage

### Phase 3: Cross-Monitor Operations
- [ ] **Display targeting**: Precise screen selection and window migration
- [ ] **Boundary calculations**: Cross-screen positioning and spanning
- [ ] **Virtual desktop integration**: macOS Spaces and Mission Control integration
- [ ] **Monitor profiles**: Saved configurations for different monitor setups

### Phase 4: Enhanced Integration
- [ ] **Application-aware positioning**: App-specific window management rules
- [ ] **Workflow automation**: Complex window arrangement sequences
- [ ] **Property-based tests**: Hypothesis validation for coordinate mathematics
- [ ] **TESTING.md update**: Advanced window management test coverage

## 🔧 Implementation Files & Specifications
```
src/server/tools/advanced_window_tools.py    # Advanced window management tool
src/core/displays.py                         # Multi-monitor type definitions
src/window/advanced_positioning.py           # Complex positioning algorithms
src/window/grid_manager.py                   # Grid layout and arrangement systems
src/window/workspace_manager.py              # Workspace and layout management
tests/tools/test_advanced_window_tools.py    # Unit and integration tests
tests/property_tests/test_window_math.py     # Property-based coordinate validation
```

### km_window_manager_advanced Tool Specification
```python
@mcp.tool()
async def km_window_manager_advanced(
    operation: str,                          # grid_layout|workspace_setup|cross_monitor_move|smart_arrange
    window_targets: List[str],               # List of application/window identifiers
    layout_pattern: Optional[str] = None,    # Grid pattern (3x3, 4x2, custom)
    target_displays: Optional[List[int]] = None,  # Target display indices
    workspace_name: Optional[str] = None,    # Named workspace configuration
    positioning_rules: Optional[Dict] = None, # Custom positioning rules
    preserve_ratios: bool = True,            # Maintain aspect ratios
    animate_transitions: bool = False,       # Smooth window animations
    save_layout: bool = False,               # Save layout for future use
    ctx = None
) -> Dict[str, Any]:
```

### Advanced Display Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple
from enum import Enum

class DisplayArrangement(Enum):
    """Multi-monitor arrangement patterns."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    L_SHAPE = "l_shape"
    CUSTOM = "custom"

class WindowGridPattern(Enum):
    """Predefined window grid patterns."""
    GRID_2X2 = "2x2"
    GRID_3X3 = "3x3"
    GRID_4X2 = "4x2"
    GRID_2X3 = "2x3"
    SIDEBAR_MAIN = "sidebar_main"
    MAIN_SIDEBAR = "main_sidebar"
    THIRDS = "thirds"
    QUARTERS = "quarters"
    CUSTOM = "custom"

@dataclass(frozen=True)
class DisplayInfo:
    """Comprehensive display information."""
    display_id: int
    name: str
    resolution: Tuple[int, int]  # (width, height)
    position: Tuple[int, int]    # (x, y) in global coordinates
    scale_factor: float
    is_main: bool
    color_space: str
    
    @require(lambda self: self.resolution[0] > 0 and self.resolution[1] > 0)
    @require(lambda self: self.scale_factor > 0.0)
    def __post_init__(self):
        pass
    
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get display bounds as (x, y, width, height)."""
        return (self.position[0], self.position[1], self.resolution[0], self.resolution[1])
    
    def center(self) -> Tuple[int, int]:
        """Get display center coordinates."""
        return (
            self.position[0] + self.resolution[0] // 2,
            self.position[1] + self.resolution[1] // 2
        )

@dataclass(frozen=True)
class GridCell:
    """Grid cell specification for window positioning."""
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1
    
    @require(lambda self: self.row >= 0 and self.column >= 0)
    @require(lambda self: self.row_span > 0 and self.column_span > 0)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class WindowLayout:
    """Advanced window layout specification."""
    name: str
    display_target: Optional[int] = None
    grid_pattern: WindowGridPattern = WindowGridPattern.GRID_2X2
    window_assignments: Dict[str, GridCell] = field(default_factory=dict)
    custom_positions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    
    @require(lambda self: len(self.name) > 0)
    def __post_init__(self):
        pass

class AdvancedWindowManager:
    """Advanced multi-monitor window management."""
    
    def __init__(self):
        self._display_cache: Optional[List[DisplayInfo]] = None
        self._layout_cache: Dict[str, WindowLayout] = {}
    
    async def detect_displays(self, force_refresh: bool = False) -> Either[WindowError, List[DisplayInfo]]:
        """Detect and analyze all connected displays."""
        if self._display_cache is None or force_refresh:
            # Use macOS APIs to enumerate displays
            displays = await self._enumerate_system_displays()
            self._display_cache = displays
        
        return Either.right(self._display_cache)
    
    async def calculate_grid_positions(
        self,
        display: DisplayInfo,
        pattern: WindowGridPattern,
        window_count: int
    ) -> Either[WindowError, List[Tuple[int, int, int, int]]]:
        """Calculate window positions for grid layout."""
        bounds = display.bounds()
        
        if pattern == WindowGridPattern.GRID_2X2:
            return self._calculate_2x2_grid(bounds, window_count)
        elif pattern == WindowGridPattern.GRID_3X3:
            return self._calculate_3x3_grid(bounds, window_count)
        elif pattern == WindowGridPattern.SIDEBAR_MAIN:
            return self._calculate_sidebar_layout(bounds, window_count)
        else:
            return Either.left(WindowError("Unsupported grid pattern"))
    
    async def arrange_windows_on_grid(
        self,
        window_identifiers: List[str],
        target_display: Optional[int] = None,
        pattern: WindowGridPattern = WindowGridPattern.GRID_2X2
    ) -> Either[WindowError, List[Dict[str, Any]]]:
        """Arrange windows in grid pattern on specified display."""
        displays_result = await self.detect_displays()
        if displays_result.is_left():
            return Either.left(displays_result.get_left())
        
        displays = displays_result.get_right()
        target = displays[target_display] if target_display is not None else displays[0]
        
        positions_result = await self.calculate_grid_positions(target, pattern, len(window_identifiers))
        if positions_result.is_left():
            return Either.left(positions_result.get_left())
        
        positions = positions_result.get_right()
        results = []
        
        for window_id, position in zip(window_identifiers, positions):
            result = await self._position_window(window_id, position)
            results.append(result)
        
        return Either.right(results)
    
    async def move_window_to_display(
        self,
        window_identifier: str,
        target_display: int,
        preserve_relative_position: bool = True
    ) -> Either[WindowError, Dict[str, Any]]:
        """Move window to different display with optional position preservation."""
        displays_result = await self.detect_displays()
        if displays_result.is_left():
            return Either.left(displays_result.get_left())
        
        displays = displays_result.get_right()
        if target_display >= len(displays):
            return Either.left(WindowError("Invalid display index"))
        
        target = displays[target_display]
        
        if preserve_relative_position:
            # Calculate relative position on current display and map to new display
            current_pos = await self._get_window_position(window_identifier)
            if current_pos.is_left():
                return Either.left(current_pos.get_left())
            
            new_pos = self._calculate_relative_position(current_pos.get_right(), target)
            return await self._position_window(window_identifier, new_pos)
        else:
            # Center on new display
            center = target.center()
            return await self._position_window(window_identifier, center)
    
    def _calculate_2x2_grid(self, bounds: Tuple[int, int, int, int], count: int) -> Either[WindowError, List[Tuple[int, int, int, int]]]:
        """Calculate 2x2 grid positions."""
        x, y, width, height = bounds
        cell_width = width // 2
        cell_height = height // 2
        
        positions = [
            (x, y, cell_width, cell_height),                    # Top-left
            (x + cell_width, y, cell_width, cell_height),       # Top-right
            (x, y + cell_height, cell_width, cell_height),      # Bottom-left
            (x + cell_width, y + cell_height, cell_width, cell_height)  # Bottom-right
        ]
        
        return Either.right(positions[:count])
    
    def _calculate_3x3_grid(self, bounds: Tuple[int, int, int, int], count: int) -> Either[WindowError, List[Tuple[int, int, int, int]]]:
        """Calculate 3x3 grid positions."""
        x, y, width, height = bounds
        cell_width = width // 3
        cell_height = height // 3
        
        positions = []
        for row in range(3):
            for col in range(3):
                if len(positions) >= count:
                    break
                pos = (
                    x + col * cell_width,
                    y + row * cell_height,
                    cell_width,
                    cell_height
                )
                positions.append(pos)
        
        return Either.right(positions)
    
    def _calculate_sidebar_layout(self, bounds: Tuple[int, int, int, int], count: int) -> Either[WindowError, List[Tuple[int, int, int, int]]]:
        """Calculate sidebar + main area layout."""
        x, y, width, height = bounds
        sidebar_width = width // 3
        main_width = width - sidebar_width
        
        positions = [
            (x, y, sidebar_width, height),                      # Sidebar
            (x + sidebar_width, y, main_width, height)          # Main area
        ]
        
        return Either.right(positions[:count])
```

## 🔒 Security Implementation
```python
class AdvancedWindowValidator:
    """Security validation for advanced window operations."""
    
    @staticmethod
    def validate_display_index(display_index: int, available_displays: int) -> Either[SecurityError, None]:
        """Validate display index is within bounds."""
        if display_index < 0 or display_index >= available_displays:
            return Either.left(SecurityError("Invalid display index"))
        return Either.right(None)
    
    @staticmethod
    def validate_grid_dimensions(pattern: WindowGridPattern, window_count: int) -> Either[SecurityError, None]:
        """Validate grid pattern can accommodate window count."""
        max_windows = {
            WindowGridPattern.GRID_2X2: 4,
            WindowGridPattern.GRID_3X3: 9,
            WindowGridPattern.GRID_4X2: 8,
            WindowGridPattern.SIDEBAR_MAIN: 2,
            WindowGridPattern.THIRDS: 3,
            WindowGridPattern.QUARTERS: 4
        }
        
        if pattern in max_windows and window_count > max_windows[pattern]:
            return Either.left(SecurityError(f"Too many windows for {pattern.value} pattern"))
        
        return Either.right(None)
    
    @staticmethod
    def validate_window_bounds(x: int, y: int, width: int, height: int, display_bounds: Tuple[int, int, int, int]) -> Either[SecurityError, None]:
        """Validate window position is within display bounds."""
        dx, dy, dw, dh = display_bounds
        
        if x < dx or y < dy:
            return Either.left(SecurityError("Window position outside display bounds"))
        
        if x + width > dx + dw or y + height > dy + dh:
            return Either.left(SecurityError("Window extends beyond display bounds"))
        
        if width < 100 or height < 100:
            return Either.left(SecurityError("Window too small (minimum 100x100)"))
        
        return Either.right(None)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=9))
def test_grid_calculation_properties(window_count):
    """Property: Grid calculations should handle all valid window counts."""
    manager = AdvancedWindowManager()
    display = DisplayInfo(0, "Test", (1920, 1080), (0, 0), 1.0, True, "sRGB")
    
    for pattern in WindowGridPattern:
        if pattern != WindowGridPattern.CUSTOM:
            result = manager._calculate_grid_positions(display, pattern, window_count)
            if result.is_right():
                positions = result.get_right()
                assert len(positions) <= window_count
                for pos in positions:
                    assert len(pos) == 4  # (x, y, width, height)

@given(st.integers(min_value=0, max_value=3), st.integers(min_value=1, max_value=5))
def test_display_targeting_properties(display_index, available_displays):
    """Property: Display targeting should validate index bounds."""
    validation = AdvancedWindowValidator.validate_display_index(display_index, available_displays)
    
    if display_index < available_displays:
        assert validation.is_right()
    else:
        assert validation.is_left()

@given(st.integers(min_value=100, max_value=2000), st.integers(min_value=100, max_value=2000))
def test_window_bounds_properties(width, height):
    """Property: Window bounds should respect minimum sizes and display limits."""
    display_bounds = (0, 0, 1920, 1080)
    x, y = 0, 0
    
    validation = AdvancedWindowValidator.validate_window_bounds(x, y, width, height, display_bounds)
    
    if width <= 1920 and height <= 1080:
        assert validation.is_right()
    else:
        assert validation.is_left()
```

## 🏗️ Modularity Strategy
- **advanced_window_tools.py**: Main MCP tool interface (<250 lines)
- **displays.py**: Display detection and multi-monitor types (<200 lines)
- **advanced_positioning.py**: Complex positioning algorithms (<300 lines)
- **grid_manager.py**: Grid layout calculations (<250 lines)
- **workspace_manager.py**: Workspace and layout management (<200 lines)

## 📋 Advanced Window Management Examples

### Multi-Monitor Grid Layout
```python
# Example: Arrange 4 windows in 2x2 grid on external monitor
result = await advanced_window_manager.arrange_windows_on_grid(
    window_identifiers=["Safari", "Terminal", "VS Code", "Slack"],
    target_display=1,  # External monitor
    pattern=WindowGridPattern.GRID_2X2
)
```

### Cross-Monitor Window Migration
```python
# Example: Move window to different display preserving relative position
result = await advanced_window_manager.move_window_to_display(
    window_identifier="Photoshop",
    target_display=2,
    preserve_relative_position=True
)
```

### Workspace Layout Setup
```python
# Example: Setup development workspace across multiple monitors
layout = WindowLayout(
    name="Development",
    window_assignments={
        "VS Code": GridCell(0, 1, 2, 2),      # Main editor area
        "Terminal": GridCell(2, 1, 1, 2),     # Bottom panel
        "Safari": GridCell(0, 0, 3, 1),       # Left sidebar
    }
)

result = await advanced_window_manager.apply_workspace_layout(layout)
```

## ✅ Success Criteria
- Complete advanced window management with multi-monitor support and grid layouts
- Comprehensive coordinate mathematics with cross-display positioning algorithms
- Property-based tests validate coordinate calculations across all display configurations
- Integration with existing window manager (TASK_16) for enhanced functionality
- Performance: <500ms for complex multi-window arrangements, <100ms for single window operations
- Documentation: Complete API documentation with multi-monitor setup examples
- TESTING.md shows 95%+ test coverage with all coordinate mathematics and boundary tests passing
- Tool enables sophisticated desktop organization and productivity workflows

## 🔄 Integration Points
- **TASK_16 (km_window_manager)**: Extends basic window operations with advanced features
- **TASK_12 (km_app_control)**: Application control for window targeting and identification
- **TASK_21/22 (conditions/control_flow)**: Intelligent window arrangement based on conditions
- **Future Workspace Tasks**: Foundation for advanced desktop automation workflows
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- Enhances existing window manager with advanced multi-monitor capabilities
- Essential for sophisticated desktop organization and productivity workflows
- Complex coordinate mathematics require careful validation and testing
- Must maintain functional programming patterns for testability and composability
- Success here enables advanced desktop automation for power users and complex workflows
- Foundation for future workspace management and desktop organization features