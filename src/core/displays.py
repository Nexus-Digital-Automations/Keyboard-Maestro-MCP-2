"""
Advanced display and multi-monitor type definitions for sophisticated window management.

This module provides comprehensive display detection, coordinate mathematics, and
multi-monitor topology analysis for advanced window positioning and desktop
organization workflows.

Security: Screen bounds validation and coordinate system integrity
Performance: Efficient display enumeration with intelligent caching
Type Safety: Complete branded type system with contract-driven development
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Tuple
from enum import Enum
import asyncio

from src.core.either import Either
from src.core.errors import ValidationError, WindowError
from src.core.contracts import require, ensure


class DisplayArrangement(Enum):
    """Multi-monitor arrangement patterns for topology analysis."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    L_SHAPE = "l_shape"
    T_SHAPE = "t_shape"
    CUSTOM = "custom"


class WindowGridPattern(Enum):
    """Predefined window grid patterns for systematic organization."""
    GRID_2X2 = "2x2"
    GRID_3X3 = "3x3"
    GRID_4X2 = "4x2"
    GRID_2X3 = "2x3"
    GRID_1X2 = "1x2"
    GRID_2X1 = "2x1"
    SIDEBAR_MAIN = "sidebar_main"
    MAIN_SIDEBAR = "main_sidebar"
    THIRDS_HORIZONTAL = "thirds_horizontal"
    THIRDS_VERTICAL = "thirds_vertical"
    QUARTERS = "quarters"
    CUSTOM = "custom"


@dataclass(frozen=True)
class DisplayInfo:
    """Comprehensive display information with coordinate system support."""
    display_id: int
    name: str
    resolution: Tuple[int, int]  # (width, height)
    position: Tuple[int, int]    # (x, y) in global coordinates
    scale_factor: float
    is_main: bool
    color_space: str
    refresh_rate: Optional[float] = None
    
    @require(lambda self: self.resolution[0] > 0 and self.resolution[1] > 0)
    @require(lambda self: self.scale_factor > 0.0)
    @require(lambda self: len(self.name) > 0)
    def __post_init__(self):
        """Contract validation for display specification."""
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
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within display bounds."""
        dx, dy, dw, dh = self.bounds()
        return dx <= x < dx + dw and dy <= y < dy + dh
    
    def overlaps_with(self, other: 'DisplayInfo') -> bool:
        """Check if this display overlaps with another."""
        x1, y1, w1, h1 = self.bounds()
        x2, y2, w2, h2 = other.bounds()
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


@dataclass(frozen=True)
class GridCell:
    """Grid cell specification for precise window positioning."""
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1
    padding: int = 0  # Pixels of padding around cell
    
    @require(lambda self: self.row >= 0 and self.column >= 0)
    @require(lambda self: self.row_span > 0 and self.column_span > 0)
    @require(lambda self: self.padding >= 0)
    def __post_init__(self):
        """Contract validation for grid cell specification."""
        pass
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary representation."""
        return {
            "row": self.row,
            "column": self.column,
            "row_span": self.row_span,
            "column_span": self.column_span,
            "padding": self.padding
        }


@dataclass(frozen=True)
class WindowLayout:
    """Advanced window layout specification with display targeting."""
    name: str
    display_target: Optional[int] = None
    grid_pattern: WindowGridPattern = WindowGridPattern.GRID_2X2
    window_assignments: Dict[str, GridCell] = field(default_factory=dict)
    custom_positions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    global_padding: int = 10
    preserve_aspect_ratios: bool = True
    
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: self.global_padding >= 0)
    def __post_init__(self):
        """Contract validation for window layout specification."""
        pass
    
    def get_window_count(self) -> int:
        """Get total number of windows in layout."""
        return len(self.window_assignments) + len(self.custom_positions)


@dataclass(frozen=True)
class DisplayTopology:
    """Multi-monitor topology analysis and relationship mapping."""
    displays: List[DisplayInfo]
    main_display_id: int
    arrangement: DisplayArrangement
    total_bounds: Tuple[int, int, int, int]  # (min_x, min_y, total_width, total_height)
    
    @require(lambda self: len(self.displays) > 0)
    @require(lambda self: any(d.display_id == self.main_display_id for d in self.displays))
    def __post_init__(self):
        """Contract validation for display topology."""
        pass
    
    def get_display_by_id(self, display_id: int) -> Optional[DisplayInfo]:
        """Get display by ID."""
        for display in self.displays:
            if display.display_id == display_id:
                return display
        return None
    
    def get_main_display(self) -> DisplayInfo:
        """Get main display information."""
        for display in self.displays:
            if display.is_main:
                return display
        # Fallback to first display
        return self.displays[0]
    
    def get_display_at_point(self, x: int, y: int) -> Optional[DisplayInfo]:
        """Get display containing the specified point."""
        for display in self.displays:
            if display.contains_point(x, y):
                return display
        return None
    
    def calculate_relative_position(
        self, 
        source_display: DisplayInfo, 
        target_display: DisplayInfo, 
        relative_x: float, 
        relative_y: float
    ) -> Tuple[int, int]:
        """Calculate absolute position from relative coordinates."""
        # Convert relative position (0.0-1.0) to absolute coordinates
        target_x = int(target_display.position[0] + relative_x * target_display.resolution[0])
        target_y = int(target_display.position[1] + relative_y * target_display.resolution[1])
        
        return (target_x, target_y)


class DisplayManager:
    """Advanced display detection and management with caching."""
    
    def __init__(self):
        self._display_cache: Optional[DisplayTopology] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl: float = 5.0  # 5 seconds cache TTL
    
    async def detect_displays(self, force_refresh: bool = False) -> Either[WindowError, DisplayTopology]:
        """
        Detect and analyze all connected displays with intelligent caching.
        
        Architecture: Event-driven display detection with topology analysis
        Security: Bounds validation and coordinate system integrity
        Performance: Intelligent caching with TTL for efficiency
        """
        import time
        
        current_time = time.time()
        
        if (not force_refresh and 
            self._display_cache is not None and 
            self._cache_timestamp is not None and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return Either.right(self._display_cache)
        
        try:
            # Use macOS NSScreen APIs to enumerate displays
            displays_result = await self._enumerate_system_displays()
            if displays_result.is_left():
                return displays_result
            
            displays = displays_result.get_right()
            
            # Analyze display topology
            topology_result = self._analyze_display_topology(displays)
            if topology_result.is_left():
                return topology_result
            
            topology = topology_result.get_right()
            
            # Update cache
            self._display_cache = topology
            self._cache_timestamp = current_time
            
            return Either.right(topology)
            
        except Exception as e:
            return Either.left(WindowError(f"Display detection failed: {str(e)}"))
    
    async def _enumerate_system_displays(self) -> Either[WindowError, List[DisplayInfo]]:
        """Enumerate system displays using macOS APIs."""
        try:
            # Mock implementation for development - replace with actual NSScreen calls
            displays = [
                DisplayInfo(
                    display_id=0,
                    name="Built-in Retina Display",
                    resolution=(2560, 1600),
                    position=(0, 0),
                    scale_factor=2.0,
                    is_main=True,
                    color_space="P3",
                    refresh_rate=60.0
                ),
                DisplayInfo(
                    display_id=1,
                    name="External Display",
                    resolution=(1920, 1080),
                    position=(2560, 0),
                    scale_factor=1.0,
                    is_main=False,
                    color_space="sRGB",
                    refresh_rate=144.0
                )
            ]
            
            return Either.right(displays)
            
        except Exception as e:
            return Either.left(WindowError(f"Failed to enumerate displays: {str(e)}"))
    
    def _analyze_display_topology(self, displays: List[DisplayInfo]) -> Either[WindowError, DisplayTopology]:
        """Analyze display arrangement and calculate topology."""
        try:
            if not displays:
                return Either.left(WindowError("No displays detected"))
            
            # Find main display
            main_display = next((d for d in displays if d.is_main), displays[0])
            
            # Calculate arrangement pattern
            arrangement = self._determine_arrangement_pattern(displays)
            
            # Calculate total bounds
            total_bounds = self._calculate_total_bounds(displays)
            
            topology = DisplayTopology(
                displays=displays,
                main_display_id=main_display.display_id,
                arrangement=arrangement,
                total_bounds=total_bounds
            )
            
            return Either.right(topology)
            
        except Exception as e:
            return Either.left(WindowError(f"Topology analysis failed: {str(e)}"))
    
    def _determine_arrangement_pattern(self, displays: List[DisplayInfo]) -> DisplayArrangement:
        """Determine the arrangement pattern of multiple displays."""
        if len(displays) == 1:
            return DisplayArrangement.CUSTOM
        
        # Simple heuristics for common arrangements
        horizontal_aligned = all(d.position[1] == displays[0].position[1] for d in displays)
        vertical_aligned = all(d.position[0] == displays[0].position[0] for d in displays)
        
        if horizontal_aligned:
            return DisplayArrangement.HORIZONTAL
        elif vertical_aligned:
            return DisplayArrangement.VERTICAL
        else:
            return DisplayArrangement.CUSTOM
    
    def _calculate_total_bounds(self, displays: List[DisplayInfo]) -> Tuple[int, int, int, int]:
        """Calculate the total bounding rectangle for all displays."""
        if not displays:
            return (0, 0, 0, 0)
        
        min_x = min(d.position[0] for d in displays)
        min_y = min(d.position[1] for d in displays)
        max_x = max(d.position[0] + d.resolution[0] for d in displays)
        max_y = max(d.position[1] + d.resolution[1] for d in displays)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)


# Grid pattern specifications for common layouts
GRID_PATTERN_SPECS = {
    WindowGridPattern.GRID_2X2: {"rows": 2, "columns": 2},
    WindowGridPattern.GRID_3X3: {"rows": 3, "columns": 3},
    WindowGridPattern.GRID_4X2: {"rows": 2, "columns": 4},
    WindowGridPattern.GRID_2X3: {"rows": 3, "columns": 2},
    WindowGridPattern.GRID_1X2: {"rows": 1, "columns": 2},
    WindowGridPattern.GRID_2X1: {"rows": 2, "columns": 1},
    WindowGridPattern.THIRDS_HORIZONTAL: {"rows": 1, "columns": 3},
    WindowGridPattern.THIRDS_VERTICAL: {"rows": 3, "columns": 1},
    WindowGridPattern.QUARTERS: {"rows": 2, "columns": 2},
}


def get_grid_dimensions(pattern: WindowGridPattern) -> Tuple[int, int]:
    """Get grid dimensions (rows, columns) for a pattern."""
    if pattern in GRID_PATTERN_SPECS:
        spec = GRID_PATTERN_SPECS[pattern]
        return (spec["rows"], spec["columns"])
    else:
        return (2, 2)  # Default fallback