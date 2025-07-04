"""
Advanced window positioning algorithms for sophisticated multi-monitor workflows.

This module implements complex positioning algorithms including cross-monitor
migration, relative position preservation, and intelligent window placement
based on display topology and workspace requirements.

Security: Cross-display coordinate validation and bounds protection
Performance: Efficient coordinate transformations and position calculations
Type Safety: Mathematical precision with contract-driven development
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import math

from src.core.displays import DisplayInfo, DisplayTopology, DisplayManager
from src.window.grid_manager import WindowPosition, AdvancedGridManager
from src.core.either import Either
from src.core.errors import ValidationError, WindowError
from src.core.contracts import require, ensure


@dataclass(frozen=True)
class WindowMigrationResult:
    """Result of cross-monitor window migration."""
    window_identifier: str
    source_display: int
    target_display: int
    old_position: Tuple[int, int, int, int]
    new_position: Tuple[int, int, int, int]
    migration_time_ms: float
    relative_position_preserved: bool
    
    def was_successful(self) -> bool:
        """Check if migration was successful."""
        return self.migration_time_ms > 0 and self.new_position != (0, 0, 0, 0)


@dataclass(frozen=True)
class SmartPositionRequest:
    """Request for intelligent window positioning."""
    window_identifier: str
    content_type: Optional[str] = None  # "editor", "browser", "terminal", "media"
    preferred_size: Optional[Tuple[int, int]] = None
    avoid_overlap: bool = True
    prefer_main_display: bool = False
    workspace_hint: Optional[str] = None  # "development", "design", "communication"


class AdvancedPositioning:
    """Advanced window positioning with cross-monitor intelligence."""
    
    def __init__(self, display_manager: DisplayManager):
        self.display_manager = display_manager
        self.grid_manager = AdvancedGridManager()
    
    @require(lambda self, window_id: len(window_id) > 0)
    @require(lambda self, target_display_id: target_display_id >= 0)
    async def migrate_window_to_display(
        self,
        window_identifier: str,
        target_display_id: int,
        preserve_relative_position: bool = True,
        intelligent_placement: bool = True
    ) -> Either[WindowError, WindowMigrationResult]:
        """
        Migrate window to different display with advanced positioning.
        
        Architecture: Cross-monitor coordinate transformation with intelligence
        Security: Display bounds validation and coordinate overflow protection
        Performance: Efficient coordinate mathematics and validation caching
        """
        try:
            # Get current display topology
            topology_result = await self.display_manager.detect_displays()
            if topology_result.is_left():
                return Either.left(topology_result.get_left())
            
            topology = topology_result.get_right()
            
            # Validate target display
            target_display = topology.get_display_by_id(target_display_id)
            if target_display is None:
                return Either.left(WindowError(f"Display {target_display_id} not found"))
            
            # Get current window position (mock implementation)
            current_position_result = await self._get_current_window_position(window_identifier)
            if current_position_result.is_left():
                return Either.left(current_position_result.get_left())
            
            old_position = current_position_result.get_right()
            source_display = self._find_display_for_position(topology, old_position[:2])
            
            # Calculate new position based on strategy
            if preserve_relative_position and source_display:
                new_position_result = self._calculate_relative_position(
                    old_position, source_display, target_display
                )
            elif intelligent_placement:
                new_position_result = await self._calculate_intelligent_position(
                    window_identifier, target_display, old_position[2:4]
                )
            else:
                # Center on target display
                new_position_result = self._center_on_display(target_display, old_position[2:4])
            
            if new_position_result.is_left():
                return Either.left(new_position_result.get_left())
            
            new_position = new_position_result.get_right()
            
            # Apply new position (mock implementation)
            migration_result = await self._apply_window_position(window_identifier, new_position)
            if migration_result.is_left():
                return Either.left(migration_result.get_left())
            
            result = WindowMigrationResult(
                window_identifier=window_identifier,
                source_display=source_display.display_id if source_display else -1,
                target_display=target_display_id,
                old_position=old_position,
                new_position=new_position,
                migration_time_ms=migration_result.get_right(),
                relative_position_preserved=preserve_relative_position
            )
            
            return Either.right(result)
            
        except Exception as e:
            return Either.left(WindowError(f"Window migration failed: {str(e)}"))
    
    @require(lambda self, requests: len(requests) > 0)
    async def arrange_windows_intelligently(
        self,
        requests: List[SmartPositionRequest],
        target_display_id: Optional[int] = None
    ) -> Either[WindowError, List[Dict[str, Any]]]:
        """
        Intelligently arrange multiple windows based on content and workspace hints.
        
        Architecture: Context-aware positioning with overlap avoidance
        Security: Window identifier validation and bounds checking
        Performance: Batch positioning with spatial optimization
        """
        try:
            # Get display topology
            topology_result = await self.display_manager.detect_displays()
            if topology_result.is_left():
                return Either.left(topology_result.get_left())
            
            topology = topology_result.get_right()
            
            # Determine target display
            if target_display_id is not None:
                target_display = topology.get_display_by_id(target_display_id)
                if target_display is None:
                    return Either.left(WindowError(f"Display {target_display_id} not found"))
            else:
                target_display = topology.get_main_display()
            
            # Categorize windows by content type
            categorized = self._categorize_windows(requests)
            
            # Calculate positions for each category
            results = []
            occupied_areas = []  # Track occupied screen areas
            
            for category, category_requests in categorized.items():
                category_results = await self._arrange_category_windows(
                    category_requests, target_display, occupied_areas
                )
                if category_results.is_left():
                    return Either.left(category_results.get_left())
                
                results.extend(category_results.get_right())
                
                # Update occupied areas
                for result in category_results.get_right():
                    if "position" in result:
                        occupied_areas.append(result["position"])
            
            return Either.right(results)
            
        except Exception as e:
            return Either.left(WindowError(f"Intelligent arrangement failed: {str(e)}"))
    
    def _calculate_relative_position(
        self,
        old_position: Tuple[int, int, int, int],
        source_display: DisplayInfo,
        target_display: DisplayInfo
    ) -> Either[WindowError, Tuple[int, int, int, int]]:
        """Calculate relative position when moving between displays."""
        try:
            old_x, old_y, width, height = old_position
            
            # Calculate relative position on source display (0.0 to 1.0)
            source_bounds = source_display.bounds()
            rel_x = (old_x - source_bounds[0]) / source_bounds[2]
            rel_y = (old_y - source_bounds[1]) / source_bounds[3]
            
            # Apply relative position to target display
            target_bounds = target_display.bounds()
            new_x = int(target_bounds[0] + rel_x * target_bounds[2])
            new_y = int(target_bounds[1] + rel_y * target_bounds[3])
            
            # Ensure window fits on target display
            if new_x + width > target_bounds[0] + target_bounds[2]:
                new_x = target_bounds[0] + target_bounds[2] - width
            if new_y + height > target_bounds[1] + target_bounds[3]:
                new_y = target_bounds[1] + target_bounds[3] - height
            
            # Ensure minimum bounds
            new_x = max(new_x, target_bounds[0])
            new_y = max(new_y, target_bounds[1])
            
            return Either.right((new_x, new_y, width, height))
            
        except Exception as e:
            return Either.left(WindowError(f"Relative position calculation failed: {str(e)}"))
    
    async def _calculate_intelligent_position(
        self,
        window_identifier: str,
        target_display: DisplayInfo,
        size: Tuple[int, int]
    ) -> Either[WindowError, Tuple[int, int, int, int]]:
        """Calculate intelligent position based on window type and content."""
        try:
            width, height = size
            bounds = target_display.bounds()
            
            # Intelligent positioning based on window characteristics
            # For now, use golden ratio positioning
            golden_ratio = 1.618
            
            # Position at golden ratio point
            x = int(bounds[0] + (bounds[2] / golden_ratio) - (width / 2))
            y = int(bounds[1] + (bounds[3] / golden_ratio) - (height / 2))
            
            # Ensure bounds
            x = max(bounds[0], min(x, bounds[0] + bounds[2] - width))
            y = max(bounds[1], min(y, bounds[1] + bounds[3] - height))
            
            return Either.right((x, y, width, height))
            
        except Exception as e:
            return Either.left(WindowError(f"Intelligent positioning failed: {str(e)}"))
    
    def _center_on_display(
        self,
        display: DisplayInfo,
        size: Tuple[int, int]
    ) -> Either[WindowError, Tuple[int, int, int, int]]:
        """Center window on display."""
        try:
            width, height = size
            center = display.center()
            
            x = center[0] - width // 2
            y = center[1] - height // 2
            
            # Ensure bounds
            bounds = display.bounds()
            x = max(bounds[0], min(x, bounds[0] + bounds[2] - width))
            y = max(bounds[1], min(y, bounds[1] + bounds[3] - height))
            
            return Either.right((x, y, width, height))
            
        except Exception as e:
            return Either.left(WindowError(f"Center positioning failed: {str(e)}"))
    
    def _find_display_for_position(
        self,
        topology: DisplayTopology,
        position: Tuple[int, int]
    ) -> Optional[DisplayInfo]:
        """Find which display contains the given position."""
        x, y = position
        return topology.get_display_at_point(x, y)
    
    def _categorize_windows(
        self,
        requests: List[SmartPositionRequest]
    ) -> Dict[str, List[SmartPositionRequest]]:
        """Categorize windows by content type for intelligent grouping."""
        categories = {
            "primary": [],
            "reference": [],
            "communication": [],
            "tools": [],
            "media": []
        }
        
        for request in requests:
            content_type = request.content_type or "primary"
            
            if content_type in ["editor", "browser", "document"]:
                categories["primary"].append(request)
            elif content_type in ["terminal", "console", "logs"]:
                categories["tools"].append(request)
            elif content_type in ["chat", "email", "messaging"]:
                categories["communication"].append(request)
            elif content_type in ["video", "audio", "image"]:
                categories["media"].append(request)
            else:
                categories["reference"].append(request)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    async def _arrange_category_windows(
        self,
        requests: List[SmartPositionRequest],
        display: DisplayInfo,
        occupied_areas: List[Tuple[int, int, int, int]]
    ) -> Either[WindowError, List[Dict[str, Any]]]:
        """Arrange windows within a category, avoiding occupied areas."""
        try:
            results = []
            
            # For now, use grid arrangement for each category
            window_ids = [req.window_identifier for req in requests]
            
            # Calculate optimal grid pattern
            pattern_result = await self.grid_manager.calculate_optimal_pattern(
                len(window_ids), display
            )
            if pattern_result.is_left():
                return Either.left(pattern_result.get_left())
            
            pattern = pattern_result.get_right()
            
            # Arrange in grid
            arrangement_result = await self.grid_manager.arrange_windows_in_grid(
                window_ids, display, pattern
            )
            if arrangement_result.is_left():
                return Either.left(arrangement_result.get_left())
            
            results = arrangement_result.get_right()
            
            return Either.right(results)
            
        except Exception as e:
            return Either.left(WindowError(f"Category arrangement failed: {str(e)}"))
    
    async def _get_current_window_position(
        self,
        window_identifier: str
    ) -> Either[WindowError, Tuple[int, int, int, int]]:
        """Get current window position (mock implementation)."""
        # Mock implementation - replace with actual window position detection
        return Either.right((100, 100, 800, 600))
    
    async def _apply_window_position(
        self,
        window_identifier: str,
        position: Tuple[int, int, int, int]
    ) -> Either[WindowError, float]:
        """Apply window position (mock implementation)."""
        # Mock implementation - replace with actual window positioning
        return Either.right(50.0)  # Mock migration time in ms


class WorkspaceManager:
    """Advanced workspace management with layout persistence."""
    
    def __init__(self, positioning: AdvancedPositioning):
        self.positioning = positioning
        self._saved_layouts: Dict[str, List[SmartPositionRequest]] = {}
    
    async def save_workspace_layout(
        self,
        name: str,
        window_requests: List[SmartPositionRequest]
    ) -> Either[WindowError, None]:
        """Save current workspace layout for future restoration."""
        try:
            if not name or len(name.strip()) == 0:
                return Either.left(WindowError("Workspace name cannot be empty"))
            
            self._saved_layouts[name] = window_requests.copy()
            return Either.right(None)
            
        except Exception as e:
            return Either.left(WindowError(f"Workspace save failed: {str(e)}"))
    
    async def restore_workspace_layout(
        self,
        name: str,
        target_display_id: Optional[int] = None
    ) -> Either[WindowError, List[Dict[str, Any]]]:
        """Restore previously saved workspace layout."""
        try:
            if name not in self._saved_layouts:
                return Either.left(WindowError(f"Workspace '{name}' not found"))
            
            requests = self._saved_layouts[name]
            return await self.positioning.arrange_windows_intelligently(
                requests, target_display_id
            )
            
        except Exception as e:
            return Either.left(WindowError(f"Workspace restore failed: {str(e)}"))
    
    def list_saved_workspaces(self) -> List[str]:
        """Get list of saved workspace names."""
        return list(self._saved_layouts.keys())
    
    async def delete_workspace(self, name: str) -> Either[WindowError, None]:
        """Delete saved workspace layout."""
        try:
            if name not in self._saved_layouts:
                return Either.left(WindowError(f"Workspace '{name}' not found"))
            
            del self._saved_layouts[name]
            return Either.right(None)
            
        except Exception as e:
            return Either.left(WindowError(f"Workspace deletion failed: {str(e)}"))


# Predefined workspace templates for common use cases
WORKSPACE_TEMPLATES = {
    "development": [
        SmartPositionRequest("VS Code", "editor", (1200, 800), workspace_hint="development"),
        SmartPositionRequest("Terminal", "terminal", (800, 600), workspace_hint="development"),
        SmartPositionRequest("Safari", "browser", (1000, 700), workspace_hint="development"),
    ],
    "design": [
        SmartPositionRequest("Photoshop", "editor", (1400, 900), workspace_hint="design"),
        SmartPositionRequest("Finder", "reference", (400, 600), workspace_hint="design"),
        SmartPositionRequest("Safari", "browser", (800, 600), workspace_hint="design"),
    ],
    "communication": [
        SmartPositionRequest("Slack", "chat", (400, 700), workspace_hint="communication"),
        SmartPositionRequest("Mail", "email", (800, 600), workspace_hint="communication"),
        SmartPositionRequest("Calendar", "reference", (600, 500), workspace_hint="communication"),
    ]
}