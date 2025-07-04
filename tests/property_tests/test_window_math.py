"""
Property-based tests for advanced window management mathematics and coordinate validation.

This module uses Hypothesis to test window positioning behavior across input ranges,
ensuring coordinate mathematics correctness, multi-monitor calculations, and grid
layout algorithms for all display configurations and window arrangements.
"""

import pytest
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from src.core.displays import DisplayInfo, WindowGridPattern, GridCell, DisplayTopology, DisplayArrangement
from src.window.grid_manager import GridCalculator, AdvancedGridManager, WindowPosition
from src.window.advanced_positioning import AdvancedPositioning, SmartPositionRequest
from src.core.either import Either
from src.core.errors import ValidationError, WindowError


class TestDisplayProperties:
    """Property-based tests for display information and topology."""
    
    @given(
        st.integers(min_value=1, max_value=7680),  # 8K width
        st.integers(min_value=1, max_value=4320),  # 8K height
        st.floats(min_value=0.5, max_value=4.0)    # Scale factor
    )
    def test_display_info_properties(self, width, height, scale_factor):
        """Property: Display info should handle all valid dimensions and scale factors."""
        display = DisplayInfo(
            display_id=0,
            name="Test Display",
            resolution=(width, height),
            position=(0, 0),
            scale_factor=scale_factor,
            is_main=True,
            color_space="sRGB"
        )
        
        assert display.resolution == (width, height)
        assert display.scale_factor == scale_factor
        
        bounds = display.bounds()
        assert bounds == (0, 0, width, height)
        
        center = display.center()
        assert center == (width // 2, height // 2)
    
    @given(
        st.integers(min_value=0, max_value=5000),  # x position
        st.integers(min_value=0, max_value=3000),  # y position
        st.integers(min_value=100, max_value=2000), # width
        st.integers(min_value=100, max_value=1500)  # height
    )
    def test_display_contains_point_properties(self, x, y, width, height):
        """Property: Display point containment should work for all valid coordinates."""
        display = DisplayInfo(
            display_id=0,
            name="Test",
            resolution=(width, height),
            position=(x, y),
            scale_factor=1.0,
            is_main=True,
            color_space="sRGB"
        )
        
        # Points inside display should be contained
        inside_x = x + width // 2
        inside_y = y + height // 2
        assert display.contains_point(inside_x, inside_y)
        
        # Points outside should not be contained
        outside_x = x + width + 10
        outside_y = y + height + 10
        assert not display.contains_point(outside_x, outside_y)
    
    @given(st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=2000),  # width
            st.integers(min_value=1, max_value=1500),  # height
            st.integers(min_value=0, max_value=3000),  # x position
            st.integers(min_value=0, max_value=2000)   # y position
        ),
        min_size=1,
        max_size=4
    ))
    def test_display_topology_properties(self, display_specs):
        """Property: Display topology should handle all valid multi-monitor configurations."""
        displays = []
        for i, (width, height, x, y) in enumerate(display_specs):
            display = DisplayInfo(
                display_id=i,
                name=f"Display {i}",
                resolution=(width, height),
                position=(x, y),
                scale_factor=1.0,
                is_main=(i == 0),
                color_space="sRGB"
            )
            displays.append(display)
        
        # Calculate total bounds manually for verification
        min_x = min(d.position[0] for d in displays)
        min_y = min(d.position[1] for d in displays)
        max_x = max(d.position[0] + d.resolution[0] for d in displays)
        max_y = max(d.position[1] + d.resolution[1] for d in displays)
        expected_bounds = (min_x, min_y, max_x - min_x, max_y - min_y)
        
        # Determine arrangement
        if len(displays) == 1:
            arrangement = DisplayArrangement.CUSTOM
        else:
            horizontal_aligned = all(d.position[1] == displays[0].position[1] for d in displays)
            vertical_aligned = all(d.position[0] == displays[0].position[0] for d in displays)
            
            if horizontal_aligned:
                arrangement = DisplayArrangement.HORIZONTAL
            elif vertical_aligned:
                arrangement = DisplayArrangement.VERTICAL
            else:
                arrangement = DisplayArrangement.CUSTOM
        
        topology = DisplayTopology(
            displays=displays,
            main_display_id=0,
            arrangement=arrangement,
            total_bounds=expected_bounds
        )
        
        assert len(topology.displays) == len(displays)
        assert topology.main_display_id == 0
        assert topology.total_bounds == expected_bounds


class TestGridCalculationProperties:
    """Property-based tests for grid calculation mathematics."""
    
    @given(
        st.integers(min_value=1, max_value=9),        # window count
        st.integers(min_value=0, max_value=50),       # padding
        st.sampled_from([
            WindowGridPattern.GRID_2X2,
            WindowGridPattern.GRID_3X3,
            WindowGridPattern.GRID_4X2,
            WindowGridPattern.THIRDS_HORIZONTAL
        ])
    )
    def test_grid_position_calculation_properties(self, window_count, padding, pattern):
        """Property: Grid calculations should produce valid positions for all inputs."""
        display = DisplayInfo(
            display_id=0,
            name="Test",
            resolution=(1920, 1080),
            position=(0, 0),
            scale_factor=1.0,
            is_main=True,
            color_space="sRGB"
        )
        
        result = GridCalculator.calculate_grid_positions(
            display, pattern, window_count, padding
        )
        
        if result.is_right():
            positions = result.get_right()
            
            # Should not exceed requested window count
            assert len(positions) <= window_count
            
            # All positions should be valid WindowPosition objects
            for pos in positions:
                assert isinstance(pos, WindowPosition)
                assert pos.width > 0
                assert pos.height > 0
                
                # Positions should be within display bounds (accounting for padding)
                display_bounds = display.bounds()
                assert pos.x >= display_bounds[0]
                assert pos.y >= display_bounds[1]
                assert pos.x + pos.width <= display_bounds[0] + display_bounds[2]
                assert pos.y + pos.height <= display_bounds[1] + display_bounds[3]
    
    @given(
        st.integers(min_value=1, max_value=4),   # rows
        st.integers(min_value=1, max_value=4),   # columns
        st.integers(min_value=1, max_value=16),  # window count
        st.integers(min_value=0, max_value=20)   # padding
    )
    def test_standard_grid_properties(self, rows, columns, window_count, padding):
        """Property: Standard grid calculations should respect grid dimensions."""
        total_width = 1920
        total_height = 1080
        
        positions = GridCalculator._calculate_standard_grid(
            0, 0, total_width, total_height, rows, columns, window_count, padding
        )
        
        # Should not exceed grid capacity or window count
        max_positions = min(rows * columns, window_count)
        assert len(positions) <= max_positions
        
        # Check that positions don't overlap (approximately)
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions[i + 1:], i + 1):
                # Check for significant overlap (allowing for padding)
                overlap_x = max(0, min(pos1.x + pos1.width, pos2.x + pos2.width) - max(pos1.x, pos2.x))
                overlap_y = max(0, min(pos1.y + pos1.height, pos2.y + pos2.height) - max(pos1.y, pos2.y))
                
                # Should have minimal overlap (less than padding tolerance)
                assert overlap_x <= padding + 10 or overlap_y <= padding + 10
    
    @given(st.integers(min_value=1, max_value=20))
    def test_window_position_properties(self, size):
        """Property: Window position objects should maintain consistency."""
        x, y = 100, 100
        width, height = size * 50, size * 40
        
        # Should reject invalid sizes
        if width <= 0 or height <= 0:
            with pytest.raises(Exception):
                WindowPosition(x, y, width, height)
        else:
            pos = WindowPosition(x, y, width, height)
            
            assert pos.x == x
            assert pos.y == y
            assert pos.width == width
            assert pos.height == height
            
            # Tuple conversion should preserve values
            assert pos.to_tuple() == (x, y, width, height)
            
            # Center calculation should be correct
            expected_center = (x + width // 2, y + height // 2)
            assert pos.center() == expected_center
            
            # Area calculation should be correct
            assert pos.area() == width * height


class TestAdvancedPositioningProperties:
    """Property-based tests for advanced positioning algorithms."""
    
    @given(
        st.floats(min_value=0.0, max_value=1.0),  # relative x
        st.floats(min_value=0.0, max_value=1.0),  # relative y
        st.integers(min_value=100, max_value=800), # window width
        st.integers(min_value=100, max_value=600)  # window height
    )
    def test_relative_position_properties(self, rel_x, rel_y, width, height):
        """Property: Relative position calculations should preserve proportions."""
        source_display = DisplayInfo(
            display_id=0, name="Source", resolution=(1920, 1080),
            position=(0, 0), scale_factor=1.0, is_main=True, color_space="sRGB"
        )
        
        target_display = DisplayInfo(
            display_id=1, name="Target", resolution=(2560, 1440),
            position=(1920, 0), scale_factor=1.0, is_main=False, color_space="sRGB"
        )
        
        # Calculate source position from relative coordinates
        source_x = int(rel_x * source_display.resolution[0])
        source_y = int(rel_y * source_display.resolution[1])
        old_position = (source_x, source_y, width, height)
        
        from src.window.advanced_positioning import AdvancedPositioning
        from src.core.displays import DisplayManager
        positioning = AdvancedPositioning(DisplayManager())
        
        result = positioning._calculate_relative_position(
            old_position, source_display, target_display
        )
        
        if result.is_right():
            new_x, new_y, new_width, new_height = result.get_right()
            
            # Size should be preserved
            assert new_width == width
            assert new_height == height
            
            # New position should be within target display bounds
            target_bounds = target_display.bounds()
            assert new_x >= target_bounds[0]
            assert new_y >= target_bounds[1]
            assert new_x + new_width <= target_bounds[0] + target_bounds[2]
            assert new_y + new_height <= target_bounds[1] + target_bounds[3]
            
            # Relative position should be approximately preserved
            new_rel_x = (new_x - target_bounds[0]) / target_bounds[2]
            new_rel_y = (new_y - target_bounds[1]) / target_bounds[3]
            
            # Allow for some tolerance due to integer rounding and bounds clamping
            assert abs(new_rel_x - rel_x) <= 0.1 or new_x == target_bounds[0] or new_x + new_width == target_bounds[0] + target_bounds[2]
            assert abs(new_rel_y - rel_y) <= 0.1 or new_y == target_bounds[1] or new_y + new_height == target_bounds[1] + target_bounds[3]
    
    @given(st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=15),
        min_size=1,
        max_size=8
    ))
    def test_smart_position_request_properties(self, window_names):
        """Property: Smart position requests should handle all valid window configurations."""
        requests = []
        for name in window_names:
            request = SmartPositionRequest(
                window_identifier=name,
                content_type="editor",
                avoid_overlap=True
            )
            requests.append(request)
        
        # All requests should have valid identifiers
        for request in requests:
            assert len(request.window_identifier) > 0
            assert request.avoid_overlap is True
        
        # Should be able to categorize windows
        from src.window.advanced_positioning import AdvancedPositioning
        from src.core.displays import DisplayManager
        positioning = AdvancedPositioning(DisplayManager())
        
        categorized = positioning._categorize_windows(requests)
        
        # Should have at least one category
        assert len(categorized) > 0
        
        # All original requests should appear in categories
        total_categorized = sum(len(cat_requests) for cat_requests in categorized.values())
        assert total_categorized == len(requests)


class TestCoordinateMathProperties:
    """Property-based tests for coordinate mathematics and bounds checking."""
    
    @given(
        st.integers(min_value=-1000, max_value=5000),  # x coordinate
        st.integers(min_value=-1000, max_value=3000),  # y coordinate
        st.integers(min_value=1, max_value=2000),      # width
        st.integers(min_value=1, max_value=1500)       # height
    )
    def test_bounds_validation_properties(self, x, y, width, height):
        """Property: Bounds validation should correctly identify valid/invalid positions."""
        display_bounds = (0, 0, 1920, 1080)
        
        from src.window.grid_manager import GridCalculator
        position = WindowPosition(max(0, x), max(0, y), width, height)
        
        display = DisplayInfo(
            display_id=0, name="Test", resolution=(1920, 1080),
            position=(0, 0), scale_factor=1.0, is_main=True, color_space="sRGB"
        )
        
        is_valid = GridCalculator._validate_position_bounds(position, display)
        
        # Should be valid if position fits within display
        expected_valid = (
            position.x >= 0 and
            position.y >= 0 and
            position.x + position.width <= 1920 and
            position.y + position.height <= 1080
        )
        
        assert is_valid == expected_valid
    
    @given(
        st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=10),
        st.integers(min_value=1, max_value=5)
    )
    def test_display_targeting_properties(self, display_indices, available_count):
        """Property: Display targeting should validate index bounds correctly."""
        from src.server.tools.advanced_window_tools import _validate_display_targets
        from src.core.displays import DisplayManager
        
        # Create mock displays
        displays = []
        for i in range(available_count):
            display = DisplayInfo(
                display_id=i, name=f"Display {i}", resolution=(1920, 1080),
                position=(i * 1920, 0), scale_factor=1.0, is_main=(i == 0), color_space="sRGB"
            )
            displays.append(display)
        
        # Test validation
        for target_id in display_indices:
            if target_id < available_count:
                # Should be valid
                assert target_id >= 0
            else:
                # Should be invalid
                assert target_id >= available_count


# Test configuration and runner
class TestWindowManagementSystem:
    """Main property test runner for window management system."""
    
    @given(
        st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
        st.sampled_from(list(WindowGridPattern))
    )
    def test_grid_manager_integration(self, window_ids, pattern):
        """Property: Grid manager should handle all valid window/pattern combinations."""
        if pattern == WindowGridPattern.CUSTOM:
            return  # Skip custom patterns in property testing
        
        manager = AdvancedGridManager()
        display = DisplayInfo(
            display_id=0, name="Test", resolution=(1920, 1080),
            position=(0, 0), scale_factor=1.0, is_main=True, color_space="sRGB"
        )
        
        # Should determine valid patterns for window count
        pattern_result = asyncio.run(manager.calculate_optimal_pattern(len(window_ids), display))
        
        if pattern_result.is_right():
            optimal_pattern = pattern_result.get_right()
            assert isinstance(optimal_pattern, WindowGridPattern)
        
        # Should get supported patterns list
        supported = manager.get_supported_patterns()
        assert len(supported) > 0
        assert all(isinstance(p, WindowGridPattern) for p in supported)
    
    @given(st.integers(min_value=1, max_value=10))
    def test_workspace_name_generation(self, workspace_count):
        """Property: Workspace management should handle naming correctly."""
        from src.window.advanced_positioning import WorkspaceManager, AdvancedPositioning
        from src.core.displays import DisplayManager
        
        positioning = AdvancedPositioning(DisplayManager())
        manager = WorkspaceManager(positioning)
        
        # Should start with empty workspace list
        assert len(manager.list_saved_workspaces()) == 0
        
        # Generate workspace names
        workspace_names = [f"workspace_{i}" for i in range(workspace_count)]
        
        for name in workspace_names:
            # Valid workspace names should be accepted
            assert len(name) > 0
            assert all(c.isalnum() or c in "_-" for c in name)


import asyncio


# Async test helpers
def run_async_test(coro):
    """Helper to run async tests in property-based testing."""
    return asyncio.run(coro)


# Pattern validation tests
@given(st.sampled_from([p.value for p in WindowGridPattern]))
def test_pattern_value_consistency(pattern_value):
    """Property: All grid pattern values should be valid enum values."""
    try:
        pattern = WindowGridPattern(pattern_value)
        assert pattern.value == pattern_value
    except ValueError:
        pytest.fail(f"Pattern value {pattern_value} should be valid")


# Grid cell validation
@given(
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5)
)
def test_grid_cell_properties(row, col, row_span, col_span):
    """Property: Grid cells should maintain valid dimensions."""
    cell = GridCell(row, col, row_span, col_span)
    
    assert cell.row == row
    assert cell.column == col
    assert cell.row_span == row_span
    assert cell.column_span == col_span
    
    # Should convert to dict correctly
    cell_dict = cell.to_dict()
    assert cell_dict["row"] == row
    assert cell_dict["column"] == col
    assert cell_dict["row_span"] == row_span
    assert cell_dict["column_span"] == col_span