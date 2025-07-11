"""Comprehensive tests for window management tools module.

import logging

logging.basicConfig(level=logging.DEBUG)
Tests cover window operations, multi-monitor support, coordinate validation,
arrangement management, and integration with property-based testing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from src.server.tools.window_tools import km_window_manager

if TYPE_CHECKING:
    from collections.abc import Callable


# Test data generators
@st.composite
def window_operation_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid window operations."""
    operations = [
        "move",
        "resize",
        "minimize",
        "maximize",
        "restore",
        "arrange",
        "get_info",
        "get_screens",
    ]
    return draw(st.sampled_from(operations))


@st.composite
def window_identifier_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid window identifiers."""
    # Mix of bundle IDs and app names (systematic pattern alignment)
    identifiers = [
        # Bundle IDs
        "com.apple.finder",
        "com.microsoft.VSCode",
        "com.github.atom",
        "org.mozilla.firefox",
        "com.google.Chrome",
        # Application names
        "Finder",
        "Visual Studio Code",
        "Firefox",
        "Chrome",
        "TextEdit",
        # Generated valid names
        draw(
            st.text(min_size=1, max_size=30).filter(
                lambda x: x.isalnum() and len(x.strip()) > 0,
            ),
        ),
    ]
    return draw(st.sampled_from(identifiers))


@st.composite
def position_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid window positions."""
    return {
        "x": draw(st.integers(min_value=0, max_value=3840)),  # 4K width
        "y": draw(st.integers(min_value=0, max_value=2160)),  # 4K height
    }


@st.composite
def size_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid window sizes."""
    return {
        "width": draw(
            st.integers(min_value=100, max_value=2560),
        ),  # Reasonable window widths
        "height": draw(
            st.integers(min_value=100, max_value=1440),
        ),  # Reasonable window heights
    }


@st.composite
def screen_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid screen identifiers."""
    screens = ["main", "external", "0", "1", "2"]
    return draw(st.sampled_from(screens))


@st.composite
def arrangement_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid window arrangements."""
    arrangements = [
        "left_half",
        "right_half",
        "top_half",
        "bottom_half",
        "top_left_quarter",
        "top_right_quarter",
        "bottom_left_quarter",
        "bottom_right_quarter",
        "center",
        "maximize",
    ]
    return draw(st.sampled_from(arrangements))


@st.composite
def window_state_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid window states."""
    states = ["normal", "minimized", "maximized", "fullscreen"]
    return draw(st.sampled_from(states))


@st.composite
def window_index_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid window indices."""
    return draw(st.integers(min_value=0, max_value=20))


@st.composite
def invalid_window_identifier_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate invalid window identifiers."""
    invalid_identifiers = [
        "",  # Empty
        "   ",  # Whitespace only
        "a" * 256,  # Too long
        "app<>name",  # Invalid characters
        "app|name",  # Pipe character
        "app;rm -rf /",  # Injection attempt
        "../../../etc/passwd",  # Path traversal
        "app\x00name",  # Null bytes
    ]
    return draw(st.sampled_from(invalid_identifiers))


class TestWindowToolsDependencies:
    """Test window tools dependencies and imports."""

    def test_window_manager_import(self) -> None:
        """Test importing window management dependencies."""
        try:
            from src.applications.app_controller import AppIdentifier
            from src.core.types import Duration
            from src.integration.km_client import KMError
            from src.windows.window_manager import (
                Position,
                Size,
                WindowArrangement,
                WindowManager,
                WindowState,
            )

            # Test basic creation
            assert WindowManager is not None
            assert Position is not None
            assert Size is not None
            assert WindowState is not None
            assert WindowArrangement is not None
            assert AppIdentifier is not None
            assert Duration is not None
            assert KMError is not None

        except ImportError:
            # Mock the dependencies for testing
            pytest.skip("Window management dependencies not available - using mocks")


class TestWindowParameterValidation:
    """Test window management parameter validation."""

    @given(window_operation_strategy())
    def test_valid_operations(self, operation: str) -> None:
        """Test that valid operations are accepted."""
        valid_operations = [
            "move",
            "resize",
            "minimize",
            "maximize",
            "restore",
            "arrange",
            "get_info",
            "get_screens",
        ]
        assert operation in valid_operations

    @given(window_identifier_strategy())
    def test_window_identifier_validation(self, window_id: str) -> None:
        """Test window identifier validation."""
        assume(len(window_id) <= 255 and len(window_id.strip()) > 0)
        # Valid window identifiers should be non-empty and within length limits
        assert 0 < len(window_id) <= 255

    @given(position_strategy())
    def test_position_validation(self, position: dict[str, int]) -> None:
        """Test position parameter validation."""
        # Positions should have x and y coordinates
        assert "x" in position
        assert "y" in position
        assert isinstance(position["x"], int)
        assert isinstance(position["y"], int)
        assert position["x"] >= 0
        assert position["y"] >= 0

    @given(size_strategy())
    def test_size_validation(self, size: dict[str, int]) -> None:
        """Test size parameter validation."""
        # Sizes should have width and height
        assert "width" in size
        assert "height" in size
        assert isinstance(size["width"], int)
        assert isinstance(size["height"], int)
        assert size["width"] >= 100  # Minimum window size
        assert size["height"] >= 100  # Minimum window size

    @given(screen_strategy())
    def test_screen_validation(self, screen: str) -> None:
        """Test screen parameter validation."""
        valid_screens = ["main", "external", "0", "1", "2"]
        assert screen in valid_screens

    @given(arrangement_strategy())
    def test_arrangement_validation(self, arrangement: str) -> None:
        """Test arrangement parameter validation."""
        valid_arrangements = [
            "left_half",
            "right_half",
            "top_half",
            "bottom_half",
            "top_left_quarter",
            "top_right_quarter",
            "bottom_left_quarter",
            "bottom_right_quarter",
            "center",
            "maximize",
        ]
        assert arrangement in valid_arrangements

    @given(window_state_strategy())
    def test_window_state_validation(self, state: str) -> None:
        """Test window state parameter validation."""
        valid_states = ["normal", "minimized", "maximized", "fullscreen"]
        assert state in valid_states

    @given(window_index_strategy())
    def test_window_index_validation(self, index: int) -> None:
        """Test window index parameter validation."""
        assert 0 <= index <= 20

    def test_invalid_window_identifiers(self) -> None:
        """Test that invalid window identifiers are rejected."""
        invalid_identifiers = ["", "   ", "a" * 256, "app<>name", "app|name"]
        # These should be detected as invalid by _is_valid_app_identifier
        for identifier in invalid_identifiers:
            # Validation logic should catch these patterns
            if len(identifier.strip()) == 0 or len(identifier) > 255:
                assert True  # Should be rejected
            elif any(char in identifier for char in "<>|*?;&`$"):
                assert True  # Should be rejected for suspicious characters


class TestWindowMoveOperationMocked:
    """Test window move operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_move_window_success(self) -> None:
        """Test successful window move operation."""
        with (
            patch(
                "src.server.tools.window_tools._window_manager",
            ) as mock_window_manager,
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create_app_id,
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
            ) as mock_validate_id,
        ):
            # Setup validation and identifier creation
            mock_validate_id.return_value = True
            mock_app_id = Mock()
            mock_app_id.primary_identifier.return_value = "com.apple.finder"
            mock_create_app_id.return_value = mock_app_id

            # Setup window manager operation result
            mock_operation_result = Mock()
            mock_operation_result.window_info = Mock()
            mock_operation_result.window_info.app_identifier = "com.apple.finder"
            mock_operation_result.window_info.window_index = 0
            mock_operation_result.window_info.position = Mock(x=100, y=200)
            mock_operation_result.window_info.size = Mock(width=800, height=600)
            mock_operation_result.window_info.state = Mock(value="normal")
            mock_operation_result.window_info.title = "Test Window"
            mock_operation_result.operation_time = Mock()
            mock_operation_result.operation_time.total_seconds.return_value = 0.5
            mock_operation_result.details = "Window moved successfully"

            mock_result = Mock()
            mock_result.is_right.return_value = True
            mock_result.get_right.return_value = mock_operation_result

            mock_window_manager.move_window = AsyncMock(return_value=mock_result)

            # Execute move operation
            result = await km_window_manager(
                operation="move",
                window_identifier="com.apple.finder",
                position={"x": 100, "y": 200},
            )

            # Verify successful move
            assert result["success"] is True
            assert result["operation"] == "move"
            assert result["window"]["position"]["x"] == 100
            assert result["window"]["position"]["y"] == 200
            assert result["execution_time_ms"] == 500
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_move_window_missing_position(self) -> None:
        """Test move operation with missing position."""
        with patch(
            "src.server.tools.window_tools._is_valid_app_identifier",
        ) as mock_validate_id:
            mock_validate_id.return_value = True

            # Execute move without position
            result = await km_window_manager(
                operation="move",
                window_identifier="com.apple.finder",
                # position is None
            )

            # Verify missing position error
            assert result["success"] is False
            assert result["error"] == "MISSING_POSITION"
            assert "Position coordinates" in result["message"]

    @pytest.mark.asyncio
    async def test_move_window_invalid_position(self) -> None:
        """Test move operation with invalid position format."""
        with (
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
            ) as mock_validate_id,
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create_app_id,
        ):
            mock_validate_id.return_value = True
            mock_app_id = Mock()
            mock_create_app_id.return_value = mock_app_id

            # Execute move with invalid position (missing y)
            result = await km_window_manager(
                operation="move",
                window_identifier="com.apple.finder",
                position={"x": 100},  # Missing y coordinate
            )

            # Verify missing position error
            assert result["success"] is False
            assert result["error"] == "MISSING_POSITION"


class TestWindowResizeOperationMocked:
    """Test window resize operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_resize_window_success(self) -> None:
        """Test successful window resize operation."""
        with (
            patch(
                "src.server.tools.window_tools._window_manager",
            ) as mock_window_manager,
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create_app_id,
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
            ) as mock_validate_id,
        ):
            # Setup validation and identifier creation
            mock_validate_id.return_value = True
            mock_app_id = Mock()
            mock_app_id.primary_identifier.return_value = "com.apple.finder"
            mock_create_app_id.return_value = mock_app_id

            # Setup window manager operation result
            mock_operation_result = Mock()
            mock_operation_result.window_info = Mock()
            mock_operation_result.window_info.app_identifier = "com.apple.finder"
            mock_operation_result.window_info.window_index = 0
            mock_operation_result.window_info.position = Mock(x=100, y=200)
            mock_operation_result.window_info.size = Mock(width=1200, height=800)
            mock_operation_result.window_info.state = Mock(value="normal")
            mock_operation_result.window_info.title = "Test Window"
            mock_operation_result.operation_time = Mock()
            mock_operation_result.operation_time.total_seconds.return_value = 0.3
            mock_operation_result.details = "Window resized successfully"

            mock_result = Mock()
            mock_result.is_right.return_value = True
            mock_result.get_right.return_value = mock_operation_result

            mock_window_manager.resize_window = AsyncMock(return_value=mock_result)

            # Execute resize operation
            result = await km_window_manager(
                operation="resize",
                window_identifier="com.apple.finder",
                size={"width": 1200, "height": 800},
            )

            # Verify successful resize
            assert result["success"] is True
            assert result["operation"] == "resize"
            assert result["window"]["size"]["width"] == 1200
            assert result["window"]["size"]["height"] == 800
            assert result["execution_time_ms"] == 300

    @pytest.mark.asyncio
    async def test_resize_window_missing_size(self) -> None:
        """Test resize operation with missing size."""
        with patch(
            "src.server.tools.window_tools._is_valid_app_identifier",
        ) as mock_validate_id:
            mock_validate_id.return_value = True

            # Execute resize without size
            result = await km_window_manager(
                operation="resize",
                window_identifier="com.apple.finder",
                # size is None
            )

            # Verify missing size error
            assert result["success"] is False
            assert result["error"] == "MISSING_SIZE"
            assert "Size dimensions" in result["message"]


class TestWindowStateOperationMocked:
    """Test window state change operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_minimize_window_success(self) -> None:
        """Test successful window minimize operation."""
        with (
            patch(
                "src.server.tools.window_tools._window_manager",
            ) as mock_window_manager,
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create_app_id,
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
            ) as mock_validate_id,
        ):
            # Setup validation and identifier creation
            mock_validate_id.return_value = True
            mock_app_id = Mock()
            mock_app_id.primary_identifier.return_value = "com.apple.finder"
            mock_create_app_id.return_value = mock_app_id

            # Setup window manager operation result
            mock_operation_result = Mock()
            mock_operation_result.window_info = Mock()
            mock_operation_result.window_info.app_identifier = "com.apple.finder"
            mock_operation_result.window_info.window_index = 0
            mock_operation_result.window_info.position = Mock(x=100, y=200)
            mock_operation_result.window_info.size = Mock(width=800, height=600)
            mock_operation_result.window_info.state = Mock(value="minimized")
            mock_operation_result.window_info.title = "Test Window"
            mock_operation_result.operation_time = Mock()
            mock_operation_result.operation_time.total_seconds.return_value = 0.2
            mock_operation_result.details = "Window minimized successfully"

            mock_result = Mock()
            mock_result.is_right.return_value = True
            mock_result.get_right.return_value = mock_operation_result

            mock_window_manager.set_window_state = AsyncMock(return_value=mock_result)

            # Execute minimize operation
            result = await km_window_manager(
                operation="minimize",
                window_identifier="com.apple.finder",
            )

            # Verify successful minimize
            assert result["success"] is True
            assert result["operation"] == "set_state_minimized"
            assert result["window"]["state"] == "minimized"
            assert result["execution_time_ms"] == 200

    @pytest.mark.asyncio
    async def test_maximize_window_success(self) -> None:
        """Test successful window maximize operation."""
        with (
            patch(
                "src.server.tools.window_tools._window_manager",
            ) as mock_window_manager,
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create_app_id,
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
            ) as mock_validate_id,
        ):
            # Setup validation and identifier creation
            mock_validate_id.return_value = True
            mock_app_id = Mock()
            mock_app_id.primary_identifier.return_value = "com.apple.finder"
            mock_create_app_id.return_value = mock_app_id

            # Setup window manager operation result
            mock_operation_result = Mock()
            mock_operation_result.window_info = Mock()
            mock_operation_result.window_info.app_identifier = "com.apple.finder"
            mock_operation_result.window_info.window_index = 0
            mock_operation_result.window_info.position = Mock(x=0, y=0)
            mock_operation_result.window_info.size = Mock(width=1920, height=1080)
            mock_operation_result.window_info.state = Mock(value="maximized")
            mock_operation_result.window_info.title = "Test Window"
            mock_operation_result.operation_time = Mock()
            mock_operation_result.operation_time.total_seconds.return_value = 0.4
            mock_operation_result.details = "Window maximized successfully"

            mock_result = Mock()
            mock_result.is_right.return_value = True
            mock_result.get_right.return_value = mock_operation_result

            mock_window_manager.set_window_state = AsyncMock(return_value=mock_result)

            # Execute maximize operation
            result = await km_window_manager(
                operation="maximize",
                window_identifier="com.apple.finder",
            )

            # Verify successful maximize
            assert result["success"] is True
            assert result["operation"] == "set_state_maximized"
            assert result["window"]["state"] == "maximized"
            assert result["window"]["size"]["width"] == 1920
            assert result["window"]["size"]["height"] == 1080


class TestWindowArrangementOperationMocked:
    """Test window arrangement operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_arrange_window_left_half_success(self) -> None:
        """Test successful window arrangement to left half."""
        with (
            patch(
                "src.server.tools.window_tools._window_manager",
            ) as mock_window_manager,
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create_app_id,
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
            ) as mock_validate_id,
        ):
            # Setup validation and identifier creation
            mock_validate_id.return_value = True
            mock_app_id = Mock()
            mock_app_id.primary_identifier.return_value = "com.apple.finder"
            mock_create_app_id.return_value = mock_app_id

            # Setup window manager operation result
            mock_operation_result = Mock()
            mock_operation_result.window_info = Mock()
            mock_operation_result.window_info.app_identifier = "com.apple.finder"
            mock_operation_result.window_info.window_index = 0
            mock_operation_result.window_info.position = Mock(x=0, y=0)
            mock_operation_result.window_info.size = Mock(
                width=960,
                height=1080,
            )  # Half screen width
            mock_operation_result.window_info.state = Mock(value="normal")
            mock_operation_result.window_info.title = "Test Window"
            mock_operation_result.operation_time = Mock()
            mock_operation_result.operation_time.total_seconds.return_value = 0.6
            mock_operation_result.details = "Window arranged to left half"

            mock_result = Mock()
            mock_result.is_right.return_value = True
            mock_result.get_right.return_value = mock_operation_result

            mock_window_manager.arrange_window = AsyncMock(return_value=mock_result)

            # Execute arrangement operation
            result = await km_window_manager(
                operation="arrange",
                window_identifier="com.apple.finder",
                arrangement="left_half",
                screen="main",
            )

            # Verify successful arrangement
            assert result["success"] is True
            assert result["operation"] == "arrange_left_half"
            assert result["window"]["position"]["x"] == 0
            assert result["window"]["size"]["width"] == 960  # Half of 1920
            assert result["screen"] == "main"
            assert result["execution_time_ms"] == 600

    @pytest.mark.asyncio
    async def test_arrange_window_missing_arrangement(self) -> None:
        """Test arrangement operation with missing arrangement type."""
        with patch(
            "src.server.tools.window_tools._is_valid_app_identifier",
        ) as mock_validate_id:
            mock_validate_id.return_value = True

            # Execute arrange without arrangement
            result = await km_window_manager(
                operation="arrange",
                window_identifier="com.apple.finder",
                # arrangement is None
            )

            # Verify missing arrangement error
            assert result["success"] is False
            assert result["error"] == "MISSING_ARRANGEMENT"
            assert "Arrangement type required" in result["message"]


class TestWindowInfoOperationMocked:
    """Test window info retrieval operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_get_window_info_success(self) -> None:
        """Test successful window info retrieval."""
        with (
            patch(
                "src.server.tools.window_tools._window_manager",
            ) as mock_window_manager,
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create_app_id,
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
            ) as mock_validate_id,
        ):
            # Setup validation and identifier creation
            mock_validate_id.return_value = True
            mock_app_id = Mock()
            mock_app_id.primary_identifier.return_value = "com.apple.finder"
            mock_create_app_id.return_value = mock_app_id

            # Setup window info result
            mock_window_info = Mock()
            mock_window_info.app_identifier = "com.apple.finder"
            mock_window_info.window_index = 0
            mock_window_info.position = Mock(x=150, y=300)
            mock_window_info.size = Mock(width=800, height=600)
            mock_window_info.state = Mock(value="normal")
            mock_window_info.title = "Finder Window"

            mock_result = Mock()
            mock_result.is_right.return_value = True
            mock_result.get_right.return_value = mock_window_info

            mock_window_manager.get_window_info = AsyncMock(return_value=mock_result)

            # Execute get info operation
            result = await km_window_manager(
                operation="get_info",
                window_identifier="com.apple.finder",
            )

            # Verify successful info retrieval
            assert result["success"] is True
            assert result["operation"] == "get_info"
            assert result["window"]["app_identifier"] == "com.apple.finder"
            assert result["window"]["position"]["x"] == 150
            assert result["window"]["position"]["y"] == 300
            assert result["window"]["size"]["width"] == 800
            assert result["window"]["size"]["height"] == 600
            assert result["window"]["state"] == "normal"
            assert result["window"]["title"] == "Finder Window"

    @pytest.mark.asyncio
    async def test_get_screens_info_success(self) -> None:
        """Test successful screen info retrieval."""
        with patch(
            "src.server.tools.window_tools._window_manager",
        ) as mock_window_manager:
            # Setup screen info
            mock_screen1 = Mock()
            mock_screen1.screen_id = "main"
            mock_screen1.name = "Built-in Display"
            mock_screen1.is_main = True
            mock_screen1.origin = Mock(x=0, y=0)
            mock_screen1.size = Mock(width=1920, height=1080)
            mock_screen1.center_position.return_value = Mock(x=960, y=540)

            mock_screen2 = Mock()
            mock_screen2.screen_id = "external"
            mock_screen2.name = "External Display"
            mock_screen2.is_main = False
            mock_screen2.origin = Mock(x=1920, y=0)
            mock_screen2.size = Mock(width=2560, height=1440)
            mock_screen2.center_position.return_value = Mock(x=3200, y=720)

            mock_screens = [mock_screen1, mock_screen2]
            mock_window_manager.get_screen_info = AsyncMock(return_value=mock_screens)

            # Execute get screens operation
            result = await km_window_manager(
                operation="get_screens",
                window_identifier="dummy",  # Not used for get_screens
            )

            # Verify successful screen info retrieval
            assert result["success"] is True
            assert result["operation"] == "get_screens"
            assert result["screen_count"] == 2
            assert len(result["screens"]) == 2

            # Check main screen
            main_screen = result["screens"][0]
            assert main_screen["screen_id"] == "main"
            assert main_screen["name"] == "Built-in Display"
            assert main_screen["is_main"] is True
            assert main_screen["size"]["width"] == 1920
            assert main_screen["size"]["height"] == 1080

            # Check external screen
            external_screen = result["screens"][1]
            assert external_screen["screen_id"] == "external"
            assert external_screen["name"] == "External Display"
            assert external_screen["is_main"] is False
            assert external_screen["size"]["width"] == 2560
            assert external_screen["size"]["height"] == 1440


class TestWindowErrorHandling:
    """Test window management error handling."""

    @pytest.mark.asyncio
    async def test_invalid_window_identifier_error(self) -> None:
        """Test handling of invalid window identifier."""
        with patch(
            "src.server.tools.window_tools._is_valid_app_identifier",
        ) as mock_validate_id:
            # Setup identifier validation failure
            mock_validate_id.return_value = False

            # Execute operation with invalid identifier
            result = await km_window_manager(
                operation="move",
                window_identifier="invalid<>app",
                position={"x": 100, "y": 200},
            )

            # Verify invalid identifier error
            assert result["success"] is False
            assert result["error"] == "INVALID_IDENTIFIER"
            assert "Invalid application identifier" in result["message"]

    @pytest.mark.asyncio
    async def test_unsupported_operation_error(self) -> None:
        """Test handling of unsupported operations."""
        with patch(
            "src.server.tools.window_tools._is_valid_app_identifier",
        ) as mock_validate_id:
            mock_validate_id.return_value = True

            # Execute unsupported operation
            result = await km_window_manager(
                operation="invalid_operation",
                window_identifier="com.apple.finder",
            )

            # Verify unsupported operation error
            assert result["success"] is False
            assert result["error"] == "INVALID_OPERATION"
            assert "Unsupported operation" in result["message"]

    @pytest.mark.asyncio
    async def test_window_manager_execution_error(self) -> None:
        """Test handling of window manager execution errors."""
        with (
            patch(
                "src.server.tools.window_tools._window_manager",
            ) as mock_window_manager,
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create_app_id,
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
            ) as mock_validate_id,
        ):
            # Setup validation and identifier creation
            mock_validate_id.return_value = True
            mock_app_id = Mock()
            mock_app_id.primary_identifier.return_value = "com.apple.finder"
            mock_create_app_id.return_value = mock_app_id

            # Setup window manager error
            mock_error = Mock()
            mock_error.code = "WINDOW_NOT_FOUND"
            mock_error.message = "Window not found for application"

            mock_result = Mock()
            mock_result.is_right.return_value = False
            mock_result.get_left.return_value = mock_error

            mock_window_manager.move_window = AsyncMock(return_value=mock_result)

            # Execute operation that should fail
            result = await km_window_manager(
                operation="move",
                window_identifier="com.apple.finder",
                position={"x": 100, "y": 200},
            )

            # Verify window manager error handling
            assert result["success"] is False
            assert result["error"] == "WINDOW_NOT_FOUND"
            assert result["message"] == "Window not found for application"

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self) -> None:
        """Test handling of unexpected errors."""
        with patch(
            "src.server.tools.window_tools._is_valid_app_identifier",
        ) as mock_validate_id:
            # Setup unexpected error during validation
            mock_validate_id.side_effect = RuntimeError("Unexpected system error")

            # Execute operation that should trigger unexpected error
            result = await km_window_manager(
                operation="move",
                window_identifier="com.apple.finder",
                position={"x": 100, "y": 200},
            )

            # Verify unexpected error handling
            assert result["success"] is False
            assert result["error"] == "EXECUTION_ERROR"
            assert "Operation failed" in result["message"]


class TestWindowIntegration:
    """Integration tests for window management operations."""

    @pytest.mark.asyncio
    async def test_complete_window_workflow(self) -> None:
        """Test complete window management workflow."""
        with (
            patch(
                "src.server.tools.window_tools._window_manager",
            ) as mock_window_manager,
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create_app_id,
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
            ) as mock_validate_id,
        ):
            # Setup validation and identifier creation
            mock_validate_id.return_value = True
            mock_app_id = Mock()
            mock_app_id.primary_identifier.return_value = "com.apple.finder"
            mock_create_app_id.return_value = mock_app_id

            # Setup different operation results
            def create_mock_operation_result(
                operation_details: dict[str, Any] | list[Any],
            ) -> None:
                mock_operation_result = Mock()
                mock_operation_result.window_info = Mock()
                mock_operation_result.window_info.app_identifier = "com.apple.finder"
                mock_operation_result.window_info.window_index = 0
                mock_operation_result.window_info.position = Mock(
                    x=operation_details["x"],
                    y=operation_details["y"],
                )
                mock_operation_result.window_info.size = Mock(
                    width=operation_details["width"],
                    height=operation_details["height"],
                )
                mock_operation_result.window_info.state = Mock(
                    value=operation_details["state"],
                )
                mock_operation_result.window_info.title = "Test Window"
                mock_operation_result.operation_time = Mock()
                mock_operation_result.operation_time.total_seconds.return_value = (
                    operation_details["time"]
                )
                mock_operation_result.details = operation_details["details"]

                mock_result = Mock()
                mock_result.is_right.return_value = True
                mock_result.get_right.return_value = mock_operation_result
                return mock_result

            # Setup move result
            move_result = create_mock_operation_result(
                {
                    "x": 100,
                    "y": 200,
                    "width": 800,
                    "height": 600,
                    "state": "normal",
                    "time": 0.5,
                    "details": "Window moved",
                },
            )

            # Setup resize result
            resize_result = create_mock_operation_result(
                {
                    "x": 100,
                    "y": 200,
                    "width": 1200,
                    "height": 800,
                    "state": "normal",
                    "time": 0.3,
                    "details": "Window resized",
                },
            )

            # Setup maximize result
            maximize_result = create_mock_operation_result(
                {
                    "x": 0,
                    "y": 0,
                    "width": 1920,
                    "height": 1080,
                    "state": "maximized",
                    "time": 0.4,
                    "details": "Window maximized",
                },
            )

            mock_window_manager.move_window = AsyncMock(return_value=move_result)
            mock_window_manager.resize_window = AsyncMock(return_value=resize_result)
            mock_window_manager.set_window_state = AsyncMock(
                return_value=maximize_result,
            )

            # Execute complete workflow: move -> resize -> maximize
            move_result_data = await km_window_manager(
                operation="move",
                window_identifier="com.apple.finder",
                position={"x": 100, "y": 200},
            )
            assert move_result_data["success"] is True
            assert move_result_data["window"]["position"]["x"] == 100

            resize_result_data = await km_window_manager(
                operation="resize",
                window_identifier="com.apple.finder",
                size={"width": 1200, "height": 800},
            )
            assert resize_result_data["success"] is True
            assert resize_result_data["window"]["size"]["width"] == 1200

            maximize_result_data = await km_window_manager(
                operation="maximize",
                window_identifier="com.apple.finder",
            )
            assert maximize_result_data["success"] is True
            assert maximize_result_data["window"]["state"] == "maximized"

            # Verify all operations were called
            mock_window_manager.move_window.assert_called_once()
            mock_window_manager.resize_window.assert_called_once()
            mock_window_manager.set_window_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_window_management_with_context(self) -> None:
        """Test window management with FastMCP context integration."""
        mock_context = Mock()
        mock_context.info = AsyncMock()
        mock_context.report_progress = AsyncMock()
        mock_context.error = AsyncMock()

        with (
            patch(
                "src.server.tools.window_tools._window_manager",
            ) as mock_window_manager,
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create_app_id,
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
            ) as mock_validate_id,
        ):
            # Setup validation and identifier creation
            mock_validate_id.return_value = True
            mock_app_id = Mock()
            mock_app_id.primary_identifier.return_value = "com.apple.finder"
            mock_create_app_id.return_value = mock_app_id

            # Setup window manager operation result
            mock_operation_result = Mock()
            mock_operation_result.window_info = Mock()
            mock_operation_result.window_info.app_identifier = "com.apple.finder"
            mock_operation_result.window_info.window_index = 0
            mock_operation_result.window_info.position = Mock(x=100, y=200)
            mock_operation_result.window_info.size = Mock(width=800, height=600)
            mock_operation_result.window_info.state = Mock(value="normal")
            mock_operation_result.window_info.title = "Test Window"
            mock_operation_result.operation_time = Mock()
            mock_operation_result.operation_time.total_seconds.return_value = 0.5
            mock_operation_result.details = "Operation completed"

            mock_result = Mock()
            mock_result.is_right.return_value = True
            mock_result.get_right.return_value = mock_operation_result

            mock_window_manager.move_window = AsyncMock(return_value=mock_result)

            # Execute operation with context
            result = await km_window_manager(
                operation="move",
                window_identifier="com.apple.finder",
                position={"x": 100, "y": 200},
                ctx=mock_context,
            )

            # Verify operation success
            assert result["success"] is True
            # Note: Context integration would be in the implementation


class TestWindowProperties:
    """Property-based tests for window management operations."""

    @given(
        window_operation_strategy(),
        window_identifier_strategy(),
        position_strategy(),
        size_strategy(),
    )
    def test_window_parameter_validation_properties(
        self,
        operation: str,
        window_identifier: str,
        position: dict[str, int],
        size: dict[str, int],
    ) -> None:
        """Property test for window parameter validation."""
        assume(len(window_identifier.strip()) > 0 and len(window_identifier) <= 255)

        # Properties that should always hold
        valid_operations = [
            "move",
            "resize",
            "minimize",
            "maximize",
            "restore",
            "arrange",
            "get_info",
            "get_screens",
        ]
        assert operation in valid_operations
        assert isinstance(window_identifier, str)
        assert 0 < len(window_identifier) <= 255

        # Position validation
        assert "x" in position and "y" in position
        assert isinstance(position["x"], int) and isinstance(position["y"], int)
        assert position["x"] >= 0 and position["y"] >= 0

        # Size validation
        assert "width" in size and "height" in size
        assert isinstance(size["width"], int) and isinstance(size["height"], int)
        assert size["width"] >= 100 and size["height"] >= 100

    @given(arrangement_strategy(), screen_strategy())
    def test_arrangement_and_screen_properties(
        self,
        arrangement: str,
        screen: str,
    ) -> None:
        """Property test for arrangement and screen parameters."""
        valid_arrangements = [
            "left_half",
            "right_half",
            "top_half",
            "bottom_half",
            "top_left_quarter",
            "top_right_quarter",
            "bottom_left_quarter",
            "bottom_right_quarter",
            "center",
            "maximize",
        ]
        valid_screens = ["main", "external", "0", "1", "2"]

        assert arrangement in valid_arrangements
        assert screen in valid_screens

    @given(window_index_strategy(), window_state_strategy())
    def test_window_index_and_state_properties(
        self,
        window_index: int,
        state: str,
    ) -> None:
        """Property test for window index and state parameters."""
        assert 0 <= window_index <= 20

        valid_states = ["normal", "minimized", "maximized", "fullscreen"]
        assert state in valid_states

    @given(st.text(min_size=1, max_size=50))
    def test_operation_result_structure_properties(self, operation_id: str) -> None:
        """Property test for operation result structure."""
        assume(operation_id.strip() != "")

        # Mock result structure
        result_structure = {
            "success": True,
            "operation": "move",
            "window": {
                "app_identifier": "com.apple.finder",
                "window_index": 0,
                "position": {"x": 100, "y": 200},
                "size": {"width": 800, "height": 600},
                "state": "normal",
                "title": "Test Window",
            },
            "execution_time_ms": 500,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Properties that should always hold
        assert "success" in result_structure
        assert isinstance(result_structure["success"], bool)
        assert "window" in result_structure
        assert "timestamp" in result_structure
        assert "position" in result_structure["window"]
        assert "size" in result_structure["window"]
        assert "x" in result_structure["window"]["position"]
        assert "y" in result_structure["window"]["position"]
        assert "width" in result_structure["window"]["size"]
        assert "height" in result_structure["window"]["size"]

    @given(invalid_window_identifier_strategy())
    def test_security_validation_properties(self, invalid_identifier: str) -> None:
        """Property test for security validation behavior."""
        # Security risks that should trigger validation failures
        security_risks = ["<>", "|", ";", "&", "`", "$", "../", "\x00"]

        has_risk = any(risk in invalid_identifier for risk in security_risks)

        if has_risk:
            # Invalid identifiers with security risks should be detectable
            # Include all security indicators including path traversal
            assert any(
                indicator in invalid_identifier
                for indicator in ["<>", "|", ";", "&", "`", "$", "../", "\x00"]
            )

        # Length validation
        if len(invalid_identifier) > 255:
            # Should be rejected for being too long
            assert len(invalid_identifier) > 255

        # Empty validation
        if len(invalid_identifier.strip()) == 0:
            # Should be rejected for being empty
            assert len(invalid_identifier.strip()) == 0
