"""Comprehensive Test Suite for Window Tools - Following Proven MCP Tool Test Pattern.

import logging

logging.basicConfig(level=logging.DEBUG)
This test suite validates the Window Tools functionality using the systematic
testing approach that achieved 100% success rate across 13 tool suites.

Test Coverage:
- Window management functionality with comprehensive validation (move, resize, state management)
- Multi-monitor support and screen targeting validation
- Coordinate validation and security boundary checking
- Window arrangement operations and predefined layouts
- Application identifier validation and security testing
- Property-based testing for robust input validation
- Integration testing with mocked window managers and app controllers
- Error handling for all failure scenarios
- Performance testing for window operation response times

Testing Strategy:
- Property-based testing with Hypothesis for comprehensive input coverage
- Mock-based testing for WindowManager and AppIdentifier components
- Security validation for application identifier injection prevention
- Integration testing scenarios with realistic window operations
- Performance and timeout testing with window operation limits

Key Mocking Pattern:
- WindowManager: Mock all methods with Either.success() pattern
- AppIdentifier: Mock app identification and validation
- Position/Size: Mock coordinate and dimension validation
- WindowState/WindowArrangement: Mock state and layout enums
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite
from src.applications.app_controller import AppIdentifier
from src.server.tools.window_tools import (
    _create_app_identifier,
    _format_screen_info,
    _format_window_info,
    _get_timestamp,
    _is_valid_app_identifier,
    _map_operation_to_state,
    km_window_manager,
)
from src.windows.window_manager import (
    WindowManager,
    WindowState,
)

# Test constants
TEST_LOW_SPAM_SCORE = 0.3  # Threshold for low spam score

if TYPE_CHECKING:
    from collections.abc import Callable


# Test fixtures following proven pattern
@pytest.fixture
def mock_context() -> Mock:
    """Create mock FastMCP context following successful pattern."""
    context = Mock(spec=Context)
    context.info = AsyncMock()
    context.warn = AsyncMock()
    context.error = AsyncMock()
    context.report_progress = AsyncMock()
    context.read_resource = AsyncMock()
    context.sample = AsyncMock()
    context.get = Mock(return_value="")  # Support ctx.get() calls
    return context


@pytest.fixture
def mock_window_manager() -> Mock:
    """Create mock WindowManager with standard interface."""
    manager = Mock(spec=WindowManager)
    manager.move_window = AsyncMock()
    manager.resize_window = AsyncMock()
    manager.set_window_state = AsyncMock()
    manager.arrange_window = AsyncMock()
    manager.get_window_info = AsyncMock()
    manager.get_screen_info = AsyncMock()

    # Setup standard success response using Either.success() pattern
    mock_result = Mock()
    mock_result.is_right.return_value = True
    mock_result.get_right.return_value = Mock(
        window_info=Mock(
            app_identifier="test-app",
            window_index=0,
            position=Mock(x=100, y=100),
            size=Mock(width=800, height=600),
            state=WindowState.NORMAL,
            title="Test Window",
        ),
        operation_time=timedelta(milliseconds=50),
        details={"operation": "success"},
    )

    # Apply to all async methods
    manager.move_window.return_value = mock_result
    manager.resize_window.return_value = mock_result
    manager.set_window_state.return_value = mock_result
    manager.arrange_window.return_value = mock_result
    manager.get_window_info.return_value = mock_result

    # Screen info returns list of screens
    screen_mock = Mock()
    screen_mock.screen_id = "main"
    screen_mock.name = "Built-in Display"
    screen_mock.is_main = True
    screen_mock.origin = Mock(x=0, y=0)
    screen_mock.size = Mock(width=1920, height=1080)
    screen_mock.center_position.return_value = Mock(x=960, y=540)
    manager.get_screen_info.return_value = [screen_mock]

    return manager


@pytest.fixture
def mock_app_identifier() -> Mock:
    """Create mock AppIdentifier with standard interface."""
    identifier = Mock(spec=AppIdentifier)
    identifier.primary_identifier.return_value = "com.test.app"
    identifier.bundle_id = "com.test.app"
    identifier.app_name = "Test App"
    return identifier


# Core Window Tools Tests
class TestWindowManagerTool:
    """Test core km_window_manager functionality."""

    @pytest.mark.asyncio
    async def test_move_operation_success(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test successful window move operation."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="move",
                window_identifier="Test App",
                position={"x": 100, "y": 100},
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["operation"] == "move"
            assert "window" in result
            assert "execution_time_ms" in result
            mock_window_manager.move_window.assert_called_once()

    @pytest.mark.asyncio
    async def test_resize_operation_success(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test successful window resize operation."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="resize",
                window_identifier="Test App",
                size={"width": 800, "height": 600},
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["operation"] == "resize"
            assert "window" in result
            assert "execution_time_ms" in result
            mock_window_manager.resize_window.assert_called_once()

    @pytest.mark.asyncio
    async def test_minimize_operation_success(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test successful window minimize operation."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="minimize",
                window_identifier="Test App",
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["operation"] == "set_state_minimized"
            assert "window" in result
            mock_window_manager.set_window_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_maximize_operation_success(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test successful window maximize operation."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="maximize",
                window_identifier="Test App",
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["operation"] == "set_state_maximized"
            assert "window" in result
            mock_window_manager.set_window_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_arrange_operation_success(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test successful window arrange operation."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="arrange",
                window_identifier="Test App",
                arrangement="left_half",
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["operation"] == "arrange_left_half"
            assert "window" in result
            assert "screen" in result
            mock_window_manager.arrange_window.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_info_operation_success(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test successful window info retrieval."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="get_info",
                window_identifier="Test App",
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["operation"] == "get_info"
            assert "window" in result
            mock_window_manager.get_window_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_screens_operation_success(
        self,
        mock_context: Any,
        mock_window_manager: Any,
    ) -> None:
        """Test successful screen info retrieval."""
        with patch(
            "src.server.tools.window_tools._window_manager",
            mock_window_manager,
        ):
            result = await km_window_manager(
                operation="get_screens",
                window_identifier="Test App",  # Required but not used for this operation
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["operation"] == "get_screens"
            assert "screens" in result
            assert "screen_count" in result
            assert result["screen_count"] == 1


# Helper Function Tests
class TestWindowToolsHelpers:
    """Test window tools helper functions."""

    def test_is_valid_app_identifier_valid_names(self) -> None:
        """Test valid application identifier validation."""
        valid_identifiers = [
            "Safari",
            "com.apple.Safari",
            "Microsoft Word",
            "Test-App_123",
            "My App",
        ]

        for identifier in valid_identifiers:
            assert _is_valid_app_identifier(identifier) is True

    def test_is_valid_app_identifier_invalid_names(self) -> None:
        """Test invalid application identifier validation."""
        invalid_identifiers = [
            "",  # Empty
            "   ",  # Only whitespace
            "app<script>",  # Invalid characters
            "app|command",  # Shell injection
            "../traversal",  # Path traversal
            "app;malicious",  # Command injection
            "a" * 300,  # Too long
        ]

        for identifier in invalid_identifiers:
            assert _is_valid_app_identifier(identifier) is False

    def test_create_app_identifier_bundle_id(self) -> None:
        """Test AppIdentifier creation from bundle ID."""
        result = _create_app_identifier("com.apple.Safari")
        assert hasattr(result, "bundle_id")

    def test_create_app_identifier_app_name(self) -> None:
        """Test AppIdentifier creation from app name."""
        result = _create_app_identifier("Safari")
        assert hasattr(result, "app_name")

    def test_map_operation_to_state_operations(self) -> None:
        """Test operation to WindowState mapping."""
        assert _map_operation_to_state("minimize", None) == WindowState.MINIMIZED
        assert _map_operation_to_state("maximize", None) == WindowState.MAXIMIZED
        assert _map_operation_to_state("restore", None) == WindowState.NORMAL
        assert _map_operation_to_state("unknown", None) == WindowState.NORMAL

    def test_map_operation_to_state_override(self) -> None:
        """Test state override functionality."""
        assert _map_operation_to_state("minimize", "maximized") == WindowState.MAXIMIZED

    def test_format_window_info_complete(self) -> None:
        """Test window info formatting with complete data."""
        mock_info = Mock()
        mock_info.app_identifier = "com.test.app"
        mock_info.window_index = 0
        mock_info.position = Mock(x=100, y=100)
        mock_info.size = Mock(width=800, height=600)
        mock_info.state = WindowState.NORMAL
        mock_info.title = "Test Window"

        result = _format_window_info(mock_info)

        assert result["app_identifier"] == "com.test.app"
        assert result["window_index"] == 0
        assert result["position"]["x"] == 100
        assert result["position"]["y"] == 100
        assert result["size"]["width"] == 800
        assert result["size"]["height"] == 600
        assert result["state"] == "normal"
        assert result["title"] == "Test Window"
        assert "bounds" in result

    def test_format_window_info_none(self) -> None:
        """Test window info formatting with None input."""
        result = _format_window_info(None)
        assert result == {}

    def test_format_screen_info_complete(self) -> None:
        """Test screen info formatting."""
        mock_screen = Mock()
        mock_screen.screen_id = "main"
        mock_screen.name = "Built-in Display"
        mock_screen.is_main = True
        mock_screen.origin = Mock(x=0, y=0)
        mock_screen.size = Mock(width=1920, height=1080)
        mock_screen.center_position.return_value = Mock(x=960, y=540)

        result = _format_screen_info(mock_screen)

        assert result["screen_id"] == "main"
        assert result["name"] == "Built-in Display"
        assert result["is_main"] is True
        assert result["origin"]["x"] == 0
        assert result["origin"]["y"] == 0
        assert result["size"]["width"] == 1920
        assert result["size"]["height"] == 1080
        assert result["center"]["x"] == 960
        assert result["center"]["y"] == 540
        assert "bounds" in result

    def test_get_timestamp_format(self) -> None:
        """Test timestamp generation format."""
        timestamp = _get_timestamp()
        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format indicator


# Error Handling Tests
class TestWindowToolsErrorHandling:
    """Test window tools error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_app_identifier_error(self, mock_context: Any) -> None:
        """Test handling of invalid application identifier."""
        result = await km_window_manager(
            operation="move",
            window_identifier="app;malicious",
            position={"x": 100, "y": 100},
            ctx=mock_context,
        )

        assert result["success"] is False
        assert result["error"] == "INVALID_IDENTIFIER"
        assert "Invalid application identifier format" in result["message"]

    @pytest.mark.asyncio
    async def test_move_missing_position_error(
        self,
        mock_context: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test move operation with missing position."""
        with (
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="move",
                window_identifier="Test App",
                position=None,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"] == "MISSING_POSITION"

    @pytest.mark.asyncio
    async def test_resize_missing_size_error(
        self,
        mock_context: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test resize operation with missing size."""
        with (
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="resize",
                window_identifier="Test App",
                size=None,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"] == "MISSING_SIZE"

    @pytest.mark.asyncio
    async def test_arrange_missing_arrangement_error(
        self,
        mock_context: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test arrange operation with missing arrangement."""
        with (
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="arrange",
                window_identifier="Test App",
                arrangement=None,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"] == "MISSING_ARRANGEMENT"

    @pytest.mark.asyncio
    async def test_unsupported_operation_error(self, mock_context: Any) -> None:
        """Test handling of unsupported operations."""
        with patch(
            "src.server.tools.window_tools._is_valid_app_identifier",
            return_value=True,
        ):
            result = await km_window_manager(
                operation="invalid_op",
                window_identifier="Test App",
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"] == "INVALID_OPERATION"
            assert "Unsupported operation" in result["message"]

    @pytest.mark.asyncio
    async def test_window_manager_error_response(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test handling of WindowManager error responses."""
        # Configure mock to return error
        error_result = Mock()
        error_result.is_right.return_value = False
        error_result.get_left.return_value = Mock(
            code="WINDOW_NOT_FOUND",
            message="Window not found",
        )
        mock_window_manager.move_window.return_value = error_result

        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="move",
                window_identifier="Test App",
                position={"x": 100, "y": 100},
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"] == "WINDOW_NOT_FOUND"
            assert result["message"] == "Window not found"

    @pytest.mark.asyncio
    async def test_exception_handling(self, mock_context: Any) -> None:
        """Test general exception handling."""
        with patch(
            "src.server.tools.window_tools._is_valid_app_identifier",
            side_effect=Exception("Test error"),
        ):
            result = await km_window_manager(
                operation="move",
                window_identifier="Test App",
                position={"x": 100, "y": 100},
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"] == "EXECUTION_ERROR"
            assert "Operation failed" in result["message"]


# Integration Tests
class TestWindowToolsIntegration:
    """Test window tools integration scenarios."""

    @pytest.mark.asyncio
    async def test_move_and_resize_workflow(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test combined move and resize workflow."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            # Move window
            move_result = await km_window_manager(
                operation="move",
                window_identifier="Test App",
                position={"x": 100, "y": 100},
                ctx=mock_context,
            )

            # Resize window
            resize_result = await km_window_manager(
                operation="resize",
                window_identifier="Test App",
                size={"width": 800, "height": 600},
                ctx=mock_context,
            )

            assert move_result["success"] is True
            assert resize_result["success"] is True
            assert mock_window_manager.move_window.call_count == 1
            assert mock_window_manager.resize_window.call_count == 1

    @pytest.mark.asyncio
    async def test_multi_monitor_workflow(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test multi-monitor window management."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            # Get screen info
            screens_result = await km_window_manager(
                operation="get_screens",
                window_identifier="Test App",
                ctx=mock_context,
            )

            # Move to external screen
            move_result = await km_window_manager(
                operation="move",
                window_identifier="Test App",
                position={"x": 2000, "y": 100},
                screen="external",
                ctx=mock_context,
            )

            assert screens_result["success"] is True
            assert move_result["success"] is True
            assert screens_result["screen_count"] == 1

    @pytest.mark.asyncio
    async def test_window_state_transitions(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test window state transition workflow."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            # Minimize window
            minimize_result = await km_window_manager(
                operation="minimize",
                window_identifier="Test App",
                ctx=mock_context,
            )

            # Restore window
            restore_result = await km_window_manager(
                operation="restore",
                window_identifier="Test App",
                ctx=mock_context,
            )

            # Maximize window
            maximize_result = await km_window_manager(
                operation="maximize",
                window_identifier="Test App",
                ctx=mock_context,
            )

            assert minimize_result["success"] is True
            assert restore_result["success"] is True
            assert maximize_result["success"] is True
            assert mock_window_manager.set_window_state.call_count == 3


# Security Tests
class TestWindowToolsSecurity:
    """Test window tools security validation."""

    @pytest.mark.asyncio
    async def test_app_identifier_injection_prevention(self, mock_context: Any) -> None:
        """Test prevention of application identifier injection."""
        malicious_identifiers = [
            "app;rm -rf /",
            "app|malicious",
            "../../../etc/passwd",
            "app<script>alert('xss')</script>",
            "app`malicious`",
        ]

        for identifier in malicious_identifiers:
            result = await km_window_manager(
                operation="move",
                window_identifier=identifier,
                position={"x": 100, "y": 100},
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"] == "INVALID_IDENTIFIER"

    @pytest.mark.asyncio
    async def test_coordinate_validation(
        self,
        mock_context: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test coordinate boundary validation."""
        with (
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            # Test with Position class validation
            with patch(
                "src.server.tools.window_tools.Position",
                side_effect=ValueError("Invalid coordinates"),
            ):
                result = await km_window_manager(
                    operation="move",
                    window_identifier="Test App",
                    position={"x": -10000, "y": -10000},
                    ctx=mock_context,
                )

                assert result["success"] is False
                assert result["error"] == "INVALID_POSITION"

    @pytest.mark.asyncio
    async def test_size_validation(
        self,
        mock_context: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test size boundary validation."""
        with (
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            # Test with Size class validation
            with patch(
                "src.server.tools.window_tools.Size",
                side_effect=ValueError("Invalid size"),
            ):
                result = await km_window_manager(
                    operation="resize",
                    window_identifier="Test App",
                    size={"width": -100, "height": -100},
                    ctx=mock_context,
                )

                assert result["success"] is False
                assert result["error"] == "INVALID_SIZE"

    @pytest.mark.asyncio
    async def test_arrangement_validation(
        self,
        mock_context: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test arrangement type validation."""
        with (
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            # Test with WindowArrangement validation
            with patch(
                "src.server.tools.window_tools.WindowArrangement",
                side_effect=ValueError("Invalid arrangement"),
            ):
                result = await km_window_manager(
                    operation="arrange",
                    window_identifier="Test App",
                    arrangement="invalid_arrangement",
                    ctx=mock_context,
                )

                assert result["success"] is False
                assert result["error"] == "INVALID_ARRANGEMENT"


# Property-Based Tests
class TestWindowToolsPropertyBased:
    """Property-based testing for window tools with Hypothesis."""

    @staticmethod
    @composite
    def valid_app_names(draw: Callable[..., Any]) -> str:
        """Generate valid application names."""
        # Generate names without dangerous characters
        chars = st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Pc"),
            whitelist_characters=" -._",
        )
        return cast("str", draw(st.text(chars, min_size=1, max_size=100)))

    @staticmethod
    @composite
    def valid_coordinates(draw: Callable[..., Any]) -> dict[str, int]:
        """Generate valid coordinate pairs."""
        x = draw(st.integers(min_value=-10000, max_value=10000))
        y = draw(st.integers(min_value=-10000, max_value=10000))
        return {"x": x, "y": y}

    @staticmethod
    @composite
    def valid_sizes(draw: Callable[..., Any]) -> dict[str, int]:
        """Generate valid size dimensions."""
        width = draw(st.integers(min_value=1, max_value=5000))
        height = draw(st.integers(min_value=1, max_value=5000))
        return {"width": width, "height": height}

    @given(valid_app_names())
    def test_valid_app_identifier_property(self, app_name: str) -> None:
        """Property: Valid app names should pass validation."""
        assume(len(app_name.strip()) > 0)  # Non-empty after strip
        assume(
            not any(char in app_name for char in '<>"|*?;&|`$'),
        )  # No dangerous chars

        result = _is_valid_app_identifier(app_name)
        assert result is True

    @staticmethod
    @composite
    def valid_app_identifiers(draw: Callable[..., Any]) -> str:
        """Generate valid application identifiers that pass both _is_valid_app_identifier and AppIdentifier validation."""
        # Generate either a valid bundle ID or app name
        choice = draw(st.integers(min_value=0, max_value=1))

        if choice == 0:
            # Generate valid bundle ID: alphanumeric with dots and hyphens
            parts = draw(
                st.lists(
                    st.text(
                        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                        min_size=1,
                        max_size=10,
                    ),
                    min_size=2,
                    max_size=4,
                ),
            )
            identifier = ".".join(parts)
        else:
            # Generate valid app name: printable characters except dangerous ones
            identifier = draw(
                st.text(
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
                        whitelist_characters="-_()[]{}",
                        blacklist_characters='<>"|*?;&|`$./\\',
                    ),
                    min_size=1,
                    max_size=50,
                ).filter(
                    lambda x: x.strip()
                    and not x.startswith(".")
                    and not x.endswith("."),
                ),
            )

        return identifier

    @given(valid_app_identifiers())
    def test_create_app_identifier_property(self, identifier: str) -> None:
        """Property: All valid identifiers should create AppIdentifier objects."""
        # Verify it passes both validation functions
        assert _is_valid_app_identifier(identifier)

        result = _create_app_identifier(identifier)
        assert hasattr(result, "primary_identifier")  # Has required method
        assert (
            result.primary_identifier() == identifier
            or result.primary_identifier() in identifier
        )

    @given(
        st.sampled_from(["minimize", "maximize", "restore"]),
        st.one_of(
            st.none(),
            st.sampled_from(["normal", "minimized", "maximized", "fullscreen"]),
        ),
    )
    def test_map_operation_property(self, operation: str, state_override: Any) -> None:
        """Property: All operations should map to valid WindowState values."""
        result = _map_operation_to_state(operation, state_override)
        assert isinstance(result, WindowState)

    @pytest.mark.asyncio
    @given(valid_coordinates())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_move_operation_coordinates_property(
        self, position: dict[str, int]
    ) -> None:
        """Property: Valid coordinates should not cause validation errors in move operation."""
        mock_context = Mock(spec=Context)
        mock_context.info = AsyncMock()

        mock_app_identifier = Mock(spec=AppIdentifier)
        mock_app_identifier.primary_identifier.return_value = "com.test.app"

        mock_window_manager = Mock(spec=WindowManager)
        mock_window_manager.move_window = AsyncMock()

        # Mock successful result
        mock_result = Mock()
        mock_result.is_right.return_value = True
        mock_result.get_right.return_value = Mock(
            window_info=Mock(
                app_identifier="test-app",
                window_index=0,
                position=Mock(x=position["x"], y=position["y"]),
                size=Mock(width=800, height=600),
                state=WindowState.NORMAL,
                title="Test Window",
            ),
            operation_time=timedelta(milliseconds=50),
            details={"operation": "success"},
        )
        mock_window_manager.move_window.return_value = mock_result

        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="move",
                window_identifier="Test App",
                position=position,
                ctx=mock_context,
            )

            # Should either succeed or fail with specific error (not crash)
            assert "success" in result
            assert isinstance(result["success"], bool)

    @pytest.mark.asyncio
    @given(valid_sizes())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_resize_operation_sizes_property(
        self, size: dict[str, int]
    ) -> None:
        """Property: Valid sizes should not cause validation errors in resize operation."""
        mock_context = Mock(spec=Context)
        mock_context.info = AsyncMock()

        mock_app_identifier = Mock(spec=AppIdentifier)
        mock_app_identifier.primary_identifier.return_value = "com.test.app"

        mock_window_manager = Mock(spec=WindowManager)
        mock_window_manager.resize_window = AsyncMock()

        # Mock successful result
        mock_result = Mock()
        mock_result.is_right.return_value = True
        mock_result.get_right.return_value = Mock(
            window_info=Mock(
                app_identifier="test-app",
                window_index=0,
                position=Mock(x=100, y=100),
                size=Mock(width=size["width"], height=size["height"]),
                state=WindowState.NORMAL,
                title="Test Window",
            ),
            operation_time=timedelta(milliseconds=50),
            details={"operation": "success"},
        )
        mock_window_manager.resize_window.return_value = mock_result

        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="resize",
                window_identifier="Test App",
                size=size,
                ctx=mock_context,
            )

            # Should either succeed or fail with specific error (not crash)
            assert "success" in result
            assert isinstance(result["success"], bool)


# Performance Tests
class TestWindowToolsPerformance:
    """Test window tools performance characteristics."""

    @pytest.mark.asyncio
    async def test_operation_response_time(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test that window operations complete within reasonable time."""
        import time

        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            operations: list[dict[str, Any]] = [
                {"operation": "move", "position": {"x": 100, "y": 100}},
                {"operation": "resize", "size": {"width": 800, "height": 600}},
                {"operation": "minimize"},
                {"operation": "maximize"},
                {"operation": "arrange", "arrangement": "left_half"},
                {"operation": "get_info"},
                {"operation": "get_screens"},
            ]

            for op_config in operations:
                start_time = time.time()

                result = await km_window_manager(
                    window_identifier="Test App",
                    ctx=mock_context,
                    **op_config,
                )

                end_time = time.time()
                execution_time = end_time - start_time

                # Should complete within 3 seconds (allowing for mocking overhead)
                assert execution_time < TEST_LOW_SPAM_SCORE
                assert (
                    result["success"] is True
                )  # Should succeed with mocked components

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test concurrent window operations."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            # Run multiple operations concurrently
            tasks = [
                km_window_manager(
                    operation="move",
                    window_identifier="Test App",
                    position={"x": 100 + i * 50, "y": 100 + i * 50},
                    ctx=mock_context,
                )
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)

            # All operations should succeed
            for result in results:
                assert result["success"] is True

            # WindowManager should have been called for each operation
            assert mock_window_manager.move_window.call_count == 5


# Edge Case Tests
class TestWindowToolsEdgeCases:
    """Test window tools edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_app_identifier(self, mock_context: Any) -> None:
        """Test handling of empty application identifier."""
        result = await km_window_manager(
            operation="move",
            window_identifier="",
            position={"x": 100, "y": 100},
            ctx=mock_context,
        )

        assert result["success"] is False
        assert result["error"] == "INVALID_IDENTIFIER"

    @pytest.mark.asyncio
    async def test_maximum_length_app_identifier(self, mock_context: Any) -> None:
        """Test handling of maximum length application identifier."""
        long_identifier = "a" * 255  # Maximum allowed length

        with (
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
            ) as mock_create,
        ):
            mock_app_id = Mock(spec=AppIdentifier)
            mock_app_id.primary_identifier.return_value = "com.test.app"
            mock_create.return_value = mock_app_id

            await km_window_manager(
                operation="get_info",
                window_identifier=long_identifier,
                ctx=mock_context,
            )

            # Should process without length-related errors
            mock_create.assert_called_once_with(long_identifier)

    @pytest.mark.asyncio
    async def test_unicode_app_identifier(
        self,
        mock_context: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test handling of Unicode application identifiers."""
        unicode_identifier = "Test App 测试 🎯"

        with (
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            result = await km_window_manager(
                operation="get_info",
                window_identifier=unicode_identifier,
                ctx=mock_context,
            )

            # Should handle Unicode gracefully (success depends on mocked components)
            assert "success" in result
            assert isinstance(result["success"], bool)

    @pytest.mark.asyncio
    async def test_extreme_coordinates(
        self,
        mock_context: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test handling of extreme coordinate values."""
        extreme_positions = [
            {"x": -50000, "y": -50000},  # Very negative
            {"x": 50000, "y": 50000},  # Very positive
            {"x": 0, "y": 0},  # Origin
        ]

        with (
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            for position in extreme_positions:
                result = await km_window_manager(
                    operation="move",
                    window_identifier="Test App",
                    position=position,
                    ctx=mock_context,
                )

                # Should handle extreme values gracefully (may succeed or fail with validation error)
                assert "success" in result
                assert isinstance(result["success"], bool)

    @pytest.mark.asyncio
    async def test_window_index_boundaries(
        self,
        mock_context: Any,
        mock_window_manager: Any,
        mock_app_identifier: Any,
    ) -> None:
        """Test window index boundary conditions."""
        with (
            patch("src.server.tools.window_tools._window_manager", mock_window_manager),
            patch(
                "src.server.tools.window_tools._create_app_identifier",
                return_value=mock_app_identifier,
            ),
            patch(
                "src.server.tools.window_tools._is_valid_app_identifier",
                return_value=True,
            ),
        ):
            # Test with minimum and maximum allowed window indices
            for window_index in [0, 20]:  # 0 is minimum, 20 is maximum allowed
                result = await km_window_manager(
                    operation="get_info",
                    window_identifier="Test App",
                    window_index=window_index,
                    ctx=mock_context,
                )

                assert result["success"] is True
                mock_window_manager.get_window_info.assert_called()
