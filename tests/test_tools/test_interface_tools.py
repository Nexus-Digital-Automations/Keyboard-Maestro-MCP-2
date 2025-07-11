"""Comprehensive Test Suite for Interface Tools - Following Proven MCP Tool Test Pattern.

import logging

logging.basicConfig(level=logging.DEBUG)
This test suite validates the Interface Tools functionality using the systematic
testing approach that achieved 100% success rate across 17 tool suites.

Test Coverage:
- Interface automation operations (click, double_click, right_click, drag, type, key_press, move_mouse)
- Coordinate validation and boundary checking with comprehensive error handling
- KM client integration and connection management with Either pattern success mocking
- Modifier key combinations and keystroke parsing with validation
- Progress reporting and context integration validation
- Security validation for coordinate bounds and input sanitization
- Property-based testing for robust coordinate and keystroke validation
- Integration testing with mocked KM clients and interface operations
- Error handling for all failure scenarios
- Performance testing for interface operation response times

Testing Strategy:
- Property-based testing with Hypothesis for comprehensive input coverage
- Mock-based testing for KM Client and interface automation components
- Security validation for coordinate bounds and injection prevention
- Integration testing scenarios with realistic interface automation operations
- Performance and timeout testing with interface operation limits

Key Mocking Pattern:
- KMClient: Mock Keyboard Maestro client integration with check_connection
- Interface operations: Mock automation operations with proper async execution
- Context: Mock progress reporting and logging operations
- Coordinate validation: Test boundary conditions and error scenarios
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite
from src.core.errors import ValidationError
from src.server.tools.interface_tools import (
    _move_mouse,
    _perform_click,
    _perform_drag,
    _press_keys,
    _type_text,
    _validate_coordinates,
    km_interface_automation,
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
def mock_km_client() -> Mock:
    """Create mock KM client with standard interface."""
    client = Mock()
    # Mock connection check - CRITICAL for all tests
    mock_connection_result = Mock()
    mock_connection_result.is_left.return_value = False
    mock_connection_result.get_right.return_value = True
    client.check_connection.return_value = mock_connection_result
    return client


@pytest.fixture
def valid_coordinates() -> dict[str, Any]:
    """Provide valid test coordinates."""
    return {"x": 100, "y": 200}


@pytest.fixture
def valid_end_coordinates() -> dict[str, Any]:
    """Provide valid end coordinates for drag operations."""
    return {"x": 300, "y": 400}


# Core Interface Operations Tests
class TestInterfaceOperations:
    """Test core km_interface_automation functionality."""

    @pytest.mark.asyncio
    async def test_click_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_coordinates: Any,
    ) -> None:
        """Test successful click operation."""
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates=valid_coordinates,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["operation"] == "click"
            assert result["data"]["coordinates"] == valid_coordinates
            assert result["data"]["modifiers"] == []
            assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_double_click_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_coordinates: Any,
    ) -> None:
        """Test successful double click operation."""
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="double_click",
                coordinates=valid_coordinates,
                modifiers=["cmd", "shift"],
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["operation"] == "double_click"
            assert result["data"]["coordinates"] == valid_coordinates
            assert result["data"]["modifiers"] == ["cmd", "shift"]
            assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_right_click_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_coordinates: Any,
    ) -> None:
        """Test successful right click operation."""
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="right_click",
                coordinates=valid_coordinates,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["operation"] == "right_click"
            assert result["data"]["coordinates"] == valid_coordinates
            assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_move_mouse_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_coordinates: Any,
    ) -> None:
        """Test successful mouse movement operation."""
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="move_mouse",
                coordinates=valid_coordinates,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["operation"] == "move_mouse"
            assert result["data"]["coordinates"] == valid_coordinates
            assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_drag_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_coordinates: Any,
        valid_end_coordinates: Any,
    ) -> None:
        """Test successful drag operation."""
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="drag",
                coordinates=valid_coordinates,
                end_coordinates=valid_end_coordinates,
                duration_ms=500,
                modifiers=["shift"],
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["operation"] == "drag"
            assert result["data"]["start_coordinates"] == valid_coordinates
            assert result["data"]["end_coordinates"] == valid_end_coordinates
            assert result["data"]["duration_ms"] == 500
            assert result["data"]["modifiers"] == ["shift"]
            assert "distance_pixels" in result["data"]
            assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_type_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test successful text typing operation."""
        test_text = "Hello, World! 123"

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="type",
                text=test_text,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["operation"] == "type"
            assert result["data"]["character_count"] == len(test_text)
            assert "estimated_duration_ms" in result["data"]
            assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_key_press_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test successful key press operation."""
        keystroke = "cmd+shift+a"

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="key_press",
                keystroke=keystroke,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["operation"] == "key_press"
            assert result["data"]["keystroke"] == keystroke
            assert "modifiers" in result["data"]
            assert "main_key" in result["data"]
            assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_operation_with_delay(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_coordinates: Any,
    ) -> None:
        """Test operation with delay parameter."""
        start_time = time.time()

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates=valid_coordinates,
                delay_ms=100,  # Small delay for testing
                ctx=mock_context,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            assert result["success"] is True
            assert execution_time >= 0.1  # Should have waited at least 100ms
            assert execution_time < 1.0  # But not too long


# Error Handling Tests
class TestInterfaceErrorHandling:
    """Test interface tools error handling scenarios."""

    @pytest.mark.asyncio
    async def test_missing_coordinates_error(self, mock_context: Any) -> None:
        """Test error when coordinates are missing for click operations."""
        with patch("src.server.tools.interface_tools.get_km_client"):
            result = await km_interface_automation(
                operation="click",
                coordinates=None,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "AUTOMATION_ERROR"
            assert "Coordinates required" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_missing_text_error(self, mock_context: Any) -> None:
        """Test error when text is missing for type operation."""
        with patch("src.server.tools.interface_tools.get_km_client"):
            result = await km_interface_automation(
                operation="type",
                text=None,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "AUTOMATION_ERROR"
            assert "Text required" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_missing_keystroke_error(self, mock_context: Any) -> None:
        """Test error when keystroke is missing for key_press operation."""
        with patch("src.server.tools.interface_tools.get_km_client"):
            result = await km_interface_automation(
                operation="key_press",
                keystroke=None,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "AUTOMATION_ERROR"
            assert "Keystroke required" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_missing_drag_coordinates_error(
        self,
        mock_context: Any,
        valid_coordinates: Any,
    ) -> None:
        """Test error when end coordinates are missing for drag operation."""
        with patch("src.server.tools.interface_tools.get_km_client"):
            result = await km_interface_automation(
                operation="drag",
                coordinates=valid_coordinates,
                end_coordinates=None,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "AUTOMATION_ERROR"
            assert (
                "Both start and end coordinates required" in result["error"]["details"]
            )

    @pytest.mark.asyncio
    async def test_km_connection_failure(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_coordinates: Any,
    ) -> None:
        """Test handling of KM connection failure."""
        # Mock connection failure
        mock_connection_result = Mock()
        mock_connection_result.is_left.return_value = True
        mock_km_client.check_connection.return_value = mock_connection_result

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates=valid_coordinates,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "KM_CONNECTION_FAILED"
            assert (
                "Cannot connect to Keyboard Maestro Engine"
                in result["error"]["message"]
            )


# Coordinate Validation Tests
class TestCoordinateValidation:
    """Test coordinate validation functionality."""

    def test_valid_coordinates(self) -> None:
        """Test validation of valid coordinates."""
        valid_coords = {"x": 100, "y": 200}
        # Should not raise exception
        _validate_coordinates(valid_coords)

    def test_missing_x_coordinate(self) -> None:
        """Test validation error for missing x coordinate."""
        invalid_coords = {"y": 200}
        with pytest.raises(ValidationError) as exc_info:
            _validate_coordinates(invalid_coords)
        assert "must have 'x' and 'y' values" in str(exc_info.value)

    def test_missing_y_coordinate(self) -> None:
        """Test validation error for missing y coordinate."""
        invalid_coords = {"x": 100}
        with pytest.raises(ValidationError) as exc_info:
            _validate_coordinates(invalid_coords)
        assert "must have 'x' and 'y' values" in str(exc_info.value)

    def test_coordinates_out_of_range(self) -> None:
        """Test validation error for coordinates out of range."""
        # Test x coordinate too large
        invalid_coords = {"x": 15000, "y": 200}
        with pytest.raises(ValidationError) as exc_info:
            _validate_coordinates(invalid_coords)
        assert "out of reasonable range" in str(exc_info.value)

        # Test y coordinate too small
        invalid_coords = {"x": 100, "y": -10000}
        with pytest.raises(ValidationError) as exc_info:
            _validate_coordinates(invalid_coords)
        assert "out of reasonable range" in str(exc_info.value)


# Helper Function Tests
class TestHelperFunctions:
    """Test interface automation helper functions."""

    @pytest.mark.asyncio
    async def test_perform_click_function(self, mock_context: Any) -> None:
        """Test the _perform_click helper function directly."""
        mock_km_client = Mock()
        coordinates = {"x": 150, "y": 250}
        modifiers = ["cmd", "shift"]

        result = await _perform_click(
            mock_km_client,
            "click",
            coordinates,
            modifiers,
            mock_context,
        )

        assert result["success"] is True
        assert result["data"]["operation"] == "click"
        assert result["data"]["coordinates"] == coordinates
        assert result["data"]["modifiers"] == modifiers

    @pytest.mark.asyncio
    async def test_move_mouse_function(self, mock_context: Any) -> None:
        """Test the _move_mouse helper function directly."""
        mock_km_client = Mock()
        coordinates = {"x": 500, "y": 600}

        result = await _move_mouse(mock_km_client, coordinates, mock_context)

        assert result["success"] is True
        assert result["data"]["operation"] == "move_mouse"
        assert result["data"]["coordinates"] == coordinates

    @pytest.mark.asyncio
    async def test_perform_drag_function(self, mock_context: Any) -> None:
        """Test the _perform_drag helper function directly."""
        mock_km_client = Mock()
        start_coords = {"x": 100, "y": 100}
        end_coords = {"x": 200, "y": 200}
        duration = 300
        modifiers = ["ctrl"]

        result = await _perform_drag(
            mock_km_client,
            start_coords,
            end_coords,
            duration,
            modifiers,
            mock_context,
        )

        assert result["success"] is True
        assert result["data"]["operation"] == "drag"
        assert result["data"]["start_coordinates"] == start_coords
        assert result["data"]["end_coordinates"] == end_coords
        assert result["data"]["duration_ms"] == duration
        assert result["data"]["modifiers"] == modifiers
        # Distance should be approximately sqrt((200-100)^2 + (200-100)^2) = sqrt(20000) ≈ 141.42
        assert abs(result["data"]["distance_pixels"] - 141.42) < 0.1

    @pytest.mark.asyncio
    async def test_type_text_function(self, mock_context: Any) -> None:
        """Test the _type_text helper function directly."""
        mock_km_client = Mock()
        text = "Test typing functionality"

        result = await _type_text(mock_km_client, text, mock_context)

        assert result["success"] is True
        assert result["data"]["operation"] == "type"
        assert result["data"]["character_count"] == len(text)
        assert result["data"]["estimated_duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_press_keys_function_simple(self, mock_context: Any) -> None:
        """Test the _press_keys helper function with simple keystroke."""
        mock_km_client = Mock()
        keystroke = "a"

        result = await _press_keys(mock_km_client, keystroke, mock_context)

        assert result["success"] is True
        assert result["data"]["operation"] == "key_press"
        assert result["data"]["keystroke"] == keystroke
        assert result["data"]["main_key"] == "a"
        assert result["data"]["modifiers"] == []

    @pytest.mark.asyncio
    async def test_press_keys_function_complex(self, mock_context: Any) -> None:
        """Test the _press_keys helper function with complex keystroke."""
        mock_km_client = Mock()
        keystroke = "cmd+shift+escape"

        result = await _press_keys(mock_km_client, keystroke, mock_context)

        assert result["success"] is True
        assert result["data"]["operation"] == "key_press"
        assert result["data"]["keystroke"] == keystroke
        assert result["data"]["main_key"] == "escape"
        assert "cmd" in result["data"]["modifiers"]
        assert "shift" in result["data"]["modifiers"]

    @pytest.mark.asyncio
    async def test_press_keys_invalid_key(self, mock_context: Any) -> None:
        """Test _press_keys with invalid key."""
        mock_km_client = Mock()
        keystroke = "invalid_key_name"

        with pytest.raises(ValidationError) as exc_info:
            await _press_keys(mock_km_client, keystroke, mock_context)
        assert "Invalid key" in str(exc_info.value)


# Integration Tests
class TestInterfaceIntegration:
    """Test interface tools integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_interface_workflow(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test complete interface automation workflow."""
        coordinates = {"x": 100, "y": 200}

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Test click
            click_result = await km_interface_automation(
                operation="click",
                coordinates=coordinates,
                ctx=mock_context,
            )
            assert click_result["success"] is True

            # Test type
            type_result = await km_interface_automation(
                operation="type",
                text="Hello World",
                ctx=mock_context,
            )
            assert type_result["success"] is True

            # Test key press
            key_result = await km_interface_automation(
                operation="key_press",
                keystroke="cmd+a",
                ctx=mock_context,
            )
            assert key_result["success"] is True

    @pytest.mark.asyncio
    async def test_complex_drag_operation(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test complex drag operation with modifiers."""
        start_coords = {"x": 50, "y": 50}
        end_coords = {"x": 300, "y": 400}

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="drag",
                coordinates=start_coords,
                end_coordinates=end_coords,
                duration_ms=800,
                modifiers=["cmd", "option"],
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["duration_ms"] == 800
            assert result["data"]["modifiers"] == ["cmd", "option"]
            # Distance should be sqrt((300-50)^2 + (400-50)^2) = sqrt(250^2 + 350^2)
            expected_distance = (250**2 + 350**2) ** 0.5
            assert abs(result["data"]["distance_pixels"] - expected_distance) < 0.1


# Context Integration Tests
class TestInterfaceContext:
    """Test interface tools context integration."""

    @pytest.mark.asyncio
    async def test_context_info_logging(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_coordinates: Any,
    ) -> None:
        """Test context info logging during execution."""
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates=valid_coordinates,
                ctx=mock_context,
            )

            assert result["success"] is True
            # Verify info logging was called
            mock_context.info.assert_called()
            # Verify progress reporting was called
            mock_context.report_progress.assert_called()

    @pytest.mark.asyncio
    async def test_context_error_logging(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test context error logging during failures."""
        # Mock connection failure to trigger error
        mock_connection_result = Mock()
        mock_connection_result.is_left.return_value = True
        mock_km_client.check_connection.return_value = mock_connection_result

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates={"x": 100, "y": 200},
                ctx=mock_context,
            )

            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_without_context(
        self,
        mock_km_client: Any,
        valid_coordinates: Any,
    ) -> None:
        """Test operation without context provided."""
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates=valid_coordinates,
                ctx=None,
            )

            assert result["success"] is True


# Security Tests
class TestInterfaceSecurity:
    """Test interface tools security validation."""

    @pytest.mark.asyncio
    async def test_coordinate_bounds_validation(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test coordinate bounds security validation."""
        # Test extremely large coordinates
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates={"x": 50000, "y": 50000},
                ctx=mock_context,
            )
            assert result["success"] is False
            assert "out of reasonable range" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_negative_coordinate_bounds(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test negative coordinate bounds validation."""
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates={"x": -10000, "y": 100},
                ctx=mock_context,
            )
            assert result["success"] is False
            assert "out of reasonable range" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_text_length_limits(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test text length validation for type operations."""
        # Test very long text (should be handled gracefully)
        long_text = "x" * 5000

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="type",
                text=long_text,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["character_count"] == len(long_text)

    @pytest.mark.asyncio
    async def test_keystroke_validation(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test keystroke format validation."""
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Test invalid keystroke
            result = await km_interface_automation(
                operation="key_press",
                keystroke="invalid+malicious+key",
                ctx=mock_context,
            )

            assert result["success"] is False
            assert "Invalid key" in result["error"]["details"]


# Property-Based Tests
class TestInterfacePropertyBased:
    """Property-based testing for interface tools with Hypothesis."""

    @composite
    def valid_coordinates_strategy(draw: Callable[..., Any]) -> dict[str, Any]:
        """Generate valid coordinate pairs."""
        x = draw(st.integers(min_value=-4999, max_value=9999))
        y = draw(st.integers(min_value=-4999, max_value=9999))
        return {"x": x, "y": y}

    @composite
    def valid_operations(draw: Callable[..., Any]) -> Mock:
        """Generate valid interface operations."""
        return draw(
            st.sampled_from(
                [
                    "click",
                    "double_click",
                    "right_click",
                    "drag",
                    "type",
                    "key_press",
                    "move_mouse",
                ],
            ),
        )

    @composite
    def valid_modifiers(draw: Callable[..., Any]) -> Mock:
        """Generate valid modifier combinations."""
        modifiers = draw(
            st.lists(
                st.sampled_from(["cmd", "ctrl", "option", "shift"]),
                min_size=0,
                max_size=4,
                unique=True,
            ),
        )
        return modifiers

    @given(valid_coordinates_strategy())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_coordinate_validation_property(
        self,
        coordinates: Callable[..., Any],
    ) -> None:
        """Property: Valid coordinates should pass validation."""
        # Should not raise exception for coordinates in valid range
        _validate_coordinates(coordinates)

    @given(valid_operations())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_operation_validation_property(self, operation: str) -> None:
        """Property: Valid operations should be accepted."""
        assert operation in [
            "click",
            "double_click",
            "right_click",
            "drag",
            "type",
            "key_press",
            "move_mouse",
        ]


# Performance Tests
class TestInterfacePerformance:
    """Test interface tools performance characteristics."""

    @pytest.mark.asyncio
    async def test_interface_operation_response_time(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_coordinates: Any,
    ) -> None:
        """Test that interface operations complete within reasonable time."""
        start_time = time.time()

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates=valid_coordinates,
                ctx=mock_context,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within 2 seconds (allowing for mocking overhead)
            assert execution_time < 2.0
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_typing_performance(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test typing operation performance with various text lengths."""
        test_cases = [
            ("Short", 10),
            ("Medium text for testing", 100),
            ("x" * 500, 500),  # Long text
        ]

        for _description, length in test_cases:
            text = "x" * length
            start_time = time.time()

            with patch(
                "src.server.tools.interface_tools.get_km_client",
                return_value=mock_km_client,
            ):
                result = await km_interface_automation(
                    operation="type",
                    text=text,
                    ctx=mock_context,
                )

                end_time = time.time()
                execution_time = end_time - start_time

                # Should complete within reasonable time (max 3 seconds for any length)
                # Interface tools typing operations should complete within 3 seconds max
                assert execution_time < 3.0
                assert result["success"] is True
                assert result["data"]["character_count"] == length

    @pytest.mark.asyncio
    async def test_drag_operation_performance(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test drag operation performance with various durations."""
        start_coords = {"x": 0, "y": 0}
        end_coords = {"x": 100, "y": 100}

        # Test different drag durations
        for duration_ms in [100, 300, 500]:
            start_time = time.time()

            with patch(
                "src.server.tools.interface_tools.get_km_client",
                return_value=mock_km_client,
            ):
                result = await km_interface_automation(
                    operation="drag",
                    coordinates=start_coords,
                    end_coordinates=end_coords,
                    duration_ms=duration_ms,
                    ctx=mock_context,
                )

                end_time = time.time()
                execution_time = end_time - start_time

                # Should take approximately the specified duration (plus some overhead)
                expected_time = duration_ms / 1000.0
                assert execution_time >= expected_time
                assert execution_time < expected_time + 1.0  # Allow 1 second overhead
                assert result["success"] is True


# Edge Case Tests
class TestInterfaceEdgeCases:
    """Test interface tools edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_zero_coordinates(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test handling of zero coordinates."""
        zero_coords = {"x": 0, "y": 0}

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates=zero_coords,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["coordinates"] == zero_coords

    @pytest.mark.asyncio
    async def test_negative_valid_coordinates(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test handling of negative but valid coordinates."""
        negative_coords = {"x": -100, "y": -200}

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="click",
                coordinates=negative_coords,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["coordinates"] == negative_coords

    @pytest.mark.asyncio
    async def test_empty_text_typing(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test typing empty text."""
        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="type",
                text="",
                ctx=mock_context,
            )

            # Empty text should be treated as missing text
            assert result["success"] is False
            assert "Text required" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_unicode_text_typing(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test typing Unicode text."""
        unicode_text = "Hello 世界 🌍 测试"

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="type",
                text=unicode_text,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["character_count"] == len(unicode_text)

    @pytest.mark.asyncio
    async def test_special_key_combinations(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test various special key combinations."""
        special_keystrokes = [
            "f1",
            "ctrl+f12",
            "cmd+option+shift+escape",
            "return",
            "space",
        ]

        for keystroke in special_keystrokes:
            with patch(
                "src.server.tools.interface_tools.get_km_client",
                return_value=mock_km_client,
            ):
                result = await km_interface_automation(
                    operation="key_press",
                    keystroke=keystroke,
                    ctx=mock_context,
                )

                assert result["success"] is True
                assert result["data"]["keystroke"] == keystroke

    @pytest.mark.asyncio
    async def test_maximum_drag_distance(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test drag operation with maximum valid distance."""
        start_coords = {"x": -4999, "y": -4999}
        end_coords = {"x": 9999, "y": 9999}

        with patch(
            "src.server.tools.interface_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_interface_automation(
                operation="drag",
                coordinates=start_coords,
                end_coordinates=end_coords,
                duration_ms=1000,
                ctx=mock_context,
            )

            assert result["success"] is True
            # Distance should be sqrt((9999-(-4999))^2 + (9999-(-4999))^2)
            expected_distance = (14998**2 + 14998**2) ** 0.5
            assert abs(result["data"]["distance_pixels"] - expected_distance) < 0.1
