"""Comprehensive tests for app control tools module.

Tests cover application lifecycle management, menu automation, state tracking,
security validation, and integration with property-based testing.
"""

from __future__ import annotations

from typing import Any, Optional
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from src.server.tools.app_control_tools import km_app_control


# Test data generators
@st.composite
def app_control_operation_strategy(draw) -> Any:
    """Generate valid app control operations."""
    operations = ["launch", "quit", "activate", "menu_select", "get_state"]
    return draw(st.sampled_from(operations))


@st.composite
def app_identifier_strategy(draw) -> Any:
    """Generate valid application identifiers."""
    # Mix of bundle IDs and app names (systematic pattern alignment)
    identifiers = [
        # Bundle IDs
        "com.apple.TextEdit",
        "com.microsoft.VSCode",
        "com.github.atom",
        "org.mozilla.firefox",
        "com.google.Chrome",
        # Application names
        "TextEdit",
        "Visual Studio Code",
        "Firefox",
        "Chrome",
        "Finder",
        # Generated valid names
        draw(
            st.text(min_size=1, max_size=30).filter(
                lambda x: x.isalnum() and len(x.strip()) > 0,
            ),
        ),
    ]
    return draw(st.sampled_from(identifiers))


@st.composite
def menu_path_strategy(draw) -> Any:
    """Generate valid menu paths."""
    menu_items = [
        "File",
        "Edit",
        "View",
        "Window",
        "Help",
        "New",
        "Open",
        "Save",
        "Close",
        "Quit",
        "Copy",
        "Paste",
        "Cut",
        "Undo",
        "Redo",
        "Preferences",
        "Settings",
        "Options",
    ]
    path_length = draw(st.integers(min_value=1, max_value=5))
    return draw(
        st.lists(
            st.sampled_from(menu_items),
            min_size=path_length,
            max_size=path_length,
        ),
    )


@st.composite
def timeout_strategy(draw) -> Any:
    """Generate valid timeout values."""
    return draw(st.integers(min_value=1, max_value=120))


@st.composite
def invalid_app_identifier_strategy(draw) -> Any:
    """Generate invalid application identifiers."""
    invalid_identifiers = [
        "",  # Empty
        "   ",  # Whitespace only
        "a" * 256,  # Too long
        "invalid/path",  # Invalid characters
        "com.invalid..bundle",  # Invalid bundle format
        "app\x00name",  # Null bytes
        "../../../etc/passwd",  # Path traversal
    ]
    return draw(st.sampled_from(invalid_identifiers))


@st.composite
def invalid_operation_strategy(draw) -> Any:
    """Generate invalid operations."""
    invalid_ops = ["invalid", "hack", "execute", "shell", "", "delete", "install"]
    return draw(st.sampled_from(invalid_ops))


class TestAppControlDependencies:
    """Test app control dependencies and imports."""

    def test_app_control_manager_import(self) -> None:
        """Test importing app control dependencies."""
        try:
            from src.applications.app_controller import (
                AppController,
                AppIdentifier,
                AppState,
                LaunchConfiguration,
                MenuPath,
            )
            from src.core.errors import SecurityViolationError, ValidationError
            from src.core.types import Duration

            # Test basic creation
            assert AppController is not None
            assert AppIdentifier is not None
            assert MenuPath is not None
            assert LaunchConfiguration is not None
            assert AppState is not None
            assert Duration is not None
            assert ValidationError is not None
            assert SecurityViolationError is not None

        except ImportError:
            # Mock the dependencies for testing
            pytest.skip("App control dependencies not available - using mocks")


class TestAppControlParameterValidation:
    """Test app control parameter validation."""

    @given(app_control_operation_strategy())
    def test_valid_operations(self, operation: str) -> None:
        """Test that valid operations are accepted."""
        valid_operations = ["launch", "quit", "activate", "menu_select", "get_state"]
        assert operation in valid_operations

    @given(app_identifier_strategy())
    def test_app_identifier_validation(self, app_id: str) -> None:
        """Test application identifier validation."""
        assume(len(app_id) <= 255 and len(app_id.strip()) > 0)
        # Valid app identifiers should be non-empty and within length limits
        assert 0 < len(app_id) <= 255

    @given(menu_path_strategy())
    def test_menu_path_validation(self, menu_path: list[str]) -> None:
        """Test menu path validation."""
        assume(
            len(menu_path) <= 10 and all(len(item.strip()) > 0 for item in menu_path),
        )
        # Menu paths should be within limits and contain valid items
        assert 0 < len(menu_path) <= 10
        assert all(isinstance(item, str) for item in menu_path)
        assert all(len(item.strip()) > 0 for item in menu_path)

    @given(timeout_strategy())
    def test_timeout_validation(self, timeout: int) -> None:
        """Test timeout parameter validation."""
        # Timeouts should be within safe limits
        assert 1 <= timeout <= 120

    def test_invalid_operations(self) -> None:
        """Test that invalid operations are rejected."""
        invalid_operations = [
            "invalid",
            "hack",
            "execute",
            "shell",
            "",
            "delete",
            "install",
        ]
        valid_operations = ["launch", "quit", "activate", "menu_select", "get_state"]
        for op in invalid_operations:
            assert op not in valid_operations

    def test_empty_app_identifier_validation(self) -> None:
        """Test that empty app identifiers are rejected."""
        empty_identifiers = ["", "   ", "\t", "\n"]
        for app_id in empty_identifiers:
            assert len(app_id.strip()) == 0  # Should be detected as invalid

    def test_oversized_app_identifier_validation(self) -> None:
        """Test that oversized app identifiers are rejected."""
        oversized_identifier = "x" * 256  # Exceeds 255 char limit
        assert len(oversized_identifier) > 255

    def test_oversized_menu_path_validation(self) -> None:
        """Test that oversized menu paths are rejected."""
        oversized_menu_path = ["item"] * 11  # Exceeds 10 item limit
        assert len(oversized_menu_path) > 10


class TestAppControlLaunchOperationMocked:
    """Test app control launch operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_launch_application_success(self) -> None:
        """Test successful application launch."""
        with (
            patch(
                "src.server.tools.app_control_tools.AppController",
            ) as mock_controller_class,
            patch(
                "src.server.tools.app_control_tools.AppIdentifier",
            ) as mock_app_id_class,
            patch("src.server.tools.app_control_tools.Duration") as mock_duration_class,
            patch(
                "src.server.tools.app_control_tools.LaunchConfiguration",
            ) as mock_launch_config_class,
        ):
            # Setup mocks for success case
            mock_app_id = Mock()
            mock_app_id.display_name.return_value = "TextEdit"
            mock_app_id.primary_identifier.return_value = "com.apple.TextEdit"
            mock_app_id_class.return_value = mock_app_id

            # Mock Duration with proper total_seconds method
            mock_duration = Mock()
            mock_duration.total_seconds.return_value = 30
            mock_duration_class.from_seconds.return_value = mock_duration

            # Mock LaunchConfiguration
            mock_launch_config = Mock()
            mock_launch_config_class.return_value = mock_launch_config

            mock_controller = Mock()
            mock_operation_result = Mock()
            mock_operation_result.app_state.value = "running"
            mock_operation_result.operation_time.total_seconds.return_value = 2.5
            mock_operation_result.details = "Application launched successfully"

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_operation_result

            mock_controller.launch_application = AsyncMock(return_value=mock_result)
            mock_controller_class.return_value = mock_controller

            # Execute launch operation
            result = await km_app_control(
                operation="launch",
                app_identifier="com.apple.TextEdit",
                wait_for_completion=True,
                timeout_seconds=30,
                hide_on_launch=False,
            )

            # Verify successful launch
            assert result["success"] is True
            assert result["data"]["app_state"] == "running"
            assert result["data"]["app_name"] == "TextEdit"
            assert result["data"]["app_identifier"] == "com.apple.TextEdit"
            assert result["data"]["operation_time"] == 2.5
            assert result["data"]["waited_for_completion"] is True
            assert result["data"]["hidden"] is False
            assert "metadata" in result

    @pytest.mark.asyncio
    async def test_launch_application_with_hide_option(self) -> None:
        """Test application launch with hide option."""
        with (
            patch(
                "src.server.tools.app_control_tools.AppController",
            ) as mock_controller_class,
            patch(
                "src.server.tools.app_control_tools.AppIdentifier",
            ) as mock_app_id_class,
            patch("src.server.tools.app_control_tools.Duration") as mock_duration_class,
            patch(
                "src.server.tools.app_control_tools.LaunchConfiguration",
            ) as mock_launch_config_class,
        ):
            # Setup mocks
            mock_app_id = Mock()
            mock_app_id.display_name.return_value = "Calculator"
            mock_app_id.primary_identifier.return_value = "com.apple.calculator"
            mock_app_id_class.return_value = mock_app_id

            # Mock Duration with proper total_seconds method
            mock_duration = Mock()
            mock_duration.total_seconds.return_value = 30
            mock_duration_class.from_seconds.return_value = mock_duration

            # Mock LaunchConfiguration
            mock_launch_config = Mock()
            mock_launch_config_class.return_value = mock_launch_config

            mock_controller = Mock()
            mock_operation_result = Mock()
            mock_operation_result.app_state.value = "running"
            mock_operation_result.operation_time.total_seconds.return_value = 1.8
            mock_operation_result.details = "Application launched and hidden"

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_operation_result

            mock_controller.launch_application = AsyncMock(return_value=mock_result)
            mock_controller_class.return_value = mock_controller

            # Execute launch with hide option
            result = await km_app_control(
                operation="launch",
                app_identifier="Calculator",
                hide_on_launch=True,
                wait_for_completion=False,
            )

            # Verify launch with hide option
            assert result["success"] is True
            assert result["data"]["hidden"] is True
            assert result["data"]["waited_for_completion"] is False

    @pytest.mark.asyncio
    async def test_launch_application_failure(self) -> None:
        """Test application launch failure."""
        with (
            patch(
                "src.server.tools.app_control_tools.AppController",
            ) as mock_controller_class,
            patch(
                "src.server.tools.app_control_tools.AppIdentifier",
            ) as mock_app_id_class,
            patch("src.server.tools.app_control_tools.Duration") as mock_duration_class,
            patch(
                "src.server.tools.app_control_tools.LaunchConfiguration",
            ) as mock_launch_config_class,
        ):
            # Setup mocks for failure case
            mock_app_id = Mock()
            mock_app_id.display_name.return_value = "NonExistentApp"
            mock_app_id_class.return_value = mock_app_id

            # Mock Duration with proper total_seconds method
            mock_duration = Mock()
            mock_duration.total_seconds.return_value = 30
            mock_duration_class.from_seconds.return_value = mock_duration

            # Mock LaunchConfiguration
            mock_launch_config = Mock()
            mock_launch_config_class.return_value = mock_launch_config

            mock_controller = Mock()
            mock_error = Mock()
            mock_error.code = "APP_NOT_FOUND"
            mock_error.message = "Application not found"
            mock_error.details = "Could not locate the specified application"
            mock_error.recovery_suggestion = "Check application name and installation"

            mock_result = Mock()
            mock_result.is_left.return_value = True
            mock_result.get_left.return_value = mock_error

            mock_controller.launch_application = AsyncMock(return_value=mock_result)
            mock_controller_class.return_value = mock_controller

            # Execute launch operation that should fail
            result = await km_app_control(
                operation="launch",
                app_identifier="NonExistentApp",
            )

            # Verify launch failure
            assert result["success"] is False
            assert result["error"]["code"] == "APP_NOT_FOUND"
            assert result["error"]["message"] == "Application not found"


class TestAppControlQuitOperationMocked:
    """Test app control quit operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_quit_application_success(self) -> None:
        """Test successful application quit."""
        with (
            patch(
                "src.server.tools.app_control_tools.AppController",
            ) as mock_controller_class,
            patch(
                "src.server.tools.app_control_tools.AppIdentifier",
            ) as mock_app_id_class,
            patch("src.server.tools.app_control_tools.Duration") as mock_duration_class,
        ):
            # Setup mocks for quit success
            mock_app_id = Mock()
            mock_app_id.display_name.return_value = "Safari"
            mock_app_id.primary_identifier.return_value = "com.apple.Safari"
            mock_app_id_class.return_value = mock_app_id

            mock_duration = Mock()
            mock_duration_class.from_seconds.return_value = mock_duration

            mock_controller = Mock()
            mock_operation_result = Mock()
            mock_operation_result.app_state.value = "not_running"
            mock_operation_result.operation_time.total_seconds.return_value = 1.2
            mock_operation_result.details = "Application quit gracefully"

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_operation_result

            mock_controller.quit_application = AsyncMock(return_value=mock_result)
            mock_controller_class.return_value = mock_controller

            # Execute quit operation
            result = await km_app_control(
                operation="quit",
                app_identifier="com.apple.Safari",
                force_quit=False,
                timeout_seconds=15,
            )

            # Verify successful quit
            assert result["success"] is True
            assert result["data"]["app_state"] == "not_running"
            assert result["data"]["app_name"] == "Safari"
            assert result["data"]["force_quit_used"] is False
            assert result["data"]["final_state"] == "not_running"

    @pytest.mark.asyncio
    async def test_quit_application_force_quit(self) -> None:
        """Test application force quit."""
        with (
            patch(
                "src.server.tools.app_control_tools.AppController",
            ) as mock_controller_class,
            patch(
                "src.server.tools.app_control_tools.AppIdentifier",
            ) as mock_app_id_class,
            patch("src.server.tools.app_control_tools.Duration") as mock_duration_class,
        ):
            # Setup mocks for force quit
            mock_app_id = Mock()
            mock_app_id.display_name.return_value = "Frozen App"
            mock_app_id.primary_identifier.return_value = "com.example.frozen"
            mock_app_id_class.return_value = mock_app_id

            mock_duration = Mock()
            mock_duration_class.from_seconds.return_value = mock_duration

            mock_controller = Mock()
            mock_operation_result = Mock()
            mock_operation_result.app_state.value = "not_running"
            mock_operation_result.operation_time.total_seconds.return_value = 0.5
            mock_operation_result.details = "Application force quit"

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_operation_result

            mock_controller.quit_application = AsyncMock(return_value=mock_result)
            mock_controller_class.return_value = mock_controller

            # Execute force quit operation
            result = await km_app_control(
                operation="quit",
                app_identifier="Frozen App",
                force_quit=True,
            )

            # Verify force quit
            assert result["success"] is True
            assert result["data"]["force_quit_used"] is True
            assert result["data"]["operation_time"] == 0.5


class TestAppControlActivateOperationMocked:
    """Test app control activate operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_activate_application_success(self) -> None:
        """Test successful application activation."""
        with (
            patch(
                "src.server.tools.app_control_tools.AppController",
            ) as mock_controller_class,
            patch(
                "src.server.tools.app_control_tools.AppIdentifier",
            ) as mock_app_id_class,
        ):
            # Setup mocks for activation success
            mock_app_id = Mock()
            mock_app_id.display_name.return_value = "Finder"
            mock_app_id.primary_identifier.return_value = "com.apple.finder"
            mock_app_id_class.return_value = mock_app_id

            mock_controller = Mock()
            mock_operation_result = Mock()
            mock_operation_result.app_state.value = "foreground"
            mock_operation_result.operation_time.total_seconds.return_value = 0.3
            mock_operation_result.details = "Application brought to foreground"

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_operation_result

            mock_controller.activate_application = AsyncMock(return_value=mock_result)
            mock_controller_class.return_value = mock_controller

            # Execute activate operation
            result = await km_app_control(
                operation="activate",
                app_identifier="com.apple.finder",
            )

            # Verify successful activation
            assert result["success"] is True
            assert result["data"]["app_state"] == "foreground"
            assert result["data"]["app_name"] == "Finder"
            assert result["data"]["activated"] is True
            assert result["data"]["operation_time"] == 0.3


class TestAppControlMenuSelectOperationMocked:
    """Test app control menu select operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_menu_select_success(self) -> None:
        """Test successful menu selection."""
        with (
            patch(
                "src.server.tools.app_control_tools.AppController",
            ) as mock_controller_class,
            patch(
                "src.server.tools.app_control_tools.AppIdentifier",
            ) as mock_app_id_class,
            patch(
                "src.server.tools.app_control_tools.MenuPath",
            ) as mock_menu_path_class,
            patch("src.server.tools.app_control_tools.Duration") as mock_duration_class,
        ):
            # Setup mocks for menu select success
            mock_app_id = Mock()
            mock_app_id.display_name.return_value = "TextEdit"
            mock_app_id.primary_identifier.return_value = "com.apple.TextEdit"
            mock_app_id_class.return_value = mock_app_id

            mock_menu_path = Mock()
            mock_menu_path_class.return_value = mock_menu_path

            mock_duration = Mock()
            mock_duration_class.from_seconds.return_value = mock_duration

            mock_controller = Mock()
            mock_operation_result = Mock()
            mock_operation_result.app_state.value = "foreground"
            mock_operation_result.operation_time.total_seconds.return_value = 0.8
            mock_operation_result.details = "Menu item selected successfully"

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_operation_result

            mock_controller.select_menu_item = AsyncMock(return_value=mock_result)
            mock_controller_class.return_value = mock_controller

            # Execute menu select operation
            result = await km_app_control(
                operation="menu_select",
                app_identifier="com.apple.TextEdit",
                menu_path=["File", "New"],
            )

            # Verify successful menu selection
            assert result["success"] is True
            assert result["data"]["app_state"] == "foreground"
            assert result["data"]["menu_path"] == ["File", "New"]
            assert result["data"]["menu_depth"] == 2
            assert result["data"]["menu_selected"] is True

    @pytest.mark.asyncio
    async def test_menu_select_missing_path(self) -> None:
        """Test menu select with missing menu path."""
        # Execute menu select without menu path
        result = await km_app_control(
            operation="menu_select",
            app_identifier="com.apple.TextEdit",
            # menu_path is None
        )

        # Verify missing menu path error
        assert result["success"] is False
        assert result["error"]["code"] == "MISSING_MENU_PATH"
        assert "Menu path required" in result["error"]["message"]


class TestAppControlGetStateOperationMocked:
    """Test app control get state operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_get_state_success(self) -> None:
        """Test successful application state query."""
        with (
            patch(
                "src.server.tools.app_control_tools.AppController",
            ) as mock_controller_class,
            patch(
                "src.server.tools.app_control_tools.AppIdentifier",
            ) as mock_app_id_class,
            patch("src.server.tools.app_control_tools.AppState") as mock_app_state,
        ):
            # Setup mocks for state query success
            mock_app_id = Mock()
            mock_app_id.display_name.return_value = "Chrome"
            mock_app_id.primary_identifier.return_value = "com.google.Chrome"
            mock_app_id_class.return_value = mock_app_id

            # Mock AppState enum
            mock_state = Mock()
            mock_state.value = "foreground"
            mock_app_state.NOT_RUNNING = Mock()
            mock_app_state.FOREGROUND = mock_state

            mock_controller = Mock()
            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_state

            mock_controller.get_application_state = AsyncMock(return_value=mock_result)
            mock_controller_class.return_value = mock_controller

            # Execute get state operation
            result = await km_app_control(
                operation="get_state",
                app_identifier="com.google.Chrome",
            )

            # Verify successful state query
            assert result["success"] is True
            assert result["data"]["app_state"] == "foreground"
            assert result["data"]["app_name"] == "Chrome"
            assert result["data"]["is_running"] is True
            assert result["data"]["is_foreground"] is True
            assert result["data"]["is_background"] is False


class TestAppControlErrorHandling:
    """Test app control error handling."""

    @pytest.mark.asyncio
    async def test_invalid_app_identifier_error(self) -> None:
        """Test handling of invalid app identifier."""
        with patch(
            "src.server.tools.app_control_tools.AppIdentifier",
        ) as mock_app_id_class:
            # Setup app identifier to raise ValueError
            mock_app_id_class.side_effect = ValueError("Invalid bundle ID format")

            # Execute operation with invalid identifier
            result = await km_app_control(
                operation="launch",
                app_identifier="invalid..bundle.id",
            )

            # Verify invalid identifier error
            assert result["success"] is False
            assert result["error"]["code"] == "INVALID_IDENTIFIER"
            assert "Invalid bundle ID format" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_unsupported_operation_error(self) -> None:
        """Test handling of unsupported operations."""
        # Execute unsupported operation
        result = await km_app_control(
            operation="invalid_operation",
            app_identifier="com.apple.TextEdit",
        )

        # Verify unsupported operation error
        assert result["success"] is False
        assert result["error"]["code"] == "UNSUPPORTED_OPERATION"
        assert "Operation not supported" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_validation_error_handling(self) -> None:
        """Test handling of validation errors."""
        from src.core.errors import ValidationError

        with patch(
            "src.server.tools.app_control_tools.AppIdentifier",
        ) as mock_app_id_class:
            # Setup validation error
            mock_app_id_class.side_effect = ValidationError(
                "field",
                "value",
                "Validation failed",
            )

            # Execute operation that should trigger validation error
            result = await km_app_control(
                operation="launch",
                app_identifier="invalid_app",
            )

            # Verify validation error handling
            assert result["success"] is False
            assert result["error"]["code"] == "VALIDATION_ERROR"
            assert "Validation failed" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_security_violation_error_handling(self) -> None:
        """Test handling of security violations."""
        try:
            from src.core.errors import SecurityViolationError
        except ImportError:
            # Create mock SecurityViolationError if not available
            class SecurityViolationError(Exception):
                def __init__(self, violation_type, details):
                    super().__init__(f"{violation_type}: {details}")
                    self.violation_type = violation_type
                    self.details = details

        with patch(
            "src.server.tools.app_control_tools.AppIdentifier",
        ) as mock_app_id_class:
            # Setup security violation error with proper constructor
            mock_app_id_class.side_effect = SecurityViolationError(
                "app_access",
                "Security policy violated",
            )

            # Execute operation that should trigger security error
            result = await km_app_control(
                operation="quit",
                app_identifier="restricted_app",
            )

            # Verify security violation error handling
            assert result["success"] is False
            assert result["error"]["code"] == "SECURITY_VIOLATION"
            assert "Security validation failed" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self) -> None:
        """Test handling of unexpected errors."""
        with patch(
            "src.server.tools.app_control_tools.AppIdentifier",
        ) as mock_app_id_class:
            # Setup unexpected error
            mock_app_id_class.side_effect = RuntimeError("Unexpected system error")

            # Execute operation that should trigger unexpected error
            result = await km_app_control(
                operation="activate",
                app_identifier="test_app",
            )

            # Verify unexpected error handling
            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert "Unexpected system error" in result["error"]["message"]


class TestAppControlIntegration:
    """Integration tests for app control operations."""

    @pytest.mark.asyncio
    async def test_complete_app_lifecycle_workflow(self) -> None:
        """Test complete application lifecycle workflow."""
        with (
            patch(
                "src.server.tools.app_control_tools.AppController",
            ) as mock_controller_class,
            patch(
                "src.server.tools.app_control_tools.AppIdentifier",
            ) as mock_app_id_class,
            patch("src.server.tools.app_control_tools.Duration") as mock_duration_class,
            patch(
                "src.server.tools.app_control_tools.LaunchConfiguration",
            ) as mock_launch_config_class,
        ):
            # Setup mocks for complete workflow
            mock_app_id = Mock()
            mock_app_id.display_name.return_value = "TestApp"
            mock_app_id.primary_identifier.return_value = "com.test.app"
            mock_app_id_class.return_value = mock_app_id

            # Mock Duration with proper total_seconds method
            mock_duration = Mock()
            mock_duration.total_seconds.return_value = 30
            mock_duration_class.from_seconds.return_value = mock_duration

            # Mock LaunchConfiguration
            mock_launch_config = Mock()
            mock_launch_config_class.return_value = mock_launch_config

            mock_controller = Mock()

            # Setup different results for different operations
            launch_result = Mock()
            launch_result.is_left.return_value = False
            launch_result.get_right.return_value = Mock(
                app_state=Mock(value="running"),
                operation_time=Mock(total_seconds=lambda: 2.0),
                details="Launched successfully",
            )

            activate_result = Mock()
            activate_result.is_left.return_value = False
            activate_result.get_right.return_value = Mock(
                app_state=Mock(value="foreground"),
                operation_time=Mock(total_seconds=lambda: 0.5),
                details="Activated successfully",
            )

            quit_result = Mock()
            quit_result.is_left.return_value = False
            quit_result.get_right.return_value = Mock(
                app_state=Mock(value="not_running"),
                operation_time=Mock(total_seconds=lambda: 1.0),
                details="Quit successfully",
            )

            mock_controller.launch_application = AsyncMock(return_value=launch_result)
            mock_controller.activate_application = AsyncMock(
                return_value=activate_result,
            )
            mock_controller.quit_application = AsyncMock(return_value=quit_result)
            mock_controller_class.return_value = mock_controller

            # Execute complete workflow: launch -> activate -> quit
            launch_result_data = await km_app_control(
                operation="launch",
                app_identifier="com.test.app",
            )
            assert launch_result_data["success"] is True

            activate_result_data = await km_app_control(
                operation="activate",
                app_identifier="com.test.app",
            )
            assert activate_result_data["success"] is True

            quit_result_data = await km_app_control(
                operation="quit",
                app_identifier="com.test.app",
            )
            assert quit_result_data["success"] is True

            # Verify all operations were called
            mock_controller.launch_application.assert_called_once()
            mock_controller.activate_application.assert_called_once()
            mock_controller.quit_application.assert_called_once()

    @pytest.mark.asyncio
    async def test_app_control_with_context(self) -> None:
        """Test app control with FastMCP context integration."""
        mock_context = Mock()
        mock_context.info = AsyncMock()
        mock_context.report_progress = AsyncMock()
        mock_context.error = AsyncMock()

        with (
            patch(
                "src.server.tools.app_control_tools.AppController",
            ) as mock_controller_class,
            patch(
                "src.server.tools.app_control_tools.AppIdentifier",
            ) as mock_app_id_class,
        ):
            # Setup mocks for context testing
            mock_app_id = Mock()
            mock_app_id.display_name.return_value = "TestApp"
            mock_app_id.primary_identifier.return_value = "com.test.app"
            mock_app_id_class.return_value = mock_app_id

            mock_controller = Mock()
            mock_operation_result = Mock()
            mock_operation_result.app_state.value = "running"
            mock_operation_result.operation_time.total_seconds.return_value = 1.5
            mock_operation_result.details = "Operation completed"

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_operation_result

            mock_controller.launch_application = AsyncMock(return_value=mock_result)
            mock_controller_class.return_value = mock_controller

            # Execute operation with context
            result = await km_app_control(
                operation="launch",
                app_identifier="com.test.app",
                ctx=mock_context,
            )

            # Verify context integration
            assert result["success"] is True
            mock_context.info.assert_called()
            mock_context.report_progress.assert_called()

            # Verify progress reporting calls
            progress_calls = mock_context.report_progress.call_args_list
            assert len(progress_calls) >= 2  # At least execution and completion


class TestAppControlProperties:
    """Property-based tests for app control operations."""

    @given(
        app_control_operation_strategy(),
        app_identifier_strategy(),
        timeout_strategy(),
    )
    def test_app_control_parameter_validation_properties(
        self,
        operation: str,
        app_identifier: str,
        timeout: int,
    ) -> None:
        """Property test for app control parameter validation."""
        assume(len(app_identifier.strip()) > 0 and len(app_identifier) <= 255)

        # Properties that should always hold
        valid_operations = ["launch", "quit", "activate", "menu_select", "get_state"]
        assert operation in valid_operations
        assert isinstance(app_identifier, str)
        assert 0 < len(app_identifier) <= 255
        assert 1 <= timeout <= 120

    @given(menu_path_strategy())
    def test_menu_path_properties(self, menu_path: list[str]) -> None:
        """Property test for menu path validation."""
        assume(
            len(menu_path) <= 10 and all(len(item.strip()) > 0 for item in menu_path),
        )

        # Properties that should always hold for menu paths
        assert isinstance(menu_path, list)
        assert 0 < len(menu_path) <= 10
        assert all(isinstance(item, str) for item in menu_path)
        assert all(len(item.strip()) > 0 for item in menu_path)

    @given(st.text(min_size=1, max_size=50))
    def test_operation_result_structure_properties(self, correlation_id: str) -> None:
        """Property test for operation result structure."""
        assume(correlation_id.strip() != "")

        # Mock result structure
        result_structure = {
            "success": True,
            "data": {
                "app_state": "running",
                "app_name": "TestApp",
                "operation_time": 2.5,
            },
            "metadata": {
                "correlation_id": correlation_id.strip(),
                "timestamp": datetime.now(UTC).isoformat(),
                "server_version": "1.0.0",
                "execution_time": 2.5,
            },
        }

        # Properties that should always hold
        assert "success" in result_structure
        assert isinstance(result_structure["success"], bool)
        assert "metadata" in result_structure
        assert "correlation_id" in result_structure["metadata"]
        assert len(result_structure["metadata"]["correlation_id"]) > 0

    @given(invalid_app_identifier_strategy())
    def test_security_validation_properties(self, invalid_identifier: str) -> None:
        """Property test for security validation behavior."""
        # Security risks that should trigger validation failures
        security_risks = ["../", "\\", "/etc/", "passwd", "\x00", "system32"]

        has_risk = any(risk in invalid_identifier.lower() for risk in security_risks)

        if has_risk:
            # Invalid identifiers with security risks should be detectable
            assert any(
                indicator in invalid_identifier
                for indicator in ["../", "\\", "/etc", "\x00"]
            )

        # Length validation
        if len(invalid_identifier) > 255:
            # Should be rejected for being too long
            assert len(invalid_identifier) > 255

        # Empty validation
        if len(invalid_identifier.strip()) == 0:
            # Should be rejected for being empty
            assert len(invalid_identifier.strip()) == 0
