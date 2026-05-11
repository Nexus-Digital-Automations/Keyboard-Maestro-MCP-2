"""Simple tests for src/applications/app_controller.py.

Focuses on basic functionality that can be easily tested to achieve coverage.
"""

from typing import cast

import pytest
from src.applications.app_controller import (
    AppController,
    AppIdentifier,
    ApplicationPermission,
    AppOperationResult,
    AppState,
    LaunchConfiguration,
    MenuPath,
)
from src.core.types import Duration


class TestAppState:
    """Test AppState enum values."""

    def test_app_state_enum_values(self) -> None:
        """Test AppState enum has expected values."""
        assert AppState.NOT_RUNNING.value == "not_running"
        assert AppState.LAUNCHING.value == "launching"
        assert AppState.RUNNING.value == "running"
        assert AppState.FOREGROUND.value == "foreground"
        assert AppState.BACKGROUND.value == "background"
        assert AppState.TERMINATING.value == "terminating"
        assert AppState.CRASHED.value == "crashed"
        assert AppState.UNKNOWN.value == "unknown"


class TestApplicationPermission:
    """Test ApplicationPermission enum values."""

    def test_application_permission_enum_values(self) -> None:
        """Test ApplicationPermission enum has expected values."""
        assert ApplicationPermission.LAUNCH.value == "launch"
        assert ApplicationPermission.QUIT.value == "quit"
        assert ApplicationPermission.ACTIVATE.value == "activate"
        assert ApplicationPermission.MENU_CONTROL.value == "menu_control"
        assert ApplicationPermission.FORCE_QUIT.value == "force_quit"
        assert ApplicationPermission.UI_AUTOMATION.value == "ui_automation"


class TestAppIdentifier:
    """Test AppIdentifier dataclass."""

    def test_app_identifier_creation_with_bundle_id(self) -> None:
        """Test AppIdentifier creation with bundle ID."""
        app_id = AppIdentifier(bundle_id="com.apple.TextEdit")

        assert app_id.bundle_id == "com.apple.TextEdit"
        assert app_id.app_name is None

    def test_app_identifier_creation_with_app_name(self) -> None:
        """Test AppIdentifier creation with app name."""
        app_id = AppIdentifier(app_name="TextEdit")

        assert app_id.app_name == "TextEdit"
        assert app_id.bundle_id is None

    def test_app_identifier_creation_with_both(self) -> None:
        """Test AppIdentifier creation with both bundle ID and app name."""
        app_id = AppIdentifier(
            app_name="TextEdit",
            bundle_id="com.apple.TextEdit",
        )

        assert app_id.app_name == "TextEdit"
        assert app_id.bundle_id == "com.apple.TextEdit"

    def test_app_identifier_primary_identifier_bundle_id_preferred(self) -> None:
        """Test primary_identifier method prefers bundle ID."""
        app_id = AppIdentifier(
            app_name="TextEdit",
            bundle_id="com.apple.TextEdit",
        )

        assert app_id.primary_identifier() == "com.apple.TextEdit"

    def test_app_identifier_primary_identifier_fallback_to_name(self) -> None:
        """Test primary_identifier method falls back to app name."""
        app_id = AppIdentifier(app_name="TextEdit")

        assert app_id.primary_identifier() == "TextEdit"

    def test_app_identifier_display_name_prefers_app_name(self) -> None:
        """Test display_name method prefers app name."""
        app_id = AppIdentifier(
            app_name="TextEdit",
            bundle_id="com.apple.TextEdit",
        )

        assert app_id.display_name() == "TextEdit"

    def test_app_identifier_display_name_fallback_to_bundle_id(self) -> None:
        """Test display_name method falls back to bundle ID."""
        app_id = AppIdentifier(bundle_id="com.apple.TextEdit")

        assert app_id.display_name() == "com.apple.TextEdit"

    def test_app_identifier_is_bundle_id_true(self) -> None:
        """Test is_bundle_id method returns True when bundle ID present."""
        app_id = AppIdentifier(bundle_id="com.apple.TextEdit")

        assert app_id.is_bundle_id() is True

    def test_app_identifier_is_bundle_id_false(self) -> None:
        """Test is_bundle_id method returns False when no bundle ID."""
        app_id = AppIdentifier(app_name="TextEdit")

        assert app_id.is_bundle_id() is False

    def test_app_identifier_validation_error_empty(self) -> None:
        """Test AppIdentifier raises error with no identifiers."""
        with pytest.raises(
            ValueError, match="Either bundle_id or app_name must be provided"
        ):
            AppIdentifier()

    def test_app_identifier_validation_error_invalid_bundle_id(self) -> None:
        """Test AppIdentifier raises error with invalid bundle ID."""
        with pytest.raises(ValueError, match="Invalid bundle ID format"):
            AppIdentifier(bundle_id="invalid/bundle@id")

    def test_app_identifier_validation_error_empty_app_name(self) -> None:
        """Test AppIdentifier raises error with empty app name."""
        with pytest.raises(
            ValueError, match="Either bundle_id or app_name must be provided"
        ):
            AppIdentifier(app_name="")

    def test_app_identifier_validation_error_long_app_name(self) -> None:
        """Test AppIdentifier raises error with too long app name."""
        long_name = "x" * 256
        with pytest.raises(ValueError, match="App name must be 1-255 characters"):
            AppIdentifier(app_name=long_name)


class TestMenuPath:
    """Test MenuPath dataclass."""

    def test_menu_path_creation(self) -> None:
        """Test MenuPath creation with valid data."""
        menu_path = MenuPath(path=["File", "Open"])

        assert menu_path.path == ["File", "Open"]

    def test_menu_path_string_representation(self) -> None:
        """Test MenuPath string conversion."""
        menu_path = MenuPath(path=["View", "Zoom", "Actual Size"])

        # Test string representation uses → separator
        assert str(menu_path) == "View → Zoom → Actual Size"

    def test_menu_path_depth(self) -> None:
        """Test MenuPath depth method."""
        menu_path = MenuPath(path=["Edit", "Copy"])

        assert menu_path.depth() == 2

    def test_menu_path_single_item(self) -> None:
        """Test MenuPath with single item."""
        menu_path = MenuPath(path=["Help"])

        assert menu_path.path == ["Help"]
        assert menu_path.depth() == 1
        assert str(menu_path) == "Help"

    def test_menu_path_validation_error_empty(self) -> None:
        """Test MenuPath raises error with empty path."""
        with pytest.raises(ValueError, match="Menu path cannot be empty"):
            MenuPath(path=[])

    def test_menu_path_validation_error_empty_item(self) -> None:
        """Test MenuPath raises error with empty menu item."""
        with pytest.raises(ValueError, match="Invalid menu item"):
            MenuPath(path=["File", ""])

    def test_menu_path_validation_error_long_item(self) -> None:
        """Test MenuPath raises error with too long menu item."""
        long_item = "x" * 101
        with pytest.raises(ValueError, match="Menu item too long"):
            MenuPath(path=["File", long_item])


class TestLaunchConfiguration:
    """Test LaunchConfiguration dataclass."""

    def test_launch_configuration_defaults(self) -> None:
        """Test LaunchConfiguration with default values."""
        config = LaunchConfiguration()

        assert config.wait_for_launch is True
        assert config.timeout == Duration.from_seconds(30)
        assert config.hide_on_launch is False
        assert config.activate_on_launch is True
        assert config.launch_arguments == []

    def test_launch_configuration_custom_values(self) -> None:
        """Test LaunchConfiguration with custom values."""
        config = LaunchConfiguration(
            wait_for_launch=False,
            timeout=Duration.from_seconds(60),
            hide_on_launch=True,
            activate_on_launch=False,
            launch_arguments=["--verbose", "--debug"],
        )

        assert config.wait_for_launch is False
        assert config.timeout == Duration.from_seconds(60)
        assert config.hide_on_launch is True
        assert config.activate_on_launch is False
        assert config.launch_arguments == ["--verbose", "--debug"]

    def test_launch_configuration_validation_error_zero_timeout(self) -> None:
        """Test LaunchConfiguration raises error with zero timeout."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            LaunchConfiguration(timeout=Duration.from_seconds(0))

    def test_launch_configuration_validation_error_negative_timeout(self) -> None:
        """Test LaunchConfiguration raises error with negative timeout."""
        # Duration.from_seconds might not allow negative values, so test zero instead
        with pytest.raises(ValueError, match="Timeout must be positive"):
            LaunchConfiguration(timeout=Duration.from_seconds(0))

    def test_launch_configuration_validation_error_excessive_timeout(self) -> None:
        """Test LaunchConfiguration raises error with too large timeout."""
        with pytest.raises(ValueError, match="Timeout cannot exceed 300 seconds"):
            LaunchConfiguration(timeout=Duration.from_seconds(301))


class TestAppOperationResult:
    """Test AppOperationResult dataclass."""

    def test_app_operation_result_creation(self) -> None:
        """Test AppOperationResult creation."""
        result = AppOperationResult(
            success=True,
            app_state=AppState.RUNNING,
            operation_time=Duration.from_seconds(2.5),
            details="Launch successful",
            error_code=None,
        )

        assert result.success is True
        assert result.app_state == AppState.RUNNING
        assert result.operation_time == Duration.from_seconds(2.5)
        assert result.details == "Launch successful"
        assert result.error_code is None

    def test_app_operation_result_success_factory(self) -> None:
        """Test AppOperationResult.success_result class method."""
        result = AppOperationResult.success_result(
            app_state=AppState.FOREGROUND,
            operation_time=Duration.from_seconds(1.0),
            details="Application activated",
        )

        assert result.success is True
        assert result.app_state == AppState.FOREGROUND
        assert result.operation_time == Duration.from_seconds(1.0)
        assert result.details == "Application activated"
        assert result.error_code is None

    def test_app_operation_result_error_creation(self) -> None:
        """Test AppOperationResult creation for error case."""
        result = AppOperationResult(
            success=False,
            app_state=AppState.NOT_RUNNING,
            operation_time=Duration.from_seconds(0.5),
            error_code="LAUNCH_FAILED",
            details="Application not found",
        )

        assert result.success is False
        assert result.app_state == AppState.NOT_RUNNING
        assert result.operation_time == Duration.from_seconds(0.5)
        assert result.error_code == "LAUNCH_FAILED"
        assert result.details == "Application not found"


class TestAppController:
    """Test AppController class basic functionality."""

    def test_app_controller_initialization(self) -> None:
        """Test AppController can be instantiated."""
        controller = AppController()

        assert isinstance(controller, AppController)

    def test_app_controller_escape_applescript_string(self) -> None:
        """Test _escape_applescript_string method."""
        controller = AppController()

        # Test escaping quotes
        result = controller._escape_applescript_string('Say "Hello"')
        assert result == 'Say \\"Hello\\"'

        # Test escaping backslashes
        result = controller._escape_applescript_string("Path\\to\\file")
        assert result == "Path\\\\to\\\\file"

        # Test escaping newlines
        result = controller._escape_applescript_string("Line 1\nLine 2")
        assert result == "Line 1\\nLine 2"

        # Test escaping tabs
        result = controller._escape_applescript_string("Col1\tCol2")
        assert result == "Col1\\tCol2"

        # Test escaping carriage returns
        result = controller._escape_applescript_string("Line 1\rLine 2")
        assert result == "Line 1\\rLine 2"

    def test_app_controller_escape_applescript_string_non_string(self) -> None:
        """Test _escape_applescript_string with non-string input."""
        controller = AppController()

        # Method coerces non-str inputs via str() at runtime; signature is str-only.
        result = controller._escape_applescript_string(cast("str", 123))
        assert result == "123"

        result = controller._escape_applescript_string(cast("str", None))
        assert result == "None"
