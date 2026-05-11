"""Focused tests for src/applications/app_controller.py.

This module provides targeted tests for the app_controller module to achieve high coverage
toward the mandatory 95% threshold.
"""

from unittest.mock import AsyncMock, patch

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
from src.integration.km_client import Either


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

    def test_app_identifier_creation(self) -> None:
        """Test AppIdentifier creation with valid data."""
        app_id = AppIdentifier(
            app_name="TextEdit",
            bundle_id="com.apple.TextEdit",
        )

        assert app_id.app_name == "TextEdit"
        assert app_id.bundle_id == "com.apple.TextEdit"

    def test_app_identifier_validation_valid_name(self) -> None:
        """Test AppIdentifier with valid app name."""
        app_id = AppIdentifier(
            app_name="Valid App Name",
            bundle_id="com.example.validapp",
        )

        # Should not raise validation errors
        assert app_id.app_name == "Valid App Name"

    def test_app_identifier_validation_valid_bundle_id(self) -> None:
        """Test AppIdentifier with valid bundle ID."""
        app_id = AppIdentifier(
            app_name="App",
            bundle_id="com.company.application",
        )

        # Should not raise validation errors
        assert app_id.bundle_id == "com.company.application"

    def test_app_identifier_equality(self) -> None:
        """Test AppIdentifier equality comparison."""
        app_id1 = AppIdentifier(
            app_name="TextEdit",
            bundle_id="com.apple.TextEdit",
        )
        app_id2 = AppIdentifier(
            app_name="TextEdit",
            bundle_id="com.apple.TextEdit",
        )

        assert app_id1 == app_id2


class TestMenuPath:
    """Test MenuPath dataclass."""

    def test_menu_path_creation(self) -> None:
        """Test MenuPath creation with valid data."""
        menu_path = MenuPath(
            path=["File", "Open"],
        )

        assert menu_path.path == ["File", "Open"]

    def test_menu_path_string_representation(self) -> None:
        """Test MenuPath string representation."""
        menu_path = MenuPath(path=["Edit", "Copy"])

        assert menu_path.path == ["Edit", "Copy"]
        assert str(menu_path) == "Edit → Copy"

    def test_menu_path_depth(self) -> None:
        """Test MenuPath depth method."""
        menu_path = MenuPath(path=["View", "Zoom", "Actual Size"])

        assert menu_path.depth() == 3

    def test_menu_path_empty_validation(self) -> None:
        """Test MenuPath validation for empty path."""
        with pytest.raises(ValueError, match="Menu path cannot be empty"):
            MenuPath(path=[])


class TestLaunchConfiguration:
    """Test LaunchConfiguration dataclass."""

    def test_launch_configuration_creation(self) -> None:
        """Test LaunchConfiguration creation with valid data."""
        config = LaunchConfiguration(
            wait_for_launch=True,
            timeout=Duration.from_seconds(30),
            hide_on_launch=False,
            activate_on_launch=True,
            launch_arguments=["--verbose", "--debug"],
        )

        assert config.wait_for_launch is True
        assert config.timeout == Duration.from_seconds(30)
        assert config.hide_on_launch is False
        assert config.activate_on_launch is True
        assert config.launch_arguments == ["--verbose", "--debug"]

    def test_launch_configuration_defaults(self) -> None:
        """Test LaunchConfiguration with default values."""
        config = LaunchConfiguration()

        assert config.wait_for_launch is True
        assert config.timeout == Duration.from_seconds(30)
        assert config.hide_on_launch is False
        assert config.activate_on_launch is True
        assert config.launch_arguments == []

    def test_launch_configuration_custom_timeout(self) -> None:
        """Test LaunchConfiguration with custom timeout."""
        custom_timeout = Duration.from_seconds(60)
        config = LaunchConfiguration(timeout=custom_timeout)

        assert config.timeout == custom_timeout

    def test_launch_configuration_with_arguments(self) -> None:
        """Test LaunchConfiguration with command line arguments."""
        args = ["--file", "document.txt", "--read-only"]
        config = LaunchConfiguration(launch_arguments=args)

        assert config.launch_arguments == args

    def test_launch_configuration_timeout_validation(self) -> None:
        """Test LaunchConfiguration timeout validation."""
        # Valid timeout
        config = LaunchConfiguration(timeout=Duration.from_seconds(60))
        assert config.timeout == Duration.from_seconds(60)

        # Test that validation occurs during construction
        with pytest.raises(ValueError):
            LaunchConfiguration(timeout=Duration.from_seconds(0))  # Invalid timeout


class TestAppOperationResult:
    """Test AppOperationResult dataclass."""

    def test_app_operation_result_success(self) -> None:
        """Test AppOperationResult for successful operation."""
        result = AppOperationResult(
            success=True,
            app_state=AppState.RUNNING,
            operation_time=Duration.from_seconds(2.5),
            details="Application launched successfully",
        )

        assert result.success is True
        assert result.app_state == AppState.RUNNING
        assert result.details == "Application launched successfully"
        assert result.operation_time == Duration.from_seconds(2.5)
        assert result.error_code is None

    def test_app_operation_result_failure(self) -> None:
        """Test AppOperationResult for failed operation."""
        result = AppOperationResult(
            success=False,
            app_state=AppState.NOT_RUNNING,
            operation_time=Duration.from_seconds(1.0),
            details="Failed to launch application",
            error_code="LAUNCH_FAILED",
        )

        assert result.success is False
        assert result.app_state == AppState.NOT_RUNNING
        assert result.details == "Failed to launch application"
        assert result.operation_time == Duration.from_seconds(1.0)
        assert result.error_code == "LAUNCH_FAILED"

    def test_app_operation_result_success_classmethod(self) -> None:
        """Test AppOperationResult.success_result class method."""
        operation_time = Duration.from_seconds(1.5)
        result = AppOperationResult.success_result(
            app_state=AppState.FOREGROUND,
            operation_time=operation_time,
            details="Operation completed successfully",
        )

        assert result.success is True
        assert result.app_state == AppState.FOREGROUND
        assert result.operation_time == operation_time
        assert result.details == "Operation completed successfully"
        assert result.error_code is None

    def test_app_operation_result_failure_classmethod(self) -> None:
        """Test AppOperationResult.failure_result class method."""
        operation_time = Duration.from_seconds(0.8)
        result = AppOperationResult.failure_result(
            app_state=AppState.NOT_RUNNING,
            operation_time=operation_time,
            error_code="TIMEOUT_ERROR",
            details="Operation timed out",
        )

        assert result.success is False
        assert result.app_state == AppState.NOT_RUNNING
        assert result.operation_time == operation_time
        assert result.error_code == "TIMEOUT_ERROR"
        assert result.details == "Operation timed out"


class TestAppController:
    """Test AppController class methods."""

    @pytest.fixture
    def app_controller(self) -> AppController:
        """Create AppController instance for testing."""
        return AppController()

    @pytest.fixture
    def sample_app_id(self) -> AppIdentifier:
        """Create sample AppIdentifier for testing."""
        return AppIdentifier(
            app_name="TestApp",
            bundle_id="com.example.testapp",
        )

    @pytest.fixture
    def sample_launch_config(self) -> LaunchConfiguration:
        """Create sample LaunchConfiguration for testing."""
        return LaunchConfiguration(
            wait_for_launch=True,
            timeout=Duration.from_seconds(30),
            activate_on_launch=True,
        )

    def test_app_controller_initialization(
        self, app_controller: AppController
    ) -> None:
        """Test AppController initialization."""
        assert isinstance(app_controller, AppController)

    @pytest.mark.asyncio
    async def test_launch_application_async_success(
        self,
        app_controller: AppController,
        sample_app_id: AppIdentifier,
        sample_launch_config: LaunchConfiguration,
    ) -> None:
        """Test successful application launch."""
        with patch.object(
            app_controller, "_launch_via_applescript", new_callable=AsyncMock
        ) as mock_launch:
            mock_launch.return_value = Either.right(True)

            with patch.object(
                app_controller, "_get_app_state_async", new_callable=AsyncMock
            ) as mock_state:
                mock_state.side_effect = [AppState.NOT_RUNNING, AppState.RUNNING]

                with patch.object(
                    app_controller, "_wait_for_launch", new_callable=AsyncMock
                ) as mock_wait:
                    mock_wait.return_value = Either.right(AppState.RUNNING)

                    result = await app_controller.launch_application_async(
                        sample_app_id, sample_launch_config
                    )

                    assert result.is_right()
                    operation_result = result.get_right()
                    assert operation_result.success is True
                    assert operation_result.app_state == AppState.RUNNING

    @pytest.mark.asyncio
    async def test_launch_application_validation_error(
        self, app_controller: AppController
    ) -> None:
        """Test application launch with validation error."""
        # Test with invalid app identifier (no name or bundle_id)
        with pytest.raises(
            ValueError, match="Either bundle_id or app_name must be provided"
        ):
            AppIdentifier()

    def test_synchronous_launch_application(
        self, app_controller: AppController
    ) -> None:
        """Test synchronous launch_application method."""
        # Test that synchronous method exists and handles basic input
        result = app_controller.launch_application("TextEdit")
        assert isinstance(result, bool)

    def test_synchronous_quit_application(
        self, app_controller: AppController
    ) -> None:
        """Test synchronous quit_application method."""
        # Test that synchronous method exists and handles basic input
        result = app_controller.quit_application("TextEdit")
        assert isinstance(result, bool)

    def test_get_running_applications(self, app_controller: AppController) -> None:
        """Test get_running_applications method."""
        # Test that method exists and returns a list
        result = app_controller.get_running_applications()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_application_state(
        self, app_controller: AppController, sample_app_id: AppIdentifier
    ) -> None:
        """Test get_application_state method."""
        result = await app_controller.get_application_state(sample_app_id)

        # Should return Either type
        assert hasattr(result, "is_right")
        assert hasattr(result, "is_left")

    def test_app_identifier_primary_identifier(
        self, sample_app_id: AppIdentifier
    ) -> None:
        """Test AppIdentifier primary_identifier method."""
        # Should prefer bundle_id over app_name
        assert sample_app_id.primary_identifier() == "com.example.testapp"

        # Test with only app_name
        app_id_name_only = AppIdentifier(app_name="TestApp")
        assert app_id_name_only.primary_identifier() == "TestApp"

    def test_app_identifier_display_name(
        self, sample_app_id: AppIdentifier
    ) -> None:
        """Test AppIdentifier display_name method."""
        # Should prefer app_name over bundle_id
        assert sample_app_id.display_name() == "TestApp"

        # Test with only bundle_id
        app_id_bundle_only = AppIdentifier(bundle_id="com.test.app")
        assert app_id_bundle_only.display_name() == "com.test.app"

    def test_app_identifier_is_bundle_id(
        self, sample_app_id: AppIdentifier
    ) -> None:
        """Test AppIdentifier is_bundle_id method."""
        assert sample_app_id.is_bundle_id() is True

        app_id_name_only = AppIdentifier(app_name="TestApp")
        assert app_id_name_only.is_bundle_id() is False
