"""Comprehensive tests for application control commands.

This module tests application launch, quit, and activation commands
with comprehensive security validation and edge case coverage.
"""

from unittest.mock import MagicMock, patch

import pytest
from src.commands.application import (
    ActivateApplicationCommand,
    LaunchApplicationCommand,
    QuitApplicationCommand,
    SecurityError,
    secure_subprocess_run,
)
from src.core.types import (
    CommandId,
    CommandParameters,
    Duration,
    ExecutionContext,
    Permission,
)


class TestSecureSubprocessRun:
    """Test secure subprocess execution utility."""

    @patch("shutil.which")
    def test_executable_not_found(self, mock_which):
        """Test error when executable not found."""
        mock_which.return_value = None

        with pytest.raises(SecurityError, match="Executable not found"):
            secure_subprocess_run("nonexistent", [])

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_trusted_path_execution(self, mock_run, mock_which):
        """Test execution from trusted path."""
        mock_which.return_value = "/usr/bin/ls"
        mock_process = MagicMock()
        mock_run.return_value = mock_process

        result = secure_subprocess_run("ls", ["-la"])

        assert result == mock_process
        mock_run.assert_called_once_with(["/usr/bin/ls", "-la"], check=False)

    @patch("shutil.which")
    @patch("subprocess.run")
    @patch("src.commands.application.logger")
    def test_untrusted_path_warning(self, mock_logger, mock_run, mock_which):
        """Test warning for non-standard executable path."""
        mock_which.return_value = "/home/user/bin/custom"
        mock_process = MagicMock()
        mock_run.return_value = mock_process

        result = secure_subprocess_run("custom", [])

        assert result == mock_process
        mock_logger.warning.assert_called_once()

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_argument_sanitization(self, mock_run, mock_which):
        """Test dangerous characters are removed from arguments."""
        mock_which.return_value = "/usr/bin/echo"
        mock_process = MagicMock()
        mock_run.return_value = mock_process

        secure_subprocess_run("echo", ["test;rm -rf /", "hello&&world", "a|b"])

        # Verify dangerous characters were removed
        mock_run.assert_called_once_with(
            ["/usr/bin/echo", "testrm -rf /", "helloworld", "ab"], check=False
        )


class TestLaunchApplicationCommand:
    """Test LaunchApplicationCommand functionality."""

    @pytest.fixture
    def context(self):
        """Create execution context."""
        return ExecutionContext.create_test_context(
            permissions=frozenset(
                [Permission.APPLICATION_CONTROL, Permission.SYSTEM_CONTROL]
            )
        )

    def test_launch_validation_success(self):
        """Test successful launch validation."""
        params = CommandParameters({"application_name": "Calculator"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is True

    def test_launch_validation_failures(self):
        """Test launch validation failures."""
        # No application name or path
        params = CommandParameters({})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Empty application name
        params = CommandParameters({"application_name": ""})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Dangerous application name
        params = CommandParameters({"application_name": "rm"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Invalid launch timeout - zero timeout
        params = CommandParameters(
            {"application_name": "Calculator", "launch_timeout": 0}
        )
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

    def test_get_application_name(self):
        """Test application name retrieval."""
        params = CommandParameters({"application_name": "TextEdit"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.get_application_name() == "TextEdit"

        params = CommandParameters({})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.get_application_name() == ""

    def test_get_application_path(self):
        """Test application path retrieval."""
        params = CommandParameters({"application_path": "/Applications/Calculator.app"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.get_application_path() == "/Applications/Calculator.app"

        params = CommandParameters({})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.get_application_path() is None

    def test_get_launch_arguments(self):
        """Test launch arguments retrieval."""
        params = CommandParameters(
            {
                "application_name": "app",
                "launch_arguments": ["--arg1", "value1", "--arg2"],
            }
        )
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.get_launch_arguments() == ["--arg1", "value1", "--arg2"]

        # Limit to 10 arguments
        params = CommandParameters(
            {
                "application_name": "app",
                "launch_arguments": [f"arg{i}" for i in range(20)],
            }
        )
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert len(cmd.get_launch_arguments()) == 10

    def test_get_launch_timeout(self):
        """Test launch timeout retrieval."""
        # Default timeout
        params = CommandParameters({"application_name": "app"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.get_launch_timeout() == Duration.from_seconds(30)

        # Custom timeout
        params = CommandParameters({"application_name": "app", "launch_timeout": 45})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.get_launch_timeout() == Duration.from_seconds(45)

        # Clamped timeout
        params = CommandParameters({"application_name": "app", "launch_timeout": 120})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.get_launch_timeout() == Duration.from_seconds(60)

    def test_is_safe_executable(self):
        """Test executable safety validation."""
        params = CommandParameters({"application_name": "app"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)

        # Test with non-existent file
        assert cmd._is_safe_executable("/fake/path.app") is False

        # Test with dangerous paths
        assert cmd._is_safe_executable("/usr/bin/rm") is False
        assert cmd._is_safe_executable("/bin/sh") is False
        assert cmd._is_safe_executable("/sbin/shutdown") is False

        # Mock file checks for platform-specific tests
        with patch("os.path.isfile") as mock_isfile, patch("os.access") as mock_access:
            mock_isfile.return_value = True
            mock_access.return_value = True

            import platform

            if platform.system() == "Darwin":
                # macOS applications
                assert cmd._is_safe_executable("/Applications/TestApp.app") is True
                assert (
                    cmd._is_safe_executable("/System/Applications/TestApp.app") is True
                )

    def test_security_validation(self):
        """Test security validation for dangerous inputs."""
        # Script injection in app name
        params = CommandParameters(
            {"application_name": "<script>alert('xss')</script>"}
        )
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Command injection in arguments
        params = CommandParameters(
            {"application_name": "Calculator", "launch_arguments": ["`rm -rf /`"]}
        )
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Path traversal in application path
        params = CommandParameters({"application_path": "../../../etc/passwd"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

    @patch("subprocess.Popen")
    @patch(
        "src.commands.application.LaunchApplicationCommand._resolve_application_path"
    )
    def test_launch_execute_success(self, mock_resolve, mock_popen, context):
        """Test successful application launch."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        mock_resolve.return_value = "/Applications/Calculator.app"

        params = CommandParameters({"application_name": "Calculator"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "Successfully launched" in result.output
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    @patch(
        "src.commands.application.LaunchApplicationCommand._resolve_application_path"
    )
    def test_launch_execute_with_wait(self, mock_resolve, mock_popen, context):
        """Test application launch with wait."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process
        mock_resolve.return_value = "/Applications/Calculator.app"

        params = CommandParameters(
            {
                "application_name": "Calculator",
                "wait_for_launch": True,
                "launch_timeout": 5,
            }
        )
        cmd = LaunchApplicationCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        mock_process.wait.assert_called_once_with(timeout=5.0)

    @patch(
        "src.commands.application.LaunchApplicationCommand._resolve_application_path"
    )
    def test_launch_execute_application_not_found(self, mock_resolve, context):
        """Test launch when application not found."""
        mock_resolve.return_value = None

        params = CommandParameters({"application_name": "NonExistentApp"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is False
        assert "Could not find application" in result.error_message

    def test_get_required_permissions(self):
        """Test required permissions."""
        params = CommandParameters({"application_name": "app"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.get_required_permissions() == frozenset([Permission.SYSTEM_CONTROL])

    def test_get_security_risk_level(self):
        """Test security risk level."""
        params = CommandParameters({"application_name": "app"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.get_security_risk_level() == "high"


class TestQuitApplicationCommand:
    """Test QuitApplicationCommand functionality."""

    @pytest.fixture
    def context(self):
        """Create execution context."""
        return ExecutionContext.create_test_context(
            permissions=frozenset(
                [Permission.APPLICATION_CONTROL, Permission.SYSTEM_CONTROL]
            )
        )

    def test_quit_validation_success(self):
        """Test successful quit validation."""
        params = CommandParameters({"application_name": "TextEdit"})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is True

    def test_quit_validation_failures(self):
        """Test quit validation failures."""
        # No application name
        params = CommandParameters({})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Protected system process
        params = CommandParameters({"application_name": "kernel"})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

    def test_get_force_quit(self):
        """Test force quit option retrieval."""
        # Default is False
        params = CommandParameters({"application_name": "app"})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        assert cmd.get_force_quit() is False

        # Explicit force quit
        params = CommandParameters({"application_name": "app", "force_quit": True})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        assert cmd.get_force_quit() is True

    def test_get_quit_timeout(self):
        """Test quit timeout retrieval."""
        # Default timeout
        params = CommandParameters({"application_name": "app"})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        assert cmd.get_quit_timeout() == Duration.from_seconds(10)

        # Custom timeout
        params = CommandParameters({"application_name": "app", "quit_timeout": 20})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        assert cmd.get_quit_timeout() == Duration.from_seconds(20)

    @patch("subprocess.run")
    @patch("src.commands.application.QuitApplicationCommand._find_application_pids")
    def test_quit_execute_success(self, mock_find_pids, mock_run, context):
        """Test successful application quit."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        mock_find_pids.return_value = [12345]

        params = CommandParameters({"application_name": "TextEdit"})
        cmd = QuitApplicationCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "Successfully quit" in result.output

    @patch("subprocess.run")
    @patch("src.commands.application.QuitApplicationCommand._find_application_pids")
    def test_quit_execute_force_quit(self, mock_find_pids, mock_run, context):
        """Test force quit execution."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        mock_find_pids.return_value = [12345]

        params = CommandParameters({"application_name": "HungApp", "force_quit": True})
        cmd = QuitApplicationCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        # Force quit uses different mechanism
        assert mock_run.called

    def test_quit_execute_no_permission(self, context):
        """Test quit without permission."""
        limited_context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.SCREEN_CAPTURE])
        )

        params = CommandParameters({"application_name": "app"})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        result = cmd.execute(limited_context)

        assert result.success is False
        assert "Missing required permission" in result.error_message


class TestActivateApplicationCommand:
    """Test ActivateApplicationCommand functionality."""

    @pytest.fixture
    def context(self):
        """Create execution context."""
        return ExecutionContext.create_test_context(
            permissions=frozenset(
                [
                    Permission.APPLICATION_CONTROL,
                    Permission.SYSTEM_CONTROL,
                    Permission.WINDOW_MANAGEMENT,
                ]
            )
        )

    def test_activate_validation_success(self):
        """Test successful activate validation."""
        params = CommandParameters({"application_name": "Finder"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is True

    def test_activate_validation_failures(self):
        """Test activate validation failures."""
        # No application name
        params = CommandParameters({})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Script injection
        params = CommandParameters({"application_name": "App<script>alert()</script>"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

    def test_get_create_if_not_running(self):
        """Test create if not running option."""
        # Default is False
        params = CommandParameters({"application_name": "app"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        assert cmd.get_create_if_not_running() is False

        # Explicit True
        params = CommandParameters(
            {"application_name": "app", "create_if_not_running": True}
        )
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        assert cmd.get_create_if_not_running() is True

    @patch("subprocess.run")
    @patch(
        "src.commands.application.ActivateApplicationCommand._is_application_running"
    )
    def test_activate_execute_success(self, mock_is_running, mock_run, context):
        """Test successful application activation."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = b"Success"
        mock_run.return_value = mock_process
        mock_is_running.return_value = True

        params = CommandParameters({"application_name": "Safari"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "Successfully activated" in result.output

    @patch("subprocess.run")
    @patch(
        "src.commands.application.ActivateApplicationCommand._is_application_running"
    )
    def test_activate_execute_not_running(self, mock_is_running, mock_run, context):
        """Test activation of non-running application."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = b"Application not running"
        mock_run.return_value = mock_process
        mock_is_running.return_value = False

        params = CommandParameters({"application_name": "NotRunningApp"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is False
        assert "not running" in result.error_message

    def test_get_required_permissions(self):
        """Test required permissions."""
        # Default without create_if_not_running
        params = CommandParameters({"application_name": "app"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        assert cmd.get_required_permissions() == frozenset(
            [Permission.WINDOW_MANAGEMENT]
        )

        # With create_if_not_running
        params = CommandParameters(
            {"application_name": "app", "create_if_not_running": True}
        )
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        assert cmd.get_required_permissions() == frozenset(
            [Permission.WINDOW_MANAGEMENT, Permission.SYSTEM_CONTROL]
        )

    def test_get_security_risk_level(self):
        """Test security risk level."""
        params = CommandParameters({"application_name": "app"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        assert cmd.get_security_risk_level() == "medium"


class TestApplicationCommandEdgeCases:
    """Test edge cases for application commands."""

    def test_launch_with_special_characters(self):
        """Test launch with special characters in name."""
        params = CommandParameters({"application_name": "My App (2023) - Edition"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is True

    def test_quit_system_applications(self):
        """Test quit validation for system applications."""
        # Should not allow quitting critical system apps
        params = CommandParameters({"application_name": "kernel"})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is False

    def test_activate_with_window_id(self):
        """Test activation with specific window ID."""
        params = CommandParameters({"application_name": "Chrome", "window_id": "12345"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        assert cmd.validate() is True
        assert cmd.parameters.get("window_id") == "12345"

    def test_launch_with_environment_variables(self):
        """Test launch with environment variables."""
        params = CommandParameters(
            {
                "application_name": "MyApp",
                "environment_variables": {"DEBUG": "1", "LOG_LEVEL": "verbose"},
            }
        )
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        # Environment variables not yet implemented
        assert cmd.validate() is True

    @patch("platform.system")
    def test_platform_specific_validation(self, mock_system):
        """Test platform-specific validation."""
        # Windows
        mock_system.return_value = "Windows"
        params = CommandParameters(
            {"application_path": "C:\\Program Files\\App\\app.exe"}
        )
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        # File doesn't exist, so should return False
        assert cmd._is_safe_executable("C:\\Program Files\\App\\app.exe") is False

        # macOS
        mock_system.return_value = "Darwin"

        # Mock file existence and access checks for macOS
        with patch("os.path.isfile") as mock_isfile, patch("os.access") as mock_access:
            mock_isfile.return_value = True
            mock_access.return_value = True

            params = CommandParameters(
                {"application_path": "/Applications/TextEdit.app"}
            )
            cmd = LaunchApplicationCommand(CommandId("test"), params)
            # With mocked file existence, macOS .app should be safe
            assert cmd._is_safe_executable("/Applications/TextEdit.app") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
