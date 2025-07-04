"""
Tests for application control commands.

Tests launch, quit, and activation commands with security validation
and proper contract enforcement.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.core.types import (
    CommandId, CommandParameters, ExecutionContext, Permission, Duration
)
from src.commands.application import (
    LaunchApplicationCommand, QuitApplicationCommand, ActivateApplicationCommand
)


class TestLaunchApplicationCommand:
    """Test application launch command functionality."""
    
    def test_launch_command_creation(self):
        """Test basic launch command creation."""
        params = CommandParameters({
            "application_name": "Calculator",
            "wait_for_launch": False
        })
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        
        assert cmd.get_application_name() == "Calculator"
        assert cmd.get_wait_for_launch() is False
        assert cmd.get_launch_arguments() == []
    
    def test_launch_validation_valid(self):
        """Test launch command validation with valid parameters."""
        params = CommandParameters({
            "application_name": "Calculator",
            "launch_arguments": ["--help"],
            "launch_timeout": 30.0
        })
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        
        assert cmd.validate() is True
    
    def test_launch_validation_no_app_name(self):
        """Test launch command validation with no app name."""
        params = CommandParameters({})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    def test_launch_validation_dangerous_app(self):
        """Test launch command validation with dangerous app name."""
        params = CommandParameters({"application_name": "rm"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    def test_launch_validation_invalid_timeout(self):
        """Test launch command validation with invalid timeout."""
        params = CommandParameters({
            "application_name": "Calculator",
            "launch_timeout": 100.0  # Too long
        })
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        
        # Timeout should be clamped
        assert cmd.get_launch_timeout().seconds == 60.0
    
    @patch('subprocess.Popen')
    @patch('src.commands.application.LaunchApplicationCommand._resolve_application_path')
    def test_launch_execution_success(self, mock_resolve, mock_popen):
        """Test successful application launch."""
        mock_resolve.return_value = "/Applications/Calculator.app"
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        params = CommandParameters({
            "application_name": "Calculator",
            "wait_for_launch": False
        })
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.SYSTEM_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert "Successfully launched Calculator" in result.output
        assert result.metadata["process_id"] == 1234
    
    @patch('src.commands.application.LaunchApplicationCommand._resolve_application_path')
    def test_launch_execution_app_not_found(self, mock_resolve):
        """Test launch execution when app is not found."""
        mock_resolve.return_value = None
        
        params = CommandParameters({"application_name": "NonExistentApp"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.SYSTEM_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is False
        assert "Could not find application" in result.error_message
    
    def test_launch_permissions(self):
        """Test launch command permission requirements."""
        params = CommandParameters({"application_name": "Calculator"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        
        permissions = cmd.get_required_permissions()
        assert Permission.SYSTEM_CONTROL in permissions
    
    def test_launch_security_risk(self):
        """Test launch command security risk level."""
        params = CommandParameters({"application_name": "Calculator"})
        cmd = LaunchApplicationCommand(CommandId("test"), params)
        
        assert cmd.get_security_risk_level() == "high"


class TestQuitApplicationCommand:
    """Test application quit command functionality."""
    
    def test_quit_command_creation(self):
        """Test basic quit command creation."""
        params = CommandParameters({
            "application_name": "Calculator",
            "force_quit": False
        })
        cmd = QuitApplicationCommand(CommandId("test"), params)
        
        assert cmd.get_application_name() == "Calculator"
        assert cmd.get_force_quit() is False
        assert cmd.get_process_id() is None
    
    def test_quit_validation_valid(self):
        """Test quit command validation with valid parameters."""
        params = CommandParameters({
            "application_name": "Calculator",
            "quit_timeout": 10.0
        })
        cmd = QuitApplicationCommand(CommandId("test"), params)
        
        assert cmd.validate() is True
    
    def test_quit_validation_no_identifier(self):
        """Test quit command validation with no app name or PID."""
        params = CommandParameters({})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    def test_quit_validation_protected_process(self):
        """Test quit command validation with protected process."""
        params = CommandParameters({"application_name": "kernel"})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    def test_quit_validation_invalid_pid(self):
        """Test quit command validation with invalid PID."""
        # System PID
        params = CommandParameters({"process_id": 1})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
        
        # Negative PID
        params = CommandParameters({"process_id": -1})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    @patch('subprocess.run')
    @patch('src.commands.application.QuitApplicationCommand._find_application_pids')
    def test_quit_execution_success(self, mock_find_pids, mock_subprocess):
        """Test successful application quit."""
        mock_find_pids.return_value = [1234]
        mock_subprocess.return_value.returncode = 0
        
        params = CommandParameters({
            "application_name": "Calculator",
            "force_quit": False
        })
        cmd = QuitApplicationCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.SYSTEM_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert "Successfully quit 1 process" in result.output
    
    @patch('src.commands.application.QuitApplicationCommand._find_application_pids')
    def test_quit_execution_app_not_running(self, mock_find_pids):
        """Test quit execution when app is not running."""
        mock_find_pids.return_value = []
        
        params = CommandParameters({"application_name": "Calculator"})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.SYSTEM_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is False
        assert "Application not running" in result.error_message
    
    def test_quit_permissions(self):
        """Test quit command permission requirements."""
        params = CommandParameters({"application_name": "Calculator"})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        
        permissions = cmd.get_required_permissions()
        assert Permission.SYSTEM_CONTROL in permissions
    
    def test_quit_security_risk(self):
        """Test quit command security risk level."""
        params = CommandParameters({"application_name": "Calculator"})
        cmd = QuitApplicationCommand(CommandId("test"), params)
        
        assert cmd.get_security_risk_level() == "high"


class TestActivateApplicationCommand:
    """Test application activation command functionality."""
    
    def test_activate_command_creation(self):
        """Test basic activate command creation."""
        params = CommandParameters({
            "application_name": "Calculator",
            "create_if_not_running": False
        })
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        
        assert cmd.get_application_name() == "Calculator"
        assert cmd.get_create_if_not_running() is False
        assert cmd.get_window_title() is None
    
    def test_activate_validation_valid(self):
        """Test activate command validation with valid parameters."""
        params = CommandParameters({
            "application_name": "Calculator",
            "window_title": "Calculator Window"
        })
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        
        assert cmd.validate() is True
    
    def test_activate_validation_no_app_name(self):
        """Test activate command validation with no app name."""
        params = CommandParameters({})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    @patch('subprocess.run')
    @patch('src.commands.application.ActivateApplicationCommand._is_application_running')
    @patch('src.commands.application.ActivateApplicationCommand._activate_application')
    def test_activate_execution_success(self, mock_activate, mock_is_running, mock_subprocess):
        """Test successful application activation."""
        mock_is_running.return_value = True
        mock_activate.return_value = True
        
        params = CommandParameters({"application_name": "Calculator"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.WINDOW_MANAGEMENT]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert "Successfully activated Calculator" in result.output
    
    @patch('src.commands.application.ActivateApplicationCommand._is_application_running')
    def test_activate_execution_app_not_running(self, mock_is_running):
        """Test activate execution when app is not running."""
        mock_is_running.return_value = False
        
        params = CommandParameters({
            "application_name": "Calculator",
            "create_if_not_running": False
        })
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.WINDOW_MANAGEMENT]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is False
        assert "Application not running" in result.error_message
    
    def test_activate_permissions_basic(self):
        """Test activate command permission requirements (basic)."""
        params = CommandParameters({"application_name": "Calculator"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        
        permissions = cmd.get_required_permissions()
        assert Permission.WINDOW_MANAGEMENT in permissions
        assert Permission.SYSTEM_CONTROL not in permissions
    
    def test_activate_permissions_with_launch(self):
        """Test activate command permission requirements (with launch)."""
        params = CommandParameters({
            "application_name": "Calculator",
            "create_if_not_running": True
        })
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        
        permissions = cmd.get_required_permissions()
        assert Permission.WINDOW_MANAGEMENT in permissions
        assert Permission.SYSTEM_CONTROL in permissions
    
    def test_activate_security_risk(self):
        """Test activate command security risk level."""
        params = CommandParameters({"application_name": "Calculator"})
        cmd = ActivateApplicationCommand(CommandId("test"), params)
        
        assert cmd.get_security_risk_level() == "medium"


if __name__ == "__main__":
    pytest.main([__file__])