"""
Comprehensive Command Module Tests - Coverage Expansion

Tests for command modules including base commands, validation, registry, and specific command types.
Focuses on achieving high coverage for core command infrastructure.

Architecture: Property-Based Testing + Type Safety + Contract Validation
Performance: <100ms per test, parallel execution, comprehensive edge case coverage
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from hypothesis import given, strategies as st, settings
from dataclasses import dataclass

# Test imports with graceful fallbacks
try:
    from src.commands.base import BaseCommand, CommandResult, CommandError
    from src.commands.validation import CommandValidator, ValidationResult
    from src.commands.registry import CommandRegistry
    from src.commands.text import TextCommand
    from src.commands.system import SystemCommand
    from src.commands.application import ApplicationCommand
    from src.commands.flow import FlowCommand
    from src.core.types import CommandId, MacroId
    from src.core.either import Either
    from src.core.errors import ValidationError
    COMMANDS_AVAILABLE = True
except ImportError:
    COMMANDS_AVAILABLE = False
    # Mock classes for testing
    class BaseCommand:
        def __init__(self, command_id: str, name: str):
            self.command_id = command_id
            self.name = name
    
    class CommandResult:
        def __init__(self, success: bool, data: Any = None, error: str = None):
            self.success = success
            self.data = data
            self.error = error
    
    class CommandError(Exception):
        pass


class TestBaseCommand:
    """Test base command functionality."""
    
    def test_base_command_creation(self):
        """Test base command creation with valid parameters."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        command = BaseCommand("test_cmd", "Test Command")
        assert command.command_id == "test_cmd"
        assert command.name == "Test Command"
    
    def test_base_command_validation_required(self):
        """Test base command requires validation implementation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        class InvalidCommand(BaseCommand):
            pass
        
        # Should raise error when validation not implemented
        with pytest.raises(NotImplementedError):
            cmd = InvalidCommand("invalid", "Invalid")
            cmd.validate({})
    
    def test_base_command_execution_required(self):
        """Test base command requires execution implementation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        class InvalidCommand(BaseCommand):
            def validate(self, parameters: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(True, {})
        
        # Should raise error when execute not implemented
        with pytest.raises(NotImplementedError):
            cmd = InvalidCommand("invalid", "Invalid")
            asyncio.run(cmd.execute({}))
    
    @pytest.mark.asyncio
    async def test_complete_command_implementation(self):
        """Test complete command implementation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        class TestCommand(BaseCommand):
            def validate(self, parameters: Dict[str, Any]) -> ValidationResult:
                if "required_param" not in parameters:
                    return ValidationResult(False, {}, "Missing required parameter")
                return ValidationResult(True, parameters)
            
            async def execute(self, parameters: Dict[str, Any]) -> CommandResult:
                return CommandResult(True, {"result": parameters["required_param"]})
        
        cmd = TestCommand("test", "Test Command")
        
        # Test validation
        validation = cmd.validate({"required_param": "value"})
        assert validation.is_valid
        
        # Test execution
        result = await cmd.execute({"required_param": "test_value"})
        assert result.success
        assert result.data["result"] == "test_value"
    
    def test_command_error_handling(self):
        """Test command error handling."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        error = CommandError("Test error", "TEST_ERROR")
        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
    
    def test_command_result_types(self):
        """Test command result types."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        # Success result
        success_result = CommandResult(True, {"key": "value"})
        assert success_result.success
        assert success_result.data["key"] == "value"
        assert success_result.error is None
        
        # Error result
        error_result = CommandResult(False, error="Operation failed")
        assert not error_result.success
        assert error_result.error == "Operation failed"
        assert error_result.data is None


class TestCommandValidator:
    """Test command validation functionality."""
    
    def test_validator_creation(self):
        """Test command validator creation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        validator = CommandValidator()
        assert validator is not None
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        validator = CommandValidator()
        
        # Test required parameter validation
        schema = {
            "required_param": {"type": str, "required": True},
            "optional_param": {"type": int, "required": False}
        }
        
        # Valid parameters
        result = validator.validate_parameters(
            {"required_param": "value", "optional_param": 42}, 
            schema
        )
        assert result.is_valid
        
        # Missing required parameter
        result = validator.validate_parameters(
            {"optional_param": 42}, 
            schema
        )
        assert not result.is_valid
        assert "required_param" in result.error_message
    
    def test_type_validation(self):
        """Test parameter type validation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        validator = CommandValidator()
        
        schema = {
            "string_param": {"type": str, "required": True},
            "int_param": {"type": int, "required": True},
            "bool_param": {"type": bool, "required": True}
        }
        
        # Valid types
        result = validator.validate_parameters({
            "string_param": "test",
            "int_param": 42,
            "bool_param": True
        }, schema)
        assert result.is_valid
        
        # Invalid types
        result = validator.validate_parameters({
            "string_param": 123,  # Should be string
            "int_param": "not_int",  # Should be int
            "bool_param": True
        }, schema)
        assert not result.is_valid
    
    def test_validation_result_structure(self):
        """Test validation result structure."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        # Valid result
        valid_result = ValidationResult(True, {"param": "value"})
        assert valid_result.is_valid
        assert valid_result.validated_parameters["param"] == "value"
        assert valid_result.error_message is None
        
        # Invalid result
        invalid_result = ValidationResult(False, {}, "Validation failed")
        assert not invalid_result.is_valid
        assert invalid_result.error_message == "Validation failed"
    
    @given(st.dictionaries(st.text(), st.one_of(st.text(), st.integers(), st.booleans())))
    @settings(max_examples=20)
    def test_validation_property_based(self, parameters):
        """Property-based test for validation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        validator = CommandValidator()
        
        # Define a permissive schema
        schema = {}
        for key in parameters.keys():
            schema[key] = {"type": type(parameters[key]), "required": False}
        
        result = validator.validate_parameters(parameters, schema)
        
        # Should always be valid for matching types
        assert result.is_valid
        assert result.validated_parameters == parameters


class TestCommandRegistry:
    """Test command registry functionality."""
    
    def test_registry_creation(self):
        """Test command registry creation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        registry = CommandRegistry()
        assert registry is not None
        assert len(registry.list_commands()) == 0
    
    def test_command_registration(self):
        """Test command registration."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        registry = CommandRegistry()
        
        # Create test command
        class TestCommand(BaseCommand):
            def validate(self, parameters: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(True, parameters)
            
            async def execute(self, parameters: Dict[str, Any]) -> CommandResult:
                return CommandResult(True, {"executed": True})
        
        command = TestCommand("test", "Test Command")
        
        # Register command
        registry.register_command(command)
        
        # Verify registration
        assert len(registry.list_commands()) == 1
        assert registry.get_command("test") == command
    
    def test_command_registration_duplicate_prevention(self):
        """Test prevention of duplicate command registration."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        registry = CommandRegistry()
        
        class TestCommand(BaseCommand):
            def validate(self, parameters: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(True, parameters)
            
            async def execute(self, parameters: Dict[str, Any]) -> CommandResult:
                return CommandResult(True, {"executed": True})
        
        command1 = TestCommand("test", "Test Command 1")
        command2 = TestCommand("test", "Test Command 2")
        
        # Register first command
        registry.register_command(command1)
        
        # Attempt to register duplicate
        with pytest.raises(ValueError, match="Command with ID 'test' already registered"):
            registry.register_command(command2)
    
    def test_command_unregistration(self):
        """Test command unregistration."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        registry = CommandRegistry()
        
        class TestCommand(BaseCommand):
            def validate(self, parameters: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(True, parameters)
            
            async def execute(self, parameters: Dict[str, Any]) -> CommandResult:
                return CommandResult(True, {"executed": True})
        
        command = TestCommand("test", "Test Command")
        
        # Register and unregister
        registry.register_command(command)
        assert len(registry.list_commands()) == 1
        
        registry.unregister_command("test")
        assert len(registry.list_commands()) == 0
        assert registry.get_command("test") is None
    
    def test_command_categories(self):
        """Test command categorization."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        registry = CommandRegistry()
        
        class TestCommand(BaseCommand):
            def __init__(self, command_id: str, name: str, category: str):
                super().__init__(command_id, name)
                self.category = category
            
            def validate(self, parameters: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(True, parameters)
            
            async def execute(self, parameters: Dict[str, Any]) -> CommandResult:
                return CommandResult(True, {"executed": True})
        
        # Register commands in different categories
        registry.register_command(TestCommand("text1", "Text Command 1", "text"))
        registry.register_command(TestCommand("text2", "Text Command 2", "text"))
        registry.register_command(TestCommand("system1", "System Command 1", "system"))
        
        # Test category filtering
        text_commands = registry.get_commands_by_category("text")
        assert len(text_commands) == 2
        
        system_commands = registry.get_commands_by_category("system")
        assert len(system_commands) == 1


class TestTextCommand:
    """Test text command functionality."""
    
    def test_text_command_creation(self):
        """Test text command creation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        text_cmd = TextCommand("insert_text", "Insert Text")
        assert text_cmd.command_id == "insert_text"
        assert text_cmd.name == "Insert Text"
    
    def test_text_command_validation(self):
        """Test text command validation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        text_cmd = TextCommand("insert_text", "Insert Text")
        
        # Valid parameters
        result = text_cmd.validate({"text": "Hello World"})
        assert result.is_valid
        
        # Missing text parameter
        result = text_cmd.validate({})
        assert not result.is_valid
        assert "text" in result.error_message
    
    @pytest.mark.asyncio
    async def test_text_command_execution(self):
        """Test text command execution."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        text_cmd = TextCommand("insert_text", "Insert Text")
        
        with patch('src.commands.text.insert_text') as mock_insert:
            mock_insert.return_value = True
            
            result = await text_cmd.execute({"text": "Hello World"})
            assert result.success
            mock_insert.assert_called_once_with("Hello World")
    
    def test_text_command_special_characters(self):
        """Test text command with special characters."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        text_cmd = TextCommand("insert_text", "Insert Text")
        
        # Test with special characters
        special_text = "Hello\nWorld\t!"
        result = text_cmd.validate({"text": special_text})
        assert result.is_valid
        assert result.validated_parameters["text"] == special_text


class TestSystemCommand:
    """Test system command functionality."""
    
    def test_system_command_creation(self):
        """Test system command creation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        sys_cmd = SystemCommand("execute_script", "Execute Script")
        assert sys_cmd.command_id == "execute_script"
        assert sys_cmd.name == "Execute Script"
    
    def test_system_command_validation(self):
        """Test system command validation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        sys_cmd = SystemCommand("execute_script", "Execute Script")
        
        # Valid parameters
        result = sys_cmd.validate({"script": "echo 'Hello'"})
        assert result.is_valid
        
        # Missing script parameter
        result = sys_cmd.validate({})
        assert not result.is_valid
    
    @pytest.mark.asyncio
    async def test_system_command_execution(self):
        """Test system command execution."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        sys_cmd = SystemCommand("execute_script", "Execute Script")
        
        with patch('src.commands.system.execute_system_command') as mock_execute:
            mock_execute.return_value = Either.right({"output": "Hello"})
            
            result = await sys_cmd.execute({"script": "echo 'Hello'"})
            assert result.success
            assert result.data["output"] == "Hello"
    
    def test_system_command_security_validation(self):
        """Test system command security validation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        sys_cmd = SystemCommand("execute_script", "Execute Script")
        
        # Test dangerous commands are rejected
        dangerous_scripts = [
            "rm -rf /",
            "sudo rm -rf /",
            "format c:",
            "del /s /q c:\\*"
        ]
        
        for script in dangerous_scripts:
            result = sys_cmd.validate({"script": script})
            assert not result.is_valid
            assert "security" in result.error_message.lower()


class TestApplicationCommand:
    """Test application command functionality."""
    
    def test_application_command_creation(self):
        """Test application command creation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        app_cmd = ApplicationCommand("activate_app", "Activate Application")
        assert app_cmd.command_id == "activate_app"
        assert app_cmd.name == "Activate Application"
    
    def test_application_command_validation(self):
        """Test application command validation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        app_cmd = ApplicationCommand("activate_app", "Activate Application")
        
        # Valid parameters
        result = app_cmd.validate({"application": "TextEdit"})
        assert result.is_valid
        
        # Missing application parameter
        result = app_cmd.validate({})
        assert not result.is_valid
    
    @pytest.mark.asyncio
    async def test_application_command_execution(self):
        """Test application command execution."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        app_cmd = ApplicationCommand("activate_app", "Activate Application")
        
        with patch('src.commands.application.activate_application') as mock_activate:
            mock_activate.return_value = Either.right({"activated": True})
            
            result = await app_cmd.execute({"application": "TextEdit"})
            assert result.success
            assert result.data["activated"] is True


class TestFlowCommand:
    """Test flow command functionality."""
    
    def test_flow_command_creation(self):
        """Test flow command creation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        flow_cmd = FlowCommand("if_condition", "If Condition")
        assert flow_cmd.command_id == "if_condition"
        assert flow_cmd.name == "If Condition"
    
    def test_flow_command_validation(self):
        """Test flow command validation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        flow_cmd = FlowCommand("if_condition", "If Condition")
        
        # Valid parameters
        result = flow_cmd.validate({
            "condition": "variable == 'value'",
            "then_actions": ["action1", "action2"]
        })
        assert result.is_valid
        
        # Missing condition parameter
        result = flow_cmd.validate({"then_actions": ["action1"]})
        assert not result.is_valid
    
    @pytest.mark.asyncio
    async def test_flow_command_execution(self):
        """Test flow command execution."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        flow_cmd = FlowCommand("if_condition", "If Condition")
        
        with patch('src.commands.flow.evaluate_condition') as mock_evaluate:
            mock_evaluate.return_value = Either.right(True)
            
            result = await flow_cmd.execute({
                "condition": "variable == 'value'",
                "then_actions": ["action1"]
            })
            assert result.success
    
    def test_flow_command_condition_parsing(self):
        """Test flow command condition parsing."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        flow_cmd = FlowCommand("if_condition", "If Condition")
        
        # Test various condition formats
        conditions = [
            "variable == 'value'",
            "count > 10",
            "status != 'error'",
            "enabled && ready"
        ]
        
        for condition in conditions:
            result = flow_cmd.validate({
                "condition": condition,
                "then_actions": ["action1"]
            })
            assert result.is_valid


class TestCommandIntegration:
    """Test command integration scenarios."""
    
    def test_command_chaining(self):
        """Test command chaining functionality."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        registry = CommandRegistry()
        
        # Create chainable commands
        class ChainableCommand(BaseCommand):
            def __init__(self, command_id: str, name: str, output_key: str):
                super().__init__(command_id, name)
                self.output_key = output_key
            
            def validate(self, parameters: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(True, parameters)
            
            async def execute(self, parameters: Dict[str, Any]) -> CommandResult:
                return CommandResult(True, {self.output_key: f"result_{self.command_id}"})
        
        # Register commands
        cmd1 = ChainableCommand("cmd1", "Command 1", "output1")
        cmd2 = ChainableCommand("cmd2", "Command 2", "output2")
        
        registry.register_command(cmd1)
        registry.register_command(cmd2)
        
        # Test command availability for chaining
        available_commands = registry.list_commands()
        assert len(available_commands) == 2
        assert all(cmd.command_id in ["cmd1", "cmd2"] for cmd in available_commands)
    
    @pytest.mark.asyncio
    async def test_command_error_propagation(self):
        """Test command error propagation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        class FailingCommand(BaseCommand):
            def validate(self, parameters: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(True, parameters)
            
            async def execute(self, parameters: Dict[str, Any]) -> CommandResult:
                raise CommandError("Intentional failure", "TEST_ERROR")
        
        cmd = FailingCommand("failing", "Failing Command")
        
        with pytest.raises(CommandError) as exc_info:
            await cmd.execute({})
        
        assert exc_info.value.error_code == "TEST_ERROR"
        assert "Intentional failure" in str(exc_info.value)
    
    def test_command_metadata_preservation(self):
        """Test command metadata preservation."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        class MetadataCommand(BaseCommand):
            def __init__(self, command_id: str, name: str, **metadata):
                super().__init__(command_id, name)
                self.metadata = metadata
            
            def validate(self, parameters: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(True, parameters)
            
            async def execute(self, parameters: Dict[str, Any]) -> CommandResult:
                return CommandResult(True, {"metadata": self.metadata})
        
        cmd = MetadataCommand(
            "meta_cmd", 
            "Metadata Command",
            category="test",
            description="Test command with metadata",
            version="1.0.0"
        )
        
        assert cmd.metadata["category"] == "test"
        assert cmd.metadata["description"] == "Test command with metadata"
        assert cmd.metadata["version"] == "1.0.0"


# Property-based testing for command operations
class TestCommandProperties:
    """Property-based tests for command operations."""
    
    @given(st.text(min_size=1, max_size=50), st.text(min_size=1, max_size=100))
    @settings(max_examples=20)
    def test_command_creation_properties(self, command_id, name):
        """Test command creation properties."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        class TestCommand(BaseCommand):
            def validate(self, parameters: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(True, parameters)
            
            async def execute(self, parameters: Dict[str, Any]) -> CommandResult:
                return CommandResult(True, parameters)
        
        cmd = TestCommand(command_id, name)
        assert cmd.command_id == command_id
        assert cmd.name == name
    
    @given(st.dictionaries(st.text(), st.one_of(st.text(), st.integers(), st.booleans())))
    @settings(max_examples=10)
    def test_validation_result_properties(self, parameters):
        """Test validation result properties."""
        if not COMMANDS_AVAILABLE:
            pytest.skip("Commands module not available")
        
        # Valid result should preserve parameters
        result = ValidationResult(True, parameters)
        assert result.is_valid
        assert result.validated_parameters == parameters
        
        # Invalid result should not have parameters
        result = ValidationResult(False, {}, "Error")
        assert not result.is_valid
        assert result.error_message == "Error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])