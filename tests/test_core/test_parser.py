"""
Tests for the macro parser and validation system.

This module tests parsing of macro definitions, input validation,
and security boundary enforcement.
"""

import pytest
import json

from src.core import (
    MacroParser, ParseResult, InputSanitizer, CommandValidator,
    CommandType, ValidationError, SecurityViolationError,
    parse_macro_from_json
)


class TestInputSanitizer:
    """Test cases for input sanitization."""
    
    def test_sanitize_valid_text(self):
        """Test sanitization of valid text input."""
        text = "Hello, World!"
        result = InputSanitizer.sanitize_text_input(text)
        assert result == "Hello, World!"
    
    def test_sanitize_text_with_dangerous_chars(self):
        """Test sanitization removes dangerous characters in non-strict mode."""
        text = "Hello <script>alert('xss')</script> World"
        # Non-strict mode should sanitize and return cleaned text
        result = InputSanitizer.sanitize_text_input(text, strict_mode=False)
        assert "<script>" not in result
        assert "alert" not in result
    
    def test_sanitize_text_length_limit(self):
        """Test that text length limits are enforced."""
        long_text = "a" * 10001  # Exceeds default limit
        
        with pytest.raises(ValidationError) as exc_info:
            InputSanitizer.sanitize_text_input(long_text)
        
        assert "length" in str(exc_info.value)
    
    def test_script_injection_detection(self):
        """Test detection of script injection attempts."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "eval(malicious_code)",
            "exec(system_command)",
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(SecurityViolationError):
                InputSanitizer.sanitize_text_input(malicious_input, strict_mode=True)
    
    def test_path_traversal_detection(self):
        """Test detection of path traversal attempts."""
        malicious_paths = [
            "../../../etc/passwd",
            "~/../../secret",
            "/bin/bash",
            "C:\\Windows\\System32\\cmd.exe",
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises(SecurityViolationError):
                InputSanitizer.validate_file_path(malicious_path)
    
    def test_identifier_validation(self):
        """Test validation of identifiers."""
        # Valid identifiers
        valid_ids = ["test_macro", "My Macro 123", "macro-name", "macro.name"]
        for valid_id in valid_ids:
            result = InputSanitizer.validate_identifier(valid_id)
            assert isinstance(result, str)
        
        # Invalid identifiers
        invalid_ids = ["", "a" * 256, "macro<script>", "macro;DROP TABLE"]
        for invalid_id in invalid_ids:
            with pytest.raises(ValidationError):
                InputSanitizer.validate_identifier(invalid_id)


class TestCommandValidator:
    """Test cases for command validation."""
    
    def test_validate_command_type(self):
        """Test command type validation."""
        # Valid command type
        cmd_type = CommandValidator.validate_command_type("text_input")
        assert cmd_type == CommandType.TEXT_INPUT
        
        # Invalid command type
        with pytest.raises(ValidationError):
            CommandValidator.validate_command_type("invalid_type")
    
    def test_validate_text_input_parameters(self):
        """Test validation of text input command parameters."""
        # Valid parameters
        params = {"text": "Hello World", "speed": "normal"}
        result = CommandValidator.validate_command_parameters(
            CommandType.TEXT_INPUT, params
        )
        assert result.get("text") == "Hello World"
        assert result.get("speed") == "normal"
        
        # Missing required parameter
        invalid_params = {"speed": "normal"}  # Missing 'text'
        with pytest.raises(ValidationError):
            CommandValidator.validate_command_parameters(
                CommandType.TEXT_INPUT, invalid_params
            )
        
        # Invalid speed value
        invalid_speed_params = {"text": "Hello", "speed": "invalid"}
        with pytest.raises(ValidationError):
            CommandValidator.validate_command_parameters(
                CommandType.TEXT_INPUT, invalid_speed_params
            )
    
    def test_validate_pause_parameters(self):
        """Test validation of pause command parameters."""
        # Valid parameters
        params = {"duration": 2.5}
        result = CommandValidator.validate_command_parameters(
            CommandType.PAUSE, params
        )
        assert result.get("duration") == 2.5
        
        # Invalid duration (too long)
        invalid_params = {"duration": 500}  # Exceeds max
        with pytest.raises(ValidationError):
            CommandValidator.validate_command_parameters(
                CommandType.PAUSE, invalid_params
            )
        
        # Invalid duration (non-numeric)
        invalid_params = {"duration": "invalid"}
        with pytest.raises(ValidationError):
            CommandValidator.validate_command_parameters(
                CommandType.PAUSE, invalid_params
            )
    
    def test_validate_sound_parameters(self):
        """Test validation of sound command parameters."""
        # Valid parameters
        params = {"sound_name": "beep", "volume": 75}
        result = CommandValidator.validate_command_parameters(
            CommandType.PLAY_SOUND, params
        )
        assert result.get("sound_name") == "beep"
        assert result.get("volume") == 75
        
        # Invalid sound name
        invalid_params = {"sound_name": "invalid_sound"}
        with pytest.raises(ValidationError):
            CommandValidator.validate_command_parameters(
                CommandType.PLAY_SOUND, invalid_params
            )
        
        # Invalid volume
        invalid_params = {"sound_name": "beep", "volume": 150}
        with pytest.raises(ValidationError):
            CommandValidator.validate_command_parameters(
                CommandType.PLAY_SOUND, invalid_params
            )
    
    def test_get_required_permissions(self):
        """Test retrieval of required permissions for commands."""
        from src.core import Permission
        
        # Text input requires TEXT_INPUT permission
        perms = CommandValidator.get_required_permissions(CommandType.TEXT_INPUT)
        assert Permission.TEXT_INPUT in perms
        
        # Sound requires SYSTEM_SOUND permission
        perms = CommandValidator.get_required_permissions(CommandType.PLAY_SOUND)
        assert Permission.SYSTEM_SOUND in perms
        
        # Pause requires no special permissions
        perms = CommandValidator.get_required_permissions(CommandType.PAUSE)
        assert len(perms) == 0


class TestMacroParser:
    """Test cases for macro parsing."""
    
    def test_parse_simple_macro(self):
        """Test parsing of a simple macro."""
        macro_data = {
            "name": "Test Macro",
            "id": "test_macro",
            "enabled": True,
            "commands": [
                {
                    "type": "text_input",
                    "parameters": {
                        "text": "Hello World",
                        "speed": "normal"
                    }
                }
            ]
        }
        
        parser = MacroParser()
        result = parser.parse_macro(macro_data)
        
        assert result.success
        assert result.macro_definition is not None
        assert result.macro_definition.name == "Test Macro"
        assert len(result.macro_definition.commands) == 1
    
    def test_parse_macro_with_multiple_commands(self):
        """Test parsing of macro with multiple commands."""
        macro_data = {
            "name": "Multi Command Macro",
            "commands": [
                {
                    "type": "text_input",
                    "parameters": {"text": "Hello"}
                },
                {
                    "type": "pause",
                    "parameters": {"duration": 1.0}
                },
                {
                    "type": "play_sound",
                    "parameters": {"sound_name": "beep"}
                }
            ]
        }
        
        parser = MacroParser()
        result = parser.parse_macro(macro_data)
        
        assert result.success
        assert len(result.macro_definition.commands) == 3
    
    def test_parse_macro_validation_errors(self):
        """Test parsing with validation errors."""
        # Missing name
        invalid_data = {
            "commands": [
                {"type": "text_input", "parameters": {"text": "Hello"}}
            ]
        }
        
        parser = MacroParser()
        result = parser.parse_macro(invalid_data)
        
        assert not result.success
        assert len(result.errors) > 0
    
    def test_parse_macro_empty_commands(self):
        """Test parsing macro with empty commands list."""
        macro_data = {
            "name": "Empty Macro",
            "commands": []
        }
        
        parser = MacroParser()
        result = parser.parse_macro(macro_data)
        
        assert not result.success
        assert any("commands" in str(error) for error in result.errors)
    
    def test_parse_macro_invalid_command(self):
        """Test parsing macro with invalid command."""
        macro_data = {
            "name": "Invalid Command Macro",
            "commands": [
                {
                    "type": "invalid_command_type",
                    "parameters": {}
                }
            ]
        }
        
        parser = MacroParser()
        result = parser.parse_macro(macro_data)
        
        assert not result.success


class TestJSONParsing:
    """Test cases for JSON macro parsing."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON macro definition."""
        json_data = json.dumps({
            "name": "JSON Macro",
            "commands": [
                {
                    "type": "text_input",
                    "parameters": {"text": "From JSON"}
                }
            ]
        })
        
        result = parse_macro_from_json(json_data)
        
        assert result.success
        assert result.macro_definition.name == "JSON Macro"
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        invalid_json = '{"name": "Invalid", "commands": [invalid json}'
        
        result = parse_macro_from_json(invalid_json)
        
        assert not result.success
        assert len(result.errors) > 0
        assert "json" in str(result.errors[0]).lower()
    
    def test_parse_json_with_security_issues(self):
        """Test parsing JSON with security issues."""
        json_with_script = json.dumps({
            "name": "Malicious Macro",
            "commands": [
                {
                    "type": "text_input",
                    "parameters": {
                        "text": "<script>alert('xss')</script>"
                    }
                }
            ]
        })
        
        result = parse_macro_from_json(json_with_script)
        
        # Should fail due to security validation
        assert not result.success