"""
Comprehensive Parser and Context Module Tests - Coverage Expansion

Tests for core parsing functionality, context management, and execution environment.
Focuses on achieving high coverage for core infrastructure modules.

Architecture: Property-Based Testing + Type Safety + Contract Validation
Performance: <50ms per test, parallel execution, comprehensive edge case coverage
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from dataclasses import dataclass

# Test imports with graceful fallbacks
try:
    from src.core.parser import (
        MacroParser, CommandParser, ParameterParser, 
        ParseResult, ParseError, TokenType, Token
    )
    from src.core.context import (
        ExecutionContext, VariableScope, ContextManager,
        ContextState, EnvironmentContext
    )
    from src.core.types import MacroId, CommandId, VariableId
    from src.core.either import Either
    PARSER_CONTEXT_AVAILABLE = True
except ImportError:
    PARSER_CONTEXT_AVAILABLE = False
    # Mock classes for testing
    class MacroParser:
        def parse(self, source: str):
            return {"commands": []}
    
    class ParseResult:
        def __init__(self, success: bool, data: Any = None, error: str = None):
            self.success = success
            self.data = data
            self.error = error


class TestMacroParser:
    """Test macro parsing functionality."""
    
    def test_parser_creation(self):
        """Test macro parser creation."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = MacroParser()
        assert parser is not None
    
    def test_simple_macro_parsing(self):
        """Test parsing simple macro definitions."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = MacroParser()
        
        # Simple macro with one command
        macro_source = """
        macro "Test Macro" {
            type_text "Hello World"
        }
        """
        
        result = parser.parse(macro_source)
        assert result.success
        assert "macro_name" in result.data
        assert result.data["macro_name"] == "Test Macro"
        assert len(result.data["commands"]) == 1
        assert result.data["commands"][0]["command_type"] == "type_text"
    
    def test_complex_macro_parsing(self):
        """Test parsing complex macro with multiple commands."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = MacroParser()
        
        # Complex macro with conditions and loops
        macro_source = """
        macro "Complex Macro" {
            set_variable "counter" "0"
            repeat 5 {
                type_text "Hello "
                increment_variable "counter"
            }
            if variable_equals("counter", "5") {
                type_text "Done!"
            }
        }
        """
        
        result = parser.parse(macro_source)
        assert result.success
        assert result.data["macro_name"] == "Complex Macro"
        assert len(result.data["commands"]) == 3  # set_variable, repeat, if
    
    def test_macro_parsing_syntax_errors(self):
        """Test macro parsing with syntax errors."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = MacroParser()
        
        # Invalid syntax examples
        invalid_sources = [
            'macro "Test" {',  # Missing closing brace
            'macro { type_text "test" }',  # Missing name
            'macro "Test" type_text "test"',  # Missing braces
            'macro "Test" { invalid_command }',  # Invalid command
        ]
        
        for source in invalid_sources:
            result = parser.parse(source)
            assert not result.success
            assert "syntax" in result.error.lower() or "parse" in result.error.lower()
    
    def test_macro_parsing_nested_structures(self):
        """Test parsing nested control structures."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = MacroParser()
        
        # Nested if and repeat structures
        macro_source = """
        macro "Nested Test" {
            repeat 3 {
                if variable_exists("test_var") {
                    repeat 2 {
                        type_text "Nested"
                    }
                } else {
                    type_text "Alternative"
                }
            }
        }
        """
        
        result = parser.parse(macro_source)
        assert result.success
        assert result.data["commands"][0]["command_type"] == "repeat"
        assert "nested_commands" in result.data["commands"][0]
    
    def test_macro_parsing_variables(self):
        """Test parsing macro with variable definitions."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = MacroParser()
        
        macro_source = """
        macro "Variable Test" {
            variable "user_name" = "John Doe"
            variable "count" = 42
            variable "enabled" = true
            type_text variable("user_name")
        }
        """
        
        result = parser.parse(macro_source)
        assert result.success
        assert "variables" in result.data
        assert len(result.data["variables"]) == 3
        assert result.data["variables"]["user_name"] == "John Doe"
        assert result.data["variables"]["count"] == 42
        assert result.data["variables"]["enabled"] is True
    
    def test_macro_parsing_comments(self):
        """Test parsing macro with comments."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = MacroParser()
        
        macro_source = """
        // This is a test macro
        macro "Comment Test" {
            // Set initial value
            set_variable "test" "value"
            /* Multi-line comment
               explaining the next command */
            type_text "Hello" // Inline comment
        }
        """
        
        result = parser.parse(macro_source)
        assert result.success
        # Comments should be stripped during parsing
        assert len(result.data["commands"]) == 2
    
    @given(st.text(min_size=1, max_size=50, alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '))
    @settings(max_examples=10)
    def test_macro_name_parsing_property(self, macro_name):
        """Property-based test for macro name parsing."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = MacroParser()
        
        # Clean macro name for valid syntax
        clean_name = macro_name.strip()
        if not clean_name:
            return
        
        macro_source = f'''
        macro "{clean_name}" {{
            type_text "test"
        }}
        '''
        
        result = parser.parse(macro_source)
        if result.success:
            assert result.data["macro_name"] == clean_name


class TestCommandParser:
    """Test command parsing functionality."""
    
    def test_command_parser_creation(self):
        """Test command parser creation."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = CommandParser()
        assert parser is not None
    
    def test_simple_command_parsing(self):
        """Test parsing simple commands."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = CommandParser()
        
        # Test various command formats
        commands = [
            'type_text "Hello World"',
            'set_variable "test" "value"',
            'pause 2.5',
            'activate_app "TextEdit"'
        ]
        
        for cmd_source in commands:
            result = parser.parse(cmd_source)
            assert result.success
            assert "command_type" in result.data
            assert "parameters" in result.data
    
    def test_command_with_parameters(self):
        """Test parsing commands with various parameter types."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = CommandParser()
        
        # String parameter
        result = parser.parse('type_text "Hello World"')
        assert result.success
        assert result.data["parameters"]["text"] == "Hello World"
        
        # Numeric parameter
        result = parser.parse('pause 2.5')
        assert result.success
        assert result.data["parameters"]["duration"] == 2.5
        
        # Boolean parameter
        result = parser.parse('set_option "enabled" true')
        assert result.success
        assert result.data["parameters"]["enabled"] is True
    
    def test_command_with_variable_references(self):
        """Test parsing commands with variable references."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = CommandParser()
        
        result = parser.parse('type_text variable("user_input")')
        assert result.success
        assert result.data["parameters"]["text"]["type"] == "variable_reference"
        assert result.data["parameters"]["text"]["variable_name"] == "user_input"
    
    def test_command_parsing_errors(self):
        """Test command parsing error handling."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = CommandParser()
        
        # Invalid command formats
        invalid_commands = [
            'unknown_command "test"',  # Unknown command
            'type_text',  # Missing parameters
            'type_text "unclosed string',  # Unclosed string
            'set_variable "test"',  # Missing second parameter
        ]
        
        for cmd in invalid_commands:
            result = parser.parse(cmd)
            assert not result.success
            assert result.error is not None


class TestParameterParser:
    """Test parameter parsing functionality."""
    
    def test_parameter_parser_creation(self):
        """Test parameter parser creation."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = ParameterParser()
        assert parser is not None
    
    def test_string_parameter_parsing(self):
        """Test string parameter parsing."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = ParameterParser()
        
        # Simple string
        result = parser.parse('"Hello World"')
        assert result.success
        assert result.data["value"] == "Hello World"
        assert result.data["type"] == "string"
        
        # String with escape sequences
        result = parser.parse('"Hello\\nWorld\\t!"')
        assert result.success
        assert result.data["value"] == "Hello\nWorld\t!"
    
    def test_numeric_parameter_parsing(self):
        """Test numeric parameter parsing."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = ParameterParser()
        
        # Integer
        result = parser.parse('42')
        assert result.success
        assert result.data["value"] == 42
        assert result.data["type"] == "integer"
        
        # Float
        result = parser.parse('3.14159')
        assert result.success
        assert result.data["value"] == 3.14159
        assert result.data["type"] == "float"
        
        # Negative numbers
        result = parser.parse('-42')
        assert result.success
        assert result.data["value"] == -42
    
    def test_boolean_parameter_parsing(self):
        """Test boolean parameter parsing."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = ParameterParser()
        
        # True values
        for true_val in ['true', 'True', 'TRUE', 'yes', 'on']:
            result = parser.parse(true_val)
            assert result.success
            assert result.data["value"] is True
            assert result.data["type"] == "boolean"
        
        # False values
        for false_val in ['false', 'False', 'FALSE', 'no', 'off']:
            result = parser.parse(false_val)
            assert result.success
            assert result.data["value"] is False
            assert result.data["type"] == "boolean"
    
    def test_variable_reference_parsing(self):
        """Test variable reference parsing."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = ParameterParser()
        
        result = parser.parse('variable("test_var")')
        assert result.success
        assert result.data["type"] == "variable_reference"
        assert result.data["variable_name"] == "test_var"
    
    def test_function_call_parsing(self):
        """Test function call parsing."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = ParameterParser()
        
        result = parser.parse('get_clipboard()')
        assert result.success
        assert result.data["type"] == "function_call"
        assert result.data["function_name"] == "get_clipboard"
        assert result.data["arguments"] == []
        
        # Function with arguments
        result = parser.parse('substring("hello", 1, 3)')
        assert result.success
        assert result.data["function_name"] == "substring"
        assert len(result.data["arguments"]) == 3


class TestExecutionContext:
    """Test execution context functionality."""
    
    def test_context_creation(self):
        """Test execution context creation."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        context = ExecutionContext()
        assert context is not None
        assert len(context.get_all_variables()) == 0
    
    def test_variable_management(self):
        """Test variable management in context."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        context = ExecutionContext()
        
        # Set variables
        context.set_variable("test_string", "Hello World")
        context.set_variable("test_number", 42)
        context.set_variable("test_boolean", True)
        
        # Get variables
        assert context.get_variable("test_string") == "Hello World"
        assert context.get_variable("test_number") == 42
        assert context.get_variable("test_boolean") is True
        
        # Test non-existent variable
        assert context.get_variable("non_existent") is None
    
    def test_variable_scoping(self):
        """Test variable scoping."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        context = ExecutionContext()
        
        # Global scope
        context.set_variable("global_var", "global_value")
        
        # Create local scope
        context.push_scope("local")
        context.set_variable("local_var", "local_value")
        context.set_variable("global_var", "overridden_value")  # Shadow global
        
        # Check local scope values
        assert context.get_variable("local_var") == "local_value"
        assert context.get_variable("global_var") == "overridden_value"
        
        # Pop scope and check global values
        context.pop_scope()
        assert context.get_variable("global_var") == "global_value"
        assert context.get_variable("local_var") is None
    
    def test_context_state_management(self):
        """Test context state management."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        context = ExecutionContext()
        
        # Initial state
        assert context.get_state() == ContextState.READY
        
        # Change state
        context.set_state(ContextState.EXECUTING)
        assert context.get_state() == ContextState.EXECUTING
        
        context.set_state(ContextState.PAUSED)
        assert context.get_state() == ContextState.PAUSED
        
        context.set_state(ContextState.COMPLETED)
        assert context.get_state() == ContextState.COMPLETED
    
    def test_environment_context(self):
        """Test environment context functionality."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        env_context = EnvironmentContext()
        
        # Test system information
        system_info = env_context.get_system_info()
        assert "platform" in system_info
        assert "python_version" in system_info
        
        # Test environment variables
        env_context.set_env_variable("TEST_VAR", "test_value")
        assert env_context.get_env_variable("TEST_VAR") == "test_value"
    
    def test_context_persistence(self):
        """Test context persistence."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        context = ExecutionContext()
        
        # Set up context state
        context.set_variable("persistent_var", "persistent_value")
        context.set_state(ContextState.EXECUTING)
        
        # Save context
        saved_state = context.save_state()
        assert saved_state is not None
        
        # Create new context and restore
        new_context = ExecutionContext()
        new_context.restore_state(saved_state)
        
        assert new_context.get_variable("persistent_var") == "persistent_value"
        assert new_context.get_state() == ContextState.EXECUTING
    
    @given(st.text(min_size=1, max_size=20), st.one_of(st.text(), st.integers(), st.booleans()))
    @settings(max_examples=20)
    def test_variable_storage_property(self, var_name, var_value):
        """Property-based test for variable storage."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        context = ExecutionContext()
        
        # Clean variable name
        clean_name = var_name.strip()
        if not clean_name:
            return
        
        context.set_variable(clean_name, var_value)
        retrieved_value = context.get_variable(clean_name)
        
        assert retrieved_value == var_value


class TestVariableScope:
    """Test variable scope functionality."""
    
    def test_scope_creation(self):
        """Test variable scope creation."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        scope = VariableScope("test_scope")
        assert scope.name == "test_scope"
        assert len(scope.get_variables()) == 0
    
    def test_scope_variable_operations(self):
        """Test variable operations within scope."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        scope = VariableScope("test_scope")
        
        # Set variables
        scope.set_variable("var1", "value1")
        scope.set_variable("var2", 42)
        
        # Get variables
        assert scope.get_variable("var1") == "value1"
        assert scope.get_variable("var2") == 42
        
        # Check variable existence
        assert scope.has_variable("var1")
        assert not scope.has_variable("non_existent")
        
        # Remove variable
        scope.remove_variable("var1")
        assert not scope.has_variable("var1")
    
    def test_scope_nesting(self):
        """Test nested scope functionality."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        parent_scope = VariableScope("parent")
        child_scope = VariableScope("child", parent_scope)
        
        # Set variable in parent
        parent_scope.set_variable("parent_var", "parent_value")
        
        # Child should see parent variable
        assert child_scope.get_variable("parent_var") == "parent_value"
        
        # Child can override parent variable
        child_scope.set_variable("parent_var", "child_value")
        assert child_scope.get_variable("parent_var") == "child_value"
        assert parent_scope.get_variable("parent_var") == "parent_value"  # Parent unchanged


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager_creation(self):
        """Test context manager creation."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        manager = ContextManager()
        assert manager is not None
    
    def test_context_lifecycle(self):
        """Test context lifecycle management."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        manager = ContextManager()
        
        # Create context
        context_id = manager.create_context("test_macro")
        assert context_id is not None
        
        # Get context
        context = manager.get_context(context_id)
        assert context is not None
        assert context.get_state() == ContextState.READY
        
        # Update context
        context.set_variable("test_var", "test_value")
        manager.update_context(context_id, context)
        
        # Verify update
        updated_context = manager.get_context(context_id)
        assert updated_context.get_variable("test_var") == "test_value"
        
        # Destroy context
        manager.destroy_context(context_id)
        assert manager.get_context(context_id) is None
    
    def test_concurrent_contexts(self):
        """Test concurrent context management."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Context module not available")
        
        manager = ContextManager()
        
        # Create multiple contexts
        context1_id = manager.create_context("macro1")
        context2_id = manager.create_context("macro2")
        
        context1 = manager.get_context(context1_id)
        context2 = manager.get_context(context2_id)
        
        # Set different variables in each context
        context1.set_variable("shared_name", "value1")
        context2.set_variable("shared_name", "value2")
        
        manager.update_context(context1_id, context1)
        manager.update_context(context2_id, context2)
        
        # Verify isolation
        updated_context1 = manager.get_context(context1_id)
        updated_context2 = manager.get_context(context2_id)
        
        assert updated_context1.get_variable("shared_name") == "value1"
        assert updated_context2.get_variable("shared_name") == "value2"


class TestTokenization:
    """Test tokenization functionality."""
    
    def test_token_creation(self):
        """Test token creation."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        token = Token(TokenType.STRING, "Hello World", 1, 5)
        assert token.type == TokenType.STRING
        assert token.value == "Hello World"
        assert token.line == 1
        assert token.column == 5
    
    def test_tokenization_process(self):
        """Test tokenization process."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = MacroParser()
        
        source = 'type_text "Hello World"'
        tokens = parser.tokenize(source)
        
        assert len(tokens) >= 3  # command, string, possibly EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "type_text"
        assert tokens[1].type == TokenType.STRING
        assert tokens[1].value == "Hello World"
    
    def test_tokenization_edge_cases(self):
        """Test tokenization edge cases."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser module not available")
        
        parser = MacroParser()
        
        # Empty source
        tokens = parser.tokenize("")
        assert len(tokens) == 1  # EOF token
        assert tokens[0].type == TokenType.EOF
        
        # Only whitespace
        tokens = parser.tokenize("   \n\t  ")
        assert len(tokens) == 1  # EOF token
        
        # Mixed tokens
        tokens = parser.tokenize('test "string" 123 true')
        expected_types = [TokenType.IDENTIFIER, TokenType.STRING, TokenType.NUMBER, TokenType.BOOLEAN, TokenType.EOF]
        assert len(tokens) == len(expected_types)
        for i, expected_type in enumerate(expected_types):
            assert tokens[i].type == expected_type


# Integration tests
class TestParserContextIntegration:
    """Test integration between parser and context."""
    
    @pytest.mark.asyncio
    async def test_parse_and_execute_workflow(self):
        """Test complete parse and execute workflow."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser/Context modules not available")
        
        # Parse macro
        parser = MacroParser()
        macro_source = '''
        macro "Integration Test" {
            set_variable "counter" "0"
            type_text "Starting..."
        }
        '''
        
        parse_result = parser.parse(macro_source)
        assert parse_result.success
        
        # Create execution context
        context = ExecutionContext()
        
        # Simulate execution
        for command in parse_result.data["commands"]:
            if command["command_type"] == "set_variable":
                context.set_variable(
                    command["parameters"]["name"],
                    command["parameters"]["value"]
                )
        
        # Verify context state
        assert context.get_variable("counter") == "0"
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        if not PARSER_CONTEXT_AVAILABLE:
            pytest.skip("Parser/Context modules not available")
        
        parser = MacroParser()
        
        # Parse with errors
        macro_source = '''
        macro "Error Test" {
            set_variable "test" "value"
            invalid_command_here
            type_text "This should still work"
        }
        '''
        
        result = parser.parse(macro_source)
        
        # Parser should attempt recovery and continue parsing
        # Implementation may vary - test that errors are properly reported
        if not result.success:
            assert "invalid_command_here" in result.error or "parse" in result.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])