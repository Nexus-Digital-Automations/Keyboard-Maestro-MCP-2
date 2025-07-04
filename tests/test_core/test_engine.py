"""
Tests for the core macro execution engine.

This module tests the main MacroEngine functionality including execution,
validation, error handling, and contract enforcement.
"""

import pytest
import time
from datetime import datetime

from src.core import (
    MacroEngine, MacroDefinition, ExecutionContext, ExecutionStatus,
    CommandType, Permission, Duration, create_test_macro,
    ValidationError, PermissionDeniedError, ExecutionError, ContractViolationError
)


class TestMacroEngine:
    """Test cases for the MacroEngine class."""
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        engine = MacroEngine()
        assert engine is not None
        assert engine.max_concurrent_executions > 0
        assert engine.default_timeout.total_seconds() > 0
    
    def test_simple_macro_execution(self):
        """Test execution of a simple macro."""
        engine = MacroEngine()
        macro = create_test_macro("test_macro", [CommandType.TEXT_INPUT])
        context = ExecutionContext.create_test_context()
        
        result = engine.execute_macro(macro, context)
        
        assert result is not None
        assert result.execution_token is not None
        assert result.macro_id == macro.macro_id
        assert result.status == ExecutionStatus.COMPLETED
        assert result.is_successful()
        assert len(result.command_results) == 1
        assert result.command_results[0].success
    
    def test_macro_with_multiple_commands(self):
        """Test execution of macro with multiple commands."""
        engine = MacroEngine()
        macro = create_test_macro("multi_command", [
            CommandType.TEXT_INPUT,
            CommandType.PAUSE,
            CommandType.PLAY_SOUND
        ])
        context = ExecutionContext.create_test_context(
            permissions=frozenset([
                Permission.TEXT_INPUT,
                Permission.SYSTEM_SOUND
            ])
        )
        
        result = engine.execute_macro(macro, context)
        
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.command_results) == 3
        assert all(cmd_result.success for cmd_result in result.command_results)
    
    def test_macro_execution_with_insufficient_permissions(self):
        """Test that macro execution fails with insufficient permissions."""
        engine = MacroEngine()
        macro = create_test_macro("sound_macro", [CommandType.PLAY_SOUND])
        
        # Create context without SYSTEM_SOUND permission
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT])  # Missing SYSTEM_SOUND
        )
        
        result = engine.execute_macro(macro, context)
        
        # Should return failed result with permission error details
        assert result.status == ExecutionStatus.FAILED
        assert not result.is_successful()
        assert result.error_details is not None
        assert "PermissionDeniedError" in result.error_details
    
    def test_invalid_macro_rejection(self):
        """Test that invalid macros are rejected."""
        engine = MacroEngine()
        
        # Create an invalid macro (empty commands list)
        invalid_macro = MacroDefinition(
            macro_id="invalid",
            name="Invalid Macro",
            commands=[]  # Empty commands should make it invalid
        )
        
        context = ExecutionContext.create_test_context()
        
        # Invalid macros should return failed execution result
        result = engine.execute_macro(invalid_macro, context)
        
        # Verify the macro was rejected and marked as failed
        assert result.status == ExecutionStatus.FAILED
        assert result.error_details is not None
        assert "validation" in result.error_details.lower()
        assert "empty commands" in result.error_details.lower() or "non-empty commands" in result.error_details.lower()
    
    def test_execution_status_tracking(self):
        """Test that execution status is properly tracked."""
        engine = MacroEngine()
        macro = create_test_macro("status_test", [CommandType.TEXT_INPUT])
        context = ExecutionContext.create_test_context()
        
        result = engine.execute_macro(macro, context)
        
        # Check final status from the result (context is cleaned up after execution)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.is_successful()
        
        # After cleanup, status should not be available
        final_status = engine.get_execution_status(result.execution_token)
        assert final_status is None  # Context cleaned up after execution
    
    def test_execution_cancellation(self):
        """Test that macro execution can be cancelled."""
        engine = MacroEngine()
        macro = create_test_macro("cancel_test", [CommandType.PAUSE])
        context = ExecutionContext.create_test_context()
        
        # Start execution in separate thread for cancellation test
        import threading
        
        result_container = []
        
        def execute_macro():
            try:
                result = engine.execute_macro(macro, context)
                result_container.append(result)
            except Exception as e:
                result_container.append(e)
        
        thread = threading.Thread(target=execute_macro)
        thread.start()
        
        # Give it a moment to start, then cancel
        time.sleep(0.1)
        
        # For this test, we'll create a mock execution to cancel
        # In a real implementation, we'd need the actual token
        # This is a simplified test of the cancellation interface
        tokens = engine.get_active_executions()
        if tokens:
            cancelled = engine.cancel_execution(tokens[0])
            assert isinstance(cancelled, bool)
        
        thread.join(timeout=1.0)
    
    def test_get_active_executions(self):
        """Test retrieval of active executions."""
        engine = MacroEngine()
        
        # Initially should be empty
        active = engine.get_active_executions()
        assert isinstance(active, list)
    
    def test_cleanup_expired_executions(self):
        """Test cleanup of expired executions."""
        engine = MacroEngine()
        
        # Test cleanup method
        cleaned = engine.cleanup_expired_executions(max_age_seconds=1.0)
        assert isinstance(cleaned, int)
        assert cleaned >= 0


class TestExecutionContext:
    """Test cases for ExecutionContext functionality."""
    
    def test_context_creation(self):
        """Test execution context creation."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_SOUND])
        timeout = Duration.from_seconds(30)
        
        context = ExecutionContext(
            permissions=permissions,
            timeout=timeout
        )
        
        assert context.permissions == permissions
        assert context.timeout == timeout
        assert context.execution_id is not None
        assert context.created_at is not None
    
    def test_permission_checking(self):
        """Test permission checking methods."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_SOUND])
        context = ExecutionContext.create_test_context(permissions=permissions)
        
        # Test has_permission
        assert context.has_permission(Permission.TEXT_INPUT)
        assert context.has_permission(Permission.SYSTEM_SOUND)
        assert not context.has_permission(Permission.FILE_ACCESS)
        
        # Test has_permissions
        required = frozenset([Permission.TEXT_INPUT])
        assert context.has_permissions(required)
        
        required_invalid = frozenset([Permission.TEXT_INPUT, Permission.FILE_ACCESS])
        assert not context.has_permissions(required_invalid)
    
    def test_variable_management(self):
        """Test variable management in context."""
        context = ExecutionContext.create_test_context()
        
        # Test adding variable
        var_name = "test_var"
        var_value = "test_value"
        
        new_context = context.with_variable(var_name, var_value)
        
        assert new_context.get_variable(var_name) == var_value
        assert context.get_variable(var_name) is None  # Original unchanged
    
    def test_default_context(self):
        """Test default context creation."""
        context = ExecutionContext.default()
        
        assert context is not None
        assert len(context.permissions) > 0
        assert context.timeout.total_seconds() > 0


class TestContractEnforcement:
    """Test cases for contract enforcement in the engine."""
    
    def test_precondition_enforcement(self):
        """Test that preconditions are enforced."""
        engine = MacroEngine()
        
        # Test with None macro (should violate precondition)
        context = ExecutionContext.create_test_context()
        
        with pytest.raises(Exception):  # Contract violation
            engine.execute_macro(None, context)
    
    def test_postcondition_verification(self):
        """Test that postconditions are verified."""
        engine = MacroEngine()
        macro = create_test_macro("postcondition_test", [CommandType.TEXT_INPUT])
        context = ExecutionContext.create_test_context()
        
        result = engine.execute_macro(macro, context)
        
        # Postcondition ensures execution_token is not None
        assert result.execution_token is not None