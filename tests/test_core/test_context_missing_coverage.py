"""Comprehensive tests to achieve 100% coverage of src/core/context.py.

This module provides targeted tests for uncovered lines in context.py to reach 95%+ coverage.
"""

import pytest
from src.core.context import (
    ExecutionContextManager,
    VariableManager,
    get_context_manager,
    get_variable_manager,
    security_context,
)
from src.core.types import (
    ExecutionContext,
    ExecutionStatus,
    ExecutionToken,
    Permission,
    VariableName,
)


class TestExecutionContextManager:
    """Test ExecutionContextManager comprehensive coverage."""

    def test_context_manager_initialization(self):
        """Test ExecutionContextManager initialization."""
        manager = ExecutionContextManager()
        assert manager._active_contexts == {}
        assert manager._context_status == {}
        assert manager._context_start_times == {}

    def test_register_context(self):
        """Test context registration."""
        manager = ExecutionContextManager()
        context = ExecutionContext.create_test_context()

        token = manager.register_context(context)

        assert isinstance(token, str)
        assert token in manager._active_contexts
        assert manager._active_contexts[token] == context
        assert manager.get_status(token) == ExecutionStatus.PENDING

    def test_register_context_with_variables(self):
        """Test context registration with variables."""
        manager = ExecutionContextManager()
        context = ExecutionContext.create_test_context()
        context = context.with_variable(VariableName("test_var"), "test_value")

        token = manager.register_context(context)

        assert token in manager._active_contexts
        assert (
            manager._active_contexts[token].get_variable(VariableName("test_var"))
            == "test_value"
        )

    def test_update_status(self):
        """Test status updates."""
        manager = ExecutionContextManager()
        context = ExecutionContext.create_test_context()
        token = manager.register_context(context)

        # Update to running
        manager.update_status(token, ExecutionStatus.RUNNING)
        assert manager.get_status(token) == ExecutionStatus.RUNNING

        # Update to completed
        manager.update_status(token, ExecutionStatus.COMPLETED)
        assert manager.get_status(token) == ExecutionStatus.COMPLETED

    def test_update_status_nonexistent_token(self):
        """Test updating status for nonexistent token."""
        manager = ExecutionContextManager()

        # Should not raise error for nonexistent token
        manager.update_status(ExecutionToken("nonexistent"), ExecutionStatus.RUNNING)

    def test_get_status_nonexistent_token(self):
        """Test getting status for nonexistent token."""
        manager = ExecutionContextManager()

        status = manager.get_status(ExecutionToken("nonexistent"))
        assert status is None

    def test_cleanup_context(self):
        """Test context cleanup."""
        manager = ExecutionContextManager()
        context = ExecutionContext.create_test_context()
        token = manager.register_context(context)

        # Verify context exists
        assert token in manager._active_contexts

        # Cleanup
        manager.cleanup_context(token)

        # Verify context removed
        assert token not in manager._active_contexts
        assert token not in manager._context_status

    def test_cleanup_nonexistent_context(self):
        """Test cleanup of nonexistent context."""
        manager = ExecutionContextManager()

        # Should not raise error for nonexistent token
        manager.cleanup_context(ExecutionToken("nonexistent"))

    def test_get_active_contexts(self):
        """Test getting active contexts."""
        manager = ExecutionContextManager()

        # No active contexts initially
        active = manager.get_active_contexts()
        assert active == []

        # Register some contexts
        context1 = ExecutionContext.create_test_context()
        context2 = ExecutionContext.create_test_context()
        token1 = manager.register_context(context1)
        token2 = manager.register_context(context2)

        # Both should be active
        active = manager.get_active_contexts()
        assert len(active) == 2
        assert token1 in active
        assert token2 in active

    def test_cleanup_expired_contexts(self):
        """Test cleanup of expired contexts."""
        manager = ExecutionContextManager()
        context = ExecutionContext.create_test_context()
        manager.register_context(context)

        # Cleanup with 1 hour max age (should not clean up newly created context)
        cleaned = manager.cleanup_expired_contexts(max_age_seconds=3600)
        assert cleaned == 0

        # Cleanup with very short max age (should clean up)
        cleaned = manager.cleanup_expired_contexts(max_age_seconds=0)
        assert cleaned == 1

    def test_get_context(self):
        """Test getting execution context."""
        manager = ExecutionContextManager()
        context = ExecutionContext.create_test_context()
        token = manager.register_context(context)

        retrieved_context = manager.get_context(token)
        assert retrieved_context == context

        # Test nonexistent context
        nonexistent_context = manager.get_context(ExecutionToken("nonexistent"))
        assert nonexistent_context is None


class TestVariableManager:
    """Test VariableManager comprehensive coverage."""

    def test_variable_manager_initialization(self):
        """Test VariableManager initialization."""
        manager = VariableManager()
        assert manager._global_variables == {}
        assert manager._context_variables == {}
        assert manager._protected_variables == set()

    def test_set_and_get_global_variable(self):
        """Test global variable setting and getting."""
        manager = VariableManager()

        # Set a variable
        manager.set_global_variable(VariableName("test_var"), "test_value")

        # Get the variable
        value = manager.get_global_variable(VariableName("test_var"))
        assert value == "test_value"

    def test_get_nonexistent_global_variable(self):
        """Test getting nonexistent global variable."""
        manager = VariableManager()

        value = manager.get_global_variable(VariableName("nonexistent"))
        assert value is None

    def test_protect_variable(self):
        """Test variable protection."""
        manager = VariableManager()
        var_name = VariableName("protected_var")

        # Set a variable first
        manager.set_global_variable(var_name, "initial_value")

        # Protect it
        manager.protect_variable(var_name)

        # Try to modify protected variable
        from src.core.errors import SecurityViolationError

        with pytest.raises(SecurityViolationError):
            manager.set_global_variable(var_name, "new_value")

    def test_set_context_variable(self):
        """Test setting context-specific variables."""
        manager = VariableManager()
        token = ExecutionToken("test-token")

        manager.set_context_variable(token, VariableName("ctx_var"), "ctx_value")

        # Context variables should be stored separately
        assert token in manager._context_variables
        assert manager._context_variables[token][VariableName("ctx_var")] == "ctx_value"

    def test_get_context_variable(self):
        """Test getting context-specific variables."""
        manager = VariableManager()
        token = ExecutionToken("test-token")

        # Set context variable
        manager.set_context_variable(token, VariableName("ctx_var"), "ctx_value")

        # Get context variable
        value = manager.get_context_variable(token, VariableName("ctx_var"))
        assert value == "ctx_value"

    def test_get_nonexistent_context_variable(self):
        """Test getting nonexistent context variable."""
        manager = VariableManager()
        token = ExecutionToken("test-token")

        value = manager.get_context_variable(token, VariableName("nonexistent"))
        assert value is None

    def test_cleanup_context_variables(self):
        """Test cleanup of context variables."""
        manager = VariableManager()
        token = ExecutionToken("test-token")

        # Set some context variables
        manager.set_context_variable(token, VariableName("var1"), "value1")
        manager.set_context_variable(token, VariableName("var2"), "value2")

        # Verify variables exist
        assert token in manager._context_variables

        # Cleanup
        manager.cleanup_context_variables(token)

        # Verify variables removed
        assert token not in manager._context_variables

    def test_cleanup_nonexistent_context_variables(self):
        """Test cleanup of nonexistent context variables."""
        manager = VariableManager()
        token = ExecutionToken("nonexistent")

        # Should not raise error
        manager.cleanup_context_variables(token)


class TestGlobalInstances:
    """Test global instance functions."""

    def test_get_context_manager(self):
        """Test get_context_manager function."""
        manager1 = get_context_manager()
        manager2 = get_context_manager()

        # Should return same instance (singleton)
        assert manager1 is manager2
        assert isinstance(manager1, ExecutionContextManager)

    def test_get_variable_manager(self):
        """Test get_variable_manager function."""
        manager1 = get_variable_manager()
        manager2 = get_variable_manager()

        # Should return same instance (singleton)
        assert manager1 is manager2
        assert isinstance(manager1, VariableManager)


class TestSecurityContext:
    """Test security_context function and context manager."""

    def test_security_context_success(self):
        """Test security context with sufficient permissions."""
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_CONTROL])
        )
        required_permissions = frozenset([Permission.TEXT_INPUT])

        # Should not raise exception
        with security_context(context, required_permissions):
            pass

    def test_security_context_missing_permissions(self):
        """Test security context with missing permissions."""
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT])
        )
        required_permissions = frozenset([Permission.SYSTEM_CONTROL])

        from src.core.errors import PermissionDeniedError

        # Should raise PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            with security_context(context, required_permissions):
                pass

    def test_security_context_partial_permissions(self):
        """Test security context with partial permissions."""
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT])
        )
        required_permissions = frozenset(
            [Permission.TEXT_INPUT, Permission.SYSTEM_CONTROL]
        )

        from src.core.errors import PermissionDeniedError

        # Should raise PermissionDeniedError for missing permission
        with pytest.raises(PermissionDeniedError):
            with security_context(context, required_permissions):
                pass

    def test_security_context_empty_required_permissions(self):
        """Test security context with no required permissions."""
        context = ExecutionContext.create_test_context()
        required_permissions = frozenset()

        # Should always succeed with empty requirements
        with security_context(context, required_permissions):
            pass

    def test_security_context_exception_propagation(self):
        """Test that exceptions inside security context are propagated."""
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT])
        )
        required_permissions = frozenset([Permission.TEXT_INPUT])

        class TestException(Exception):
            pass

        # Exception should be propagated
        with pytest.raises(TestException):
            with security_context(context, required_permissions):
                raise TestException("Test error")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_context_manager_with_invalid_token_format(self):
        """Test context manager with various token formats."""
        manager = ExecutionContextManager()

        # Test with empty string token
        status = manager.get_status(ExecutionToken(""))
        assert status is None

        # Test with very long token
        long_token = "a" * 1000
        status = manager.get_status(ExecutionToken(long_token))
        assert status is None

    def test_variable_manager_with_special_characters(self):
        """Test variable manager with special character variable names."""
        manager = VariableManager()

        # Test with special characters
        special_name = VariableName("var_with_special_chars_!@#$%")
        manager.set_global_variable(special_name, "special_value")

        value = manager.get_global_variable(special_name)
        assert value == "special_value"

    def test_context_manager_concurrent_operations(self):
        """Test context manager with simulated concurrent operations."""
        manager = ExecutionContextManager()

        # Register multiple contexts
        contexts = []
        tokens = []
        for _i in range(10):
            context = ExecutionContext.create_test_context()
            token = manager.register_context(context)
            contexts.append(context)
            tokens.append(token)

        # Update statuses
        for i, token in enumerate(tokens):
            status = (
                ExecutionStatus.RUNNING if i % 2 == 0 else ExecutionStatus.COMPLETED
            )
            manager.update_status(ExecutionToken(token), status)

        # Verify all contexts are tracked
        active = manager.get_active_contexts()
        assert len(active) == 10

    def test_variable_manager_large_values(self):
        """Test variable manager with large values."""
        manager = VariableManager()

        # Test with large string value
        large_value = "x" * 10000
        manager.set_global_variable(VariableName("large_var"), large_value)

        retrieved_value = manager.get_global_variable(VariableName("large_var"))
        assert retrieved_value == large_value
        assert len(retrieved_value) == 10000
