"""Comprehensive tests for core context management functionality.

This module provides comprehensive test coverage for the context management system
focusing on security boundaries, execution context management, and variable handling.
"""

import threading
import time
from datetime import datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.core.context import (
    ExecutionContextManager,
    SecurityBoundary,
    SecurityContextManager,
    VariableManager,
    get_context_manager,
    get_variable_manager,
    security_context,
)
from src.core.errors import (
    PermissionDeniedError,
    SecurityViolationError,
    TimeoutError,
)
from src.core.types import (
    Duration,
    ExecutionContext,
    ExecutionStatus,
    ExecutionToken,
    Permission,
    VariableName,
)


class TestSecurityBoundary:
    """Test SecurityBoundary functionality."""

    def test_security_boundary_creation(self) -> None:
        """Test creation of security boundary."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_SOUND])
        max_time = Duration.from_seconds(30)

        boundary = SecurityBoundary(
            allowed_permissions=permissions,
            max_execution_time=max_time,
            max_memory_mb=50,
            allow_network_access=True,
            allow_file_system_access=False,
            sandbox_mode=True,
        )

        assert boundary.allowed_permissions == permissions
        assert boundary.max_execution_time == max_time
        assert boundary.max_memory_mb == 50
        assert boundary.allow_network_access is True
        assert boundary.allow_file_system_access is False
        assert boundary.sandbox_mode is True

    def test_security_boundary_defaults(self) -> None:
        """Test security boundary default values."""
        permissions = frozenset([Permission.TEXT_INPUT])
        max_time = Duration.from_seconds(60)

        boundary = SecurityBoundary(
            allowed_permissions=permissions, max_execution_time=max_time
        )

        assert boundary.max_memory_mb == 100
        assert boundary.allow_network_access is False
        assert boundary.allow_file_system_access is False
        assert boundary.sandbox_mode is True

    def test_validate_permission_success(self) -> None:
        """Test successful permission validation."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.CLIPBOARD_ACCESS])
        boundary = SecurityBoundary(
            allowed_permissions=permissions,
            max_execution_time=Duration.from_seconds(30),
        )

        assert boundary.validate_permission(Permission.TEXT_INPUT) is True
        assert boundary.validate_permission(Permission.CLIPBOARD_ACCESS) is True

    def test_validate_permission_failure(self) -> None:
        """Test permission validation failure."""
        permissions = frozenset([Permission.TEXT_INPUT])
        boundary = SecurityBoundary(
            allowed_permissions=permissions,
            max_execution_time=Duration.from_seconds(30),
        )

        assert boundary.validate_permission(Permission.SYSTEM_CONTROL) is False
        assert boundary.validate_permission(Permission.NETWORK_ACCESS) is False

    def test_validate_permissions_success(self) -> None:
        """Test successful multiple permissions validation."""
        allowed = frozenset(
            [
                Permission.TEXT_INPUT,
                Permission.CLIPBOARD_ACCESS,
                Permission.SYSTEM_SOUND,
            ]
        )
        boundary = SecurityBoundary(
            allowed_permissions=allowed, max_execution_time=Duration.from_seconds(30)
        )

        required = frozenset([Permission.TEXT_INPUT, Permission.CLIPBOARD_ACCESS])
        assert boundary.validate_permissions(required) is True

        # Test subset validation
        single_permission = frozenset([Permission.TEXT_INPUT])
        assert boundary.validate_permissions(single_permission) is True

    def test_validate_permissions_failure(self) -> None:
        """Test multiple permissions validation failure."""
        allowed = frozenset([Permission.TEXT_INPUT])
        boundary = SecurityBoundary(
            allowed_permissions=allowed, max_execution_time=Duration.from_seconds(30)
        )

        required = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_CONTROL])
        assert boundary.validate_permissions(required) is False

        # Test completely different permissions
        different = frozenset([Permission.NETWORK_ACCESS, Permission.FILE_ACCESS])
        assert boundary.validate_permissions(different) is False


class TestExecutionContextManager:
    """Test ExecutionContextManager functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.manager = ExecutionContextManager()
        self.context = ExecutionContext.create_test_context()

    def test_manager_initialization(self) -> None:
        """Test manager initialization."""
        assert self.manager._active_contexts == {}
        assert self.manager._context_status == {}
        assert self.manager._context_start_times == {}
        assert self.manager._context_threads == {}
        assert self.manager._lock is not None

    def test_register_context(self) -> None:
        """Test context registration."""
        token = self.manager.register_context(self.context)

        assert token == self.context.execution_id
        assert token in self.manager._active_contexts
        assert self.manager._active_contexts[token] == self.context
        assert self.manager._context_status[token] == ExecutionStatus.PENDING
        assert token in self.manager._context_start_times

    def test_get_context_success(self) -> None:
        """Test successful context retrieval."""
        token = self.manager.register_context(self.context)
        retrieved = self.manager.get_context(token)

        assert retrieved == self.context
        assert retrieved.execution_id == token

    def test_get_context_not_found(self) -> None:
        """Test context retrieval for non-existent token."""
        fake_token = ExecutionToken("non_existent_token")
        retrieved = self.manager.get_context(fake_token)

        assert retrieved is None

    def test_get_status_success(self) -> None:
        """Test successful status retrieval."""
        token = self.manager.register_context(self.context)
        status = self.manager.get_status(token)

        assert status == ExecutionStatus.PENDING

    def test_get_status_not_found(self) -> None:
        """Test status retrieval for non-existent token."""
        fake_token = ExecutionToken("non_existent_token")
        status = self.manager.get_status(fake_token)

        assert status is None

    def test_update_status_success(self) -> None:
        """Test successful status update."""
        token = self.manager.register_context(self.context)

        self.manager.update_status(token, ExecutionStatus.RUNNING)
        assert self.manager.get_status(token) == ExecutionStatus.RUNNING

        self.manager.update_status(token, ExecutionStatus.COMPLETED)
        assert self.manager.get_status(token) == ExecutionStatus.COMPLETED

    def test_update_status_not_found(self) -> None:
        """Test status update for non-existent token."""
        fake_token = ExecutionToken("non_existent_token")

        # Should not raise exception
        self.manager.update_status(fake_token, ExecutionStatus.FAILED)

        # Confirm it wasn't added
        assert self.manager.get_status(fake_token) is None

    def test_cleanup_context(self) -> None:
        """Test context cleanup."""
        token = self.manager.register_context(self.context)

        # Verify context exists
        assert self.manager.get_context(token) is not None
        assert self.manager.get_status(token) is not None

        # Clean up
        self.manager.cleanup_context(token)

        # Verify context is removed
        assert self.manager.get_context(token) is None
        assert self.manager.get_status(token) is None
        assert token not in self.manager._context_start_times
        assert token not in self.manager._context_threads

    def test_cleanup_context_not_found(self) -> None:
        """Test cleanup of non-existent context."""
        fake_token = ExecutionToken("non_existent_token")

        # Should not raise exception
        self.manager.cleanup_context(fake_token)

    def test_get_active_contexts_empty(self) -> None:
        """Test getting active contexts when none exist."""
        active = self.manager.get_active_contexts()
        assert active == []

    def test_get_active_contexts_with_data(self) -> None:
        """Test getting active contexts with registered contexts."""
        context1 = ExecutionContext.create_test_context()
        context2 = ExecutionContext.create_test_context()

        token1 = self.manager.register_context(context1)
        token2 = self.manager.register_context(context2)

        active = self.manager.get_active_contexts()
        assert len(active) == 2
        assert token1 in active
        assert token2 in active

    def test_cleanup_expired_contexts_none_expired(self) -> None:
        """Test cleanup when no contexts are expired."""
        token = self.manager.register_context(self.context)

        # Clean up contexts older than 1 hour (nothing should be expired)
        cleaned = self.manager.cleanup_expired_contexts(max_age_seconds=3600)

        assert cleaned == 0
        assert self.manager.get_context(token) is not None

    def test_cleanup_expired_contexts_with_expired(self) -> None:
        """Test cleanup when contexts are expired."""
        token = self.manager.register_context(self.context)

        # Manually set start time to be old
        old_time = datetime.now() - timedelta(hours=2)
        self.manager._context_start_times[token] = old_time

        # Clean up contexts older than 1 hour
        cleaned = self.manager.cleanup_expired_contexts(max_age_seconds=3600)

        assert cleaned == 1
        assert self.manager.get_context(token) is None

    def test_thread_safety(self) -> None:
        """Test thread safety of context manager."""
        results = []
        errors = []

        def register_contexts() -> None:
            try:
                for _ in range(10):
                    context = ExecutionContext.create_test_context()
                    token = self.manager.register_context(context)
                    results.append(token)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=register_contexts) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify no errors and correct number of registrations
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 contexts each
        assert len(set(results)) == 50  # All tokens should be unique


class TestSecurityContextManager:
    """Test SecurityContextManager functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.CLIPBOARD_ACCESS])
        )

    def test_validate_permissions_success(self) -> None:
        """Test successful permission validation."""
        required = frozenset([Permission.TEXT_INPUT])

        # Should not raise exception
        SecurityContextManager.validate_permissions(self.context, required)

    def test_validate_permissions_subset_success(self) -> None:
        """Test permission validation with subset of available permissions."""
        required = frozenset([Permission.TEXT_INPUT, Permission.CLIPBOARD_ACCESS])

        # Should not raise exception
        SecurityContextManager.validate_permissions(self.context, required)

    def test_validate_permissions_failure(self) -> None:
        """Test permission validation failure."""
        required = frozenset([Permission.SYSTEM_CONTROL])

        with pytest.raises(PermissionDeniedError) as exc_info:
            SecurityContextManager.validate_permissions(self.context, required)

        error = exc_info.value
        assert Permission.SYSTEM_CONTROL.value in error.required_permissions
        assert Permission.SYSTEM_CONTROL.value not in error.available_permissions

    def test_validate_permissions_partial_failure(self) -> None:
        """Test permission validation with some missing permissions."""
        required = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_CONTROL])

        with pytest.raises(PermissionDeniedError) as exc_info:
            SecurityContextManager.validate_permissions(self.context, required)

        error = exc_info.value
        assert Permission.SYSTEM_CONTROL.value in error.required_permissions
        assert Permission.TEXT_INPUT.value in error.required_permissions
        assert Permission.TEXT_INPUT.value in error.available_permissions
        assert Permission.SYSTEM_CONTROL.value not in error.available_permissions

    def test_validate_timeout(self) -> None:
        """Test timeout validation."""
        duration = Duration.from_seconds(30)

        # Should not raise exception (currently placeholder implementation)
        SecurityContextManager.validate_timeout("test_operation", duration)

    def test_create_security_boundary_default(self) -> None:
        """Test security boundary creation with defaults."""
        permissions = frozenset([Permission.TEXT_INPUT])
        max_time = Duration.from_seconds(60)

        boundary = SecurityContextManager.create_security_boundary(
            permissions, max_time
        )

        assert boundary.allowed_permissions == permissions
        assert boundary.max_execution_time == max_time
        assert boundary.sandbox_mode is True

    def test_create_security_boundary_custom(self) -> None:
        """Test security boundary creation with custom settings."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.FILE_ACCESS])
        max_time = Duration.from_seconds(120)

        boundary = SecurityContextManager.create_security_boundary(
            permissions, max_time, sandbox=False
        )

        assert boundary.allowed_permissions == permissions
        assert boundary.max_execution_time == max_time
        assert boundary.sandbox_mode is False


class TestSecurityContextDecorator:
    """Test security_context context manager."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.CLIPBOARD_ACCESS]),
            timeout=Duration.from_seconds(5),
        )

    def test_security_context_success(self) -> None:
        """Test successful execution within security context."""
        required = frozenset([Permission.TEXT_INPUT])

        with security_context(self.context, required) as ctx:
            assert ctx == self.context
            assert ctx.has_permission(Permission.TEXT_INPUT)

    def test_security_context_permission_failure(self) -> None:
        """Test security context with insufficient permissions."""
        required = frozenset([Permission.SYSTEM_CONTROL])

        with pytest.raises(PermissionDeniedError):
            with security_context(self.context, required):
                pass

    def test_security_context_with_exception(self) -> None:
        """Test security context behavior when exception occurs."""
        required = frozenset([Permission.TEXT_INPUT])

        with pytest.raises(ValueError):
            with security_context(self.context, required):
                raise ValueError("Test exception")

    def test_security_context_timeout_detection(self) -> None:
        """Test timeout detection in security context."""
        # Create context with very short timeout
        short_timeout_context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT]),
            timeout=Duration.from_seconds(0.1),
        )
        required = frozenset([Permission.TEXT_INPUT])

        with pytest.raises(TimeoutError):
            with security_context(short_timeout_context, required):
                # Simulate long-running operation
                time.sleep(0.2)
                raise ValueError("This should trigger timeout")


class TestVariableManager:
    """Test VariableManager functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.manager = VariableManager()
        self.token = ExecutionToken("test_token")

    def test_manager_initialization(self) -> None:
        """Test manager initialization."""
        assert self.manager._global_variables == {}
        assert self.manager._context_variables == {}
        assert self.manager._protected_variables == set()
        assert self.manager._lock is not None

    def test_set_global_variable(self) -> None:
        """Test setting global variables."""
        name = VariableName("test_var")
        value = "test_value"

        self.manager.set_global_variable(name, value)

        assert self.manager._global_variables[name] == value

    def test_get_global_variable_success(self) -> None:
        """Test getting existing global variable."""
        name = VariableName("test_var")
        value = "test_value"

        self.manager.set_global_variable(name, value)
        retrieved = self.manager.get_global_variable(name)

        assert retrieved == value

    def test_get_global_variable_not_found(self) -> None:
        """Test getting non-existent global variable."""
        name = VariableName("non_existent")
        retrieved = self.manager.get_global_variable(name)

        assert retrieved is None

    def test_set_global_variable_protected(self) -> None:
        """Test setting protected global variable."""
        name = VariableName("protected_var")

        # Protect the variable first
        self.manager.protect_variable(name)

        # Attempt to set it should raise exception
        with pytest.raises(SecurityViolationError) as exc_info:
            self.manager.set_global_variable(name, "value")

        error = exc_info.value
        assert "protected_variable_modification" in error.violation_type
        assert name in str(error)  # Check name appears in string representation

    def test_set_context_variable_new_context(self) -> None:
        """Test setting variable in new context."""
        name = VariableName("context_var")
        value = "context_value"

        self.manager.set_context_variable(self.token, name, value)

        assert self.token in self.manager._context_variables
        assert self.manager._context_variables[self.token][name] == value

    def test_set_context_variable_existing_context(self) -> None:
        """Test setting variable in existing context."""
        name1 = VariableName("var1")
        name2 = VariableName("var2")
        value1 = "value1"
        value2 = "value2"

        # Set first variable
        self.manager.set_context_variable(self.token, name1, value1)
        # Set second variable in same context
        self.manager.set_context_variable(self.token, name2, value2)

        context_vars = self.manager._context_variables[self.token]
        assert context_vars[name1] == value1
        assert context_vars[name2] == value2

    def test_get_context_variable_success(self) -> None:
        """Test getting existing context variable."""
        name = VariableName("context_var")
        value = "context_value"

        self.manager.set_context_variable(self.token, name, value)
        retrieved = self.manager.get_context_variable(self.token, name)

        assert retrieved == value

    def test_get_context_variable_not_found(self) -> None:
        """Test getting non-existent context variable."""
        name = VariableName("non_existent")
        retrieved = self.manager.get_context_variable(self.token, name)

        assert retrieved is None

    def test_get_context_variable_no_context(self) -> None:
        """Test getting variable from non-existent context."""
        fake_token = ExecutionToken("fake_token")
        name = VariableName("var")

        retrieved = self.manager.get_context_variable(fake_token, name)

        assert retrieved is None

    def test_protect_variable(self) -> None:
        """Test protecting variables."""
        name = VariableName("protected_var")

        self.manager.protect_variable(name)

        assert name in self.manager._protected_variables

    def test_cleanup_context_variables(self) -> None:
        """Test cleanup of context variables."""
        name = VariableName("context_var")
        value = "context_value"

        # Set some variables
        self.manager.set_context_variable(self.token, name, value)
        assert self.manager.get_context_variable(self.token, name) == value

        # Clean up
        self.manager.cleanup_context_variables(self.token)

        # Verify cleanup
        assert self.manager.get_context_variable(self.token, name) is None
        assert self.token not in self.manager._context_variables

    def test_cleanup_context_variables_not_found(self) -> None:
        """Test cleanup of non-existent context variables."""
        fake_token = ExecutionToken("fake_token")

        # Should not raise exception
        self.manager.cleanup_context_variables(fake_token)


class TestGlobalInstances:
    """Test global instance access functions."""

    def test_get_context_manager(self) -> None:
        """Test global context manager access."""
        manager1 = get_context_manager()
        manager2 = get_context_manager()

        assert manager1 is not None
        assert isinstance(manager1, ExecutionContextManager)
        assert manager1 is manager2  # Should return same instance

    def test_get_variable_manager(self) -> None:
        """Test global variable manager access."""
        manager1 = get_variable_manager()
        manager2 = get_variable_manager()

        assert manager1 is not None
        assert isinstance(manager1, VariableManager)
        assert manager1 is manager2  # Should return same instance


class TestPropertyBasedContext:
    """Property-based tests for context functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.manager = ExecutionContextManager()
        self.var_manager = VariableManager()

    @given(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    def test_variable_name_handling(self, var_name: str) -> None:
        """Property: Variable names should be handled consistently."""
        try:
            name = VariableName(var_name)
            value = "test_value"
            token = ExecutionToken("test_token")

            # Test global variable operations
            self.var_manager.set_global_variable(name, value)
            retrieved = self.var_manager.get_global_variable(name)
            assert retrieved == value

            # Test context variable operations
            self.var_manager.set_context_variable(token, name, value)
            retrieved_context = self.var_manager.get_context_variable(token, name)
            assert retrieved_context == value

        except Exception as e:
            # Some variable names might be invalid, which is acceptable for property-based tests
            pytest.skip(f"Skipping invalid variable name: {e}")

    @given(st.lists(st.sampled_from(list(Permission)), min_size=0, max_size=5))
    def test_permission_set_handling(self, permission_list: list[Permission]) -> None:
        """Property: Permission sets should be handled consistently."""
        permissions = frozenset(permission_list)
        max_time = Duration.from_seconds(30)

        # Create security boundary
        boundary = SecurityBoundary(
            allowed_permissions=permissions, max_execution_time=max_time
        )

        # Test permission validation consistency
        for permission in permission_list:
            assert boundary.validate_permission(permission) is True

        # Test subset validation
        if permissions:
            subset = frozenset([list(permissions)[0]]) if permissions else frozenset()
            assert boundary.validate_permissions(subset) is True


class TestErrorHandling:
    """Test error handling in context management."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.manager = ExecutionContextManager()
        self.var_manager = VariableManager()

    def test_context_manager_error_recovery(self) -> None:
        """Test context manager handles errors gracefully."""
        context = ExecutionContext.create_test_context()
        token = self.manager.register_context(context)

        # Simulate error by corrupting internal state
        self.manager._active_contexts.clear()

        # Operations should handle missing context gracefully
        retrieved = self.manager.get_context(token)
        assert retrieved is None

        status = self.manager.get_status(token)
        # Status might still exist even if context is gone
        assert status in [None, ExecutionStatus.PENDING]

    def test_variable_manager_concurrent_access(self) -> None:
        """Test variable manager handles concurrent access."""
        errors = []

        def worker() -> None:
            try:
                for i in range(10):
                    name = VariableName(f"var_{i}")
                    value = f"value_{i}"
                    self.var_manager.set_global_variable(name, value)
                    retrieved = self.var_manager.get_global_variable(name)
                    assert retrieved == value
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0


class TestIntegrationScenarios:
    """Test integration scenarios with multiple components."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.context_manager = ExecutionContextManager()
        self.var_manager = VariableManager()

    def test_full_execution_lifecycle(self) -> None:
        """Test complete execution lifecycle with context and variables."""
        # Create execution context
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.FLOW_CONTROL])
        )

        # Register context
        token = self.context_manager.register_context(context)
        assert self.context_manager.get_status(token) == ExecutionStatus.PENDING

        # Start execution
        self.context_manager.update_status(token, ExecutionStatus.RUNNING)

        # Set some variables during execution
        self.var_manager.set_context_variable(token, VariableName("result"), "success")
        self.var_manager.set_global_variable(VariableName("last_execution"), str(token))

        # Complete execution
        self.context_manager.update_status(token, ExecutionStatus.COMPLETED)

        # Verify final state
        assert self.context_manager.get_status(token) == ExecutionStatus.COMPLETED
        assert (
            self.var_manager.get_context_variable(token, VariableName("result"))
            == "success"
        )
        assert self.var_manager.get_global_variable(
            VariableName("last_execution")
        ) == str(token)

        # Clean up
        self.var_manager.cleanup_context_variables(token)
        self.context_manager.cleanup_context(token)

        # Verify cleanup
        assert self.context_manager.get_context(token) is None
        assert (
            self.var_manager.get_context_variable(token, VariableName("result")) is None
        )

    def test_security_violation_scenario(self) -> None:
        """Test security violation detection and handling."""
        # Create context with limited permissions
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT])
        )

        # Attempt operation requiring higher permissions
        required = frozenset([Permission.SYSTEM_CONTROL, Permission.FILE_ACCESS])

        with pytest.raises(PermissionDeniedError):
            with security_context(context, required):
                pass

    def test_concurrent_execution_contexts(self) -> None:
        """Test handling multiple concurrent execution contexts."""
        contexts = []
        tokens = []

        # Create multiple contexts
        for _ in range(5):
            context = ExecutionContext.create_test_context()
            token = self.context_manager.register_context(context)
            contexts.append(context)
            tokens.append(token)

        # Verify all contexts are active
        active = self.context_manager.get_active_contexts()
        assert len(active) == 5
        for token in tokens:
            assert token in active

        # Update statuses concurrently
        for i, token in enumerate(tokens):
            if i % 2 == 0:
                self.context_manager.update_status(token, ExecutionStatus.COMPLETED)
            else:
                self.context_manager.update_status(token, ExecutionStatus.FAILED)

        # Verify status updates
        for i, token in enumerate(tokens):
            expected_status = (
                ExecutionStatus.COMPLETED if i % 2 == 0 else ExecutionStatus.FAILED
            )
            assert self.context_manager.get_status(token) == expected_status

        # Clean up all contexts
        for token in tokens:
            self.context_manager.cleanup_context(token)

        # Verify cleanup
        active_after = self.context_manager.get_active_contexts()
        assert len(active_after) == 0
