"""
Execution context management for the Keyboard Maestro MCP macro engine.

This module provides secure, isolated execution environments for macro operations
with comprehensive state management and security boundary enforcement.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Set, ContextManager, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time
from .types import (
    ExecutionContext, ExecutionToken, Permission, Duration, VariableName,
    ExecutionStatus, CommandResult
)
from .errors import (
    PermissionDeniedError, TimeoutError, SecurityViolationError,
    create_error_context
)
from .contracts import require, ensure


@dataclass
class SecurityBoundary:
    """Defines security constraints for execution context."""
    allowed_permissions: frozenset[Permission]
    max_execution_time: Duration
    max_memory_mb: int = 100
    allow_network_access: bool = False
    allow_file_system_access: bool = False
    sandbox_mode: bool = True
    
    def validate_permission(self, permission: Permission) -> bool:
        """Check if permission is allowed within this boundary."""
        return permission in self.allowed_permissions
    
    def validate_permissions(self, permissions: frozenset[Permission]) -> bool:
        """Check if all permissions are allowed within this boundary."""
        return permissions.issubset(self.allowed_permissions)


class ExecutionContextManager:
    """
    Manages execution contexts with security boundaries and resource tracking.
    
    This class provides thread-safe context management with comprehensive
    security enforcement and resource monitoring.
    """
    
    def __init__(self):
        self._active_contexts: Dict[ExecutionToken, ExecutionContext] = {}
        self._context_status: Dict[ExecutionToken, ExecutionStatus] = {}
        self._context_start_times: Dict[ExecutionToken, datetime] = {}
        self._context_threads: Dict[ExecutionToken, threading.Thread] = {}
        self._lock = threading.RLock()
    
    # @require(lambda self, context: context is not None, "context cannot be None")
    # @require(lambda self, context: isinstance(context, ExecutionContext), "must be ExecutionContext")
    @ensure(lambda self, context, result: result in self._active_contexts, "context must be registered")
    def register_context(self, context: ExecutionContext) -> ExecutionToken:
        """Register a new execution context."""
        with self._lock:
            token = context.execution_id
            self._active_contexts[token] = context
            self._context_status[token] = ExecutionStatus.PENDING
            self._context_start_times[token] = datetime.now()
            return token
    
    # @require(lambda self, token: token is not None, "token cannot be None")
    def get_context(self, token: ExecutionToken) -> Optional[ExecutionContext]:
        """Retrieve execution context by token."""
        with self._lock:
            return self._active_contexts.get(token)
    
    # @require(lambda self, token: token is not None, "token cannot be None")
    def get_status(self, token: ExecutionToken) -> Optional[ExecutionStatus]:
        """Get execution status for a context."""
        with self._lock:
            return self._context_status.get(token)
    
    # @require(lambda self, token: token is not None, "token cannot be None")
    # @require(lambda self, token: token in self._active_contexts, "context must exist")
    def update_status(self, token: ExecutionToken, status: ExecutionStatus) -> None:
        """Update execution status for a context."""
        with self._lock:
            if token in self._context_status:
                self._context_status[token] = status
    
    # @require(lambda self, token: token is not None, "token cannot be None")
    def cleanup_context(self, token: ExecutionToken) -> None:
        """Clean up a finished execution context."""
        with self._lock:
            self._active_contexts.pop(token, None)
            self._context_status.pop(token, None)
            self._context_start_times.pop(token, None)
            self._context_threads.pop(token, None)
    
    def get_active_contexts(self) -> List[ExecutionToken]:
        """Get list of all active execution contexts."""
        with self._lock:
            return list(self._active_contexts.keys())
    
    def cleanup_expired_contexts(self, max_age_seconds: float = 3600) -> int:
        """Clean up contexts older than max_age_seconds."""
        now = datetime.now()
        expired_tokens = []
        
        with self._lock:
            for token, start_time in self._context_start_times.items():
                age = (now - start_time).total_seconds()
                if age > max_age_seconds:
                    expired_tokens.append(token)
        
        for token in expired_tokens:
            self.cleanup_context(token)
        
        return len(expired_tokens)


class SecurityContextManager:
    """
    Manages security boundaries and permission validation.
    
    Provides secure execution environments with comprehensive permission
    checking and security violation detection.
    """
    
    @staticmethod
    @require(lambda context, permissions: context is not None, "context cannot be None")
    @require(lambda context, permissions: permissions is not None, "permissions cannot be None")
    def validate_permissions(
        context: ExecutionContext,
        required_permissions: frozenset[Permission]
    ) -> None:
        """Validate that context has required permissions."""
        if not context.has_permissions(required_permissions):
            missing = required_permissions - context.permissions
            error_context = create_error_context(
                operation="permission_validation",
                component="security_context_manager",
                required=list(required_permissions),
                available=list(context.permissions),
                missing=list(missing)
            )
            raise PermissionDeniedError(
                required_permissions=list(required_permissions),
                available_permissions=list(context.permissions),
                context=error_context
            )
    
    @staticmethod
    @require(lambda operation, max_duration: max_duration.seconds > 0, "duration must be positive")
    def validate_timeout(operation: str, max_duration: Duration) -> None:
        """Validate operation timeout constraints."""
        # Implementation would include actual timeout checking logic
        # For now, this is a placeholder for the contract structure
        pass
    
    @staticmethod
    def create_security_boundary(
        permissions: frozenset[Permission],
        max_time: Duration,
        sandbox: bool = True
    ) -> SecurityBoundary:
        """Create a security boundary with specified constraints."""
        return SecurityBoundary(
            allowed_permissions=permissions,
            max_execution_time=max_time,
            sandbox_mode=sandbox
        )


@contextmanager
def security_context(
    context: ExecutionContext,
    required_permissions: frozenset[Permission]
) -> ContextManager[ExecutionContext]:
    """
    Context manager for secure execution with permission validation.
    
    Args:
        context: Execution context to validate
        required_permissions: Permissions required for the operation
        
    Yields:
        The validated execution context
        
    Raises:
        PermissionDeniedError: If required permissions are not available
        SecurityViolationError: If security constraints are violated
    """
    # Validate permissions before entering context
    SecurityContextManager.validate_permissions(context, required_permissions)
    
    start_time = time.time()
    
    try:
        yield context
    except Exception as e:
        # Log security-relevant errors
        execution_time = time.time() - start_time
        error_context = create_error_context(
            operation="security_context_execution",
            component="security_context_manager",
            execution_time=execution_time,
            error_type=type(e).__name__
        )
        
        if execution_time > context.timeout.total_seconds():
            raise TimeoutError(
                operation="security_context",
                timeout_seconds=context.timeout.total_seconds(),
                context=error_context
            )
        
        # Re-raise the original exception
        raise
    finally:
        # Cleanup logic would go here
        pass


class VariableManager:
    """
    Manages variables within execution contexts with security controls.
    
    Provides secure variable storage and retrieval with access control
    and audit logging for sensitive operations.
    """
    
    def __init__(self):
        self._global_variables: Dict[VariableName, str] = {}
        self._context_variables: Dict[ExecutionToken, Dict[VariableName, str]] = {}
        self._protected_variables: Set[VariableName] = set()
        self._lock = threading.RLock()
    
    @require(lambda self, name, value: name is not None, "variable name cannot be None")
    @require(lambda self, name, value: value is not None, "variable value cannot be None")
    def set_global_variable(self, name: VariableName, value: str) -> None:
        """Set a global variable value."""
        with self._lock:
            if name in self._protected_variables:
                raise SecurityViolationError(
                    violation_type="protected_variable_modification",
                    details=f"Attempt to modify protected variable: {name}"
                )
            self._global_variables[name] = value
    
    @require(lambda self, name: name is not None, "variable name cannot be None")
    def get_global_variable(self, name: VariableName) -> Optional[str]:
        """Get a global variable value."""
        with self._lock:
            return self._global_variables.get(name)
    
    # @require(lambda self, token: token is not None, "token cannot be None")
    @require(lambda self, token, name, value: name is not None, "variable name cannot be None")
    @require(lambda self, token, name, value: value is not None, "variable value cannot be None")
    def set_context_variable(
        self,
        token: ExecutionToken,
        name: VariableName,
        value: str
    ) -> None:
        """Set a context-specific variable."""
        with self._lock:
            if token not in self._context_variables:
                self._context_variables[token] = {}
            self._context_variables[token][name] = value
    
    # @require(lambda self, token: token is not None, "token cannot be None")
    @require(lambda self, token, name: name is not None, "variable name cannot be None")
    def get_context_variable(
        self,
        token: ExecutionToken,
        name: VariableName
    ) -> Optional[str]:
        """Get a context-specific variable value."""
        with self._lock:
            context_vars = self._context_variables.get(token, {})
            return context_vars.get(name)
    
    @require(lambda self, name: name is not None, "variable name cannot be None")
    def protect_variable(self, name: VariableName) -> None:
        """Mark a variable as protected from modification."""
        with self._lock:
            self._protected_variables.add(name)
    
    # @require(lambda self, token: token is not None, "token cannot be None")
    def cleanup_context_variables(self, token: ExecutionToken) -> None:
        """Clean up variables for a finished context."""
        with self._lock:
            self._context_variables.pop(token, None)


# Global instances for context and variable management
_context_manager = ExecutionContextManager()
_variable_manager = VariableManager()


def get_context_manager() -> ExecutionContextManager:
    """Get the global context manager instance."""
    return _context_manager


def get_variable_manager() -> VariableManager:
    """Get the global variable manager instance."""
    return _variable_manager