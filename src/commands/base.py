"""
Base Command Contracts and Implementation

Provides the foundational contracts and base implementation for all macro commands
with design by contract validation, security boundaries, and type safety.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic, Optional, Dict, Any, FrozenSet
from dataclasses import dataclass, field
from datetime import datetime
import time

from ..core.types import (
    CommandId, ExecutionContext, CommandResult, CommandParameters, 
    Permission, Duration
)
from ..core.contracts import require, ensure
from ..core.errors import SecurityViolationError, ValidationError
from ..core.context import security_context

# Constants for security and performance limits
MAX_COMMAND_DURATION = Duration.from_seconds(300)  # 5 minutes max
MAX_TEXT_LENGTH = 10000  # Maximum text input length
MAX_LOOP_ITERATIONS = 1000  # Prevent infinite loops


class CommandContract(Protocol):
    """
    Protocol defining the contract for all macro commands.
    
    All command implementations must satisfy this contract with:
    - Pre/post condition validation
    - Permission checking
    - Security validation
    - Type safety
    """
    
    def execute(self, context: ExecutionContext) -> CommandResult:
        """
        Execute the command in the given execution context.
        
        Preconditions:
        - context must be valid with required permissions
        - command must pass validation
        
        Postconditions:
        - result indicates success or provides error information
        - execution completes within timeout limits
        """
        ...
    
    def validate(self) -> bool:
        """
        Validate command parameters and configuration.
        
        Returns:
            True if command is valid and safe to execute
        """
        ...
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """
        Get the set of permissions required to execute this command.
        
        Returns:
            Frozen set of required permissions
        """
        ...
    
    def get_security_risk_level(self) -> str:
        """
        Get the security risk level of this command.
        
        Returns:
            One of: 'low', 'medium', 'high', 'critical'
        """
        ...


@dataclass(frozen=True)
class BaseCommand(ABC):
    """
    Base implementation for all macro commands.
    
    Provides common functionality including:
    - Contract enforcement
    - Security validation
    - Performance monitoring
    - Error handling
    """
    command_id: CommandId
    parameters: CommandParameters
    created_at: datetime = field(default_factory=datetime.now)
    
    @abstractmethod
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """
        Implementation-specific execution logic.
        
        This method is called by execute() after all validation
        and security checks have passed.
        """
        pass
    
    @abstractmethod
    def _validate_impl(self) -> bool:
        """
        Implementation-specific validation logic.
        
        Should validate parameters specific to this command type.
        """
        pass
    
    @abstractmethod
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Get required permissions for this command."""
        pass
    
    def get_security_risk_level(self) -> str:
        """Default to medium risk - override for specific commands."""
        return "medium"
    
    # @require(lambda self, context: self.validate() and context.has_permissions(self.get_required_permissions()))
    # @ensure(lambda result: result.execution_time is None or result.execution_time <= MAX_COMMAND_DURATION)
    def execute(self, context: ExecutionContext) -> CommandResult:
        """
        Execute command with full contract enforcement.
        
        Handles validation, security checks, performance monitoring,
        and error recovery.
        """
        start_time = time.time()
        
        try:
            # Validate command before execution
            if not self.validate():
                return CommandResult(
                    success=False,
                    error_message="Command validation failed",
                    execution_time=Duration.from_seconds(time.time() - start_time)
                )
            
            # Check permissions
            required_permissions = self.get_required_permissions()
            if not context.has_permissions(required_permissions):
                missing = required_permissions - context.permissions
                return CommandResult(
                    success=False,
                    error_message=f"Missing required permissions: {missing}",
                    execution_time=Duration.from_seconds(time.time() - start_time)
                )
            
            # Execute within security context
            with security_context(context, required_permissions):
                result = self._execute_impl(context)
                
                # Add execution time if not already set
                if result.execution_time is None:
                    execution_time = Duration.from_seconds(time.time() - start_time)
                    result = CommandResult(
                        success=result.success,
                        output=result.output,
                        error_message=result.error_message,
                        execution_time=execution_time,
                        metadata=result.metadata
                    )
                
                return result
                
        except SecurityViolationError as e:
            return CommandResult(
                success=False,
                error_message=f"Security violation: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
        except Exception as e:
            return CommandResult(
                success=False,
                error_message=f"Command execution failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def validate(self) -> bool:
        """
        Validate command with security checks.
        
        Performs both base validation and implementation-specific validation.
        """
        try:
            # Base validation
            if not self.command_id:
                return False
            
            if not isinstance(self.parameters, CommandParameters):
                return False
            
            # Implementation-specific validation
            return self._validate_impl()
            
        except Exception:
            return False


@dataclass(frozen=True)
class NoOpCommand(BaseCommand):
    """
    No-operation command for testing and placeholders.
    
    Always succeeds and performs no actions.
    Safe to execute with minimal permissions.
    """
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute no-op - always succeeds."""
        return CommandResult(
            success=True,
            output="No operation performed",
            metadata={"command_type": "no_op"}
        )
    
    def _validate_impl(self) -> bool:
        """No-op is always valid."""
        return True
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """No-op requires no permissions."""
        return frozenset()
    
    def get_security_risk_level(self) -> str:
        """No-op has no security risk."""
        return "low"


# Utility functions for command validation

def is_safe_text_content(text: str) -> bool:
    """
    Check if text content is safe for execution.
    
    Validates against script injection, system commands,
    and other security threats.
    """
    if len(text) > MAX_TEXT_LENGTH:
        return False
    
    # Check for script injection patterns
    dangerous_patterns = [
        '<script', 'javascript:', 'vbscript:', 
        'eval(', 'exec(', 'system(', 'os.system',
        '$(', '`', 'rm -rf', 'del /f'
    ]
    
    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if pattern in text_lower:
            return False
    
    return True


def is_valid_duration(duration: Duration) -> bool:
    """Check if duration is within safe limits."""
    return Duration.ZERO < duration <= MAX_COMMAND_DURATION


def create_command_result(
    success: bool,
    output: Optional[str] = None,
    error_message: Optional[str] = None,
    **metadata
) -> CommandResult:
    """
    Helper function to create standardized command results.
    
    Args:
        success: Whether command succeeded
        output: Optional output text
        error_message: Optional error message
        **metadata: Additional metadata fields
    
    Returns:
        Properly formatted CommandResult
    """
    return CommandResult(
        success=success,
        output=output,
        error_message=error_message,
        metadata=metadata
    )