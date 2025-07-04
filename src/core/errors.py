"""
Comprehensive error hierarchy for the Keyboard Maestro MCP macro engine.

This module defines a structured error system with clear categorization,
detailed error information, and security-conscious error handling.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import uuid


class ErrorCategory(Enum):
    """Categories for error classification."""
    VALIDATION = "validation"
    SECURITY = "security"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    RESOURCE = "resource"
    SYSTEM = "system"
    CONTRACT = "contract"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ErrorContext:
    """Context information for error diagnosis."""
    operation: str
    component: str
    timestamp: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def with_metadata(self, **kwargs) -> ErrorContext:
        """Add metadata to error context."""
        new_metadata = self.metadata.copy()
        new_metadata.update(kwargs)
        return ErrorContext(
            operation=self.operation,
            component=self.component,
            timestamp=self.timestamp,
            metadata=new_metadata
        )


class MacroEngineError(Exception):
    """Base exception for all macro engine errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        recovery_suggestion: Optional[str] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.recovery_suggestion = recovery_suggestion
        self.error_code = error_code or self._generate_error_code()
    
    def _generate_error_code(self) -> str:
        """Generate unique error code."""
        return f"{self.category.value.upper()}_{self.__class__.__name__.upper()}_{str(uuid.uuid4())[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context.__dict__ if self.context else None,
            "recovery_suggestion": self.recovery_suggestion,
            "error_type": self.__class__.__name__
        }


class ValidationError(MacroEngineError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        field_name: str,
        value: Any,
        constraint: str,
        context: Optional[ErrorContext] = None
    ):
        message = f"Validation failed for field '{field_name}': {constraint}. Got: {value}"
        recovery = f"Ensure '{field_name}' meets the constraint: {constraint}"
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion=recovery
        )
        self.field_name = field_name
        self.value = value
        self.constraint = constraint


class SecurityViolationError(MacroEngineError):
    """Raised when security boundaries are violated."""
    
    def __init__(
        self,
        violation_type: str,
        details: str,
        context: Optional[ErrorContext] = None
    ):
        message = f"Security violation: {violation_type} - {details}"
        recovery = "Review security permissions and input validation"
        super().__init__(
            message=message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion=recovery
        )
        self.violation_type = violation_type


class PermissionDeniedError(MacroEngineError):
    """Raised when required permissions are not available."""
    
    def __init__(
        self,
        required_permissions: List[str],
        available_permissions: List[str],
        context: Optional[ErrorContext] = None
    ):
        missing = set(required_permissions) - set(available_permissions)
        message = f"Missing required permissions: {list(missing)}"
        recovery = f"Grant the following permissions: {list(missing)}"
        super().__init__(
            message=message,
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion=recovery
        )
        self.required_permissions = required_permissions
        self.available_permissions = available_permissions


class ExecutionError(MacroEngineError):
    """Raised when macro execution fails."""
    
    def __init__(
        self,
        operation: str,
        cause: str,
        context: Optional[ErrorContext] = None
    ):
        message = f"Execution failed for operation '{operation}': {cause}"
        recovery = f"Check the configuration and state for operation: {operation}"
        super().__init__(
            message=message,
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion=recovery
        )
        self.operation = operation
        self.cause = cause


class TimeoutError(MacroEngineError):
    """Raised when operations exceed time limits."""
    
    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        context: Optional[ErrorContext] = None
    ):
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        recovery = f"Increase timeout for '{operation}' or optimize operation performance"
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion=recovery
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class ResourceNotFoundError(MacroEngineError):
    """Raised when required resources are not available."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        context: Optional[ErrorContext] = None
    ):
        message = f"{resource_type} not found: {resource_id}"
        recovery = f"Verify that {resource_type} '{resource_id}' exists and is accessible"
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion=recovery
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class ContractViolationError(MacroEngineError):
    """Raised when design-by-contract assertions fail."""
    
    def __init__(
        self,
        contract_type: str,
        condition: str,
        context: Optional[ErrorContext] = None
    ):
        message = f"{contract_type} violated: {condition}"
        recovery = f"Ensure {contract_type.lower()} condition is met: {condition}"
        super().__init__(
            message=message,
            category=ErrorCategory.CONTRACT,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion=recovery
        )
        self.contract_type = contract_type
        self.condition = condition


class SystemError(MacroEngineError):
    """Raised for system-level errors."""
    
    def __init__(
        self,
        system_component: str,
        error_details: str,
        context: Optional[ErrorContext] = None
    ):
        message = f"System error in {system_component}: {error_details}"
        recovery = f"Check system status and logs for {system_component}"
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            recovery_suggestion=recovery
        )
        self.system_component = system_component


class SecurityError(MacroEngineError):
    """Raised for security-related errors and violations."""
    
    def __init__(
        self,
        security_code: str,
        message: str,
        context: Optional[ErrorContext] = None
    ):
        recovery = "Review security policies and input validation"
        super().__init__(
            message=f"Security error [{security_code}]: {message}",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            recovery_suggestion=recovery,
            error_code=security_code  # Pass security_code as error_code to parent
        )
        self.security_code = security_code


class IntegrationError(MacroEngineError):
    """Raised for integration-related errors with external systems."""
    
    def __init__(
        self,
        integration_code: str,
        message: str,
        context: Optional[ErrorContext] = None
    ):
        recovery = "Check external system connectivity and configuration"
        super().__init__(
            message=f"Integration error [{integration_code}]: {message}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion=recovery
        )
        self.integration_code = integration_code


class DataError(MacroEngineError):
    """Raised for data processing and management errors."""
    
    def __init__(
        self,
        data_operation: str,
        message: str,
        context: Optional[ErrorContext] = None
    ):
        recovery = "Verify data format, schema, and operation parameters"
        super().__init__(
            message=f"Data error in {data_operation}: {message}",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion=recovery
        )
        self.data_operation = data_operation
    
    def is_processing_error(self) -> bool:
        """Check if this is a data processing error."""
        return True


class CommunicationError(MacroEngineError):
    """Raised for communication errors with external systems."""
    
    def __init__(
        self,
        endpoint: str,
        message: str,
        context: Optional[ErrorContext] = None
    ):
        recovery = "Check network connectivity and endpoint availability"
        super().__init__(
            message=f"Communication error with {endpoint}: {message}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion=recovery
        )
        self.endpoint = endpoint
    
    @classmethod
    def email_send_failed(cls, reason: str, context: Optional[ErrorContext] = None) -> "CommunicationError":
        """Create an email send failure error."""
        return cls(
            endpoint="email",
            message=f"Email send failed: {reason}",
            context=context
        )
    
    @classmethod
    def execution_error(cls, reason: str, context: Optional[ErrorContext] = None) -> "CommunicationError":
        """Create a communication execution error."""
        return cls(
            endpoint="unknown",
            message=f"Communication execution failed: {reason}",
            context=context
        )


class ConfigurationError(MacroEngineError):
    """Raised for configuration-related errors."""
    
    def __init__(
        self,
        config_item: str,
        message: str,
        context: Optional[ErrorContext] = None
    ):
        recovery = f"Review configuration for {config_item}"
        super().__init__(
            message=f"Configuration error for {config_item}: {message}",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion=recovery
        )
        self.config_item = config_item


class WindowError(MacroEngineError):
    """Raised for window management and visual automation errors."""
    
    def __init__(
        self,
        window_operation: str,
        message: str,
        context: Optional[ErrorContext] = None
    ):
        recovery = "Verify window availability and system permissions"
        super().__init__(
            message=f"Window error in {window_operation}: {message}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion=recovery
        )
        self.window_operation = window_operation


class MCPError(MacroEngineError):
    """Base error for MCP-specific errors."""
    
    def __init__(
        self,
        mcp_operation: str,
        message: str,
        context: Optional[ErrorContext] = None
    ):
        recovery = "Check MCP server configuration and operation parameters"
        super().__init__(
            message=f"MCP error in {mcp_operation}: {message}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion=recovery
        )
        self.mcp_operation = mcp_operation


class IntelligenceError(MacroEngineError):
    """Raised for AI and intelligence module errors."""
    
    def __init__(
        self,
        intelligence_operation: str,
        message: str,
        context: Optional[ErrorContext] = None
    ):
        recovery = "Review AI operation parameters and input data"
        super().__init__(
            message=f"Intelligence error in {intelligence_operation}: {message}",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion=recovery
        )
        self.intelligence_operation = intelligence_operation


class RateLimitError(MacroEngineError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        resource: str,
        limit: int,
        window: str,
        retry_after: Optional[int] = None,
        context: Optional[ErrorContext] = None
    ):
        message = f"Rate limit exceeded for {resource}: {limit} per {window}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        recovery = f"Wait {retry_after or 60} seconds before retrying"
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion=recovery
        )
        self.resource = resource
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


# Utility functions for error handling
class AnalyticsError(MacroEngineError):
    """Analytics and reporting system errors."""
    
    def __init__(self, operation: str, error_details: str, context: Optional[ErrorContext] = None):
        self.operation = operation
        self.error_details = error_details
        message = f"Analytics operation '{operation}' failed: {error_details}"
        super().__init__(
            message=message,
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            context=context or create_error_context(operation=operation, component="analytics_system")
        )


def create_error_context(operation: str, component: str, **metadata) -> ErrorContext:
    """Create error context with metadata."""
    return ErrorContext(operation=operation, component=component, metadata=metadata)


def handle_error_safely(error: Exception, mask_details: bool = True) -> MacroEngineError:
    """Convert generic exceptions to MacroEngineError with optional detail masking."""
    if isinstance(error, MacroEngineError):
        return error
    
    if mask_details:
        message = "An internal error occurred"
        details = str(error)[:100] if len(str(error)) < 100 else str(error)[:100] + "..."
    else:
        message = f"Unexpected error: {str(error)}"
        details = str(error)
    
    context = create_error_context(
        operation="error_handling",
        component="error_handler",
        original_error_type=type(error).__name__,
        original_message=details
    )
    
    return SystemError(
        system_component="error_handler",
        error_details=message,
        context=context
    )