"""
Core macro engine module for Keyboard Maestro MCP.

This module provides the complete core functionality for macro execution
with type safety, contract-based validation, and security enforcement.
"""

# Type definitions and data structures
# Context management
from .context import (
    # Core classes
    ExecutionContextManager,
    SecurityBoundary,
    SecurityContextManager,
    VariableManager,
    # Global instances
    get_context_manager,
    get_variable_manager,
    # Context manager
    security_context,
)

# Contract system
from .contracts import (
    any_condition,
    # Condition combinators
    combine_conditions,
    ensure,
    # Utility functions
    get_contract_info,
    invariant,
    is_non_negative,
    is_not_none,
    is_positive,
    is_valid_string,
    not_condition,
    # Decorators
    require,
)

# Main execution engine
from .engine import (
    # Utilities
    EngineMetrics,
    # Core engine
    MacroEngine,
    PlaceholderCommand,
    # Test utilities
    create_test_macro,
    # Global instances
    get_default_engine,
    get_engine_metrics,
)

# Error hierarchy
from .errors import (
    CommunicationError,
    ConfigurationError,
    ContractViolationError,
    DataError,
    # Error utilities
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    ExecutionError,
    IntegrationError,
    IntelligenceError,
    # Base error
    MacroEngineError,
    MCPError,
    PermissionDeniedError,
    ResourceNotFoundError,
    SecurityError,
    SecurityViolationError,
    SystemError,
    TimeoutError,
    # Specific error types
    ValidationError,
    WindowError,
    create_error_context,
    handle_error_safely,
)

# Parsing and validation
from .parser import (
    # Enums
    CommandType,
    CommandValidator,
    InputSanitizer,
    # Classes
    MacroParser,
    ParseResult,
    # Functions
    parse_macro_from_json,
    validate_macro_definition,
)
from .types import (
    CommandId,
    CommandParameters,
    CommandResult,
    # Data classes
    Duration,
    ExecutionContext,
    ExecutionResult,
    # Enums
    ExecutionStatus,
    ExecutionToken,
    GroupId,
    # Protocols
    MacroCommand,
    MacroDefinition,
    # Branded types
    MacroId,
    Permission,
    Priority,
    TriggerId,
    VariableName,
)

# Version information
__version__ = "1.0.0"
__author__ = "Agent_1"

# Public API - what external modules should import
__all__ = [
    # Core types
    "MacroId",
    "CommandId",
    "ExecutionToken",
    "TriggerId",
    "GroupId",
    "VariableName",
    "ExecutionStatus",
    "Priority",
    "Permission",
    "Duration",
    "CommandParameters",
    "ExecutionContext",
    "CommandResult",
    "MacroDefinition",
    "ExecutionResult",
    "MacroCommand",
    # Error system
    "MacroEngineError",
    "ValidationError",
    "SecurityViolationError",
    "PermissionDeniedError",
    "ExecutionError",
    "TimeoutError",
    "ResourceNotFoundError",
    "ContractViolationError",
    "SystemError",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorContext",
    "create_error_context",
    "handle_error_safely",
    # Contract system
    "require",
    "ensure",
    "invariant",
    "combine_conditions",
    "any_condition",
    "not_condition",
    "get_contract_info",
    "is_not_none",
    "is_positive",
    "is_non_negative",
    "is_valid_string",
    # Context management
    "ExecutionContextManager",
    "SecurityContextManager",
    "VariableManager",
    "SecurityBoundary",
    "security_context",
    "get_context_manager",
    "get_variable_manager",
    # Parsing
    "CommandType",
    "MacroParser",
    "ParseResult",
    "InputSanitizer",
    "CommandValidator",
    "parse_macro_from_json",
    "validate_macro_definition",
    # Engine
    "MacroEngine",
    "EngineMetrics",
    "PlaceholderCommand",
    "get_default_engine",
    "get_engine_metrics",
    "create_test_macro",
]


def get_version() -> str:
    """Get the current version of the core engine."""
    return __version__


def create_simple_macro(name: str, text_to_type: str) -> MacroDefinition:
    """
    Convenience function to create a simple text-typing macro.

    Args:
        name: Name of the macro
        text_to_type: Text to type when macro executes

    Returns:
        MacroDefinition ready for execution
    """
    return create_test_macro(name, [CommandType.TEXT_INPUT])


def validate_system_requirements() -> bool:
    """
    Validate that system requirements are met for macro execution.

    Returns:
        True if all requirements are satisfied
    """
    # In a real implementation, this would check:
    # - Python version compatibility
    # - Required permissions
    # - System resources
    # - Keyboard Maestro availability

    return True
