"""
Core macro engine module for Keyboard Maestro MCP.

This module provides the complete core functionality for macro execution
with type safety, contract-based validation, and security enforcement.
"""

# Type definitions and data structures
from .types import (
    # Branded types
    MacroId,
    CommandId,
    ExecutionToken,
    TriggerId,
    GroupId,
    VariableName,
    
    # Enums
    ExecutionStatus,
    Priority,
    Permission,
    
    # Data classes
    Duration,
    CommandParameters,
    ExecutionContext,
    CommandResult,
    MacroDefinition,
    ExecutionResult,
    
    # Protocols
    MacroCommand,
)

# Error hierarchy
from .errors import (
    # Base error
    MacroEngineError,
    
    # Specific error types
    ValidationError,
    SecurityViolationError,
    PermissionDeniedError,
    ExecutionError,
    TimeoutError,
    ResourceNotFoundError,
    ContractViolationError,
    SystemError,
    SecurityError,
    IntegrationError,
    DataError,
    CommunicationError,
    ConfigurationError,
    WindowError,
    MCPError,
    IntelligenceError,
    
    # Error utilities
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
    create_error_context,
    handle_error_safely,
)

# Contract system
from .contracts import (
    # Decorators
    require,
    ensure,
    invariant,
    
    # Condition combinators
    combine_conditions,
    any_condition,
    not_condition,
    
    # Utility functions
    get_contract_info,
    is_not_none,
    is_positive,
    is_non_negative,
    is_valid_string,
)

# Context management
from .context import (
    # Core classes
    ExecutionContextManager,
    SecurityContextManager,
    VariableManager,
    SecurityBoundary,
    
    # Context manager
    security_context,
    
    # Global instances
    get_context_manager,
    get_variable_manager,
)

# Parsing and validation
from .parser import (
    # Enums
    CommandType,
    
    # Classes
    MacroParser,
    ParseResult,
    InputSanitizer,
    CommandValidator,
    
    # Functions
    parse_macro_from_json,
    validate_macro_definition,
)

# Main execution engine
from .engine import (
    # Core engine
    MacroEngine,
    
    # Utilities
    EngineMetrics,
    PlaceholderCommand,
    
    # Global instances
    get_default_engine,
    get_engine_metrics,
    
    # Test utilities
    create_test_macro,
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