"""
Test utilities package for the Keyboard Maestro MCP testing framework.

This package provides comprehensive testing utilities including:
- Hypothesis data generators for property-based testing
- Mock objects for external dependencies
- Custom assertions for validation
- Test fixtures and helpers
"""

# Import key utilities for easy access
from .generators import (
    # Basic type generators
    macro_ids,
    command_ids,
    execution_tokens,
    variable_names,
    
    # Complex generators
    durations,
    permission_sets,
    command_parameters,
    execution_contexts,
    simple_macro_definitions,
    complex_macro_definitions,
    
    # Content generators
    safe_text_content,
    malicious_text_content,
    
    # Edge case generators
    invalid_identifiers,
    edge_case_durations,
    concurrent_execution_scenarios,
    
    # Utility functions
    is_safe_for_validation,
    contains_injection_pattern
)

from .mocks import (
    # Mock classes
    MockKeyboardMaestroClient,
    MockExecutionContext,
    MockCommand,
    MockMacroEngine,
    MockFileSystem,
    MockKMResponse,
    
    # Factory functions
    create_failing_km_client,
    create_slow_km_client,
    create_reliable_km_client,
    create_mock_command_sequence,
    create_privileged_mock_context
)

from .assertions import (
    # Execution assertions
    assert_execution_successful,
    assert_execution_failed,
    assert_command_successful,
    assert_command_failed,
    
    # Security assertions
    assert_permissions_required,
    assert_security_violation_blocked,
    assert_input_sanitized,
    assert_injection_prevented,
    
    # Performance assertions
    assert_performance_within_bounds,
    assert_thread_safe_operation,
    
    # Validation assertions
    assert_macro_valid,
    assert_context_valid,
    assert_duration_valid,
    
    # Property testing assertions
    assert_property_holds,
    assert_invariant_maintained,
    assert_error_contains_context,
    
    # Context managers
    assert_no_memory_leaks,
    assert_execution_time
)

# Version information
__version__ = "1.0.0"

# Public API
__all__ = [
    # Generators
    "macro_ids",
    "command_ids", 
    "execution_tokens",
    "variable_names",
    "durations",
    "permission_sets",
    "command_parameters",
    "execution_contexts",
    "simple_macro_definitions",
    "complex_macro_definitions",
    "safe_text_content",
    "malicious_text_content",
    "invalid_identifiers",
    "edge_case_durations",
    "concurrent_execution_scenarios",
    "is_safe_for_validation",
    "contains_injection_pattern",
    
    # Mocks
    "MockKeyboardMaestroClient",
    "MockExecutionContext",
    "MockCommand",
    "MockMacroEngine", 
    "MockFileSystem",
    "MockKMResponse",
    "create_failing_km_client",
    "create_slow_km_client",
    "create_reliable_km_client",
    "create_mock_command_sequence",
    "create_privileged_mock_context",
    
    # Assertions
    "assert_execution_successful",
    "assert_execution_failed",
    "assert_command_successful",
    "assert_command_failed",
    "assert_permissions_required",
    "assert_security_violation_blocked",
    "assert_input_sanitized",
    "assert_injection_prevented",
    "assert_performance_within_bounds",
    "assert_thread_safe_operation",
    "assert_macro_valid",
    "assert_context_valid",
    "assert_duration_valid",
    "assert_property_holds",
    "assert_invariant_maintained",
    "assert_error_contains_context",
    "assert_no_memory_leaks",
    "assert_execution_time"
]


def get_test_utilities_info():
    """Get information about available test utilities."""
    return {
        "version": __version__,
        "generators": {
            "basic_types": ["macro_ids", "command_ids", "execution_tokens", "variable_names"],
            "complex_types": ["durations", "permission_sets", "execution_contexts", "macro_definitions"],
            "content": ["safe_text_content", "malicious_text_content"],
            "edge_cases": ["invalid_identifiers", "edge_case_durations"]
        },
        "mocks": {
            "km_integration": ["MockKeyboardMaestroClient"],
            "core_components": ["MockExecutionContext", "MockCommand", "MockMacroEngine"],
            "file_system": ["MockFileSystem"],
            "factories": ["create_failing_km_client", "create_reliable_km_client"]
        },
        "assertions": {
            "execution": ["assert_execution_successful", "assert_execution_failed"],
            "security": ["assert_security_violation_blocked", "assert_injection_prevented"],
            "performance": ["assert_performance_within_bounds", "assert_thread_safe_operation"],
            "validation": ["assert_macro_valid", "assert_context_valid"],
            "property_testing": ["assert_property_holds", "assert_invariant_maintained"]
        }
    }