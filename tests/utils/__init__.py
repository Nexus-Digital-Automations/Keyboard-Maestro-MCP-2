"""Test utilities package for the Keyboard Maestro MCP testing framework.

import logging

logging.basicConfig(level=logging.DEBUG)
This package provides comprehensive testing utilities including:
- Hypothesis data generators for property-based testing
- Mock objects for external dependencies
- Custom assertions for validation
- Test fixtures and helpers
"""

# Import key utilities for easy access
from __future__ import annotations

from .assertions import (
    assert_command_failed,
    assert_command_successful,
    assert_context_valid,
    assert_duration_valid,
    assert_error_contains_context,
    assert_execution_failed,
    # Execution assertions
    assert_execution_successful,
    assert_execution_time,
    assert_injection_prevented,
    assert_input_sanitized,
    assert_invariant_maintained,
    # Validation assertions
    assert_macro_valid,
    # Context managers
    assert_no_memory_leaks,
    # Performance assertions
    assert_performance_within_bounds,
    # Security assertions
    assert_permissions_required,
    # Property testing assertions
    assert_property_holds,
    assert_security_violation_blocked,
    assert_thread_safe_operation,
)
from .generators import (
    command_ids,
    command_parameters,
    complex_macro_definitions,
    concurrent_execution_scenarios,
    contains_injection_pattern,
    # Complex generators
    durations,
    edge_case_durations,
    execution_contexts,
    execution_tokens,
    # Edge case generators
    invalid_identifiers,
    # Utility functions
    is_safe_for_validation,
    # Basic type generators
    macro_ids,
    malicious_text_content,
    permission_sets,
    # Content generators
    safe_text_content,
    simple_macro_definitions,
    variable_names,
)
from .mocks import (
    MockCommand,
    MockExecutionContext,
    MockFileSystem,
    # Mock classes
    MockKeyboardMaestroClient,
    MockKMResponse,
    MockMacroEngine,
    # Factory functions
    create_failing_km_client,
    create_mock_command_sequence,
    create_privileged_mock_context,
    create_reliable_km_client,
    create_slow_km_client,
)

# Version information
__version__ = "1.0.0"

# Public API
__all__ = [
    "MockCommand",
    "MockExecutionContext",
    "MockFileSystem",
    "MockKMResponse",
    # Mocks
    "MockKeyboardMaestroClient",
    "MockMacroEngine",
    "assert_command_failed",
    "assert_command_successful",
    "assert_context_valid",
    "assert_duration_valid",
    "assert_error_contains_context",
    "assert_execution_failed",
    # Assertions
    "assert_execution_successful",
    "assert_execution_time",
    "assert_injection_prevented",
    "assert_input_sanitized",
    "assert_invariant_maintained",
    "assert_macro_valid",
    "assert_no_memory_leaks",
    "assert_performance_within_bounds",
    "assert_permissions_required",
    "assert_property_holds",
    "assert_security_violation_blocked",
    "assert_thread_safe_operation",
    "command_ids",
    "command_parameters",
    "complex_macro_definitions",
    "concurrent_execution_scenarios",
    "contains_injection_pattern",
    "create_failing_km_client",
    "create_mock_command_sequence",
    "create_privileged_mock_context",
    "create_reliable_km_client",
    "create_slow_km_client",
    "durations",
    "edge_case_durations",
    "execution_contexts",
    "execution_tokens",
    "invalid_identifiers",
    "is_safe_for_validation",
    # Generators
    "macro_ids",
    "malicious_text_content",
    "permission_sets",
    "safe_text_content",
    "simple_macro_definitions",
    "variable_names",
]


def get_test_utilities_info() -> None:
    """Get information about available test utilities."""
    return {
        "version": __version__,
        "generators": {
            "basic_types": [
                "macro_ids",
                "command_ids",
                "execution_tokens",
                "variable_names",
            ],
            "complex_types": [
                "durations",
                "permission_sets",
                "execution_contexts",
                "macro_definitions",
            ],
            "content": ["safe_text_content", "malicious_text_content"],
            "edge_cases": ["invalid_identifiers", "edge_case_durations"],
        },
        "mocks": {
            "km_integration": ["MockKeyboardMaestroClient"],
            "core_components": [
                "MockExecutionContext",
                "MockCommand",
                "MockMacroEngine",
            ],
            "file_system": ["MockFileSystem"],
            "factories": ["create_failing_km_client", "create_reliable_km_client"],
        },
        "assertions": {
            "execution": ["assert_execution_successful", "assert_execution_failed"],
            "security": [
                "assert_security_violation_blocked",
                "assert_injection_prevented",
            ],
            "performance": [
                "assert_performance_within_bounds",
                "assert_thread_safe_operation",
            ],
            "validation": ["assert_macro_valid", "assert_context_valid"],
            "property_testing": [
                "assert_property_holds",
                "assert_invariant_maintained",
            ],
        },
    }
