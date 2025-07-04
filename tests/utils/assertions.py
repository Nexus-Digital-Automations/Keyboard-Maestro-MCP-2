"""
Custom test assertions for the Keyboard Maestro MCP testing framework.

This module provides specialized assertion functions for validating
macro system behavior, security properties, and performance characteristics.
"""

from typing import Any, List, Dict, Optional, Callable, Union
import time
import re
from contextlib import contextmanager

from src.core import (
    ExecutionResult, ExecutionStatus, CommandResult, MacroDefinition,
    ExecutionContext, Duration, Permission, MacroEngineError
)


def assert_execution_successful(result: ExecutionResult, message: str = "") -> None:
    """Assert that macro execution was successful."""
    prefix = f"{message}: " if message else ""
    
    assert result is not None, f"{prefix}Execution result cannot be None"
    assert result.execution_token is not None, f"{prefix}Execution token must be present"
    assert result.status == ExecutionStatus.COMPLETED, f"{prefix}Expected COMPLETED, got {result.status}"
    assert result.is_successful(), f"{prefix}Execution should be marked as successful"
    assert result.completed_at is not None, f"{prefix}Completion time must be set"
    assert result.total_duration is not None, f"{prefix}Total duration must be measured"


def assert_execution_failed(result: ExecutionResult, 
                          expected_error: Optional[str] = None,
                          message: str = "") -> None:
    """Assert that macro execution failed with expected error."""
    prefix = f"{message}: " if message else ""
    
    assert result is not None, f"{prefix}Execution result cannot be None"
    assert result.status == ExecutionStatus.FAILED, f"{prefix}Expected FAILED, got {result.status}"
    assert not result.is_successful(), f"{prefix}Execution should not be marked as successful"
    assert result.has_error_info(), f"{prefix}Error information must be available"
    
    if expected_error:
        assert result.error_details is not None, f"{prefix}Error details must be present"
        assert expected_error in result.error_details, f"{prefix}Expected error '{expected_error}' not found in '{result.error_details}'"


def assert_command_successful(result: CommandResult, message: str = "") -> None:
    """Assert that command execution was successful."""
    prefix = f"{message}: " if message else ""
    
    assert result is not None, f"{prefix}Command result cannot be None"
    assert result.success, f"{prefix}Command should be successful"
    assert result.error_message is None, f"{prefix}No error message should be present: {result.error_message}"
    assert result.execution_time is not None, f"{prefix}Execution time must be measured"


def assert_command_failed(result: CommandResult, 
                         expected_error: Optional[str] = None,
                         message: str = "") -> None:
    """Assert that command execution failed."""
    prefix = f"{message}: " if message else ""
    
    assert result is not None, f"{prefix}Command result cannot be None"
    assert not result.success, f"{prefix}Command should not be successful"
    assert result.error_message is not None, f"{prefix}Error message must be present"
    
    if expected_error:
        assert expected_error in result.error_message, f"{prefix}Expected error '{expected_error}' not found in '{result.error_message}'"


def assert_permissions_required(context: ExecutionContext, 
                               required_permissions: List[Permission],
                               message: str = "") -> None:
    """Assert that context has all required permissions."""
    prefix = f"{message}: " if message else ""
    
    for permission in required_permissions:
        assert context.has_permission(permission), f"{prefix}Missing required permission: {permission}"


def assert_security_violation_blocked(func: Callable, 
                                    args: tuple = (),
                                    kwargs: Dict[str, Any] = None,
                                    expected_error_type: type = MacroEngineError,
                                    message: str = "") -> None:
    """Assert that a security violation is properly blocked."""
    prefix = f"{message}: " if message else ""
    kwargs = kwargs or {}
    
    try:
        result = func(*args, **kwargs)
        assert False, f"{prefix}Expected security violation to be blocked, but function succeeded with result: {result}"
    except expected_error_type:
        # Expected behavior - security violation was caught
        pass
    except Exception as e:
        assert False, f"{prefix}Expected {expected_error_type.__name__}, but got {type(e).__name__}: {e}"


def assert_input_sanitized(original_input: str, 
                          sanitized_output: str,
                          message: str = "") -> None:
    """Assert that input has been properly sanitized."""
    prefix = f"{message}: " if message else ""
    
    # Check that dangerous patterns are removed
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'eval\s*\(',
        r'exec\s*\(',
        r'\.\./',
        r'[<>"\']'
    ]
    
    for pattern in dangerous_patterns:
        original_matches = len(re.findall(pattern, original_input, re.IGNORECASE))
        sanitized_matches = len(re.findall(pattern, sanitized_output, re.IGNORECASE))
        
        assert sanitized_matches < original_matches or original_matches == 0, \
            f"{prefix}Dangerous pattern '{pattern}' not properly sanitized"


def assert_performance_within_bounds(execution_time: float,
                                   max_time: float,
                                   min_time: float = 0.0,
                                   message: str = "") -> None:
    """Assert that execution time is within acceptable bounds."""
    prefix = f"{message}: " if message else ""
    
    assert execution_time >= min_time, f"{prefix}Execution time {execution_time}s is below minimum {min_time}s"
    assert execution_time <= max_time, f"{prefix}Execution time {execution_time}s exceeds maximum {max_time}s"


def assert_macro_valid(macro: MacroDefinition, message: str = "") -> None:
    """Assert that macro definition is valid."""
    prefix = f"{message}: " if message else ""
    
    assert macro is not None, f"{prefix}Macro cannot be None"
    assert macro.macro_id is not None, f"{prefix}Macro ID must be set"
    assert macro.name is not None and macro.name.strip(), f"{prefix}Macro name must be non-empty"
    assert len(macro.commands) > 0, f"{prefix}Macro must have at least one command"
    assert macro.is_valid(), f"{prefix}Macro must pass validation"
    
    # Validate all commands
    for i, command in enumerate(macro.commands):
        assert command.validate(), f"{prefix}Command {i} must be valid"


def assert_context_valid(context: ExecutionContext, message: str = "") -> None:
    """Assert that execution context is valid."""
    prefix = f"{message}: " if message else ""
    
    assert context is not None, f"{prefix}Context cannot be None"
    assert context.execution_id is not None, f"{prefix}Execution ID must be set"
    assert context.permissions is not None, f"{prefix}Permissions must be set"
    assert context.timeout is not None, f"{prefix}Timeout must be set"
    assert context.timeout.total_seconds() > 0, f"{prefix}Timeout must be positive"
    assert context.created_at is not None, f"{prefix}Creation time must be set"


def assert_duration_valid(duration: Duration, 
                         min_seconds: float = 0.0,
                         max_seconds: float = 3600.0,
                         message: str = "") -> None:
    """Assert that duration is within valid bounds."""
    prefix = f"{message}: " if message else ""
    
    assert duration is not None, f"{prefix}Duration cannot be None"
    assert duration.total_seconds() >= min_seconds, f"{prefix}Duration {duration.total_seconds()}s is below minimum {min_seconds}s"
    assert duration.total_seconds() <= max_seconds, f"{prefix}Duration {duration.total_seconds()}s exceeds maximum {max_seconds}s"


def assert_thread_safe_operation(operation: Callable,
                                args_list: List[tuple],
                                max_workers: int = 5,
                                message: str = "") -> None:
    """Assert that operation is thread-safe."""
    import threading
    import concurrent.futures
    
    prefix = f"{message}: " if message else ""
    
    results = []
    errors = []
    lock = threading.Lock()
    
    def worker(args):
        try:
            result = operation(*args)
            with lock:
                results.append(result)
        except Exception as e:
            with lock:
                errors.append(e)
    
    # Run operations concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, args) for args in args_list[:max_workers]]
        concurrent.futures.wait(futures, timeout=10.0)
    
    # Check results
    assert len(errors) == 0, f"{prefix}Thread safety violation - errors occurred: {errors}"
    assert len(results) == len(args_list[:max_workers]), f"{prefix}Not all operations completed successfully"


@contextmanager
def assert_no_memory_leaks(max_growth_mb: float = 10.0):
    """Context manager to assert no significant memory growth."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_growth = final_memory - initial_memory
    
    assert memory_growth <= max_growth_mb, f"Memory leak detected: {memory_growth:.2f}MB growth (max: {max_growth_mb}MB)"


@contextmanager
def assert_execution_time(max_time: float, min_time: float = 0.0):
    """Context manager to assert execution time bounds."""
    start_time = time.perf_counter()
    
    yield
    
    execution_time = time.perf_counter() - start_time
    assert_performance_within_bounds(execution_time, max_time, min_time, "Execution time bounds")


def assert_property_holds(predicate: Callable[[Any], bool],
                         values: List[Any],
                         property_name: str = "property",
                         message: str = "") -> None:
    """Assert that a property holds for all values."""
    prefix = f"{message}: " if message else ""
    
    failed_values = []
    for value in values:
        try:
            if not predicate(value):
                failed_values.append(value)
        except Exception as e:
            failed_values.append(f"{value} (error: {e})")
    
    assert len(failed_values) == 0, f"{prefix}Property '{property_name}' failed for values: {failed_values}"


def assert_invariant_maintained(invariant_check: Callable[[], bool],
                               operation: Callable[[], Any],
                               invariant_name: str = "invariant",
                               message: str = "") -> Any:
    """Assert that an invariant is maintained across an operation."""
    prefix = f"{message}: " if message else ""
    
    # Check invariant before operation
    assert invariant_check(), f"{prefix}Invariant '{invariant_name}' violated before operation"
    
    # Perform operation
    result = operation()
    
    # Check invariant after operation
    assert invariant_check(), f"{prefix}Invariant '{invariant_name}' violated after operation"
    
    return result


def assert_error_contains_context(error: Exception,
                                expected_context_keys: List[str],
                                message: str = "") -> None:
    """Assert that error contains expected context information."""
    prefix = f"{message}: " if message else ""
    
    if hasattr(error, 'context') and error.context:
        context_dict = error.context.__dict__ if hasattr(error.context, '__dict__') else {}
        for key in expected_context_keys:
            assert key in context_dict, f"{prefix}Expected context key '{key}' not found in error context"
    else:
        assert False, f"{prefix}Error should contain context information"


# Specialized assertions for security testing
def assert_injection_prevented(input_text: str,
                              validation_func: Callable[[str], bool],
                              message: str = "") -> None:
    """Assert that injection attempts are prevented."""
    prefix = f"{message}: " if message else ""
    
    injection_patterns = [
        "<script>", "javascript:", "eval(", "exec(",
        "../", "; rm", "| cat", "&& format",
        "DROP TABLE", "' OR '1'='1"
    ]
    
    contains_injection = any(pattern in input_text for pattern in injection_patterns)
    
    if contains_injection:
        assert not validation_func(input_text), f"{prefix}Injection pattern in '{input_text}' should be rejected"
    else:
        # Safe input should pass validation
        assert validation_func(input_text), f"{prefix}Safe input '{input_text}' should pass validation"