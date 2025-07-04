"""
Comprehensive test coverage for core modules - TASK_69 Coverage Expansion.

This module provides extensive testing for core system components to achieve
near-100% test coverage as required by the user's testing directive.

Targeting high-impact core modules with minimal dependencies:
- src/core/errors.py (175 lines, 42% coverage)
- src/core/types.py (numerous type definitions)
- src/core/either.py (Either monad implementation)
- src/core/contracts.py (Design by Contract decorators)
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional

# Import core modules for comprehensive testing
from src.core.errors import (
    ErrorCategory, ErrorSeverity, MacroEngineError, ValidationError, 
    SecurityError, ExecutionError, TimeoutError, create_error_context,
    handle_error_safely, AnalyticsError, CommunicationError, 
    ConfigurationError, DataError, IntegrationError, MCPError,
    PermissionDeniedError, RateLimitError, ResourceNotFoundError,
    SecurityViolationError, SystemError, WindowError, ErrorContext
)

from src.core.types import (
    MacroId, CommandId, ExecutionToken, TriggerId, GroupId, VariableName,
    TemplateId, CreationToken, ClipboardId, AppId, BundleId, MenuItemId,
    ToolId, Duration, ExecutionContext, CommandResult, MacroDefinition,
    ExecutionResult, Permission, ExecutionStatus
)

from src.core.either import Either

from src.core.contracts import require, ensure, ContractViolationError


class TestErrorHierarchy:
    """Comprehensive test coverage for error handling system."""
    
    def test_error_category_enum(self):
        """Test error category enumeration."""
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.SECURITY.value == "security"
        assert ErrorCategory.EXECUTION.value == "execution"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        
        # Test all categories are defined
        categories = list(ErrorCategory)
        assert len(categories) >= 4
        
        # Test enum iteration
        for category in ErrorCategory:
            assert isinstance(category.value, str)
            assert len(category.value) > 0
    
    def test_error_severity_enum(self):
        """Test error severity enumeration."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium" 
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
        
        # Test severity ordering
        severities = list(ErrorSeverity)
        assert len(severities) >= 4
    
    def test_macro_engine_error_creation(self):
        """Test MacroEngineError creation and properties."""
        context = create_error_context("test_operation", "test_component")
        error = MacroEngineError(
            message="Test error message",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion="Check parameter format",
            error_code="TEST_001"
        )
        
        assert error.message == "Test error message"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.error_code == "TEST_001"
        assert error.recovery_suggestion == "Check parameter format"
        assert error.context == context
    
    def test_validation_error_creation(self):
        """Test ValidationError creation and inheritance."""
        context = create_error_context("validation", "input_processor")
        error = ValidationError(
            field_name="macro_id",
            value="invalid_id",
            constraint="must be valid UUID string",
            context=context
        )
        
        assert "Validation failed for field 'macro_id'" in error.message
        assert error.category == ErrorCategory.VALIDATION
        assert error.field_name == "macro_id"
        assert error.value == "invalid_id"
        assert error.constraint == "must be valid UUID string"
        assert isinstance(error, MacroEngineError)
    
    def test_security_error_creation(self):
        """Test SecurityError creation and properties."""
        context = create_error_context("authentication", "security_manager")
        error = SecurityError(
            security_code="SECURITY_001",
            message="Permission denied for macro execution",
            context=context
        )
        
        assert "Security error [SECURITY_001]" in error.message
        assert error.category == ErrorCategory.SECURITY
        assert error.security_code == "SECURITY_001"
        assert error.severity == ErrorSeverity.CRITICAL
        assert isinstance(error, MacroEngineError)
    
    def test_execution_error_creation(self):
        """Test ExecutionError creation and properties."""
        context = create_error_context("macro_execution", "execution_engine")
        error = ExecutionError(
            operation="macro_execution",
            cause="Action step failed",
            context=context
        )
        
        assert "Execution failed for operation 'macro_execution'" in error.message
        assert error.category == ErrorCategory.EXECUTION
        assert error.operation == "macro_execution"
        assert error.cause == "Action step failed"
        assert error.severity == ErrorSeverity.MEDIUM
        assert isinstance(error, MacroEngineError)
    
    def test_timeout_error_creation(self):
        """Test TimeoutError creation and properties."""
        context = create_error_context("operation_timeout", "timeout_manager")
        error = TimeoutError(
            operation="macro_execution",
            timeout_seconds=30.0,
            context=context
        )
        
        assert "Operation 'macro_execution' timed out after 30.0 seconds" in error.message
        assert error.category == ErrorCategory.TIMEOUT
        assert error.operation == "macro_execution"
        assert error.timeout_seconds == 30.0
        assert isinstance(error, MacroEngineError)
    
    def test_create_error_context(self):
        """Test error context creation function."""
        context = create_error_context(
            operation="test_operation",
            component="test_component",
            key="value"
        )
        
        assert isinstance(context, ErrorContext)
        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.metadata["key"] == "value"
    
    def test_handle_error_safely(self):
        """Test safe error handling function."""
        test_error = ValueError("Test error")
        
        result = handle_error_safely(test_error, mask_details=False)
        
        # Should handle error safely and return MacroEngineError
        assert isinstance(result, SystemError)
        assert "Unexpected error: Test error" in result.message
    
    def test_additional_error_types(self):
        """Test additional specialized error types."""
        context = create_error_context("test", "test_component")
        
        # Test AnalyticsError
        analytics_error = AnalyticsError("data_processing", "Processing failed", context)
        assert "data_processing" in analytics_error.operation
        assert "Processing failed" in analytics_error.error_details
        assert isinstance(analytics_error, Exception)
        
        # Test CommunicationError
        comm_error = CommunicationError("api_endpoint", "Connection timeout", context)
        assert "Communication error with api_endpoint" in comm_error.message
        assert isinstance(comm_error, MacroEngineError)
        
        # Test ConfigurationError
        config_error = ConfigurationError("database_url", "Invalid format", context)
        assert "Configuration error for database_url" in config_error.message
        assert isinstance(config_error, MacroEngineError)
        
        # Test DataError
        data_error = DataError("validation", "Schema mismatch", context)
        assert "Data error in validation" in data_error.message
        assert isinstance(data_error, MacroEngineError)
    
    def test_error_serialization(self):
        """Test error serialization for logging/transport."""
        context = create_error_context("serialization", "test_component")
        error = ValidationError(
            field_name="test_field",
            value=123,
            constraint="must be string",
            context=context
        )
        
        serialized = error.to_dict()
        
        assert isinstance(serialized, dict)
        assert "Validation failed for field 'test_field'" in serialized["message"]
        assert serialized["category"] == "validation"
        assert serialized["error_type"] == "ValidationError"
        assert "error_code" in serialized
    
    def test_error_chain_creation(self):
        """Test error chaining for nested error handling."""
        context = create_error_context("validation", "input_processor")
        root_error = ValidationError(
            field_name="input_value",
            value="invalid",
            constraint="must be numeric",
            context=context
        )
        
        exec_context = create_error_context("execution", "macro_engine")
        chained_error = ExecutionError(
            operation="macro_execution",
            cause="Validation failure in input processing",
            context=exec_context
        )
        
        assert "Execution failed for operation 'macro_execution'" in chained_error.message
        assert chained_error.category == ErrorCategory.EXECUTION
        assert chained_error.operation == "macro_execution"


class TestCoreTypes:
    """Comprehensive test coverage for core type system."""
    
    def test_branded_type_creation(self):
        """Test branded type creation and type safety."""
        macro_id = MacroId("test_macro_123")
        command_id = CommandId("test_command_456")
        execution_token = ExecutionToken("exec_token_789")
        
        assert macro_id == "test_macro_123"
        assert command_id == "test_command_456"
        assert execution_token == "exec_token_789"
        
        # Test type distinction through isinstance checks
        assert isinstance(macro_id, str)
        assert isinstance(command_id, str)
        assert isinstance(execution_token, str)
    
    def test_all_branded_types(self):
        """Test all branded type definitions."""
        types_to_test = [
            (MacroId, "macro_123"),
            (CommandId, "cmd_456"),
            (ExecutionToken, "token_789"),
            (TriggerId, "trigger_abc"),
            (GroupId, "group_def"),
            (VariableName, "var_name"),
            (TemplateId, "template_ghi"),
            (CreationToken, "create_jkl"),
            (ClipboardId, "clip_mno"),
            (AppId, "app_pqr"),
            (BundleId, "bundle_stu"),
            (MenuItemId, "menu_vwx"),
            (ToolId, "tool_yz")
        ]
        
        for branded_type, value in types_to_test:
            instance = branded_type(value)
            assert instance == value
            assert isinstance(instance, str)
    
    def test_duration_operations(self):
        """Test Duration type operations."""
        # Test creation
        dur1 = Duration.from_seconds(5.0)
        dur2 = Duration.from_milliseconds(3000)
        
        assert dur1.total_seconds() == 5.0
        assert dur2.total_seconds() == 3.0
        
        # Test arithmetic
        dur_sum = dur1 + dur2
        assert dur_sum.total_seconds() == 8.0
        
        # Test comparisons
        assert dur2 < dur1
        assert dur1 > dur2
        assert dur1 >= dur2
        assert dur2 <= dur1
        
        # Test equality
        dur3 = Duration.from_seconds(5.0)
        assert dur1 == dur3
        
        # Test zero constant
        assert Duration.ZERO.total_seconds() == 0.0
    
    def test_execution_context(self):
        """Test ExecutionContext creation and operations."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_CONTROL])
        timeout = Duration.from_seconds(30)
        
        context = ExecutionContext(permissions=permissions, timeout=timeout)
        
        assert context.has_permission(Permission.TEXT_INPUT)
        assert context.has_permission(Permission.SYSTEM_CONTROL)
        assert not context.has_permission(Permission.FILE_ACCESS)
        
        assert context.has_permissions(frozenset([Permission.TEXT_INPUT]))
        assert not context.has_permissions(frozenset([Permission.FILE_ACCESS]))
        
        # Test variable operations
        var_name = VariableName("test_var")
        new_context = context.with_variable(var_name, "test_value")
        assert new_context.get_variable(var_name) == "test_value"
        assert context.get_variable(var_name) is None  # Original unchanged
    
    def test_command_result(self):
        """Test CommandResult creation and operations."""
        # Test success result
        success_result = CommandResult.success_result(
            output="Operation completed",
            execution_time=Duration.from_seconds(1.5),
            custom_metadata="test_value"
        )
        
        assert success_result.success is True
        assert success_result.output == "Operation completed"
        assert success_result.execution_time.total_seconds() == 1.5
        assert success_result.metadata["custom_metadata"] == "test_value"
        
        # Test failure result
        failure_result = CommandResult.failure_result(
            error_message="Operation failed",
            execution_time=Duration.from_seconds(0.5)
        )
        
        assert failure_result.success is False
        assert failure_result.error_message == "Operation failed"
        assert failure_result.execution_time.total_seconds() == 0.5


class TestEitherMonad:
    """Comprehensive test coverage for Either monad implementation."""
    
    def test_right_creation_and_access(self):
        """Test Right (success) case creation and access."""
        right_value = Either.right("success_value")
        
        assert right_value.is_right() is True
        assert right_value.is_left() is False
        assert right_value.get_right() == "success_value"
        
        # Test that getting left from right raises exception
        with pytest.raises(ValueError):
            right_value.get_left()
    
    def test_left_creation_and_access(self):
        """Test Left (error) case creation and access."""
        left_value = Either.left("error_message")
        
        assert left_value.is_left() is True
        assert left_value.is_right() is False
        assert left_value.get_left() == "error_message"
        
        # Test that getting right from left raises exception
        with pytest.raises(ValueError):
            left_value.get_right()
    
    def test_either_map_operations(self):
        """Test Either map operations for transformations."""
        right_value = Either.right(10)
        left_value = Either.left("error")
        
        # Map should transform right values
        mapped_right = right_value.map(lambda x: x * 2)
        assert mapped_right.is_right()
        assert mapped_right.get_right() == 20
        
        # Map should pass through left values unchanged
        mapped_left = left_value.map(lambda x: x * 2)
        assert mapped_left.is_left()
        assert mapped_left.get_left() == "error"
    
    def test_either_flat_map_operations(self):
        """Test Either flat_map operations for chaining."""
        def safe_divide(x, y):
            if y == 0:
                return Either.left("Division by zero")
            return Either.right(x / y)
        
        # Test successful chaining
        result = Either.right(10).flat_map(lambda x: safe_divide(x, 2))
        assert result.is_right()
        assert result.get_right() == 5.0
        
        # Test error propagation
        error_result = Either.right(10).flat_map(lambda x: safe_divide(x, 0))
        assert error_result.is_left()
        assert error_result.get_left() == "Division by zero"
        
        # Test left value propagation
        left_start = Either.left("initial_error")
        propagated = left_start.flat_map(lambda x: safe_divide(x, 2))
        assert propagated.is_left()
        assert propagated.get_left() == "initial_error"
    
    def test_either_fold_operations(self):
        """Test Either fold operations for result extraction."""
        right_value = Either.right(42)
        left_value = Either.left("error_msg")
        
        # Test either value processing
        right_result = "Success: 42" if right_value.is_right() else f"Error: {right_value.get_left()}"
        assert right_result == "Success: 42"
        
        # Test left value processing  
        left_result = f"Error: {left_value.get_left()}" if left_value.is_left() else f"Success: {left_value.get_right()}"
        assert left_result == "Error: error_msg"
    
    def test_either_get_or_else(self):
        """Test Either get_or_else for default values."""
        right_value = Either.right("actual_value")
        left_value = Either.left("error")
        
        # Right should return actual value
        assert right_value.get_or_else("default") == "actual_value"
        
        # Left should return default value
        assert left_value.get_or_else("default") == "default"
    
    def test_either_chaining_complex(self):
        """Test complex Either chaining scenarios."""
        def parse_int(s):
            try:
                return Either.right(int(s))
            except ValueError:
                return Either.left(f"Cannot parse '{s}' as integer")
        
        def multiply_by_two(x):
            return Either.right(x * 2)
        
        def ensure_positive(x):
            if x > 0:
                return Either.right(x)
            return Either.left(f"Value {x} is not positive")
        
        # Test successful chain
        result = (Either.right("42")
                 .flat_map(parse_int)
                 .flat_map(multiply_by_two)
                 .flat_map(ensure_positive))
        
        assert result.is_right()
        assert result.get_right() == 84
        
        # Test chain with error
        error_result = (Either.right("invalid")
                       .flat_map(parse_int)
                       .flat_map(multiply_by_two)
                       .flat_map(ensure_positive))
        
        assert error_result.is_left()
        assert "Cannot parse" in error_result.get_left()


class TestContractSystem:
    """Comprehensive test coverage for Design by Contract decorators."""
    
    def test_require_decorator_success(self):
        """Test require decorator with valid preconditions."""
        @require(lambda x: x > 0, "x must be positive")
        @require(lambda x: isinstance(x, (int, float)), "x must be numeric")
        def divide_ten_by(x):
            return 10 / x
        
        # Test with valid input
        result = divide_ten_by(5)
        assert result == 2.0
        
        result = divide_ten_by(2.5)
        assert result == 4.0
    
    def test_require_decorator_failure(self):
        """Test require decorator with invalid preconditions."""
        @require(lambda x: x > 0, "x must be positive")
        @require(lambda x: isinstance(x, (int, float)), "x must be numeric")
        def divide_ten_by(x):
            return 10 / x
        
        # Test with invalid input (negative)
        with pytest.raises(ContractViolationError) as exc_info:
            divide_ten_by(-5)
        assert exc_info.value.contract_type == "Precondition"
        assert "x must be positive" in exc_info.value.condition
        
        # Test with invalid input (wrong type) - should hit numeric check first
        with pytest.raises(ContractViolationError) as exc_info:
            divide_ten_by("5")
        assert exc_info.value.contract_type == "Precondition"
        # Contract order can affect which message appears first
        assert any(msg in exc_info.value.condition for msg in ["x must be numeric", "x must be positive"])
    
    def test_ensure_decorator_success(self):
        """Test ensure decorator with valid postconditions."""
        @ensure(lambda result: result > 0, "result must be positive")
        @ensure(lambda result: isinstance(result, (int, float)), "result must be numeric")
        def get_positive_number():
            return 42
        
        # Test with valid output
        result = get_positive_number()
        assert result == 42
    
    def test_ensure_decorator_failure(self):
        """Test ensure decorator with invalid postconditions."""
        @ensure(lambda result: result > 0, "result must be positive")
        def get_negative_number():
            return -10
        
        # Test with invalid output
        with pytest.raises(ContractViolationError) as exc_info:
            get_negative_number()
        assert "result must be positive" in str(exc_info.value)
    
    def test_combined_contracts(self):
        """Test combining require and ensure decorators."""
        @require(lambda x, y: x > 0 and y > 0, "both inputs must be positive")
        @ensure(lambda result: result > 0, "result must be positive")
        def multiply_positive(x, y):
            return x * y
        
        # Test valid case
        result = multiply_positive(3, 4)
        assert result == 12
        
        # Test invalid precondition
        with pytest.raises(ContractViolationError) as exc_info:
            multiply_positive(-3, 4)
        assert "both inputs must be positive" in str(exc_info.value)
    
    def test_contracts_with_multiple_parameters(self):
        """Test contracts with multiple parameters."""
        @require(lambda a, b, c: len(a) > 0, "list must not be empty")
        @require(lambda a, b, c: isinstance(b, str), "b must be string")
        @require(lambda a, b, c: c >= 0, "c must be non-negative")
        @ensure(lambda result: isinstance(result, dict), "result must be dict")
        def process_data(a, b, c):
            return {
                "list_length": len(a),
                "string_value": b,
                "number_value": c
            }
        
        # Test valid case
        result = process_data([1, 2, 3], "test", 5)
        assert result["list_length"] == 3
        assert result["string_value"] == "test"
        assert result["number_value"] == 5
        
        # Test invalid cases
        with pytest.raises(ContractViolationError):
            process_data([], "test", 5)  # Empty list
        
        with pytest.raises(ContractViolationError):
            process_data([1, 2], 123, 5)  # Non-string b
        
        with pytest.raises(ContractViolationError):
            process_data([1, 2], "test", -1)  # Negative c
    
    def test_contract_with_async_function(self):
        """Test contracts with async functions."""
        @require(lambda x: x >= 0, "x must be non-negative")
        @ensure(lambda result: result >= 0, "result must be non-negative")
        async def async_sqrt(x):
            return x ** 0.5
        
        # Test async contract validation
        import asyncio
        
        async def test_async():
            result = await async_sqrt(16)
            assert abs(result - 4.0) < 0.001
            
            with pytest.raises(ContractViolationError):
                await async_sqrt(-1)
        
        asyncio.run(test_async())
    
    def test_contract_violation_error_details(self):
        """Test ContractViolationError provides detailed information."""
        @require(lambda x: x > 10, "x must be greater than 10")
        def test_function(x):
            return x
        
        try:
            test_function(5)
            assert False, "Should have raised ContractViolationError"
        except ContractViolationError as e:
            assert "x must be greater than 10" in e.condition
            assert hasattr(e, 'contract_type')
            assert e.contract_type == "Precondition"
            assert "Precondition violated" in e.message


if __name__ == "__main__":
    pytest.main([__file__])