"""Comprehensive tests for Either monad functionality.

This module provides comprehensive test coverage for the Either monad implementation
focusing on functional error handling, monadic operations, and type safety.
"""

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.core.either import (
    Either,
    Left,
    Right,
    sequence,
    traverse,
)


class TestEitherCreation:
    """Test Either creation methods."""

    def test_left_creation(self) -> None:
        """Test creation of Left values."""
        error = "error message"
        left: Either[str, Any] = Either.left(error)

        assert left.is_left() is True
        assert left.is_right() is False
        assert left.get_left() == error

        with pytest.raises(ValueError, match="Cannot get Right value from Left"):
            left.get_right()

    def test_right_creation(self) -> None:
        """Test creation of Right values."""
        value = "success value"
        right: Either[Any, Any] = Either.right(value)

        assert right.is_left() is False
        assert right.is_right() is True
        assert right.get_right() == value

        with pytest.raises(ValueError, match="Cannot get Left value from Right"):
            right.get_left()

    def test_direct_left_creation(self) -> None:
        """Test direct Left constructor usage."""
        error = "direct error"
        left: Left[str, Any] = Left(error)

        assert left.is_left() is True
        assert left.is_right() is False
        assert left.get_left() == error

    def test_direct_right_creation(self) -> None:
        """Test direct Right constructor usage."""
        value = "direct value"
        right: Right[Any, Any] = Right(value)

        assert right.is_left() is False
        assert right.is_right() is True
        assert right.get_right() == value


class TestEitherCompatibilityMethods:
    """Test compatibility alias methods."""

    def test_success_error_aliases(self) -> None:
        """Test success and error alias methods."""
        success_value = "success"
        error_value = "error"

        success: Either[Any, Any] = Either.success(success_value)
        error: Either[str, Any] = Either.error(error_value)

        assert success.is_right() is True
        assert success.get_right() == success_value

        assert error.is_left() is True
        assert error.get_left() == error_value

    def test_is_success_is_error_aliases(self) -> None:
        """Test is_success and is_error alias methods."""
        success: Either[Any, Any] = Either.right("value")
        error: Either[str, Any] = Either.left("error")

        assert success.is_success() is True
        assert success.is_error() is False

        assert error.is_success() is False
        assert error.is_error() is True

    def test_value_properties(self) -> None:
        """Test value and error_value property aliases."""
        success_value = "success"
        error_value = "error"

        success: Either[Any, Any] = Either.right(success_value)
        error: Either[str, Any] = Either.left(error_value)

        assert success.value == success_value
        assert error.error_value == error_value

        # Test exceptions when accessing wrong property
        with pytest.raises(ValueError):
            _ = error.value

        with pytest.raises(ValueError):
            _ = success.error_value


class TestEitherMap:
    """Test Either map operations."""

    def test_map_right(self) -> None:
        """Test mapping over Right values."""
        right: Either[Any, Any] = Either.right(5)
        mapped = right.map(lambda x: x * 2)

        assert mapped.is_right() is True
        assert mapped.get_right() == 10

    def test_map_left(self) -> None:
        """Test mapping over Left values (should remain unchanged)."""
        left: Either[str, Any] = Either.left("error")
        mapped = left.map(lambda x: x * 2)

        assert mapped.is_left() is True
        assert mapped.get_left() == "error"

    def test_map_left_transform(self) -> None:
        """Test map_left transformation."""
        left: Either[str, Any] = Either.left("error")
        mapped = left.map_left(lambda x: f"transformed {x}")

        assert mapped.is_left() is True
        assert mapped.get_left() == "transformed error"

    def test_map_left_on_right(self) -> None:
        """Test map_left on Right values (should remain unchanged)."""
        right: Either[Any, Any] = Either.right(42)
        mapped = right.map_left(lambda x: f"transformed {x}")

        assert mapped.is_right() is True
        assert mapped.get_right() == 42

    def test_map_chaining(self) -> None:
        """Test chaining multiple map operations."""
        right: Either[Any, Any] = Either.right(3)
        result = right.map(lambda x: x * 2).map(lambda x: x + 1).map(lambda x: str(x))

        assert result.is_right() is True
        assert result.get_right() == "7"


class TestEitherFlatMap:
    """Test Either flat_map (monadic bind) operations."""

    def test_flat_map_right_to_right(self) -> None:
        """Test flat_map from Right to Right."""
        right: Either[Any, Any] = Either.right(5)
        result = right.flat_map(lambda x: Either.right(x * 2))

        assert result.is_right() is True
        assert result.get_right() == 10

    def test_flat_map_right_to_left(self) -> None:
        """Test flat_map from Right to Left."""
        right: Either[Any, Any] = Either.right(5)
        result: Either[str, Any] = right.flat_map(lambda x: Either.left("error in flat_map"))

        assert result.is_left() is True
        assert result.get_left() == "error in flat_map"

    def test_flat_map_left(self) -> None:
        """Test flat_map on Left values (should remain Left)."""
        left: Either[str, Any] = Either.left("original error")
        result = left.flat_map(lambda x: Either.right(x * 2))

        assert result.is_left() is True
        assert result.get_left() == "original error"

    def test_flat_map_chaining(self) -> None:
        """Test chaining multiple flat_map operations."""
        right: Either[Any, Any] = Either.right(3)
        result = (
            right.flat_map(lambda x: Either.right(x * 2))
            .flat_map(lambda x: Either.right(x + 1))
            .flat_map(lambda x: Either.right(str(x)))
        )

        assert result.is_right() is True
        assert result.get_right() == "7"

    def test_flat_map_chain_with_error(self) -> None:
        """Test flat_map chain that includes an error."""
        right: Either[Any, Any] = Either.right(3)
        result = (
            right.flat_map(lambda x: Either.right(x * 2))
            .flat_map(lambda x: Either.left("error in middle"))
            .flat_map(lambda x: Either.right(str(x)))
        )

        assert result.is_left() is True
        assert result.get_left() == "error in middle"


class TestEitherFilter:
    """Test Either filter operations."""

    def test_filter_right_passes(self) -> None:
        """Test filter on Right value that passes predicate."""
        right: Either[Any, Any] = Either.right(10)
        filtered = right.filter(lambda x: x > 5, "too small")

        assert filtered.is_right() is True
        assert filtered.get_right() == 10

    def test_filter_right_fails(self) -> None:
        """Test filter on Right value that fails predicate."""
        right: Either[Any, Any] = Either.right(3)
        filtered = right.filter(lambda x: x > 5, "too small")

        assert filtered.is_left() is True
        assert filtered.get_left() == "too small"

    def test_filter_left(self) -> None:
        """Test filter on Left value (should remain Left)."""
        left: Either[str, Any] = Either.left("original error")
        filtered = left.filter(lambda x: x > 5, "too small")

        assert filtered.is_left() is True
        assert filtered.get_left() == "original error"


class TestEitherFold:
    """Test Either fold operations."""

    def test_fold_right(self) -> None:
        """Test fold on Right value."""
        right: Either[Any, Any] = Either.right(42)
        result = right.fold(
            lambda x: f"Error: {x}",  # left_func
            lambda x: f"Success: {x}",  # right_func
        )

        assert result == "Success: 42"

    def test_fold_left(self) -> None:
        """Test fold on Left value."""
        left: Either[str, Any] = Either.left("failed")
        result = left.fold(
            lambda x: f"Error: {x}",  # left_func
            lambda x: f"Success: {x}",  # right_func
        )

        assert result == "Error: failed"

    def test_fold_with_different_return_types(self) -> None:
        """Test fold operations returning different types."""
        right: Either[Any, Any] = Either.right(5)
        left: Either[str, Any] = Either.left("error")

        # Both return integers
        right_result = right.fold(lambda x: 0, lambda x: x * 2)
        left_result = left.fold(lambda x: 0, lambda x: x * 2)

        assert right_result == 10
        assert left_result == 0


class TestEitherUtilityMethods:
    """Test Either utility methods."""

    def test_get_or_else_right(self) -> None:
        """Test get_or_else on Right value."""
        right: Either[Any, Any] = Either.right("success")
        result = right.get_or_else("default")

        assert result == "success"

    def test_get_or_else_left(self) -> None:
        """Test get_or_else on Left value."""
        left: Either[str, Any] = Either.left("error")
        result = left.get_or_else("default")

        assert result == "default"

    def test_or_else_right(self) -> None:
        """Test or_else on Right value."""
        right: Either[Any, Any] = Either.right("success")
        alternative: Either[Any, Any] = Either.right("alternative")
        result = right.or_else(alternative)

        assert result is right
        assert result.get_right() == "success"

    def test_or_else_left(self) -> None:
        """Test or_else on Left value."""
        left: Either[str, Any] = Either.left("error")
        alternative: Either[Any, Any] = Either.right("alternative")
        result = left.or_else(alternative)

        assert result is alternative
        assert result.get_right() == "alternative"

    def test_or_else_both_left(self) -> None:
        """Test or_else when both are Left values."""
        left: Either[str, Any] = Either.left("error")
        alternative: Either[str, Any] = Either.left("alternative error")
        result = left.or_else(alternative)

        assert result is alternative
        assert result.get_left() == "alternative error"


class TestEitherTryOperations:
    """Test Either try_* static methods."""

    def test_try_operation_success(self) -> None:
        """Test try_operation with successful operation."""

        def successful_operation() -> str:
            return "success result"

        def error_handler(e: Exception) -> str:
            return f"Error: {e}"

        result = Either.try_operation(successful_operation, error_handler)

        assert result.is_right() is True
        assert result.get_right() == "success result"

    def test_try_operation_failure(self) -> None:
        """Test try_operation with failing operation."""

        def failing_operation() -> str:
            raise ValueError("operation failed")

        def error_handler(e: Exception) -> str:
            return f"Error: {e}"

        result = Either.try_operation(failing_operation, error_handler)

        assert result.is_left() is True
        assert "Error: operation failed" in result.get_left()

    def test_try_either_success(self) -> None:
        """Test try_either with successful operation."""

        def successful_operation() -> str:
            return "success result"

        result = Either.try_either(successful_operation)

        assert result.is_right() is True
        assert result.get_right() == "success result"

    def test_try_either_failure(self) -> None:
        """Test try_either with failing operation."""

        def failing_operation() -> str:
            raise ValueError("operation failed")

        result = Either.try_either(failing_operation)

        assert result.is_left() is True
        assert isinstance(result.get_left(), ValueError)
        assert str(result.get_left()) == "operation failed"

    def test_from_optional_with_value(self) -> None:
        """Test from_optional with non-None value."""
        result = Either.from_optional("value", "error if None")

        assert result.is_right() is True
        assert result.get_right() == "value"

    def test_from_optional_with_none(self) -> None:
        """Test from_optional with None value."""
        result: Either[str, Any] = Either.from_optional(None, "error if None")

        assert result.is_left() is True
        assert result.get_left() == "error if None"


class TestEitherEquality:
    """Test Either equality and representation."""

    def test_left_equality(self) -> None:
        """Test equality for Left values."""
        left1: Either[str, Any] = Either.left("error")
        left2: Either[str, Any] = Either.left("error")
        left3: Either[str, Any] = Either.left("different error")

        assert left1 == left2
        assert left1 != left3
        assert left1 != Either.right("error")

    def test_right_equality(self) -> None:
        """Test equality for Right values."""
        right1: Either[Any, Any] = Either.right("value")
        right2: Either[Any, Any] = Either.right("value")
        right3: Either[Any, Any] = Either.right("different value")

        assert right1 == right2
        assert right1 != right3
        assert right1 != Either.left("value")

    def test_representation(self) -> None:
        """Test string representation of Either values."""
        left: Either[str, Any] = Either.left("error")
        right: Either[Any, Any] = Either.right("value")

        assert repr(left) == "Left(error)"
        assert repr(right) == "Right(value)"

    def test_equality_with_non_either(self) -> None:
        """Test equality with non-Either objects."""
        left: object = Either.left("error")
        right: object = Either.right("value")

        assert left != "error"
        assert right != "value"
        assert left is not None
        assert right is not None


class TestSequenceFunction:
    """Test sequence utility function."""

    def test_sequence_all_right(self) -> None:
        """Test sequence with all Right values."""
        eithers: list[Either[Any, int]] = [Either.right(1), Either.right(2), Either.right(3)]

        result = sequence(eithers)

        assert result.is_right() is True
        assert result.get_right() == [1, 2, 3]

    def test_sequence_with_left(self) -> None:
        """Test sequence with one Left value."""
        eithers: list[Either[str, int]] = [Either.right(1), Either.left("error"), Either.right(3)]

        result = sequence(eithers)

        assert result.is_left() is True
        assert result.get_left() == "error"

    def test_sequence_multiple_lefts(self) -> None:
        """Test sequence with multiple Left values (returns first)."""
        eithers: list[Either[str, int]] = [
            Either.right(1),
            Either.left("first error"),
            Either.left("second error"),
        ]

        result = sequence(eithers)

        assert result.is_left() is True
        assert result.get_left() == "first error"

    def test_sequence_empty_list(self) -> None:
        """Test sequence with empty list."""
        eithers: list[Either[Any, Any]] = []

        result = sequence(eithers)

        assert result.is_right() is True
        assert result.get_right() == []


class TestTraverseFunction:
    """Test traverse utility function."""

    def test_traverse_all_success(self) -> None:
        """Test traverse with all successful transformations."""
        values = [1, 2, 3]

        def safe_double(x: int) -> Either[Any, int]:
            return Either.right(x * 2)

        result = traverse(values, safe_double)

        assert result.is_right() is True
        assert result.get_right() == [2, 4, 6]

    def test_traverse_with_failure(self) -> None:
        """Test traverse with one failing transformation."""
        values = [1, 2, 3]

        def failing_on_two(x: int) -> Either[str, int]:
            if x == 2:
                return Either.left(f"error on {x}")
            return Either.right(x * 2)

        result = traverse(values, failing_on_two)

        assert result.is_left() is True
        assert result.get_left() == "error on 2"

    def test_traverse_empty_list(self) -> None:
        """Test traverse with empty list."""
        values: list[int] = []

        def transform(x: int) -> Either[Any, int]:
            return Either.right(x * 2)

        result = traverse(values, transform)

        assert result.is_right() is True
        assert result.get_right() == []

    def test_traverse_complex_transformation(self) -> None:
        """Test traverse with complex transformation logic."""
        values = ["1", "2", "invalid", "4"]

        def safe_int_parse(s: str) -> Either[str, int]:
            try:
                return Either.right(int(s))
            except ValueError:
                return Either.left(f"Cannot parse '{s}' as integer")

        result = traverse(values, safe_int_parse)

        assert result.is_left() is True
        assert result.get_left() == "Cannot parse 'invalid' as integer"


class TestPropertyBasedEither:
    """Property-based tests for Either functionality."""

    @given(st.text())
    def test_left_right_roundtrip(self, value: str) -> None:
        """Property: Creating Left/Right and getting value should be identity."""
        left: Either[str, Any] = Either.left(value)
        right: Either[Any, Any] = Either.right(value)

        assert left.get_left() == value
        assert right.get_right() == value

        assert left.is_left() is True
        assert left.is_right() is False
        assert right.is_left() is False
        assert right.is_right() is True

    @given(st.integers())
    def test_map_identity(self, value: int) -> None:
        """Property: Mapping with identity function should not change the value."""
        right: Either[Any, Any] = Either.right(value)
        mapped = right.map(lambda x: x)

        assert mapped.is_right() is True
        assert mapped.get_right() == value

    @given(st.integers())
    def test_flat_map_identity(self, value: int) -> None:
        """Property: flat_map with Either.right should be identity for Right values."""
        right: Either[Any, Any] = Either.right(value)
        flat_mapped = right.flat_map(lambda x: Either.right(x))

        assert flat_mapped.is_right() is True
        assert flat_mapped.get_right() == value

    @given(st.text())
    def test_left_operations_preserve_error(self, error: str) -> None:
        """Property: Operations on Left values should preserve the error."""
        left: Either[str, Any] = Either.left(error)

        # Map operations should preserve Left
        mapped = left.map(lambda x: x.upper())
        assert mapped.is_left() is True
        assert mapped.get_left() == error

        # flat_map operations should preserve Left
        flat_mapped = left.flat_map(lambda x: Either.right(x.upper()))
        assert flat_mapped.is_left() is True
        assert flat_mapped.get_left() == error

        # Filter operations should preserve Left
        filtered = left.filter(lambda x: True, "filter error")
        assert filtered.is_left() is True
        assert filtered.get_left() == error

    @given(st.lists(st.integers(), min_size=0, max_size=10))
    def test_sequence_all_rights_property(self, values: list[int]) -> None:
        """Property: Sequence of all Right values should return Right with list."""
        eithers: list[Either[Any, int]] = [Either.right(x) for x in values]
        result = sequence(eithers)

        assert result.is_right() is True
        assert result.get_right() == values


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_get_right_on_left_raises(self) -> None:
        """Test that get_right on Left raises appropriate error."""
        left: Either[str, Any] = Either.left("error")

        with pytest.raises(ValueError) as exc_info:
            left.get_right()

        assert "Cannot get Right value from Left" in str(exc_info.value)

    def test_get_left_on_right_raises(self) -> None:
        """Test that get_left on Right raises appropriate error."""
        right: Either[Any, Any] = Either.right("value")

        with pytest.raises(ValueError) as exc_info:
            right.get_left()

        assert "Cannot get Left value from Right" in str(exc_info.value)

    def test_value_property_on_left_raises(self) -> None:
        """Test that value property on Left raises appropriate error."""
        left: Either[str, Any] = Either.left("error")

        with pytest.raises(ValueError):
            _ = left.value

    def test_error_value_property_on_right_raises(self) -> None:
        """Test that error_value property on Right raises appropriate error."""
        right: Either[Any, Any] = Either.right("value")

        with pytest.raises(ValueError):
            _ = right.error_value


class TestComplexScenarios:
    """Test complex real-world usage scenarios."""

    def test_nested_either_operations(self) -> None:
        """Test complex nested Either operations."""

        def divide_safely(a: int, b: int) -> Either[str, float]:
            if b == 0:
                return Either.left("Division by zero")
            return Either.right(a / b)

        def sqrt_safely(x: float) -> Either[str, float]:
            if x < 0:
                return Either.left("Square root of negative number")
            return Either.right(x**0.5)

        # Test successful chain
        result = divide_safely(16, 4).flat_map(lambda x: sqrt_safely(x))

        assert result.is_right() is True
        assert result.get_right() == 2.0

        # Test failure in first operation
        result_fail1 = divide_safely(16, 0).flat_map(lambda x: sqrt_safely(x))

        assert result_fail1.is_left() is True
        assert result_fail1.get_left() == "Division by zero"

        # Test failure in second operation
        result_fail2 = divide_safely(-16, 4).flat_map(lambda x: sqrt_safely(x))

        assert result_fail2.is_left() is True
        assert result_fail2.get_left() == "Square root of negative number"

    def test_error_accumulation_pattern(self) -> None:
        """Test pattern for accumulating multiple errors."""

        def validate_positive(x: int, field_name: str) -> Either[list[str], int]:
            if x <= 0:
                return Either.left([f"{field_name} must be positive"])
            return Either.right(x)

        def validate_even(x: int, field_name: str) -> Either[list[str], int]:
            if x % 2 != 0:
                return Either.left([f"{field_name} must be even"])
            return Either.right(x)

        # Test with valid value
        value = 4
        result = validate_positive(value, "value").flat_map(
            lambda x: validate_even(x, "value")
        )

        assert result.is_right() is True
        assert result.get_right() == 4

        # Test with invalid value (negative)
        result_negative = validate_positive(-2, "value").flat_map(
            lambda x: validate_even(x, "value")
        )

        assert result_negative.is_left() is True
        assert result_negative.get_left() == ["value must be positive"]

    def test_resource_management_pattern(self) -> None:
        """Test Either for safe resource management."""

        class MockResource:
            def __init__(self, name: str, should_fail: bool = False):
                self.name = name
                self.should_fail = should_fail
                self.is_open = False

            def open(self) -> "MockResource":
                if self.should_fail:
                    raise RuntimeError(f"Failed to open {self.name}")
                self.is_open = True
                return self

            def process(self) -> str:
                if not self.is_open:
                    raise RuntimeError("Resource not open")
                return f"Processed {self.name}"

            def close(self) -> None:
                self.is_open = False

        def safe_resource_operation(
            resource_name: str, should_fail: bool = False
        ) -> Either[str, str]:
            return Either.try_operation(
                operation=lambda: MockResource(resource_name, should_fail)
                .open()
                .process(),
                error_handler=lambda e: str(e),
            )

        # Test successful operation
        result = safe_resource_operation("test_resource", False)
        assert result.is_right() is True
        assert result.get_right() == "Processed test_resource"

        # Test failed operation
        result_fail = safe_resource_operation("failing_resource", True)
        assert result_fail.is_left() is True
        assert "Failed to open failing_resource" in result_fail.get_left()
