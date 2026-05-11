"""Comprehensive tests for Either monad implementation.

import logging

logging.basicConfig(level=logging.DEBUG)
Tests cover all Either functionality including Left/Right creation,
transformations, error handling, and property-based testing.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.core.either import Either


# Create convenience aliases for test readability
def Left(value: Any) -> Either[Any, Any]:
    return Either.left(value)


def Right(value: Any) -> Either[Any, Any]:
    return Either.right(value)


class TestEitherBasics:
    """Test basic Either functionality."""

    def test_left_creation(self) -> None:
        """Test Left value creation and properties."""
        error = "Something went wrong"
        left = Left(error)

        assert left.is_left() is True
        assert left.is_right() is False
        assert left.get_left() == error

        with pytest.raises(ValueError):
            left.get_right()

    def test_right_creation(self) -> None:
        """Test Right value creation and properties."""
        value = 42
        right = Right(value)

        assert right.is_left() is False
        assert right.is_right() is True
        assert right.get_right() == value

        with pytest.raises(ValueError):
            right.get_left()

    def test_left_equality(self) -> None:
        """Test Left equality comparison."""
        left1 = Left("error")
        left2 = Left("error")
        left3 = Left("different")

        assert left1 == left2
        assert left1 != left3
        assert left1 != Right("error")

    def test_right_equality(self) -> None:
        """Test Right equality comparison."""
        right1 = Right(42)
        right2 = Right(42)
        right3 = Right(43)

        assert right1 == right2
        assert right1 != right3
        assert right1 != Left(42)


class TestEitherTransformations:
    """Test Either transformations (map, flat_map, etc.)."""

    def test_right_map(self) -> None:
        """Test mapping over Right values."""
        right = Right(5)
        mapped = right.map(lambda x: x * 2)

        assert mapped.is_right() is True
        assert mapped.get_right() == 10

    def test_left_map(self) -> None:
        """Test mapping over Left values (should remain unchanged)."""
        left = Left("error")
        mapped = left.map(lambda x: x * 2)

        assert mapped.is_left() is True
        assert mapped.get_left() == "error"

    def test_right_flat_map(self) -> None:
        """Test flat mapping over Right values."""
        right = Right(5)

        # Successful transformation
        flat_mapped = right.flat_map(lambda x: Right(x * 2))
        assert flat_mapped.is_right() is True
        assert flat_mapped.get_right() == 10

        # Transformation that returns Left
        flat_mapped_error = right.flat_map(lambda x: Left("error in transformation"))
        assert flat_mapped_error.is_left() is True
        assert flat_mapped_error.get_left() == "error in transformation"

    def test_left_flat_map(self) -> None:
        """Test flat mapping over Left values (should remain unchanged)."""
        left = Left("original error")
        flat_mapped = left.flat_map(lambda x: Right(x * 2))

        assert flat_mapped.is_left() is True
        assert flat_mapped.get_left() == "original error"

    def test_map_left(self) -> None:
        """Test mapping over Left values."""
        left = Left("error")
        mapped = left.map_left(lambda x: f"Transformed: {x}")

        assert mapped.is_left() is True
        assert mapped.get_left() == "Transformed: error"

    def test_right_map_left(self) -> None:
        """Test mapping Left on Right values (should remain unchanged)."""
        right = Right(42)
        mapped = right.map_left(lambda x: f"Transformed: {x}")

        assert mapped.is_right() is True
        assert mapped.get_right() == 42


class TestEitherFiltering:
    """Test Either filtering and conditional operations."""

    def test_right_filter_success(self) -> None:
        """Test filtering Right values that pass the predicate."""
        right = Right(10)
        filtered = right.filter(lambda x: x > 5, "Value too small")

        assert filtered.is_right() is True
        assert filtered.get_right() == 10

    def test_right_filter_failure(self) -> None:
        """Test filtering Right values that fail the predicate."""
        right = Right(3)
        filtered = right.filter(lambda x: x > 5, "Value too small")

        assert filtered.is_left() is True
        assert filtered.get_left() == "Value too small"

    def test_left_filter(self) -> None:
        """Test filtering Left values (should remain unchanged)."""
        left = Left("original error")
        filtered = left.filter(lambda x: x > 5, "Value too small")

        assert filtered.is_left() is True
        assert filtered.get_left() == "original error"


class TestEitherUtilities:
    """Test Either utility methods."""

    def test_fold_right(self) -> None:
        """Test folding Right values."""
        right = Right(42)
        result = right.fold(
            lambda x: f"Error: {x}",  # left_func (positional)
            lambda x: f"Success: {x}",  # right_func (positional)
        )

        assert result == "Success: 42"

    def test_fold_left(self) -> None:
        """Test folding Left values."""
        left = Left("error")
        result = left.fold(
            lambda x: f"Error: {x}",  # left_func (positional)
            lambda x: f"Success: {x}",  # right_func (positional)
        )

        assert result == "Error: error"

    def test_or_else_right(self) -> None:
        """Test or_else on Right values."""
        right = Right(42)
        result = right.or_else(Right(99))

        assert result.is_right() is True
        assert result.get_right() == 42

    def test_or_else_left(self) -> None:
        """Test or_else on Left values."""
        left = Left("error")
        result = left.or_else(Right(99))

        assert result.is_right() is True
        assert result.get_right() == 99

    def test_get_or_else_right(self) -> None:
        """Test get_or_else on Right values."""
        right = Right(42)
        result = right.get_or_else(99)

        assert result == 42

    def test_get_or_else_left(self) -> None:
        """Test get_or_else on Left values."""
        left = Left("error")
        result = left.get_or_else(99)

        assert result == 99


class TestEitherChaining:
    """Test chaining Either operations."""

    def test_successful_chain(self) -> None:
        """Test chaining successful operations."""
        result = (
            Right(5)
            .map(lambda x: x * 2)
            .flat_map(lambda x: Right(x + 1))
            .filter(lambda x: x > 10, "Too small")
        )

        assert result.is_right() is True
        assert result.get_right() == 11

    def test_chain_with_failure(self) -> None:
        """Test chaining with a failure in the middle."""
        result = (
            Right(5)
            .map(lambda x: x * 2)
            .flat_map(lambda x: Left("Error in middle"))
            .map(lambda x: x + 1)
        )

        assert result.is_left() is True
        assert result.get_left() == "Error in middle"

    def test_chain_starting_with_left(self) -> None:
        """Test chaining starting with Left value."""
        result = (
            Left("Initial error")
            .map(lambda x: x * 2)
            .flat_map(lambda x: Right(x + 1))
            .filter(lambda x: x > 10, "Too small")
        )

        assert result.is_left() is True
        assert result.get_left() == "Initial error"


class TestEitherConstructors:
    """Test Either construction helpers."""

    def test_try_success(self) -> None:
        """Test try_either with successful operation."""
        result = Either.try_either(lambda: 10 / 2)

        assert result.is_right() is True
        assert result.get_right() == 5.0

    def test_try_failure(self) -> None:
        """Test try_either with failing operation."""
        result = Either.try_either(lambda: 10 / 0)

        assert result.is_left() is True
        assert isinstance(result.get_left(), ZeroDivisionError)

    def test_from_optional_some(self) -> None:
        """Test creating Either from Some value."""
        result = Either.from_optional(42, "No value")

        assert result.is_right() is True
        assert result.get_right() == 42

    def test_from_optional_none(self) -> None:
        """Test creating Either from None value."""
        result: Either[str, Any] = Either.from_optional(None, "No value")

        assert result.is_left() is True
        assert result.get_left() == "No value"


class TestEitherAdditional:
    """Additional tests for Either to achieve 100% coverage."""

    def test_try_operation_success(self) -> None:
        """Test try_operation with successful operation."""

        def operation() -> int:
            return 42

        def error_handler(e: Exception) -> str:
            return f"Error: {str(e)}"

        result = Either.try_operation(operation, error_handler)

        assert result.is_right() is True
        assert result.get_right() == 42

    def test_try_operation_failure(self) -> None:
        """Test try_operation with failing operation."""

        def operation() -> int:
            raise ValueError("Test error")

        def error_handler(e: Exception) -> str:
            return f"Caught: {str(e)}"

        result = Either.try_operation(operation, error_handler)

        assert result.is_left() is True
        assert result.get_left() == "Caught: Test error"

    def test_success_alias(self) -> None:
        """Test Either.success() alias for right()."""
        result: Either[Any, int] = Either.success(42)

        assert result.is_right() is True
        assert result.get_right() == 42

    def test_error_alias(self) -> None:
        """Test Either.error() alias for left()."""
        result: Either[str, Any] = Either.error("Error message")

        assert result.is_left() is True
        assert result.get_left() == "Error message"

    def test_is_success_alias(self) -> None:
        """Test is_success() alias for is_right()."""
        success = Right(42)
        failure = Left("error")

        assert success.is_success() is True
        assert failure.is_success() is False

    def test_is_error_alias(self) -> None:
        """Test is_error() alias for is_left()."""
        success = Right(42)
        failure = Left("error")

        assert success.is_error() is False
        assert failure.is_error() is True

    def test_value_property(self) -> None:
        """Test value property alias for get_right()."""
        success = Right(42)

        assert success.value == 42

        # Test on Left should raise
        failure = Left("error")
        with pytest.raises(ValueError):
            _ = failure.value

    def test_error_value_property(self) -> None:
        """Test error_value property alias for get_left()."""
        failure = Left("error message")

        assert failure.error_value == "error message"

        # Test on Right should raise
        success = Right(42)
        with pytest.raises(ValueError):
            _ = success.error_value


class TestEitherSequenceTraverse:
    """Test sequence and traverse utility functions."""

    def test_sequence_all_right(self) -> None:
        """Test sequence with all Right values."""
        from src.core.either import sequence

        eithers = [Right(1), Right(2), Right(3)]
        result = sequence(eithers)

        assert result.is_right() is True
        assert result.get_right() == [1, 2, 3]

    def test_sequence_with_left(self) -> None:
        """Test sequence with a Left value."""
        from src.core.either import sequence

        eithers = [Right(1), Left("error"), Right(3)]
        result = sequence(eithers)

        assert result.is_left() is True
        assert result.get_left() == "error"

    def test_sequence_first_left_returned(self) -> None:
        """Test that sequence returns the first Left encountered."""
        from src.core.either import sequence

        eithers = [Right(1), Left("first error"), Left("second error")]
        result = sequence(eithers)

        assert result.is_left() is True
        assert result.get_left() == "first error"

    def test_sequence_empty_list(self) -> None:
        """Test sequence with empty list."""
        from src.core.either import sequence

        result: Either[Any, list[Any]] = sequence([])

        assert result.is_right() is True
        assert result.get_right() == []

    def test_traverse_all_success(self) -> None:
        """Test traverse with all successful operations."""
        from src.core.either import traverse

        def validate(x: int) -> Either[str, int]:
            if x > 0:
                return Right(x * 2)
            return Left(f"Invalid: {x}")

        result = traverse([1, 2, 3], validate)

        assert result.is_right() is True
        assert result.get_right() == [2, 4, 6]

    def test_traverse_with_failure(self) -> None:
        """Test traverse with a failing operation."""
        from src.core.either import traverse

        def validate(x: int) -> Either[str, int]:
            if x > 0:
                return Right(x * 2)
            return Left(f"Invalid: {x}")

        result = traverse([1, -2, 3], validate)

        assert result.is_left() is True
        assert result.get_left() == "Invalid: -2"

    def test_traverse_first_error_returned(self) -> None:
        """Test that traverse returns the first error encountered."""
        from src.core.either import traverse

        def validate(x: int) -> Either[str, int]:
            if x > 0:
                return Right(x * 2)
            return Left(f"Invalid: {x}")

        result = traverse([1, -2, -3], validate)

        assert result.is_left() is True
        assert result.get_left() == "Invalid: -2"

    def test_traverse_empty_list(self) -> None:
        """Test traverse with empty list."""
        from src.core.either import traverse

        def validate(x: int) -> Either[str, int]:
            return Right(x * 2)

        result = traverse([], validate)

        assert result.is_right() is True
        assert result.get_right() == []


class TestPropertyBasedEither:
    """Property-based tests for Either laws and invariants."""

    @given(st.integers())
    def test_right_identity_law(self, value: Any) -> None:
        """Test that Right values preserve identity under map with identity function."""
        right = Right(value)
        mapped = right.map(lambda x: x)

        assert mapped == right

    @given(st.text())
    def test_left_identity_law(self, error: str | Exception) -> None:
        """Test that Left values preserve identity under map."""
        left = Left(error)
        mapped = left.map(lambda x: x * 2)

        assert mapped == left

    @given(st.integers())
    def test_composition_law(self, value: Any) -> None:
        """Test that map composition is associative."""
        right = Right(value)

        def f(x: int) -> Any:
            return x + 1

        def g(x: Any) -> Any:
            return x * 2

        # map(g ∘ f) == map(g).map(f)
        composed1 = right.map(lambda x: g(f(x)))
        composed2 = right.map(f).map(g)

        assert composed1 == composed2

    @given(st.integers())
    def test_flat_map_identity_law(self, value: Any) -> None:
        """Test flat_map identity law: Right(a).flat_map(Right) == Right(a)."""
        right = Right(value)
        flat_mapped = right.flat_map(lambda x: Right(x))

        assert flat_mapped == right

    @given(st.text())
    def test_left_flat_map_invariant(self, error: str | Exception) -> None:
        """Test that flat_map on Left values returns the same Left."""
        left = Left(error)
        flat_mapped = left.flat_map(lambda x: Right(x * 2))

        assert flat_mapped == left

    @given(st.integers(), st.integers())
    def test_associativity_law(self, value: Any, multiplier: float) -> None:
        """Test flat_map associativity law."""
        right = Right(value)

        def f(x: int) -> Any:
            return Right(x + 1)

        def g(x: Any) -> Any:
            return Right(x * multiplier)

        # right.flat_map(f).flat_map(g) == right.flat_map(lambda x: f(x).flat_map(g))
        left_side = right.flat_map(f).flat_map(g)
        right_side = right.flat_map(lambda x: f(x).flat_map(g))

        assert left_side == right_side


class TestEitherTypeAnnotations:
    """Test Either with various types."""

    def test_string_either(self) -> None:
        """Test Either with string types."""
        success: Either[str, str] = Right("success")
        failure: Either[str, str] = Left("failure")

        assert success.is_right() is True
        assert failure.is_left() is True

    def test_complex_types(self) -> None:
        """Test Either with complex types."""
        data = {"key": "value", "number": 42}
        success: Either[str, dict] = Right(data)

        assert success.is_right() is True
        assert success.get_right() == data

    def test_nested_either(self) -> None:
        """Test nested Either operations."""

        def divide(x: float, y: float) -> Either[str, float]:
            if y == 0:
                return Left("Division by zero")
            return Right(x / y)

        def sqrt(x: float) -> Either[str, float]:
            if x < 0:
                return Left("Negative square root")
            return Right(x**0.5)

        # Chain operations
        result = (
            Right(16.0).flat_map(lambda x: divide(x, 4.0)).flat_map(lambda x: sqrt(x))
        )

        assert result.is_right() is True
        assert result.get_right() == 2.0

        # Chain with failure
        result_fail = (
            Right(16.0).flat_map(lambda x: divide(x, 0.0)).flat_map(lambda x: sqrt(x))
        )

        assert result_fail.is_left() is True
        assert result_fail.get_left() == "Division by zero"
