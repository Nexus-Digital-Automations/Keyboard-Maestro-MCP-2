"""
Comprehensive tests for Either monad implementation.

Tests cover all Either functionality including Left/Right creation,
transformations, error handling, and property-based testing.
"""

import pytest
from typing import Union
from hypothesis import given, strategies as st

from src.core.either import Either, Left, Right


class TestEitherBasics:
    """Test basic Either functionality."""
    
    def test_left_creation(self):
        """Test Left value creation and properties."""
        error = "Something went wrong"
        left = Left(error)
        
        assert left.is_left() is True
        assert left.is_right() is False
        assert left.get_left() == error
        
        with pytest.raises(ValueError):
            left.get_right()
    
    def test_right_creation(self):
        """Test Right value creation and properties."""
        value = 42
        right = Right(value)
        
        assert right.is_left() is False
        assert right.is_right() is True
        assert right.get_right() == value
        
        with pytest.raises(ValueError):
            right.get_left()
    
    def test_left_equality(self):
        """Test Left equality comparison."""
        left1 = Left("error")
        left2 = Left("error")
        left3 = Left("different")
        
        assert left1 == left2
        assert left1 != left3
        assert left1 != Right("error")
    
    def test_right_equality(self):
        """Test Right equality comparison."""
        right1 = Right(42)
        right2 = Right(42)
        right3 = Right(43)
        
        assert right1 == right2
        assert right1 != right3
        assert right1 != Left(42)


class TestEitherTransformations:
    """Test Either transformations (map, flat_map, etc.)."""
    
    def test_right_map(self):
        """Test mapping over Right values."""
        right = Right(5)
        mapped = right.map(lambda x: x * 2)
        
        assert mapped.is_right() is True
        assert mapped.get_right() == 10
    
    def test_left_map(self):
        """Test mapping over Left values (should remain unchanged)."""
        left = Left("error")
        mapped = left.map(lambda x: x * 2)
        
        assert mapped.is_left() is True
        assert mapped.get_left() == "error"
    
    def test_right_flat_map(self):
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
    
    def test_left_flat_map(self):
        """Test flat mapping over Left values (should remain unchanged)."""
        left = Left("original error")
        flat_mapped = left.flat_map(lambda x: Right(x * 2))
        
        assert flat_mapped.is_left() is True
        assert flat_mapped.get_left() == "original error"
    
    def test_map_left(self):
        """Test mapping over Left values."""
        left = Left("error")
        mapped = left.map_left(lambda x: f"Transformed: {x}")
        
        assert mapped.is_left() is True
        assert mapped.get_left() == "Transformed: error"
    
    def test_right_map_left(self):
        """Test mapping Left on Right values (should remain unchanged)."""
        right = Right(42)
        mapped = right.map_left(lambda x: f"Transformed: {x}")
        
        assert mapped.is_right() is True
        assert mapped.get_right() == 42


class TestEitherFiltering:
    """Test Either filtering and conditional operations."""
    
    def test_right_filter_success(self):
        """Test filtering Right values that pass the predicate."""
        right = Right(10)
        filtered = right.filter(lambda x: x > 5, "Value too small")
        
        assert filtered.is_right() is True
        assert filtered.get_right() == 10
    
    def test_right_filter_failure(self):
        """Test filtering Right values that fail the predicate."""
        right = Right(3)
        filtered = right.filter(lambda x: x > 5, "Value too small")
        
        assert filtered.is_left() is True
        assert filtered.get_left() == "Value too small"
    
    def test_left_filter(self):
        """Test filtering Left values (should remain unchanged)."""
        left = Left("original error")
        filtered = left.filter(lambda x: x > 5, "Value too small")
        
        assert filtered.is_left() is True
        assert filtered.get_left() == "original error"


class TestEitherUtilities:
    """Test Either utility methods."""
    
    def test_fold_right(self):
        """Test folding Right values."""
        right = Right(42)
        result = right.fold(
            left_func=lambda x: f"Error: {x}",
            right_func=lambda x: f"Success: {x}"
        )
        
        assert result == "Success: 42"
    
    def test_fold_left(self):
        """Test folding Left values."""
        left = Left("error")
        result = left.fold(
            left_func=lambda x: f"Error: {x}",
            right_func=lambda x: f"Success: {x}"
        )
        
        assert result == "Error: error"
    
    def test_or_else_right(self):
        """Test or_else on Right values."""
        right = Right(42)
        result = right.or_else(Right(99))
        
        assert result.is_right() is True
        assert result.get_right() == 42
    
    def test_or_else_left(self):
        """Test or_else on Left values."""
        left = Left("error")
        result = left.or_else(Right(99))
        
        assert result.is_right() is True
        assert result.get_right() == 99
    
    def test_get_or_else_right(self):
        """Test get_or_else on Right values."""
        right = Right(42)
        result = right.get_or_else(99)
        
        assert result == 42
    
    def test_get_or_else_left(self):
        """Test get_or_else on Left values."""
        left = Left("error")
        result = left.get_or_else(99)
        
        assert result == 99


class TestEitherChaining:
    """Test chaining Either operations."""
    
    def test_successful_chain(self):
        """Test chaining successful operations."""
        result = (Right(5)
                 .map(lambda x: x * 2)
                 .flat_map(lambda x: Right(x + 1))
                 .filter(lambda x: x > 10, "Too small"))
        
        assert result.is_right() is True
        assert result.get_right() == 11
    
    def test_chain_with_failure(self):
        """Test chaining with a failure in the middle."""
        result = (Right(5)
                 .map(lambda x: x * 2)
                 .flat_map(lambda x: Left("Error in middle"))
                 .map(lambda x: x + 1))
        
        assert result.is_left() is True
        assert result.get_left() == "Error in middle"
    
    def test_chain_starting_with_left(self):
        """Test chaining starting with Left value."""
        result = (Left("Initial error")
                 .map(lambda x: x * 2)
                 .flat_map(lambda x: Right(x + 1))
                 .filter(lambda x: x > 10, "Too small"))
        
        assert result.is_left() is True
        assert result.get_left() == "Initial error"


class TestEitherConstructors:
    """Test Either construction helpers."""
    
    def test_try_success(self):
        """Test try_either with successful operation."""
        result = Either.try_either(lambda: 10 / 2)
        
        assert result.is_right() is True
        assert result.get_right() == 5.0
    
    def test_try_failure(self):
        """Test try_either with failing operation."""
        result = Either.try_either(lambda: 10 / 0)
        
        assert result.is_left() is True
        assert isinstance(result.get_left(), ZeroDivisionError)
    
    def test_from_optional_some(self):
        """Test creating Either from Some value."""
        result = Either.from_optional(42, "No value")
        
        assert result.is_right() is True
        assert result.get_right() == 42
    
    def test_from_optional_none(self):
        """Test creating Either from None value."""
        result = Either.from_optional(None, "No value")
        
        assert result.is_left() is True
        assert result.get_left() == "No value"


class TestPropertyBasedEither:
    """Property-based tests for Either laws and invariants."""
    
    @given(st.integers())
    def test_right_identity_law(self, value):
        """Test that Right values preserve identity under map with identity function."""
        right = Right(value)
        mapped = right.map(lambda x: x)
        
        assert mapped == right
    
    @given(st.text())
    def test_left_identity_law(self, error):
        """Test that Left values preserve identity under map."""
        left = Left(error)
        mapped = left.map(lambda x: x * 2)
        
        assert mapped == left
    
    @given(st.integers())
    def test_composition_law(self, value):
        """Test that map composition is associative."""
        right = Right(value)
        f = lambda x: x + 1
        g = lambda x: x * 2
        
        # map(g ∘ f) == map(g).map(f)
        composed1 = right.map(lambda x: g(f(x)))
        composed2 = right.map(f).map(g)
        
        assert composed1 == composed2
    
    @given(st.integers())
    def test_flat_map_identity_law(self, value):
        """Test flat_map identity law: Right(a).flat_map(Right) == Right(a)."""
        right = Right(value)
        flat_mapped = right.flat_map(lambda x: Right(x))
        
        assert flat_mapped == right
    
    @given(st.text())
    def test_left_flat_map_invariant(self, error):
        """Test that flat_map on Left values returns the same Left."""
        left = Left(error)
        flat_mapped = left.flat_map(lambda x: Right(x * 2))
        
        assert flat_mapped == left
    
    @given(st.integers(), st.integers())
    def test_associativity_law(self, value, multiplier):
        """Test flat_map associativity law."""
        right = Right(value)
        f = lambda x: Right(x + 1)
        g = lambda x: Right(x * multiplier)
        
        # right.flat_map(f).flat_map(g) == right.flat_map(lambda x: f(x).flat_map(g))
        left_side = right.flat_map(f).flat_map(g)
        right_side = right.flat_map(lambda x: f(x).flat_map(g))
        
        assert left_side == right_side


class TestEitherTypeAnnotations:
    """Test Either with various types."""
    
    def test_string_either(self):
        """Test Either with string types."""
        success: Either[str, str] = Right("success")
        failure: Either[str, str] = Left("failure")
        
        assert success.is_right() is True
        assert failure.is_left() is True
    
    def test_complex_types(self):
        """Test Either with complex types."""
        data = {"key": "value", "number": 42}
        success: Either[str, dict] = Right(data)
        
        assert success.is_right() is True
        assert success.get_right() == data
    
    def test_nested_either(self):
        """Test nested Either operations."""
        def divide(x: float, y: float) -> Either[str, float]:
            if y == 0:
                return Left("Division by zero")
            return Right(x / y)
        
        def sqrt(x: float) -> Either[str, float]:
            if x < 0:
                return Left("Negative square root")
            return Right(x ** 0.5)
        
        # Chain operations
        result = (Right(16.0)
                 .flat_map(lambda x: divide(x, 4.0))
                 .flat_map(lambda x: sqrt(x)))
        
        assert result.is_right() is True
        assert result.get_right() == 2.0
        
        # Chain with failure
        result_fail = (Right(16.0)
                      .flat_map(lambda x: divide(x, 0.0))
                      .flat_map(lambda x: sqrt(x)))
        
        assert result_fail.is_left() is True
        assert result_fail.get_left() == "Division by zero"