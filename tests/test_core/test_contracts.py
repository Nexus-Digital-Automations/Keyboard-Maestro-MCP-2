"""Tests for the design-by-contract system.

import logging

logging.basicConfig(level=logging.DEBUG)
This module tests contract decorators, validation, and error handling
for preconditions, postconditions, and invariants.
"""

from __future__ import annotations

from typing import Any

import pytest
from src.core import (
    ContractViolationError,
    any_condition,
    combine_conditions,
    ensure,
    get_contract_info,
    invariant,
    is_non_negative,
    is_not_none,
    is_positive,
    is_valid_string,
    not_condition,
    require,
)


class TestContractDecorators:
    """Test cases for contract decorators."""

    def test_require_decorator_success(self) -> None:
        """Test successful precondition validation."""

        @require(lambda x: x > 0, "x must be positive")
        def sqrt(x: Any) -> Any:
            return x**0.5

        result = sqrt(4)
        assert result == 2.0

    def test_require_decorator_failure(self) -> None:
        """Test precondition failure."""

        @require(lambda x: x > 0, "x must be positive")
        def sqrt(x: Any) -> Any:
            return x**0.5

        with pytest.raises(ContractViolationError) as exc_info:
            sqrt(-1)

        assert "Precondition" in str(exc_info.value)
        assert "x must be positive" in str(exc_info.value)

    def test_ensure_decorator_success(self) -> None:
        """Test successful postcondition validation."""

        @ensure(lambda x, result: result >= 0, "result must be non-negative")
        def abs_value(x: Any) -> Any:
            return abs(x)

        result = abs_value(-5)
        assert result == 5

    def test_ensure_decorator_failure(self) -> None:
        """Test postcondition failure."""

        @ensure(lambda x, result: result >= 0, "result must be non-negative")
        def broken_abs(x: Any) -> Any:
            return -abs(x)  # Intentionally broken

        with pytest.raises(ContractViolationError) as exc_info:
            broken_abs(5)

        assert "Postcondition" in str(exc_info.value)
        assert "result must be non-negative" in str(exc_info.value)

    def test_multiple_contracts(self) -> None:
        """Test function with multiple contracts."""

        @require(lambda x: x >= 0, "x must be non-negative")
        @require(lambda x: isinstance(x, int | float), "x must be numeric")
        @ensure(lambda x, result: result >= 0, "result must be non-negative")
        def sqrt(x: Any) -> Any:
            return x**0.5

        # Valid case
        result = sqrt(9)
        assert result == 3.0

        # Invalid input type
        with pytest.raises(ContractViolationError):
            sqrt("invalid")

        # Invalid input value
        with pytest.raises(ContractViolationError):
            sqrt(-1)

    def test_contract_with_multiple_parameters(self) -> None:
        """Test contracts with multiple parameters."""

        @require(lambda a, b: a > 0 and b > 0, "both parameters must be positive")
        @ensure(lambda a, b, result: result == a / b, "result must equal a/b")
        def divide(a: Any, b: Any) -> Any:
            return a / b

        result = divide(10, 2)
        assert result == 5.0

        with pytest.raises(ContractViolationError):
            divide(-1, 2)


class TestInvariantDecorator:
    """Test cases for class invariant decorator."""

    def test_invariant_success(self) -> None:
        """Test successful invariant validation."""

        @invariant(lambda self: self.balance >= 0, "balance cannot be negative")
        class Account:
            def __init__(self, balance: Any) -> None:
                self.balance = balance

            def deposit(self, amount: Any) -> None:
                self.balance += amount

            def withdraw(self, amount: Any) -> None:
                if amount <= self.balance:
                    self.balance -= amount

        account = Account(100)
        assert account.balance == 100

        account.deposit(50)
        assert account.balance == 150

        account.withdraw(30)
        assert account.balance == 120

    def test_invariant_violation_in_constructor(self) -> None:
        """Test invariant violation during construction."""

        @invariant(lambda self: self.balance >= 0, "balance cannot be negative")
        class Account:
            def __init__(self, balance: Any) -> None:
                self.balance = balance

        with pytest.raises(ContractViolationError):
            Account(-50)  # Should violate invariant

    def test_invariant_violation_in_method(self) -> None:
        """Test invariant violation during method execution."""

        @invariant(lambda self: self.balance >= 0, "balance cannot be negative")
        class Account:
            def __init__(self, balance: Any) -> None:
                self.balance = balance

            def bad_withdraw(self, amount: Any) -> None:
                self.balance -= amount  # No validation

        account = Account(100)

        with pytest.raises(ContractViolationError):
            account.bad_withdraw(150)  # Would make balance negative


class TestConditionCombiners:
    """Test cases for condition combining functions."""

    def test_combine_conditions(self) -> bool:
        """Test combining conditions with AND logic."""

        def is_positive(x: Any) -> bool:
            return x > 0

        def is_even(x: Any) -> bool:
            return x % 2 == 0

        combined = combine_conditions(is_positive, is_even)

        assert combined(4)  # Positive and even
        assert not combined(-2)  # Negative but even
        assert not combined(3)  # Positive but odd
        assert not combined(-3)  # Negative and odd

    def test_any_condition(self) -> bool:
        """Test combining conditions with OR logic."""

        def is_negative(x: Any) -> bool:
            return x < 0

        def is_even(x: Any) -> bool:
            return x % 2 == 0

        any_cond = any_condition(is_negative, is_even)

        assert any_cond(-3)  # Negative
        assert any_cond(4)  # Even
        assert any_cond(-2)  # Both
        assert not any_cond(3)  # Neither

    def test_not_condition(self) -> bool:
        """Test negating a condition."""

        def is_positive(x: Any) -> bool:
            return x > 0

        is_not_positive = not_condition(is_positive)

        assert is_not_positive(-1)
        assert is_not_positive(0)
        assert not is_not_positive(1)


class TestHelperConditions:
    """Test cases for helper condition functions."""

    def test_is_not_none(self) -> None:
        """Test is_not_none helper."""
        assert is_not_none("value")
        assert is_not_none(0)
        assert is_not_none([])
        assert not is_not_none(None)

    def test_is_positive(self) -> None:
        """Test is_positive helper."""
        assert is_positive(1)
        assert is_positive(0.1)
        assert not is_positive(0)
        assert not is_positive(-1)

    def test_is_non_negative(self) -> None:
        """Test is_non_negative helper."""
        assert is_non_negative(1)
        assert is_non_negative(0)
        assert not is_non_negative(-1)

    def test_is_valid_string(self) -> None:
        """Test is_valid_string helper."""
        assert is_valid_string("hello")
        assert is_valid_string("a")
        assert not is_valid_string("")  # Below min_length

        # Test with custom limits
        assert is_valid_string("abc", min_length=2, max_length=5)
        assert not is_valid_string("a", min_length=2, max_length=5)
        assert not is_valid_string("abcdef", min_length=2, max_length=5)

        # Test with non-string
        assert not is_valid_string(123)


class TestContractInfo:
    """Test cases for contract information extraction."""

    def test_get_contract_info_function(self) -> None:
        """Test getting contract info from a function."""

        @require(lambda x: x > 0, "x must be positive")
        @ensure(lambda x, result: result > 0, "result must be positive")
        def sqrt(x: Any) -> Any:
            return x**0.5

        info = get_contract_info(sqrt)

        assert info["has_contracts"]
        assert info["preconditions"] == 1
        assert info["postconditions"] == 1
        assert info["invariants"] == 0

    def test_get_contract_info_class(self) -> None:
        """Test getting contract info from a class."""

        @invariant(lambda self: self.value >= 0, "value must be non-negative")
        class TestClass:
            def __init__(self, value: Any) -> None:
                self.value = value

        info = get_contract_info(TestClass)

        assert info["has_contracts"]
        assert info["preconditions"] == 0
        assert info["postconditions"] == 0
        assert info["invariants"] == 1

    def test_get_contract_info_no_contracts(self) -> None:
        """Test getting contract info from function without contracts."""

        def simple_function(x: Any) -> Any:
            return x * 2

        info = get_contract_info(simple_function)

        assert not info["has_contracts"]
        assert info["preconditions"] == 0
        assert info["postconditions"] == 0
        assert info["invariants"] == 0


class TestContractErrorHandling:
    """Test cases for contract error handling."""

    def test_contract_error_details(self) -> None:
        """Test that contract errors contain proper details."""

        @require(lambda x: x > 0, "x must be positive")
        def test_func(x: Any) -> None:
            return x

        with pytest.raises(ContractViolationError) as exc_info:
            test_func(-1)

        error = exc_info.value
        assert error.contract_type == "Precondition"
        assert error.condition == "x must be positive"
        assert error.context is not None

    def test_contract_error_context(self) -> None:
        """Test that contract errors include execution context."""

        @require(lambda x, y: x > 0 and y > 0, "both parameters must be positive")
        def test_func(x: Any, y: Any) -> None:
            return x + y

        with pytest.raises(ContractViolationError) as exc_info:
            test_func(1, -1)

        error = exc_info.value
        assert error.context is not None
        assert error.context.operation == "precondition_check"
        assert error.context.component == "contract_validator"
