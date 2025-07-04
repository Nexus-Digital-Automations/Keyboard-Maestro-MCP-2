"""
Tests for the design-by-contract system.

This module tests contract decorators, validation, and error handling
for preconditions, postconditions, and invariants.
"""

import pytest

from src.core import (
    require, ensure, invariant, combine_conditions, any_condition, not_condition,
    get_contract_info, is_not_none, is_positive, is_non_negative, is_valid_string,
    ContractViolationError
)


class TestContractDecorators:
    """Test cases for contract decorators."""
    
    def test_require_decorator_success(self):
        """Test successful precondition validation."""
        @require(lambda x: x > 0, "x must be positive")
        def sqrt(x):
            return x ** 0.5
        
        result = sqrt(4)
        assert result == 2.0
    
    def test_require_decorator_failure(self):
        """Test precondition failure."""
        @require(lambda x: x > 0, "x must be positive")
        def sqrt(x):
            return x ** 0.5
        
        with pytest.raises(ContractViolationError) as exc_info:
            sqrt(-1)
        
        assert "Precondition" in str(exc_info.value)
        assert "x must be positive" in str(exc_info.value)
    
    def test_ensure_decorator_success(self):
        """Test successful postcondition validation."""
        @ensure(lambda x, result: result >= 0, "result must be non-negative")
        def abs_value(x):
            return abs(x)
        
        result = abs_value(-5)
        assert result == 5
    
    def test_ensure_decorator_failure(self):
        """Test postcondition failure."""
        @ensure(lambda x, result: result >= 0, "result must be non-negative")
        def broken_abs(x):
            return -abs(x)  # Intentionally broken
        
        with pytest.raises(ContractViolationError) as exc_info:
            broken_abs(5)
        
        assert "Postcondition" in str(exc_info.value)
        assert "result must be non-negative" in str(exc_info.value)
    
    def test_multiple_contracts(self):
        """Test function with multiple contracts."""
        @require(lambda x: x >= 0, "x must be non-negative")
        @require(lambda x: isinstance(x, (int, float)), "x must be numeric")
        @ensure(lambda x, result: result >= 0, "result must be non-negative")
        def sqrt(x):
            return x ** 0.5
        
        # Valid case
        result = sqrt(9)
        assert result == 3.0
        
        # Invalid input type
        with pytest.raises(ContractViolationError):
            sqrt("invalid")
        
        # Invalid input value
        with pytest.raises(ContractViolationError):
            sqrt(-1)
    
    def test_contract_with_multiple_parameters(self):
        """Test contracts with multiple parameters."""
        @require(lambda a, b: a > 0 and b > 0, "both parameters must be positive")
        @ensure(lambda a, b, result: result == a / b, "result must equal a/b")
        def divide(a, b):
            return a / b
        
        result = divide(10, 2)
        assert result == 5.0
        
        with pytest.raises(ContractViolationError):
            divide(-1, 2)


class TestInvariantDecorator:
    """Test cases for class invariant decorator."""
    
    def test_invariant_success(self):
        """Test successful invariant validation."""
        @invariant(lambda self: self.balance >= 0, "balance cannot be negative")
        class Account:
            def __init__(self, balance):
                self.balance = balance
            
            def deposit(self, amount):
                self.balance += amount
            
            def withdraw(self, amount):
                if amount <= self.balance:
                    self.balance -= amount
        
        account = Account(100)
        assert account.balance == 100
        
        account.deposit(50)
        assert account.balance == 150
        
        account.withdraw(30)
        assert account.balance == 120
    
    def test_invariant_violation_in_constructor(self):
        """Test invariant violation during construction."""
        @invariant(lambda self: self.balance >= 0, "balance cannot be negative")
        class Account:
            def __init__(self, balance):
                self.balance = balance
        
        with pytest.raises(ContractViolationError):
            Account(-50)  # Should violate invariant
    
    def test_invariant_violation_in_method(self):
        """Test invariant violation during method execution."""
        @invariant(lambda self: self.balance >= 0, "balance cannot be negative")
        class Account:
            def __init__(self, balance):
                self.balance = balance
            
            def bad_withdraw(self, amount):
                self.balance -= amount  # No validation
        
        account = Account(100)
        
        with pytest.raises(ContractViolationError):
            account.bad_withdraw(150)  # Would make balance negative


class TestConditionCombiners:
    """Test cases for condition combining functions."""
    
    def test_combine_conditions(self):
        """Test combining conditions with AND logic."""
        def is_positive(x):
            return x > 0
        
        def is_even(x):
            return x % 2 == 0
        
        combined = combine_conditions(is_positive, is_even)
        
        assert combined(4) == True   # Positive and even
        assert combined(-2) == False # Negative but even
        assert combined(3) == False  # Positive but odd
        assert combined(-3) == False # Negative and odd
    
    def test_any_condition(self):
        """Test combining conditions with OR logic."""
        def is_negative(x):
            return x < 0
        
        def is_even(x):
            return x % 2 == 0
        
        any_cond = any_condition(is_negative, is_even)
        
        assert any_cond(-3) == True  # Negative
        assert any_cond(4) == True   # Even
        assert any_cond(-2) == True  # Both
        assert any_cond(3) == False  # Neither
    
    def test_not_condition(self):
        """Test negating a condition."""
        def is_positive(x):
            return x > 0
        
        is_not_positive = not_condition(is_positive)
        
        assert is_not_positive(-1) == True
        assert is_not_positive(0) == True
        assert is_not_positive(1) == False


class TestHelperConditions:
    """Test cases for helper condition functions."""
    
    def test_is_not_none(self):
        """Test is_not_none helper."""
        assert is_not_none("value") == True
        assert is_not_none(0) == True
        assert is_not_none([]) == True
        assert is_not_none(None) == False
    
    def test_is_positive(self):
        """Test is_positive helper."""
        assert is_positive(1) == True
        assert is_positive(0.1) == True
        assert is_positive(0) == False
        assert is_positive(-1) == False
    
    def test_is_non_negative(self):
        """Test is_non_negative helper."""
        assert is_non_negative(1) == True
        assert is_non_negative(0) == True
        assert is_non_negative(-1) == False
    
    def test_is_valid_string(self):
        """Test is_valid_string helper."""
        assert is_valid_string("hello") == True
        assert is_valid_string("a") == True
        assert is_valid_string("") == False  # Below min_length
        
        # Test with custom limits
        assert is_valid_string("abc", min_length=2, max_length=5) == True
        assert is_valid_string("a", min_length=2, max_length=5) == False
        assert is_valid_string("abcdef", min_length=2, max_length=5) == False
        
        # Test with non-string
        assert is_valid_string(123) == False


class TestContractInfo:
    """Test cases for contract information extraction."""
    
    def test_get_contract_info_function(self):
        """Test getting contract info from a function."""
        @require(lambda x: x > 0, "x must be positive")
        @ensure(lambda x, result: result > 0, "result must be positive")
        def sqrt(x):
            return x ** 0.5
        
        info = get_contract_info(sqrt)
        
        assert info["has_contracts"] == True
        assert info["preconditions"] == 1
        assert info["postconditions"] == 1
        assert info["invariants"] == 0
    
    def test_get_contract_info_class(self):
        """Test getting contract info from a class."""
        @invariant(lambda self: self.value >= 0, "value must be non-negative")
        class TestClass:
            def __init__(self, value):
                self.value = value
        
        info = get_contract_info(TestClass)
        
        assert info["has_contracts"] == True
        assert info["preconditions"] == 0
        assert info["postconditions"] == 0
        assert info["invariants"] == 1
    
    def test_get_contract_info_no_contracts(self):
        """Test getting contract info from function without contracts."""
        def simple_function(x):
            return x * 2
        
        info = get_contract_info(simple_function)
        
        assert info["has_contracts"] == False
        assert info["preconditions"] == 0
        assert info["postconditions"] == 0
        assert info["invariants"] == 0


class TestContractErrorHandling:
    """Test cases for contract error handling."""
    
    def test_contract_error_details(self):
        """Test that contract errors contain proper details."""
        @require(lambda x: x > 0, "x must be positive")
        def test_func(x):
            return x
        
        with pytest.raises(ContractViolationError) as exc_info:
            test_func(-1)
        
        error = exc_info.value
        assert error.contract_type == "Precondition"
        assert error.condition == "x must be positive"
        assert error.context is not None
    
    def test_contract_error_context(self):
        """Test that contract errors include execution context."""
        @require(lambda x, y: x > 0 and y > 0, "both parameters must be positive")
        def test_func(x, y):
            return x + y
        
        with pytest.raises(ContractViolationError) as exc_info:
            test_func(1, -1)
        
        error = exc_info.value
        assert error.context is not None
        assert error.context.operation == "precondition_check"
        assert error.context.component == "contract_validator"