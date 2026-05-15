"""Expanded comprehensive tests for contracts module to achieve 100% coverage.

This module extends the existing contracts tests to cover the remaining uncovered lines
focusing on async function contract violations, keyword argument handling, and
invariant error context management.
"""

import asyncio

import pytest
from src.core.contracts import (
    ContractValidator,
    ensure,
    invariant,
    require,
)
from src.core.errors import ContractViolationError


class TestContractValidatorKeywordArgs:
    """Test ContractValidator with keyword arguments coverage."""

    def test_evaluate_condition_with_keyword_args(self) -> None:
        """Test evaluate_condition with keyword arguments - covers lines 46-47."""

        def condition_with_kwargs(value: int, multiplier: int = 2) -> bool:
            return value * multiplier > 10

        # Test with keyword arguments that match parameter names
        result = ContractValidator.evaluate_condition(
            condition_with_kwargs,
            args=(5,),  # value=5
            kwargs={"multiplier": 3},  # multiplier=3, 5*3=15 > 10
            result=None,
        )
        assert result is True

        # Test with keyword arguments that don't match parameter names (should be ignored)
        result = ContractValidator.evaluate_condition(
            condition_with_kwargs,
            args=(3,),  # value=3
            kwargs={"unknown_param": 999, "multiplier": 2},  # 3*2=6 <= 10
            result=None,
        )
        assert result is False

    def test_evaluate_condition_mixed_args_kwargs(self) -> None:
        """Test evaluate_condition with mixed positional and keyword arguments."""

        def complex_condition(x: int, y: int, z: int = 1) -> bool:
            return x + y + z == 10

        # Test: x=3 (positional), y=4 (positional), z=3 (keyword)
        result = ContractValidator.evaluate_condition(
            complex_condition,
            args=(3, 4),  # x=3, y=4
            kwargs={"z": 3},  # z=3, total: 3+4+3=10
            result=None,
        )
        assert result is True

        # Test keyword arguments overriding defaults
        result = ContractValidator.evaluate_condition(
            complex_condition,
            args=(2, 5),  # x=2, y=5
            kwargs={"z": 4},  # z=4, total: 2+5+4=11 != 10
            result=None,
        )
        assert result is False


class TestAsyncContractViolations:
    """Test async function contract violations for complete coverage."""

    @pytest.mark.asyncio
    async def test_async_precondition_violation(self) -> None:
        """Test async function precondition violation - covers lines 97-110."""

        @require(lambda x: x > 0, "Value must be positive")
        async def async_positive_operation(x: int) -> int:
            return x * 2

        # Test successful case first
        result = await async_positive_operation(5)
        assert result == 10

        # Test precondition violation
        with pytest.raises(ContractViolationError) as exc_info:
            await async_positive_operation(-1)

        error = exc_info.value
        assert error.contract_type == "Precondition"
        assert "Value must be positive" in error.condition
        assert error.context is not None
        assert "precondition_check" in error.context.operation

    @pytest.mark.asyncio
    async def test_async_postcondition_violation(self) -> None:
        """Test async function postcondition violation - covers lines 170-191."""

        @ensure(lambda result: result > 0, "Result must be positive")
        async def async_possibly_negative_operation(x: int) -> int:
            # This function might return negative results, violating postcondition
            return x - 10

        # Test successful case
        result = await async_possibly_negative_operation(15)
        assert result == 5

        # Test postcondition violation
        with pytest.raises(ContractViolationError) as exc_info:
            await async_possibly_negative_operation(5)  # Result will be -5

        error = exc_info.value
        assert error.contract_type == "Postcondition"
        assert "Result must be positive" in error.condition
        assert error.context is not None
        assert "postcondition_check" in error.context.operation
        assert "result_type" in error.context.metadata

    @pytest.mark.asyncio
    async def test_async_complex_contract_chain(self) -> None:
        """Test async function with both precondition and postcondition."""

        @require(lambda x: x >= 0, "Input must be non-negative")
        @ensure(lambda result: result >= 0, "Result must be non-negative")
        async def async_safe_sqrt(x: float) -> float:
            return x**0.5

        # Test successful execution
        result = await async_safe_sqrt(16.0)
        assert result == 4.0

        # Test precondition violation
        with pytest.raises(ContractViolationError) as exc_info:
            await async_safe_sqrt(-4.0)

        error = exc_info.value
        assert error.contract_type == "Precondition"


class TestInvariantErrorContext:
    """Test invariant error context handling - covers lines 292-295."""

    def test_invariant_violation_with_context(self) -> None:
        """Test invariant violation error context enhancement."""

        @invariant(lambda self: self.value >= 0)
        class PositiveValue:
            def __init__(self, value: int) -> None:
                self.value = value

            def subtract(self, amount: int) -> None:
                """Method that might violate invariant."""
                self.value -= amount

            def add(self, amount: int) -> None:
                """Method that should maintain invariant."""
                self.value += amount

        # Test successful operations
        obj = PositiveValue(10)
        obj.add(5)
        assert obj.value == 15

        obj.subtract(3)
        assert obj.value == 12

        # Test invariant violation with error context
        with pytest.raises(ContractViolationError) as exc_info:
            obj.subtract(20)  # This would make value negative

        error = exc_info.value
        assert error.contract_type == "Invariant"
        # Check that method context was added to error - covers lines 292-295
        assert "subtract:" in str(error.context) or "subtract" in str(error)

    def test_invariant_multiple_method_violations(self) -> None:
        """Test invariant violations in different methods have proper context."""

        @invariant(lambda self: len(self.items) <= self.max_capacity)
        class LimitedContainer:
            def __init__(self, max_capacity: int) -> None:
                self.items = []
                self.max_capacity = max_capacity

            def add_item(self, item: str) -> None:
                self.items.append(item)

            def add_multiple(self, items: list[str]) -> None:
                self.items.extend(items)

        container = LimitedContainer(max_capacity=2)

        # Test successful addition
        container.add_item("item1")
        container.add_item("item2")
        assert len(container.items) == 2

        # Test invariant violation in add_item method
        with pytest.raises(ContractViolationError) as exc_info:
            container.add_item("item3")  # Exceeds capacity

        error = exc_info.value
        assert error.contract_type == "Invariant"

        # Test invariant violation in add_multiple method
        container2 = LimitedContainer(max_capacity=1)
        with pytest.raises(ContractViolationError) as exc_info:
            container2.add_multiple(["item1", "item2"])  # Exceeds capacity

        error = exc_info.value
        assert error.contract_type == "Invariant"


class TestComplexContractScenarios:
    """Test complex contract scenarios for edge case coverage."""

    def test_condition_evaluation_with_result_parameter(self) -> None:
        """Test condition evaluation that uses result parameter."""

        def condition_using_result(x: int, result: int) -> bool:
            return result == x * 2

        # Test with result parameter
        is_valid = ContractValidator.evaluate_condition(
            condition_using_result,
            args=(5,),  # x=5
            kwargs={},
            result=10,  # result=10, 10 == 5*2 is True
        )
        assert is_valid is True

        # Test with wrong result
        is_valid = ContractValidator.evaluate_condition(
            condition_using_result,
            args=(5,),  # x=5
            kwargs={},
            result=9,  # result=9, 9 != 5*2 is False
        )
        assert is_valid is False

    def test_condition_evaluation_exception_handling(self) -> None:
        """Test that condition evaluation exceptions are handled gracefully."""

        def problematic_condition(x: int) -> bool:
            # This will raise ZeroDivisionError when x=0
            return 10 / x > 1

        # Test normal case
        result = ContractValidator.evaluate_condition(
            problematic_condition,
            args=(5,),  # 10/5 = 2 > 1 is True
            kwargs={},
            result=None,
        )
        assert result is True

        # Test exception case - should return False, not raise exception
        result = ContractValidator.evaluate_condition(
            problematic_condition,
            args=(0,),  # This will cause ZeroDivisionError
            kwargs={},
            result=None,
        )
        assert result is False  # Exception caught and treated as contract violation

    @pytest.mark.asyncio
    async def test_async_contract_with_complex_conditions(self) -> None:
        """Test async contracts with complex multi-parameter conditions."""

        @require(
            lambda data, threshold: len(data) > 0 and threshold > 0,
            "Data must be non-empty and threshold positive",
        )
        @ensure(
            lambda result: isinstance(result, list) and len(result) >= 0,
            "Result must be a non-empty list",
        )
        async def async_filter_data(data: list[int], threshold: int) -> list[int]:
            # Simulate async processing
            await asyncio.sleep(0.001)
            return [x for x in data if x > threshold]

        # Test successful execution
        result = await async_filter_data([1, 5, 10, 15], 7)
        assert result == [10, 15]

        # Test precondition violation - empty data
        with pytest.raises(ContractViolationError):
            await async_filter_data([], 5)

        # Test precondition violation - zero threshold
        with pytest.raises(ContractViolationError):
            await async_filter_data([1, 2, 3], 0)


class TestContractIntegration:
    """Test integration between different contract types."""

    def test_combined_contracts_on_class(self) -> None:
        """Test class with multiple types of contracts."""

        @invariant(lambda self: self.balance >= 0)
        class BankAccount:
            def __init__(self, initial_balance: float) -> None:
                self.balance = initial_balance

            @require(lambda self, amount: amount > 0, "Deposit amount must be positive")
            @ensure(
                lambda self, amount, result: self.balance >= 0,
                "Balance must remain non-negative after deposit",
            )
            def deposit(self, amount: float) -> float:
                old_balance = self.balance
                self.balance += amount
                return old_balance

            @require(
                lambda self, amount: amount > 0 and amount <= self.balance,
                "Withdrawal amount must be positive and not exceed balance",
            )
            @ensure(
                lambda self, amount, result: self.balance >= 0,
                "Balance must remain non-negative after withdrawal",
            )
            def withdraw(self, amount: float) -> float:
                old_balance = self.balance
                self.balance -= amount
                return old_balance

        # Test successful operations
        account = BankAccount(100.0)

        old_balance = account.deposit(50.0)
        assert old_balance == 100.0
        assert account.balance == 150.0

        old_balance = account.withdraw(30.0)
        assert old_balance == 150.0
        assert account.balance == 120.0

        # Test various contract violations

        # Deposit precondition violation
        with pytest.raises(ContractViolationError):
            account.deposit(-10.0)

        # Withdrawal precondition violation (amount too large)
        with pytest.raises(ContractViolationError):
            account.withdraw(200.0)

        # Withdrawal precondition violation (negative amount)
        with pytest.raises(ContractViolationError):
            account.withdraw(-5.0)
