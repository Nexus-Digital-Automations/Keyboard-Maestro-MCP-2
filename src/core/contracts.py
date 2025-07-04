"""
Design by Contract implementation for the Keyboard Maestro MCP macro engine.

This module provides decorators for preconditions, postconditions, and invariants
to ensure robust and reliable macro execution with comprehensive validation.
"""

from __future__ import annotations
from typing import Callable, Any, TypeVar, Optional, Dict, List
from functools import wraps
import inspect
from .errors import ContractViolationError, create_error_context


# Type variables
F = TypeVar('F', bound=Callable[..., Any])


class ContractValidator:
    """Utility class for contract validation logic."""
    
    @staticmethod
    def evaluate_condition(
        condition: Callable[..., bool],
        args: tuple,
        kwargs: dict,
        result: Any = None
    ) -> bool:
        """Safely evaluate a contract condition."""
        try:
            # Get the condition's signature to determine parameter binding
            sig = inspect.signature(condition)
            
            # Try to bind all available arguments to the condition
            condition_kwargs = {}
            param_names = list(sig.parameters.keys())
            
            # Map positional arguments to parameter names
            for i, arg in enumerate(args):
                if i < len(param_names) and param_names[i] != 'result':
                    condition_kwargs[param_names[i]] = arg
            
            # Add keyword arguments
            for key, value in kwargs.items():
                if key in param_names:
                    condition_kwargs[key] = value
            
            # For postconditions, add result as a parameter if expected
            if result is not None and 'result' in param_names:
                condition_kwargs['result'] = result
            
            return condition(**condition_kwargs)
        except Exception:
            # If condition evaluation fails, consider it a contract violation
            return False
    
    @staticmethod
    def extract_function_context(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract context information from function call."""
        return {
            "function_name": func.__name__,
            "module": func.__module__,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys()),
        }


def require(
    condition: Callable[..., bool],
    message: str = "Precondition failed"
) -> Callable[[F], F]:
    """
    Precondition contract decorator.
    
    Args:
        condition: Function that takes the same parameters as the decorated function
                  and returns True if the precondition is satisfied
        message: Error message to display when precondition fails
    
    Example:
        @require(lambda x: x > 0, "x must be positive")
        def sqrt(x: float) -> float:
            return x ** 0.5
    """
    def decorator(func: F) -> F:
        import asyncio
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Evaluate precondition
            if not ContractValidator.evaluate_condition(condition, args, kwargs):
                context = create_error_context(
                    operation="precondition_check",
                    component="contract_validator",
                    **ContractValidator.extract_function_context(func, args, kwargs)
                )
                raise ContractViolationError(
                    contract_type="Precondition",
                    condition=message,
                    context=context
                )
            
            # Execute the original async function
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Evaluate precondition
            if not ContractValidator.evaluate_condition(condition, args, kwargs):
                context = create_error_context(
                    operation="precondition_check",
                    component="contract_validator",
                    **ContractValidator.extract_function_context(func, args, kwargs)
                )
                raise ContractViolationError(
                    contract_type="Precondition",
                    condition=message,
                    context=context
                )
            
            # Execute the original sync function
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        # Add contract metadata
        wrapper.__contracts__ = getattr(func, '__contracts__', {})
        wrapper.__contracts__['preconditions'] = wrapper.__contracts__.get('preconditions', [])
        wrapper.__contracts__['preconditions'].append((condition, message))
        
        return wrapper
    return decorator


def ensure(
    condition: Callable[..., bool],
    message: str = "Postcondition failed"
) -> Callable[[F], F]:
    """
    Postcondition contract decorator.
    
    Args:
        condition: Function that takes the same parameters as the decorated function
                  plus a 'result' parameter and returns True if postcondition is satisfied
        message: Error message to display when postcondition fails
    
    Example:
        @ensure(lambda x, result: result >= 0, "result must be non-negative")
        def abs_value(x: float) -> float:
            return abs(x)
    """
    def decorator(func: F) -> F:
        import asyncio
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute the original async function
            result = await func(*args, **kwargs)
            
            # Evaluate postcondition
            if not ContractValidator.evaluate_condition(condition, args, kwargs, result):
                context = create_error_context(
                    operation="postcondition_check",
                    component="contract_validator",
                    result_type=type(result).__name__,
                    **ContractValidator.extract_function_context(func, args, kwargs)
                )
                raise ContractViolationError(
                    contract_type="Postcondition",
                    condition=message,
                    context=context
                )
            
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Execute the original sync function
            result = func(*args, **kwargs)
            
            # Evaluate postcondition
            if not ContractValidator.evaluate_condition(condition, args, kwargs, result):
                context = create_error_context(
                    operation="postcondition_check",
                    component="contract_validator",
                    result_type=type(result).__name__,
                    **ContractValidator.extract_function_context(func, args, kwargs)
                )
                raise ContractViolationError(
                    contract_type="Postcondition",
                    condition=message,
                    context=context
                )
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        # Add contract metadata
        wrapper.__contracts__ = getattr(func, '__contracts__', {})
        wrapper.__contracts__['postconditions'] = wrapper.__contracts__.get('postconditions', [])
        wrapper.__contracts__['postconditions'].append((condition, message))
        
        return wrapper
    return decorator


def invariant(
    condition: Callable[..., bool],
    message: str = "Class invariant violated"
) -> Callable[[type], type]:
    """
    Class invariant decorator for ensuring object state consistency.
    
    Args:
        condition: Function that takes an instance (self) and returns True
                  if the invariant holds
        message: Error message to display when invariant fails
    
    Example:
        @invariant(lambda self: self.balance >= 0, "balance cannot be negative")
        class Account:
            def __init__(self, balance: float):
                self.balance = balance
    """
    def class_decorator(cls: type) -> type:
        # Store original methods
        original_init = cls.__init__
        original_methods = {}
        
        # Collect all public methods
        for name in dir(cls):
            if not name.startswith('_') and callable(getattr(cls, name)):
                original_methods[name] = getattr(cls, name)
        
        def check_invariant(instance):
            """Check the class invariant."""
            if not condition(instance):
                context = create_error_context(
                    operation="invariant_check",
                    component="contract_validator",
                    class_name=cls.__name__,
                    instance_id=id(instance)
                )
                raise ContractViolationError(
                    contract_type="Invariant",
                    condition=message,
                    context=context
                )
        
        # Wrap __init__ to check invariant after construction
        @wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            check_invariant(self)
        
        # Wrap public methods to check invariant before and after
        def wrap_method(method_name, method):
            @wraps(method)
            def wrapped_method(self, *args, **kwargs):
                # Check invariant before method execution
                check_invariant(self)
                
                # Execute method
                result = method(self, *args, **kwargs)
                
                # Check invariant after method execution
                check_invariant(self)
                
                return result
            return wrapped_method
        
        # Apply wrapping
        cls.__init__ = wrapped_init
        for name, method in original_methods.items():
            setattr(cls, name, wrap_method(name, method))
        
        # Add contract metadata
        cls.__contracts__ = getattr(cls, '__contracts__', {})
        cls.__contracts__['invariants'] = cls.__contracts__.get('invariants', [])
        cls.__contracts__['invariants'].append((condition, message))
        
        return cls
    
    return class_decorator


def combine_conditions(*conditions: Callable[..., bool]) -> Callable[..., bool]:
    """
    Combine multiple conditions with AND logic.
    
    Args:
        conditions: Multiple condition functions to combine
        
    Returns:
        A single condition function that returns True only if all conditions are True
    """
    def combined_condition(*args, **kwargs) -> bool:
        return all(cond(*args, **kwargs) for cond in conditions)
    return combined_condition


def any_condition(*conditions: Callable[..., bool]) -> Callable[..., bool]:
    """
    Combine multiple conditions with OR logic.
    
    Args:
        conditions: Multiple condition functions to combine
        
    Returns:
        A single condition function that returns True if any condition is True
    """
    def combined_condition(*args, **kwargs) -> bool:
        return any(cond(*args, **kwargs) for cond in conditions)
    return combined_condition


def not_condition(condition: Callable[..., bool]) -> Callable[..., bool]:
    """
    Negate a condition.
    
    Args:
        condition: Condition function to negate
        
    Returns:
        A condition function that returns the opposite of the input condition
    """
    def negated_condition(*args, **kwargs) -> bool:
        return not condition(*args, **kwargs)
    return negated_condition


def get_contract_info(func_or_class) -> Dict[str, Any]:
    """
    Extract contract information from a function or class.
    
    Args:
        func_or_class: Function or class with contract decorators
        
    Returns:
        Dictionary containing contract information
    """
    contracts = getattr(func_or_class, '__contracts__', {})
    
    info = {
        "has_contracts": bool(contracts),
        "preconditions": len(contracts.get('preconditions', [])),
        "postconditions": len(contracts.get('postconditions', [])),
        "invariants": len(contracts.get('invariants', [])),
    }
    
    if contracts:
        info["contract_details"] = contracts
    
    return info


# Commonly used condition helpers
def is_not_none(value: Any) -> bool:
    """Helper condition to check if value is not None."""
    return value is not None


def is_positive(value: float) -> bool:
    """Helper condition to check if numeric value is positive."""
    return value > 0


def is_non_negative(value: float) -> bool:
    """Helper condition to check if numeric value is non-negative."""
    return value >= 0


def is_valid_string(value: str, min_length: int = 1, max_length: Optional[int] = None) -> bool:
    """Helper condition to validate string length."""
    if not isinstance(value, str):
        return False
    if len(value) < min_length:
        return False
    if max_length is not None and len(value) > max_length:
        return False
    return True