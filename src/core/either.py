"""
Either monad implementation for functional error handling.

This module provides a type-safe way to handle operations that can fail,
using the Either type to represent success (Right) or failure (Left) states.
"""

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Union, Any
from abc import ABC, abstractmethod

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')

class Either(Generic[A, B], ABC):
    """
    Either monad for functional error handling.
    
    Either[A, B] represents a value that can be either:
    - Left[A]: An error or failure value of type A
    - Right[B]: A success value of type B
    """
    
    @abstractmethod
    def is_left(self) -> bool:
        """Check if this is a Left (error) value."""
        pass
    
    @abstractmethod
    def is_right(self) -> bool:
        """Check if this is a Right (success) value."""
        pass
    
    @abstractmethod
    def get_left(self) -> A:
        """Get the Left value. Raises ValueError if called on Right."""
        pass
    
    @abstractmethod
    def get_right(self) -> B:
        """Get the Right value. Raises ValueError if called on Left."""
        pass
    
    @abstractmethod
    def map(self, f: Callable[[B], C]) -> Either[A, C]:
        """Transform the Right value, leaving Left unchanged."""
        pass
    
    @abstractmethod
    def flat_map(self, f: Callable[[B], Either[A, C]]) -> Either[A, C]:
        """Monadic bind operation."""
        pass
    
    @abstractmethod
    def map_left(self, f: Callable[[A], C]) -> Either[C, B]:
        """Transform the Left value, leaving Right unchanged."""
        pass
    
    @abstractmethod
    def filter(self, predicate: Callable[[B], bool], error: A) -> Either[A, B]:
        """Filter the Right value using predicate, returning Left with error if fails."""
        pass
    
    @abstractmethod
    def fold(self, left_func: Callable[[A], C], right_func: Callable[[B], C]) -> C:
        """Fold the Either value using appropriate function for Left or Right."""
        pass
    
    def get_or_else(self, default: B) -> B:
        """Get the Right value or return default if Left."""
        if self.is_right():
            return self.get_right()
        return default
    
    def or_else(self, alternative: Either[A, B]) -> Either[A, B]:
        """Return this Either if Right, otherwise return alternative."""
        if self.is_right():
            return self
        return alternative
    
    @staticmethod
    def left(value: A) -> Either[A, B]:
        """Create a Left (error) value."""
        return _Left(value)
    
    @staticmethod
    def right(value: B) -> Either[A, B]:
        """Create a Right (success) value."""
        return _Right(value)
    
    @staticmethod
    def try_operation(operation: Callable[[], B], error_handler: Callable[[Exception], A]) -> Either[A, B]:
        """
        Execute an operation that might throw an exception.
        
        Args:
            operation: Function that might throw an exception
            error_handler: Function to convert exception to error value
            
        Returns:
            Either[A, B] containing the result or error
        """
        try:
            result = operation()
            return Either.right(result)
        except Exception as e:
            error = error_handler(e)
            return Either.left(error)
    
    @staticmethod
    def try_either(operation: Callable[[], B]) -> Either[Exception, B]:
        """
        Execute an operation that might throw an exception.
        
        Args:
            operation: Function that might throw an exception
            
        Returns:
            Either[Exception, B] containing the result or exception
        """
        try:
            result = operation()
            return Either.right(result)
        except Exception as e:
            return Either.left(e)
    
    @staticmethod
    def from_optional(value: Union[B, None], error: A) -> Either[A, B]:
        """
        Create Either from optional value.
        
        Args:
            value: Optional value (None or actual value)
            error: Error to use if value is None
            
        Returns:
            Either[A, B] containing the result or error
        """
        if value is None:
            return Either.left(error)
        return Either.right(value)
    
    # Compatibility methods for existing code
    @staticmethod
    def success(value: B) -> Either[A, B]:
        """Create a successful Either (alias for right)."""
        return Either.right(value)
    
    @staticmethod
    def error(value: A) -> Either[A, B]:
        """Create an error Either (alias for left)."""
        return Either.left(value)
    
    def is_success(self) -> bool:
        """Check if this is a success value (alias for is_right)."""
        return self.is_right()
    
    def is_error(self) -> bool:
        """Check if this is an error value (alias for is_left)."""
        return self.is_left()
    
    @property
    def value(self) -> B:
        """Get the success value (alias for get_right)."""
        return self.get_right()
    
    @property
    def error_value(self) -> A:
        """Get the error value (alias for get_left)."""
        return self.get_left()

class _Left(Either[A, B]):
    """Left (error) implementation of Either."""
    
    def __init__(self, value: A):
        self._value = value
    
    def is_left(self) -> bool:
        return True
    
    def is_right(self) -> bool:
        return False
    
    def get_left(self) -> A:
        return self._value
    
    def get_right(self) -> B:
        raise ValueError("Cannot get Right value from Left")
    
    def map(self, f: Callable[[B], C]) -> Either[A, C]:
        return _Left(self._value)
    
    def flat_map(self, f: Callable[[B], Either[A, C]]) -> Either[A, C]:
        return _Left(self._value)
    
    def map_left(self, f: Callable[[A], C]) -> Either[C, B]:
        return _Left(f(self._value))
    
    def filter(self, predicate: Callable[[B], bool], error: A) -> Either[A, B]:
        """Filter operation on Left returns the same Left."""
        return _Left(self._value)
    
    def fold(self, left_func: Callable[[A], C], right_func: Callable[[B], C]) -> C:
        """Fold Left value using left_func."""
        return left_func(self._value)
    
    def __repr__(self) -> str:
        return f"Left({self._value})"
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Left) and self._value == other._value

class _Right(Either[A, B]):
    """Right (success) implementation of Either."""
    
    def __init__(self, value: B):
        self._value = value
    
    def is_left(self) -> bool:
        return False
    
    def is_right(self) -> bool:
        return True
    
    def get_left(self) -> A:
        raise ValueError("Cannot get Left value from Right")
    
    def get_right(self) -> B:
        return self._value
    
    def map(self, f: Callable[[B], C]) -> Either[A, C]:
        return _Right(f(self._value))
    
    def flat_map(self, f: Callable[[B], Either[A, C]]) -> Either[A, C]:
        return f(self._value)
    
    def map_left(self, f: Callable[[A], C]) -> Either[C, B]:
        return _Right(self._value)
    
    def filter(self, predicate: Callable[[B], bool], error: A) -> Either[A, B]:
        """Filter Right value using predicate."""
        if predicate(self._value):
            return _Right(self._value)
        return _Left(error)
    
    def fold(self, left_func: Callable[[A], C], right_func: Callable[[B], C]) -> C:
        """Fold Right value using right_func."""
        return right_func(self._value)
    
    def __repr__(self) -> str:
        return f"Right({self._value})"
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Right) and self._value == other._value

# Convenience functions for common operations
def sequence(eithers: list[Either[A, B]]) -> Either[A, list[B]]:
    """
    Convert a list of Either values to Either of list.
    
    If all values are Right, returns Right with list of values.
    If any value is Left, returns the first Left encountered.
    """
    results = []
    for either in eithers:
        if either.is_left():
            return either
        results.append(either.get_right())
    return Either.right(results)

def traverse(values: list[A], f: Callable[[A], Either[B, C]]) -> Either[B, list[C]]:
    """
    Apply a function returning Either to each value in a list.
    
    Returns Either[B, list[C]] where:
    - Right contains list of all successful results
    - Left contains the first error encountered
    """
    results = []
    for value in values:
        result = f(value)
        if result.is_left():
            return result
        results.append(result.get_right())
    return Either.right(results)

# Public aliases for Left and Right for direct usage
Left = _Left
Right = _Right