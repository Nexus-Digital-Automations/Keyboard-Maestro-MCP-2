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