"""
Circuit Breaker Pattern Implementation - TASK_64 Phase 2 Implementation

Advanced fault tolerance and resilience patterns for API orchestration with
Design by Contract patterns, type safety, and comprehensive monitoring.

Architecture: State machine + Failure tracking + Recovery mechanisms
Performance: <10ms state transitions, <1ms failure detection
Security: Failure isolation, resource protection, and attack mitigation
"""

from __future__ import annotations
import asyncio
import time
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import asynccontextmanager

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.errors import ValidationError, SecurityError
from ..core.types import ServiceID, APIEndpoint

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing recovery


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker."""
    TIMEOUT = "timeout"
    ERROR_RESPONSE = "error_response"
    EXCEPTION = "exception"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    NETWORK = "network"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    name: str
    failure_threshold: int = 5           # Failures before opening
    recovery_timeout_seconds: int = 60   # Time before trying recovery
    success_threshold: int = 2           # Successes needed to close from half-open
    timeout_seconds: float = 30.0        # Call timeout
    expected_error_rate: float = 0.05    # Expected error rate (5%)
    rolling_window_seconds: int = 300    # Rolling window for failure tracking
    max_concurrent_calls: int = 100      # Maximum concurrent calls
    slow_call_threshold_ms: float = 1000 # Threshold for slow calls
    slow_call_rate_threshold: float = 0.5 # Slow call rate threshold
    
    def __post_init__(self):
        if self.failure_threshold <= 0:
            raise ValidationError("failure_threshold", self.failure_threshold, "Must be positive")
        if self.recovery_timeout_seconds <= 0:
            raise ValidationError("recovery_timeout_seconds", self.recovery_timeout_seconds, "Must be positive")


@dataclass
class CallResult:
    """Result of a circuit breaker protected call."""
    success: bool
    duration_ms: float
    failure_type: Optional[FailureType] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker operational metrics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    slow_calls: int = 0
    current_concurrent_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


class CircuitBreaker:
    """
    Advanced circuit breaker implementation with comprehensive failure tracking.
    
    Implements the circuit breaker pattern for fault tolerance in API calls,
    with sophisticated failure detection, recovery mechanisms, and monitoring.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        require(lambda: config.name, "Config must have name")
        
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.call_history: List[CallResult] = []
        self.state_change_time = datetime.now(UTC)
        self.half_open_successes = 0
        self.concurrent_calls = 0
        self.lock = asyncio.Lock()
        
        logger.info(f"Circuit breaker '{config.name}' initialized")
    
    @require(lambda func: callable(func), "Function must be callable")
    @ensure(lambda result: result is not None, "Returns call result")
    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Either[str, Any]:
        """Execute function call with circuit breaker protection."""
        async with self.lock:
            # Check if call should be rejected
            rejection_reason = self._should_reject_call()
            if rejection_reason:
                self.metrics.rejected_calls += 1
                return Either.left(f"Circuit breaker rejected call: {rejection_reason}")
            
            # Track concurrent call
            self.concurrent_calls += 1
            self.metrics.current_concurrent_calls = self.concurrent_calls
        
        start_time = time.time()
        call_result = None
        
        try:
            # Execute call with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            # Record successful call
            duration_ms = (time.time() - start_time) * 1000
            call_result = CallResult(
                success=True,
                duration_ms=duration_ms
            )
            
            await self._record_call_result(call_result)
            return Either.right(result)
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            call_result = CallResult(
                success=False,
                duration_ms=duration_ms,
                failure_type=FailureType.TIMEOUT,
                error_message=f"Call timed out after {self.config.timeout_seconds}s"
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            failure_type = self._classify_exception(e)
            call_result = CallResult(
                success=False,
                duration_ms=duration_ms,
                failure_type=failure_type,
                error_message=str(e)
            )
        
        finally:
            # Always decrement concurrent calls
            async with self.lock:
                self.concurrent_calls = max(0, self.concurrent_calls - 1)
                self.metrics.current_concurrent_calls = self.concurrent_calls
        
        if call_result and not call_result.success:
            await self._record_call_result(call_result)
            return Either.left(call_result.error_message or "Call failed")
        
        return Either.left("Unknown error occurred")
    
    def _should_reject_call(self) -> Optional[str]:
        """Check if call should be rejected based on current state."""
        # Check concurrent call limit
        if self.concurrent_calls >= self.config.max_concurrent_calls:
            return f"Too many concurrent calls ({self.concurrent_calls}/{self.config.max_concurrent_calls})"
        
        # State-based rejection
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            time_since_state_change = datetime.now(UTC) - self.state_change_time
            if time_since_state_change.total_seconds() >= self.config.recovery_timeout_seconds:
                self._transition_to_half_open()
                return None  # Allow call in half-open state
            return f"Circuit breaker is open, recovery in {self.config.recovery_timeout_seconds - time_since_state_change.total_seconds():.1f}s"
        
        elif self.state == CircuitState.HALF_OPEN:
            # In half-open, only allow limited calls
            if self.concurrent_calls > 0:
                return "Circuit breaker is half-open, testing with single call"
        
        return None
    
    async def _record_call_result(self, result: CallResult) -> None:
        """Record call result and update circuit breaker state."""
        async with self.lock:
            # Update metrics
            self.metrics.total_calls += 1
            
            if result.success:
                self.metrics.successful_calls += 1
                self.metrics.last_success_time = result.timestamp
                
                # Check for slow call
                if result.duration_ms > self.config.slow_call_threshold_ms:
                    self.metrics.slow_calls += 1
                
                # Handle state transitions on success
                if self.state == CircuitState.HALF_OPEN:
                    self.half_open_successes += 1
                    if self.half_open_successes >= self.config.success_threshold:
                        self._transition_to_closed()
            else:
                self.metrics.failed_calls += 1
                self.metrics.last_failure_time = result.timestamp
                
                # Handle state transitions on failure
                if self.state == CircuitState.HALF_OPEN:
                    self._transition_to_open()
                elif self.state == CircuitState.CLOSED:
                    if self._should_open_circuit():
                        self._transition_to_open()
            
            # Add to call history (maintain rolling window)
            self.call_history.append(result)
            self._cleanup_old_call_history()
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened based on recent failures."""
        # Get recent calls within rolling window
        cutoff_time = datetime.now(UTC) - timedelta(seconds=self.config.rolling_window_seconds)
        recent_calls = [call for call in self.call_history if call.timestamp >= cutoff_time]
        
        if len(recent_calls) < self.config.failure_threshold:
            return False
        
        # Calculate failure rate
        failed_calls = [call for call in recent_calls if not call.success]
        failure_rate = len(failed_calls) / len(recent_calls)
        
        # Check failure threshold
        if len(failed_calls) >= self.config.failure_threshold:
            return True
        
        # Check if failure rate exceeds expected rate significantly
        if failure_rate > self.config.expected_error_rate * 3:  # 3x expected rate
            return True
        
        # Check slow call rate
        slow_calls = [call for call in recent_calls if call.duration_ms > self.config.slow_call_threshold_ms]
        if len(recent_calls) > 0:
            slow_call_rate = len(slow_calls) / len(recent_calls)
            if slow_call_rate > self.config.slow_call_rate_threshold:
                return True
        
        return False
    
    def _transition_to_open(self) -> None:
        """Transition circuit breaker to open state."""
        if self.state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker '{self.config.name}' opened")
            self.state = CircuitState.OPEN
            self.state_change_time = datetime.now(UTC)
            self.metrics.state_changes += 1
            self.half_open_successes = 0
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to half-open state."""
        if self.state != CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker '{self.config.name}' half-opened for testing")
            self.state = CircuitState.HALF_OPEN
            self.state_change_time = datetime.now(UTC)
            self.metrics.state_changes += 1
            self.half_open_successes = 0
    
    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to closed state."""
        if self.state != CircuitState.CLOSED:
            logger.info(f"Circuit breaker '{self.config.name}' closed")
            self.state = CircuitState.CLOSED
            self.state_change_time = datetime.now(UTC)
            self.metrics.state_changes += 1
            self.half_open_successes = 0
    
    def _classify_exception(self, exception: Exception) -> FailureType:
        """Classify exception type for circuit breaker decision making."""
        exception_name = exception.__class__.__name__.lower()
        exception_msg = str(exception).lower()
        
        if "timeout" in exception_name or "timeout" in exception_msg:
            return FailureType.TIMEOUT
        elif "network" in exception_msg or "connection" in exception_msg:
            return FailureType.NETWORK
        elif "auth" in exception_msg or "unauthorized" in exception_msg:
            return FailureType.AUTHENTICATION
        elif "rate" in exception_msg and "limit" in exception_msg:
            return FailureType.RATE_LIMIT
        elif "http" in exception_name or any(code in exception_msg for code in ["400", "500", "502", "503", "504"]):
            return FailureType.ERROR_RESPONSE
        else:
            return FailureType.EXCEPTION
    
    def _cleanup_old_call_history(self) -> None:
        """Remove old call history outside rolling window."""
        cutoff_time = datetime.now(UTC) - timedelta(seconds=self.config.rolling_window_seconds * 2)
        self.call_history = [call for call in self.call_history if call.timestamp >= cutoff_time]
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state and metrics."""
        recent_calls = self._get_recent_calls()
        failure_rate = self._calculate_failure_rate(recent_calls)
        slow_call_rate = self._calculate_slow_call_rate(recent_calls)
        
        return {
            "name": self.config.name,
            "state": self.state.value,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "rejected_calls": self.metrics.rejected_calls,
                "slow_calls": self.metrics.slow_calls,
                "current_concurrent_calls": self.metrics.current_concurrent_calls,
                "state_changes": self.metrics.state_changes,
                "failure_rate": failure_rate,
                "slow_call_rate": slow_call_rate
            },
            "state_info": {
                "state_change_time": self.state_change_time.isoformat(),
                "time_in_current_state": (datetime.now(UTC) - self.state_change_time).total_seconds(),
                "half_open_successes": self.half_open_successes if self.state == CircuitState.HALF_OPEN else None,
                "next_recovery_attempt": (self.state_change_time + timedelta(seconds=self.config.recovery_timeout_seconds)).isoformat() if self.state == CircuitState.OPEN else None
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout_seconds": self.config.recovery_timeout_seconds,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "max_concurrent_calls": self.config.max_concurrent_calls
            }
        }
    
    def _get_recent_calls(self) -> List[CallResult]:
        """Get calls within the rolling window."""
        cutoff_time = datetime.now(UTC) - timedelta(seconds=self.config.rolling_window_seconds)
        return [call for call in self.call_history if call.timestamp >= cutoff_time]
    
    def _calculate_failure_rate(self, calls: List[CallResult]) -> float:
        """Calculate failure rate for given calls."""
        if not calls:
            return 0.0
        failed_calls = len([call for call in calls if not call.success])
        return failed_calls / len(calls)
    
    def _calculate_slow_call_rate(self, calls: List[CallResult]) -> float:
        """Calculate slow call rate for given calls."""
        if not calls:
            return 0.0
        slow_calls = len([call for call in calls if call.duration_ms > self.config.slow_call_threshold_ms])
        return slow_calls / len(calls)
    
    async def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        async with self.lock:
            logger.info(f"Resetting circuit breaker '{self.config.name}'")
            self.state = CircuitState.CLOSED
            self.state_change_time = datetime.now(UTC)
            self.half_open_successes = 0
            self.call_history.clear()
            self.metrics = CircuitBreakerMetrics()
    
    async def force_open(self) -> None:
        """Force circuit breaker to open state."""
        async with self.lock:
            logger.warning(f"Force opening circuit breaker '{self.config.name}'")
            self._transition_to_open()


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Provides centralized management and monitoring of circuit breakers
    across different services and API endpoints.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig(name="default")
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig(name=name)
            self.circuit_breakers[name] = CircuitBreaker(config)
        
        return self.circuit_breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove circuit breaker."""
        if name in self.circuit_breakers:
            del self.circuit_breakers[name]
            return True
        return False
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers."""
        return {
            name: breaker.get_state()
            for name, breaker in self.circuit_breakers.items()
        }
    
    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.circuit_breakers.values():
            await breaker.reset()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all circuit breakers."""
        states = {"closed": 0, "open": 0, "half_open": 0}
        total_calls = 0
        total_failures = 0
        
        for breaker in self.circuit_breakers.values():
            states[breaker.state.value] += 1
            total_calls += breaker.metrics.total_calls
            total_failures += breaker.metrics.failed_calls
        
        return {
            "total_circuit_breakers": len(self.circuit_breakers),
            "states": states,
            "overall_metrics": {
                "total_calls": total_calls,
                "total_failures": total_failures,
                "overall_failure_rate": total_failures / total_calls if total_calls > 0 else 0.0
            }
        }


# Global registry
_circuit_breaker_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get or create the global circuit breaker registry."""
    global _circuit_breaker_registry
    if _circuit_breaker_registry is None:
        _circuit_breaker_registry = CircuitBreakerRegistry()
    return _circuit_breaker_registry


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for protecting functions with circuit breaker."""
    def decorator(func: Callable):
        registry = get_circuit_breaker_registry()
        breaker = registry.get_or_create(name, config)
        
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


@asynccontextmanager
async def circuit_breaker_context(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Context manager for circuit breaker protection."""
    registry = get_circuit_breaker_registry()
    breaker = registry.get_or_create(name, config)
    
    try:
        yield breaker
    finally:
        # Circuit breaker continues to exist in registry
        pass


# Utility functions for common circuit breaker configurations

def create_api_circuit_breaker(service_name: str, timeout_seconds: float = 30.0) -> CircuitBreakerConfig:
    """Create circuit breaker config optimized for API calls."""
    return CircuitBreakerConfig(
        name=f"api_{service_name}",
        failure_threshold=5,
        recovery_timeout_seconds=60,
        success_threshold=2,
        timeout_seconds=timeout_seconds,
        expected_error_rate=0.05,
        rolling_window_seconds=300,
        max_concurrent_calls=50,
        slow_call_threshold_ms=5000,
        slow_call_rate_threshold=0.3
    )


def create_database_circuit_breaker(db_name: str) -> CircuitBreakerConfig:
    """Create circuit breaker config optimized for database connections."""
    return CircuitBreakerConfig(
        name=f"db_{db_name}",
        failure_threshold=3,
        recovery_timeout_seconds=30,
        success_threshold=1,
        timeout_seconds=10.0,
        expected_error_rate=0.01,
        rolling_window_seconds=180,
        max_concurrent_calls=20,
        slow_call_threshold_ms=1000,
        slow_call_rate_threshold=0.5
    )


def create_external_service_circuit_breaker(service_name: str) -> CircuitBreakerConfig:
    """Create circuit breaker config optimized for external service calls."""
    return CircuitBreakerConfig(
        name=f"external_{service_name}",
        failure_threshold=10,
        recovery_timeout_seconds=120,
        success_threshold=3,
        timeout_seconds=60.0,
        expected_error_rate=0.1,
        rolling_window_seconds=600,
        max_concurrent_calls=30,
        slow_call_threshold_ms=10000,
        slow_call_rate_threshold=0.4
    )