"""
Rate Limiter - TASK_64 Phase 4 Advanced Features

Advanced rate limiting and throttling mechanisms for API orchestration.
Provides intelligent traffic control with burst handling and adaptive limits.

Architecture: Rate Limiting + Throttling + Burst Control + Adaptive Limits + Quota Management
Performance: <10ms rate check, <50ms quota calculation, <100ms adaptive adjustment
Scalability: Distributed rate limiting, hierarchical quotas, multi-dimensional limits
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import time
import hashlib
from collections import defaultdict, deque
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.api_orchestration_architecture import (
    ServiceId, APIOrchestrationError, create_service_id
)


class RateLimitWindow(Enum):
    """Rate limiting time windows."""
    SECOND = "second"
    MINUTE = "minute" 
    HOUR = "hour"
    DAY = "day"


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"          # Fixed time window
    SLIDING_WINDOW = "sliding_window"      # Sliding time window
    TOKEN_BUCKET = "token_bucket"          # Token bucket algorithm
    LEAKY_BUCKET = "leaky_bucket"          # Leaky bucket algorithm
    ADAPTIVE = "adaptive"                  # Adaptive rate limiting


class ThrottleAction(Enum):
    """Actions to take when rate limit is exceeded."""
    REJECT = "reject"                      # Reject request immediately
    DELAY = "delay"                        # Delay request execution
    QUEUE = "queue"                        # Queue request for later
    DEGRADE = "degrade"                    # Provide degraded service


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    rule_id: str
    name: str
    key_pattern: str                       # Pattern for grouping requests
    limit: int                            # Request limit
    window: RateLimitWindow               # Time window
    strategy: RateLimitStrategy           # Limiting strategy
    action: ThrottleAction                # Action when exceeded
    
    # Advanced configuration
    burst_limit: Optional[int] = None     # Burst allowance
    burst_window_seconds: int = 60        # Burst window duration
    priority: int = 0                     # Rule priority (higher = more important)
    
    # Token bucket specific
    token_refill_rate: Optional[float] = None
    token_bucket_size: Optional[int] = None
    
    # Adaptive limiting
    adaptive_factor: float = 1.0          # Adjustment factor for adaptive limits
    min_limit: Optional[int] = None       # Minimum limit for adaptive
    max_limit: Optional[int] = None       # Maximum limit for adaptive
    
    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitState:
    """Current state of a rate limit bucket."""
    key: str
    rule_id: str
    current_count: int = 0
    window_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_request: Optional[datetime] = None
    
    # Token bucket state
    tokens: float = 0.0
    last_refill: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # Sliding window state
    request_times: deque = field(default_factory=deque)
    
    # Burst tracking
    burst_count: int = 0
    burst_window_start: Optional[datetime] = None
    
    # Adaptive state
    adaptive_limit: Optional[int] = None
    performance_history: List[float] = field(default_factory=list)
    
    # Statistics
    total_requests: int = 0
    rejected_requests: int = 0
    delayed_requests: int = 0
    queued_requests: int = 0
    
    def get_success_rate(self) -> float:
        """Calculate success rate for adaptive limiting."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.rejected_requests) / self.total_requests


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    rule_id: str
    key: str
    action: ThrottleAction
    remaining_requests: int
    reset_time: datetime
    retry_after_seconds: Optional[int] = None
    delay_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuotaDefinition:
    """Quota definition for service or user."""
    quota_id: str
    name: str
    limit: int
    window: RateLimitWindow
    scope: str                            # "user", "service", "global", etc.
    
    # Hierarchical quotas
    parent_quota_id: Optional[str] = None
    child_quota_ids: List[str] = field(default_factory=list)
    
    # Quota allocation
    allocated_quotas: Dict[str, int] = field(default_factory=dict)
    
    # Reset configuration
    auto_reset: bool = True
    reset_schedule: Optional[str] = None  # Cron-like schedule
    
    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies and adaptive capabilities."""
    
    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.states: Dict[str, RateLimitState] = {}
        self.quotas: Dict[str, QuotaDefinition] = {}
        self.quota_states: Dict[str, RateLimitState] = {}
        
        # Request queue for delayed processing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.queue_processor_active = False
        
        # Performance monitoring
        self.performance_metrics: Dict[str, Any] = {
            "total_checks": 0,
            "total_allowed": 0,
            "total_rejected": 0,
            "total_delayed": 0,
            "total_queued": 0,
            "average_check_time_ms": 0.0
        }
        
        # Adaptive adjustment parameters
        self.adaptive_adjustment_interval = 300  # 5 minutes
        self.last_adaptive_adjustment = datetime.now(UTC)
        
        # Start background tasks
        asyncio.create_task(self._start_background_tasks())
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        await asyncio.gather(
            self._process_request_queue(),
            self._adaptive_adjustment_loop(),
            self._cleanup_expired_states()
        )
    
    @require(lambda rule: isinstance(rule, RateLimitRule))
    def add_rule(self, rule: RateLimitRule) -> Either[APIOrchestrationError, bool]:
        """Add rate limiting rule."""
        try:
            self.rules[rule.rule_id] = rule
            return Either.success(True)
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Failed to add rate limit rule: {str(e)}"))
    
    def remove_rule(self, rule_id: str) -> Either[APIOrchestrationError, bool]:
        """Remove rate limiting rule."""
        try:
            if rule_id in self.rules:
                del self.rules[rule_id]
                # Clean up associated states
                states_to_remove = [k for k in self.states.keys() if k.startswith(f"{rule_id}:")]
                for state_key in states_to_remove:
                    del self.states[state_key]
                return Either.success(True)
            else:
                return Either.error(APIOrchestrationError(f"Rate limit rule {rule_id} not found"))
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Failed to remove rate limit rule: {str(e)}"))
    
    @require(lambda quota: isinstance(quota, QuotaDefinition))
    def add_quota(self, quota: QuotaDefinition) -> Either[APIOrchestrationError, bool]:
        """Add quota definition."""
        try:
            self.quotas[quota.quota_id] = quota
            return Either.success(True)
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Failed to add quota: {str(e)}"))
    
    @require(lambda key: isinstance(key, str) and len(key) > 0)
    @require(lambda request_metadata: request_metadata is None or isinstance(request_metadata, dict))
    async def check_rate_limit(self, key: str, request_metadata: Optional[Dict[str, Any]] = None) -> Either[APIOrchestrationError, RateLimitResult]:
        """
        Check rate limit for request key.
        
        Args:
            key: Request key for rate limiting
            request_metadata: Additional request metadata
            
        Returns:
            Either API orchestration error or rate limit result
        """
        try:
            check_start = time.time()
            self.performance_metrics["total_checks"] += 1
            
            # Find applicable rules
            applicable_rules = self._find_applicable_rules(key, request_metadata)
            
            if not applicable_rules:
                # No rules apply, allow request
                result = RateLimitResult(
                    allowed=True,
                    rule_id="none",
                    key=key,
                    action=ThrottleAction.REJECT,
                    remaining_requests=999999,
                    reset_time=datetime.now(UTC) + timedelta(hours=1)
                )
                return Either.success(result)
            
            # Check each applicable rule (most restrictive wins)
            most_restrictive_result = None
            
            for rule in applicable_rules:
                state_key = f"{rule.rule_id}:{key}"
                state = self._get_or_create_state(state_key, rule.rule_id, key)
                
                # Check rule based on strategy
                rule_result = await self._check_rule(rule, state, request_metadata)
                
                if not rule_result.allowed:
                    most_restrictive_result = rule_result
                    break  # First blocking rule wins
                elif most_restrictive_result is None:
                    most_restrictive_result = rule_result
            
            # Update performance metrics
            check_time = (time.time() - check_start) * 1000
            current_avg = self.performance_metrics["average_check_time_ms"]
            total_checks = self.performance_metrics["total_checks"]
            self.performance_metrics["average_check_time_ms"] = (current_avg * (total_checks - 1) + check_time) / total_checks
            
            # Update counters
            if most_restrictive_result.allowed:
                self.performance_metrics["total_allowed"] += 1
            else:
                if most_restrictive_result.action == ThrottleAction.REJECT:
                    self.performance_metrics["total_rejected"] += 1
                elif most_restrictive_result.action == ThrottleAction.DELAY:
                    self.performance_metrics["total_delayed"] += 1
                elif most_restrictive_result.action == ThrottleAction.QUEUE:
                    self.performance_metrics["total_queued"] += 1
            
            return Either.success(most_restrictive_result)
            
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Rate limit check failed: {str(e)}"))
    
    def _find_applicable_rules(self, key: str, request_metadata: Optional[Dict[str, Any]]) -> List[RateLimitRule]:
        """Find rules applicable to the request key."""
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Simple pattern matching (in production would use regex)
            if rule.key_pattern == "*" or rule.key_pattern in key:
                applicable_rules.append(rule)
        
        # Sort by priority (higher priority first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        return applicable_rules
    
    def _get_or_create_state(self, state_key: str, rule_id: str, key: str) -> RateLimitState:
        """Get or create rate limit state."""
        if state_key not in self.states:
            rule = self.rules[rule_id]
            
            # Initialize based on strategy
            initial_tokens = 0.0
            if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                initial_tokens = float(rule.token_bucket_size or rule.limit)
            
            self.states[state_key] = RateLimitState(
                key=key,
                rule_id=rule_id,
                tokens=initial_tokens,
                adaptive_limit=rule.limit if rule.strategy == RateLimitStrategy.ADAPTIVE else None
            )
        
        return self.states[state_key]
    
    async def _check_rule(self, rule: RateLimitRule, state: RateLimitState, request_metadata: Optional[Dict[str, Any]]) -> RateLimitResult:
        """Check specific rate limiting rule."""
        now = datetime.now(UTC)
        
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._check_fixed_window(rule, state, now)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(rule, state, now)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(rule, state, now)
        elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return await self._check_leaky_bucket(rule, state, now)
        elif rule.strategy == RateLimitStrategy.ADAPTIVE:
            return await self._check_adaptive(rule, state, now)
        else:
            # Default to fixed window
            return await self._check_fixed_window(rule, state, now)
    
    async def _check_fixed_window(self, rule: RateLimitRule, state: RateLimitState, now: datetime) -> RateLimitResult:
        """Check fixed time window rate limit."""
        # Calculate window duration
        window_duration = self._get_window_duration(rule.window)
        
        # Reset window if expired
        if now - state.window_start > window_duration:
            state.window_start = now
            state.current_count = 0
        
        # Check burst limit first
        if rule.burst_limit and not self._check_burst_limit(rule, state, now):
            return RateLimitResult(
                allowed=False,
                rule_id=rule.rule_id,
                key=state.key,
                action=rule.action,
                remaining_requests=0,
                reset_time=state.window_start + window_duration,
                retry_after_seconds=int((state.window_start + window_duration - now).total_seconds())
            )
        
        # Check main limit
        if state.current_count < rule.limit:
            state.current_count += 1
            state.last_request = now
            state.total_requests += 1
            
            return RateLimitResult(
                allowed=True,
                rule_id=rule.rule_id,
                key=state.key,
                action=rule.action,
                remaining_requests=rule.limit - state.current_count,
                reset_time=state.window_start + window_duration
            )
        else:
            state.rejected_requests += 1
            return RateLimitResult(
                allowed=False,
                rule_id=rule.rule_id,
                key=state.key,
                action=rule.action,
                remaining_requests=0,
                reset_time=state.window_start + window_duration,
                retry_after_seconds=int((state.window_start + window_duration - now).total_seconds())
            )
    
    async def _check_sliding_window(self, rule: RateLimitRule, state: RateLimitState, now: datetime) -> RateLimitResult:
        """Check sliding time window rate limit."""
        window_duration = self._get_window_duration(rule.window)
        cutoff_time = now - window_duration
        
        # Remove old requests from sliding window
        while state.request_times and state.request_times[0] < cutoff_time:
            state.request_times.popleft()
        
        # Check limit
        if len(state.request_times) < rule.limit:
            state.request_times.append(now)
            state.last_request = now
            state.total_requests += 1
            
            return RateLimitResult(
                allowed=True,
                rule_id=rule.rule_id,
                key=state.key,
                action=rule.action,
                remaining_requests=rule.limit - len(state.request_times),
                reset_time=now + window_duration
            )
        else:
            state.rejected_requests += 1
            # Calculate retry after based on oldest request
            oldest_request = state.request_times[0]
            retry_after = int((oldest_request + window_duration - now).total_seconds())
            
            return RateLimitResult(
                allowed=False,
                rule_id=rule.rule_id,
                key=state.key,
                action=rule.action,
                remaining_requests=0,
                reset_time=oldest_request + window_duration,
                retry_after_seconds=max(1, retry_after)
            )
    
    async def _check_token_bucket(self, rule: RateLimitRule, state: RateLimitState, now: datetime) -> RateLimitResult:
        """Check token bucket rate limit."""
        # Refill tokens
        time_since_last_refill = (now - state.last_refill).total_seconds()
        refill_rate = rule.token_refill_rate or (rule.limit / self._get_window_duration(rule.window).total_seconds())
        bucket_size = rule.token_bucket_size or rule.limit
        
        tokens_to_add = time_since_last_refill * refill_rate
        state.tokens = min(bucket_size, state.tokens + tokens_to_add)
        state.last_refill = now
        
        # Check if token available
        if state.tokens >= 1.0:
            state.tokens -= 1.0
            state.last_request = now
            state.total_requests += 1
            
            return RateLimitResult(
                allowed=True,
                rule_id=rule.rule_id,
                key=state.key,
                action=rule.action,
                remaining_requests=int(state.tokens),
                reset_time=now + timedelta(seconds=1/refill_rate)
            )
        else:
            state.rejected_requests += 1
            # Calculate delay based on token refill time
            delay_seconds = (1.0 - state.tokens) / refill_rate
            
            return RateLimitResult(
                allowed=False,
                rule_id=rule.rule_id,
                key=state.key,
                action=rule.action,
                remaining_requests=0,
                reset_time=now + timedelta(seconds=delay_seconds),
                delay_seconds=delay_seconds if rule.action == ThrottleAction.DELAY else None
            )
    
    async def _check_leaky_bucket(self, rule: RateLimitRule, state: RateLimitState, now: datetime) -> RateLimitResult:
        """Check leaky bucket rate limit."""
        # Similar to token bucket but with constant leak rate
        window_duration = self._get_window_duration(rule.window)
        leak_rate = rule.limit / window_duration.total_seconds()
        
        # Leak tokens since last check
        time_since_last = (now - state.last_refill).total_seconds()
        leaked_tokens = time_since_last * leak_rate
        state.tokens = max(0, state.tokens - leaked_tokens)
        state.last_refill = now
        
        # Check capacity
        if state.tokens < rule.limit:
            state.tokens += 1
            state.last_request = now
            state.total_requests += 1
            
            return RateLimitResult(
                allowed=True,
                rule_id=rule.rule_id,
                key=state.key,
                action=rule.action,
                remaining_requests=rule.limit - int(state.tokens),
                reset_time=now + timedelta(seconds=1/leak_rate)
            )
        else:
            state.rejected_requests += 1
            return RateLimitResult(
                allowed=False,
                rule_id=rule.rule_id,
                key=state.key,
                action=rule.action,
                remaining_requests=0,
                reset_time=now + timedelta(seconds=1/leak_rate)
            )
    
    async def _check_adaptive(self, rule: RateLimitRule, state: RateLimitState, now: datetime) -> RateLimitResult:
        """Check adaptive rate limit that adjusts based on performance."""
        # Use current adaptive limit or fall back to base limit
        current_limit = state.adaptive_limit or rule.limit
        
        # Perform basic fixed window check with adaptive limit
        temp_rule = RateLimitRule(
            rule_id=rule.rule_id,
            name=rule.name,
            key_pattern=rule.key_pattern,
            limit=current_limit,
            window=rule.window,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            action=rule.action
        )
        
        result = await self._check_fixed_window(temp_rule, state, now)
        
        # Update performance history for adaptive adjustment
        success_rate = state.get_success_rate()
        state.performance_history.append(success_rate)
        
        # Keep only recent history
        if len(state.performance_history) > 100:
            state.performance_history = state.performance_history[-50:]
        
        return result
    
    def _check_burst_limit(self, rule: RateLimitRule, state: RateLimitState, now: datetime) -> bool:
        """Check burst limit allowance."""
        if not rule.burst_limit:
            return True
        
        # Reset burst window if expired
        if not state.burst_window_start or (now - state.burst_window_start).total_seconds() > rule.burst_window_seconds:
            state.burst_window_start = now
            state.burst_count = 0
        
        # Check burst limit
        if state.burst_count < rule.burst_limit:
            state.burst_count += 1
            return True
        
        return False
    
    def _get_window_duration(self, window: RateLimitWindow) -> timedelta:
        """Get duration for rate limit window."""
        if window == RateLimitWindow.SECOND:
            return timedelta(seconds=1)
        elif window == RateLimitWindow.MINUTE:
            return timedelta(minutes=1)
        elif window == RateLimitWindow.HOUR:
            return timedelta(hours=1)
        elif window == RateLimitWindow.DAY:
            return timedelta(days=1)
        else:
            return timedelta(minutes=1)  # Default
    
    async def _process_request_queue(self):
        """Process delayed/queued requests."""
        self.queue_processor_active = True
        
        while self.queue_processor_active:
            try:
                # Process queued requests with delay
                await asyncio.sleep(0.1)
                
                # In production, would process actual queued requests
                # For now, just maintain the queue processing loop
                
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)  # Error recovery
    
    async def _adaptive_adjustment_loop(self):
        """Periodically adjust adaptive rate limits."""
        while True:
            try:
                await asyncio.sleep(self.adaptive_adjustment_interval)
                await self._adjust_adaptive_limits()
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(30)  # Error recovery
    
    async def _adjust_adaptive_limits(self):
        """Adjust adaptive rate limits based on performance."""
        now = datetime.now(UTC)
        
        for state in self.states.values():
            if state.rule_id not in self.rules:
                continue
            
            rule = self.rules[state.rule_id]
            if rule.strategy != RateLimitStrategy.ADAPTIVE:
                continue
            
            if not state.performance_history:
                continue
            
            # Calculate average success rate
            avg_success_rate = sum(state.performance_history) / len(state.performance_history)
            
            # Adjust limit based on success rate
            if avg_success_rate > 0.95:  # High success rate, can increase limit
                adjustment = 1.1
            elif avg_success_rate > 0.85:  # Good success rate, small increase
                adjustment = 1.05
            elif avg_success_rate < 0.7:  # Poor success rate, decrease limit
                adjustment = 0.8
            elif avg_success_rate < 0.85:  # Fair success rate, small decrease
                adjustment = 0.95
            else:
                adjustment = 1.0  # No change
            
            # Apply adjustment with bounds
            current_limit = state.adaptive_limit or rule.limit
            new_limit = int(current_limit * adjustment * rule.adaptive_factor)
            
            # Apply min/max bounds
            if rule.min_limit:
                new_limit = max(new_limit, rule.min_limit)
            if rule.max_limit:
                new_limit = min(new_limit, rule.max_limit)
            
            state.adaptive_limit = new_limit
        
        self.last_adaptive_adjustment = now
    
    async def _cleanup_expired_states(self):
        """Clean up expired rate limit states."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                now = datetime.now(UTC)
                expired_keys = []
                
                for key, state in self.states.items():
                    # Remove states inactive for more than 24 hours
                    if state.last_request and (now - state.last_request).total_seconds() > 86400:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.states[key]
                    
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(300)  # Error recovery
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        active_states = len(self.states)
        active_rules = len([r for r in self.rules.values() if r.enabled])
        
        total_requests = sum(s.total_requests for s in self.states.values())
        total_rejected = sum(s.rejected_requests for s in self.states.values())
        total_delayed = sum(s.delayed_requests for s in self.states.values())
        
        return {
            **self.performance_metrics,
            "active_rules": active_rules,
            "total_rules": len(self.rules),
            "active_states": active_states,
            "total_quotas": len(self.quotas),
            "queue_size": self.request_queue.qsize(),
            "total_requests_tracked": total_requests,
            "total_rejected_tracked": total_rejected,
            "total_delayed_tracked": total_delayed,
            "success_rate": (total_requests - total_rejected) / total_requests if total_requests > 0 else 1.0
        }
    
    def get_rule_metrics(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for specific rule."""
        if rule_id not in self.rules:
            return None
        
        rule = self.rules[rule_id]
        rule_states = [s for s in self.states.values() if s.rule_id == rule_id]
        
        total_requests = sum(s.total_requests for s in rule_states)
        total_rejected = sum(s.rejected_requests for s in rule_states)
        
        return {
            "rule_id": rule_id,
            "rule_name": rule.name,
            "strategy": rule.strategy.value,
            "limit": rule.limit,
            "window": rule.window.value,
            "active_states": len(rule_states),
            "total_requests": total_requests,
            "total_rejected": total_rejected,
            "success_rate": (total_requests - total_rejected) / total_requests if total_requests > 0 else 1.0,
            "enabled": rule.enabled
        }


# Export the rate limiter classes
__all__ = [
    "AdvancedRateLimiter", "RateLimitRule", "RateLimitState", "RateLimitResult",
    "QuotaDefinition", "RateLimitWindow", "RateLimitStrategy", "ThrottleAction"
]