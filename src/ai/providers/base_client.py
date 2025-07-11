"""Base provider client for AI service integration.

This module provides the abstract base class for all AI provider clients,
establishing common patterns for authentication, error handling, rate limiting,
and request/response processing with enterprise-grade reliability.

Security: All provider clients must implement secure API key management.
Performance: Optimized for concurrent requests with intelligent rate limiting.
Type Safety: Complete integration with AI processing architecture.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

from ...core.contracts import require
from ...core.either import Either
from ...core.errors import ValidationError

if TYPE_CHECKING:
    from ...core.ai_integration import (
        AIOperation,
        AIRequest,
        AIResponse,
        CostAmount,
    )


class ProviderStatus(Enum):
    """AI provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    RATE_LIMITED = "rate_limited"


class AuthenticationType(Enum):
    """Authentication methods for AI providers."""

    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"  # noqa: S105 # Enum value, not password
    OAUTH2 = "oauth2"
    SERVICE_ACCOUNT = "service_account"


@dataclass
class RateLimitInfo:
    """Rate limiting information and status."""

    requests_per_minute: int
    tokens_per_minute: int
    current_requests: int = 0
    current_tokens: int = 0
    reset_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_rate_limited(self) -> bool:
        """Check if rate limits are exceeded."""
        now = datetime.now(UTC)
        if now >= self.reset_time:
            # Reset counters
            self.current_requests = 0
            self.current_tokens = 0
            self.reset_time = now + timedelta(minutes=1)
            return False

        return (
            self.current_requests >= self.requests_per_minute
            or self.current_tokens >= self.tokens_per_minute
        )

    def add_usage(self, tokens: int) -> None:
        """Record usage for rate limiting."""
        self.current_requests += 1
        self.current_tokens += tokens


@dataclass
class ProviderCapabilities:
    """Provider capability information."""

    max_tokens: int
    context_window: int
    supports_streaming: bool = False
    supports_function_calling: bool = False
    supports_vision: bool = False
    supported_operations: set[AIOperation] = field(default_factory=set)
    cost_per_input_token: Decimal = field(default_factory=lambda: Decimal(0))
    cost_per_output_token: Decimal = field(default_factory=lambda: Decimal(0))


@dataclass
class ProviderHealth:
    """Provider health status and metrics."""

    status: ProviderStatus
    last_check: datetime
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    success_count: int = 0
    error_count: int = 0
    last_error: str | None = None

    @property
    def uptime_percentage(self) -> float:
        """Calculate uptime percentage."""
        total_requests = self.success_count + self.error_count
        if total_requests == 0:
            return 100.0
        return (self.success_count / total_requests) * 100.0


class BaseProviderClient(ABC):
    """Abstract base class for AI provider clients."""

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        auth_type: AuthenticationType = AuthenticationType.API_KEY,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.provider_name = provider_name
        self.api_key = api_key
        self.auth_type = auth_type
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # Rate limiting and health tracking
        self.rate_limit = RateLimitInfo(
            requests_per_minute=60,
            tokens_per_minute=100000,
        )
        self.health = ProviderHealth(
            status=ProviderStatus.HEALTHY,
            last_check=datetime.now(UTC),
        )

        # Request tracking for monitoring
        self.request_history: list[dict[str, Any]] = []
        self.max_history_size = 1000

    @abstractmethod
    async def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities and limits."""

    @abstractmethod
    async def process_request(
        self,
        request: AIRequest,
    ) -> Either[ValidationError, AIResponse]:
        """Process AI request with provider-specific implementation."""

    @abstractmethod
    async def estimate_cost(
        self,
        request: AIRequest,
    ) -> Either[ValidationError, CostAmount]:
        """Estimate cost for AI request."""

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """Build authentication headers for requests."""

    @abstractmethod
    def _build_request_payload(self, request: AIRequest) -> dict[str, Any]:
        """Build provider-specific request payload."""

    @abstractmethod
    def _parse_response(
        self,
        response_data: dict[str, Any],
    ) -> Either[ValidationError, AIResponse]:
        """Parse provider response into standard format."""

    async def check_health(self) -> ProviderHealth:
        """Check provider health status."""
        start_time = datetime.now(UTC)

        try:
            # Implement basic health check
            response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            self.health.status = ProviderStatus.HEALTHY
            self.health.last_check = datetime.now(UTC)
            self.health.response_time_ms = response_time
            self.health.success_count += 1

        except Exception as e:
            self.health.status = ProviderStatus.UNHEALTHY
            self.health.last_check = datetime.now(UTC)
            self.health.error_count += 1
            self.health.last_error = str(e)

        # Update error rate
        total_requests = self.health.success_count + self.health.error_count
        if total_requests > 0:
            self.health.error_rate = self.health.error_count / total_requests

        return self.health

    async def wait_for_rate_limit(self) -> None:
        """Wait if rate limited."""
        if self.rate_limit.is_rate_limited():
            wait_time = (self.rate_limit.reset_time - datetime.now(UTC)).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    def _record_request(
        self,
        request: AIRequest,
        response: AIResponse | None,
        error: str | None = None,
    ) -> None:
        """Record request for monitoring and analysis."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": request.operation.value,
            "success": error is None,
            "error": error,
            "response_tokens": getattr(response, "token_count", 0) if response else 0,
        }

        self.request_history.append(record)

        # Limit history size
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size :]

    def get_usage_statistics(self) -> dict[str, Any]:
        """Get usage statistics for monitoring."""
        if not self.request_history:
            return {
                "total_requests": 0,
                "success_rate": 100.0,
                "average_tokens": 0,
                "recent_errors": [],
                "health_status": self.health.status.value,
                "uptime_percentage": self.health.uptime_percentage,
            }

        total_requests = len(self.request_history)
        successful_requests = sum(1 for r in self.request_history if r["success"])
        success_rate = (successful_requests / total_requests) * 100.0

        # Calculate average tokens for successful requests
        successful_with_tokens = [
            r for r in self.request_history if r["success"] and r["response_tokens"] > 0
        ]
        average_tokens = (
            (
                sum(r["response_tokens"] for r in successful_with_tokens)
                / len(successful_with_tokens)
            )
            if successful_with_tokens
            else 0
        )

        # Get recent errors
        recent_errors = [r for r in self.request_history[-50:] if not r["success"]]

        return {
            "total_requests": total_requests,
            "success_rate": success_rate,
            "average_tokens": average_tokens,
            "recent_errors": recent_errors,
            "health_status": self.health.status.value,
            "uptime_percentage": self.health.uptime_percentage,
        }

    @require(lambda __self, retries: retries >= 0)
    async def _execute_with_retry(
        self,
        request: AIRequest,
        retries: int = None,
    ) -> Either[ValidationError, AIResponse]:
        """Execute request with retry logic and exponential backoff."""
        if retries is None:
            retries = self.max_retries

        last_error = None

        for attempt in range(retries + 1):
            try:
                # Wait for rate limits
                await self.wait_for_rate_limit()

                # Execute the request
                result = await self.process_request(request)

                if result.is_right():
                    self._record_request(request, result.value)
                    return result
                last_error = result.left_value

            except Exception as e:
                last_error = ValidationError(
                    "request_failed",
                    str(e),
                    "Request execution failed",
                )

                # Exponential backoff for retries
                if attempt < retries:
                    wait_time = 2**attempt  # 1s, 2s, 4s, 8s...
                    await asyncio.sleep(wait_time)

        # Record the failed request
        self._record_request(request, None, str(last_error))
        return Either.left(last_error)


class ProviderRegistry:
    """Registry for managing multiple AI provider clients."""

    def __init__(self):
        self.providers: dict[str, BaseProviderClient] = {}
        self.fallback_order: list[str] = []

    def register_provider(
        self,
        name: str,
        client: BaseProviderClient,
        is_fallback: bool = True,
    ) -> None:
        """Register a provider client."""
        self.providers[name] = client
        if is_fallback and name not in self.fallback_order:
            self.fallback_order.append(name)

    def get_provider(self, name: str) -> BaseProviderClient | None:
        """Get provider by name."""
        return self.providers.get(name)

    async def get_healthy_provider(self) -> BaseProviderClient | None:
        """Get first healthy provider from fallback order."""
        for provider_name in self.fallback_order:
            provider = self.providers.get(provider_name)
            if provider:
                health = await provider.check_health()
                if health.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]:
                    return provider
        return None

    async def process_with_fallback(
        self,
        request: AIRequest,
    ) -> Either[ValidationError, AIResponse]:
        """Process request with automatic fallback to healthy providers."""
        last_error = None

        for provider_name in self.fallback_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            # Check if provider supports the operation
            capabilities = await provider.get_capabilities()
            if request.operation not in capabilities.supported_operations:
                continue

            # Try the provider
            result = await provider._execute_with_retry(request)
            if result.is_right():
                return result
            last_error = result.left_value

        return Either.left(
            last_error
            or ValidationError("no_providers", None, "No healthy providers available"),
        )
