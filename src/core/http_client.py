"""HTTP client infrastructure for secure web request operations.

This module implements a comprehensive HTTP client with security-first design,
supporting all major HTTP methods, authentication types, and response processing.
Includes SSRF protection, input validation, and comprehensive error handling.

Security: URL validation, SSRF protection, response size limits
Performance: Connection pooling, timeout management, efficient parsing
Type Safety: Complete branded type system with contract validation
"""

from __future__ import annotations

import ipaddress
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import httpx

from src.core.constants import (
    ASCII_PRINTABLE_MIN,
    HTTP_CLIENT_ERROR_MIN,
    HTTP_HEADER_LENGTH_MAX,
    HTTP_HEADER_VALUE_MAX,
    HTTP_PORT_MAX,
    HTTP_PORT_MIN,
    HTTP_RATE_LIMIT_PER_MINUTE,
    HTTP_REDIRECT_MAX,
    HTTP_REDIRECT_MIN,
    HTTP_SERVER_ERROR_MIN,
    HTTP_SUCCESS_MAX,
    HTTP_SUCCESS_MIN,
    MAX_HTTP_STATUS_CODE,
    MAX_TIMEOUT_SECONDS,
    MIN_HTTP_STATUS_CODE,
    MIN_TIMEOUT_SECONDS,
    RESPONSE_SIZE_BYTES_1KB,
    RESPONSE_SIZE_BYTES_1MB,
)
from src.core.contracts import ensure, require
from src.core.either import Either
from src.core.errors import MCPError, SecurityError, ValidationError


class HTTPMethod(Enum):
    """Supported HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthenticationType(Enum):
    """Supported authentication types."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"  # noqa: S105 # Enum value, not password
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom_header"


class ResponseFormat(Enum):
    """Response format options."""

    AUTO = "auto"
    JSON = "json"
    TEXT = "text"
    XML = "xml"
    BINARY = "binary"


@dataclass(frozen=True)
class HTTPRequest:
    """Type-safe HTTP request specification."""

    url: str
    method: HTTPMethod = HTTPMethod.GET
    headers: dict[str, str] = field(default_factory=dict)
    data: str | bytes | dict[str, Any] | None = None
    params: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_response_size: int = 10485760  # 10MB

    def __post_init__(self):
        """Contract validation for HTTP request."""
        if not self.url or not self.url.strip():
            raise ValueError("URL cannot be empty")

        if not self.url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")

        if not (MIN_TIMEOUT_SECONDS <= self.timeout_seconds <= MAX_TIMEOUT_SECONDS):
            raise ValueError(
                f"Timeout must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS} seconds",
            )

        MAX_RESPONSE_SIZE_100MB = 100 * RESPONSE_SIZE_BYTES_1MB
        if not (
            RESPONSE_SIZE_BYTES_1KB <= self.max_response_size <= MAX_RESPONSE_SIZE_100MB
        ):
            raise ValueError(
                f"Max response size must be between {RESPONSE_SIZE_BYTES_1KB} bytes and {MAX_RESPONSE_SIZE_100MB} bytes",
            )

    def to_httpx_kwargs(self) -> dict[str, Any]:
        """Convert to httpx client kwargs."""
        kwargs = {
            "method": self.method.value,
            "url": self.url,
            "headers": dict(self.headers),
            "params": dict(self.params),
            "timeout": self.timeout_seconds,
            "follow_redirects": self.follow_redirects,
        }

        if self.data is not None:
            if isinstance(self.data, dict):
                kwargs["json"] = self.data
            elif isinstance(self.data, str):
                kwargs["content"] = self.data.encode("utf-8")
            else:
                kwargs["content"] = self.data

        return kwargs


@dataclass(frozen=True)
class HTTPResponse:
    """HTTP response container with comprehensive metadata."""

    status_code: int
    headers: dict[str, str]
    content: str | bytes | dict[str, Any]
    url: str
    method: str
    duration_ms: float
    content_type: str = ""
    content_length: int = 0

    def __post_init__(self):
        """Contract validation for HTTP response."""
        if not (MIN_HTTP_STATUS_CODE <= self.status_code <= MAX_HTTP_STATUS_CODE):
            raise ValueError(
                f"Status code must be between {MIN_HTTP_STATUS_CODE} and {MAX_HTTP_STATUS_CODE}",
            )

        if self.duration_ms < 0:
            raise ValueError("Duration must be non-negative")

    def is_success(self) -> bool:
        """Check if response indicates success."""
        return HTTP_SUCCESS_MIN <= self.status_code < HTTP_SUCCESS_MAX

    def is_client_error(self) -> bool:
        """Check if response indicates client error."""
        return HTTP_CLIENT_ERROR_MIN <= self.status_code < HTTP_SERVER_ERROR_MIN

    def is_server_error(self) -> bool:
        """Check if response indicates server error."""
        return HTTP_SERVER_ERROR_MIN <= self.status_code < MAX_HTTP_STATUS_CODE + 1

    def is_redirect(self) -> bool:
        """Check if response is a redirect."""
        return HTTP_REDIRECT_MIN <= self.status_code < HTTP_REDIRECT_MAX

    def get_json(self) -> Either[ValidationError, dict[str, Any]]:
        """Safely parse JSON content."""
        if not isinstance(self.content, dict):
            if isinstance(self.content, str):
                try:
                    return Either.right(json.loads(self.content))
                except json.JSONDecodeError as e:
                    return Either.left(
                        ValidationError(
                            field_name="content",
                            value=str(self.content)[:100],
                            constraint=f"Invalid JSON: {e!s}",
                        ),
                    )
            else:
                return Either.left(
                    ValidationError(
                        field_name="content",
                        value=type(self.content).__name__,
                        constraint="Content is not JSON parseable",
                    ),
                )

        return Either.right(self.content)


class HTTPSecurityValidator:
    """Security-first validation for HTTP requests."""

    # Forbidden hosts to prevent SSRF attacks
    FORBIDDEN_HOSTS = {
        "metadata.google.internal",
        "169.254.169.254",  # AWS/Azure metadata
        "100.100.100.200",  # Alibaba metadata
        "metadata.digitalocean.com",
        "metadata.packet.net",
        "metadata.oracle.com",
    }

    # Dangerous headers that could enable attacks
    FORBIDDEN_HEADERS = {
        "x-forwarded-for",
        "x-real-ip",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-cluster-client-ip",
    }

    @staticmethod
    def validate_url_security(
        url: str,
        allow_localhost: bool = False,
    ) -> Either[SecurityError, str]:
        """Comprehensive URL security validation."""
        try:
            parsed = urlparse(url.strip())
        except Exception as e:
            return Either.left(
                SecurityError("URL_PARSE_ERROR", f"Failed to parse URL: {e!s}"),
            )

        # Validate scheme
        if parsed.scheme not in ["http", "https"]:
            return Either.left(
                SecurityError(
                    "INVALID_SCHEME",
                    f"Scheme '{parsed.scheme}' not allowed. Use http or https.",
                ),
            )

        # Require HTTPS for external requests (except localhost)
        # SIM102 fix: Combine conditions in single if statement
        if (
            parsed.scheme != "https"
            and not allow_localhost
            and parsed.hostname not in ["localhost", "127.0.0.1", "::1"]
        ):
            return Either.left(
                SecurityError("INSECURE_URL", "HTTPS required for external URLs"),
            )

        # Check forbidden hosts
        if parsed.hostname in HTTPSecurityValidator.FORBIDDEN_HOSTS:
            return Either.left(
                SecurityError(
                    "FORBIDDEN_HOST",
                    f"Access to host '{parsed.hostname}' is not allowed",
                ),
            )

        # Validate against private IP ranges (SSRF protection)
        if parsed.hostname and not allow_localhost:
            try:
                ip = ipaddress.ip_address(parsed.hostname)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return Either.left(
                        SecurityError(
                            "PRIVATE_IP_ACCESS",
                            "Access to private IP addresses is not allowed",
                        ),
                    )
            except ValueError:
                # Not an IP address, which is fine for domain names
                pass

        # Validate port
        if parsed.port:
            if parsed.port < HTTP_PORT_MIN or parsed.port > HTTP_PORT_MAX:
                return Either.left(
                    SecurityError("INVALID_PORT", f"Port {parsed.port} is not valid"),
                )

            # Restrict certain ports that could enable attacks
            dangerous_ports = {22, 23, 25, 53, 135, 139, 445, 993, 995}
            if parsed.port in dangerous_ports:
                return Either.left(
                    SecurityError(
                        "DANGEROUS_PORT",
                        f"Access to port {parsed.port} is restricted",
                    ),
                )

        return Either.right(url.strip())

    @staticmethod
    def validate_headers(
        headers: dict[str, str],
    ) -> Either[SecurityError, dict[str, str]]:
        """Validate request headers for security."""
        validated_headers = {}

        for name, value in headers.items():
            # Check for forbidden headers
            if name.lower() in HTTPSecurityValidator.FORBIDDEN_HEADERS:
                return Either.left(
                    SecurityError(
                        "FORBIDDEN_HEADER",
                        f"Header '{name}' is not allowed",
                    ),
                )

            # Validate header name
            if not name or len(name) > HTTP_HEADER_LENGTH_MAX:
                return Either.left(
                    SecurityError(
                        "INVALID_HEADER_NAME",
                        f"Header name '{name}' is invalid or too long",
                    ),
                )

            # Validate header value
            if not isinstance(value, str):
                value = str(value)

            if len(value) > HTTP_HEADER_VALUE_MAX:  # 8KB limit per header
                return Either.left(
                    SecurityError(
                        "HEADER_VALUE_TOO_LONG",
                        f"Header '{name}' value exceeds 8KB limit",
                    ),
                )

            # Remove control characters
            clean_value = "".join(
                char
                for char in value
                if ord(char) >= ASCII_PRINTABLE_MIN or char in "\t\n\r"
            )
            validated_headers[name] = clean_value

        return Either.right(validated_headers)

    @staticmethod
    def sanitize_response_data(
        data: Any,
        max_depth: int = 10,
        max_string_length: int = 10000,
    ) -> Any:
        """Recursively sanitize response data."""
        if max_depth <= 0:
            return "[MAX_DEPTH_EXCEEDED]"

        if isinstance(data, dict):
            return {
                str(k)[:100]: HTTPSecurityValidator.sanitize_response_data(
                    v,
                    max_depth - 1,
                    max_string_length,
                )
                for k, v in list(data.items())[:1000]  # Limit dict size
            }
        if isinstance(data, list):
            return [
                HTTPSecurityValidator.sanitize_response_data(
                    item,
                    max_depth - 1,
                    max_string_length,
                )
                for item in data[:1000]  # Limit list size
            ]
        if isinstance(data, str):
            # Limit string length and remove null bytes
            sanitized = data[:max_string_length].replace("\x00", "").replace("\x1b", "")
            return sanitized
        if isinstance(data, int | float | bool):
            return data
        if data is None:
            return None
        # Convert unknown types to string representation
        return str(data)[:max_string_length]


class HTTPClient:
    """Secure HTTP client with comprehensive validation and error handling."""

    def __init__(self, allow_localhost: bool = False):
        """Initialize HTTP client with security configuration."""
        self.allow_localhost = allow_localhost
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = HTTPRateLimiter()

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: str,
        exc_val: Exception | str,
        exc_tb: Exception | str,
    ):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                verify=True,  # SSL verification enabled
                follow_redirects=False,  # We handle redirects manually for security
            )

    @require(lambda __self, request: isinstance(request, HTTPRequest))
    @ensure(
        lambda result: result.is_right()
        or isinstance(result.get_left(), SecurityError | MCPError),
    )
    async def execute_request(
        self,
        request: HTTPRequest,
        auth_headers: dict[str, str] | None = None,
    ) -> Either[SecurityError | MCPError, HTTPResponse]:
        """Execute HTTP request with comprehensive security validation."""
        try:
            await self._ensure_client()

            # Security validation
            url_validation = HTTPSecurityValidator.validate_url_security(
                request.url,
                self.allow_localhost,
            )
            if url_validation.is_left():
                return Either.left(url_validation.get_left())

            validated_url = url_validation.get_right()

            # Merge and validate headers
            all_headers = {**request.headers}
            if auth_headers:
                all_headers.update(auth_headers)

            header_validation = HTTPSecurityValidator.validate_headers(all_headers)
            if header_validation.is_left():
                return Either.left(header_validation.get_left())

            validated_headers = header_validation.get_right()

            # Rate limiting check
            rate_limit_result = await self._rate_limiter.check_rate_limit(validated_url)
            if rate_limit_result.is_left():
                return Either.left(rate_limit_result.get_left())

            # Execute request
            start_time = time.time()

            try:
                # Create request with validated parameters
                http_request = HTTPRequest(
                    url=validated_url,
                    method=request.method,
                    headers=validated_headers,
                    data=request.data,
                    params=request.params,
                    timeout_seconds=request.timeout_seconds,
                    verify_ssl=request.verify_ssl,
                    follow_redirects=False,  # Handle manually
                    max_response_size=request.max_response_size,
                )

                kwargs = http_request.to_httpx_kwargs()

                response = await self._client.request(**kwargs)
                duration_ms = (time.time() - start_time) * 1000

                # Process response
                return await self._process_response(response, request, duration_ms)

            except httpx.TimeoutException:
                return Either.left(
                    MCPError(
                        "REQUEST_TIMEOUT",
                        f"Request timed out after {request.timeout_seconds} seconds",
                    ),
                )
            except httpx.NetworkError as e:
                return Either.left(
                    MCPError("NETWORK_ERROR", f"Network error: {e!s}"),
                )
            except httpx.HTTPStatusError as e:
                return Either.left(
                    MCPError(
                        "HTTP_ERROR",
                        f"HTTP error {e.response.status_code}: {e.response.text}",
                    ),
                )

        except Exception as e:
            return Either.left(
                MCPError(
                    "REQUEST_ERROR",
                    f"Unexpected error executing request: {e!s}",
                ),
            )

    async def _process_response(
        self,
        response: httpx.Response,
        request: HTTPRequest,
        duration_ms: float,
    ) -> Either[MCPError, HTTPResponse]:
        """Process HTTP response with security validation."""
        try:
            # Check response size
            content_length = len(response.content)
            if content_length > request.max_response_size:
                return Either.left(
                    MCPError(
                        "RESPONSE_TOO_LARGE",
                        f"Response size {content_length} exceeds limit {request.max_response_size}",
                    ),
                )

            # Get content type
            content_type = response.headers.get("content-type", "").lower()

            # Parse content based on content type
            if "application/json" in content_type:
                try:
                    content = response.json()
                    # Sanitize JSON content
                    content = HTTPSecurityValidator.sanitize_response_data(content)
                except json.JSONDecodeError:
                    # Fall back to text if JSON parsing fails
                    content = response.text
            elif content_type.startswith("text/"):
                content = response.text
            else:
                # For binary content, keep as bytes but limit size
                content = response.content

            return Either.right(
                HTTPResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=content,
                    url=str(response.url),
                    method=request.method.value,
                    duration_ms=duration_ms,
                    content_type=content_type,
                    content_length=content_length,
                ),
            )

        except Exception as e:
            return Either.left(
                MCPError(
                    "RESPONSE_PROCESSING_ERROR",
                    f"Failed to process response: {e!s}",
                ),
            )


class HTTPRateLimiter:
    """Rate limiter to prevent abuse and excessive requests."""

    def __init__(self, max_requests_per_minute: int = HTTP_RATE_LIMIT_PER_MINUTE):
        self.max_requests_per_minute = max_requests_per_minute
        self._request_times: dict[str, list[float]] = {}

    async def check_rate_limit(self, url: str) -> Either[MCPError, None]:
        """Check if request is within rate limits."""
        try:
            # Extract host for rate limiting
            parsed = urlparse(url)
            host = parsed.hostname or "unknown"

            current_time = time.time()
            minute_ago = current_time - 60

            # Clean old requests
            if host in self._request_times:
                self._request_times[host] = [
                    req_time
                    for req_time in self._request_times[host]
                    if req_time > minute_ago
                ]
            else:
                self._request_times[host] = []

            # Check rate limit
            if len(self._request_times[host]) >= self.max_requests_per_minute:
                return Either.left(
                    MCPError(
                        "RATE_LIMIT_EXCEEDED",
                        f"Rate limit exceeded for {host}. Max {self.max_requests_per_minute} requests per minute.",
                    ),
                )

            # Record this request
            self._request_times[host].append(current_time)

            return Either.right(None)

        except Exception:
            # Don't fail requests due to rate limiter errors
            return Either.right(None)
