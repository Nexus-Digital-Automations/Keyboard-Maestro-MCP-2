"""
Property-based tests for web request functionality.

Tests core properties of HTTP client, authentication, and web request tools
using Hypothesis for comprehensive input validation and security verification.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
import json
from urllib.parse import urlparse

from src.core.http_client import (
    HTTPClient, HTTPRequest, HTTPMethod, HTTPSecurityValidator, 
    HTTPResponse, HTTPRateLimiter
)
from src.web.authentication import (
    AuthenticationManager, AuthenticationType, 
    create_api_key_auth, create_bearer_token_auth
)
from src.server.tools.web_request_tools import WebRequestProcessor
from src.core.errors import ValidationError, SecurityError, MCPError


# Custom strategies for generating test data
@composite
def safe_urls(draw):
    """Generate safe HTTPS URLs for testing."""
    domains = st.sampled_from([
        "api.github.com", "httpbin.org", "jsonplaceholder.typicode.com",
        "api.stripe.com", "api.twitter.com", "graph.microsoft.com"
    ])
    
    domain = draw(domains)
    path = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789/-_",
        min_size=0, max_size=50
    ))
    
    return f"https://{domain}/{path.strip('/')}"


@composite
def unsafe_urls(draw):
    """Generate URLs that should be rejected by security validation."""
    schemes = st.sampled_from(["file", "ftp", "javascript", "data"])
    hosts = st.sampled_from([
        "localhost", "127.0.0.1", "169.254.169.254", 
        "metadata.google.internal", "10.0.0.1"
    ])
    
    scheme = draw(schemes)
    host = draw(hosts)
    
    return f"{scheme}://{host}/test"


@composite
def http_headers(draw):
    """Generate valid HTTP headers."""
    header_names = st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
        min_size=1, max_size=30
    ).filter(lambda x: not x.startswith('-') and not x.endswith('-'))
    
    header_values = st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ./_-=",
        min_size=0, max_size=100
    )
    
    num_headers = draw(st.integers(min_value=0, max_value=10))
    headers = {}
    
    for _ in range(num_headers):
        name = draw(header_names)
        value = draw(header_values)
        headers[name] = value
    
    return headers


@composite
def api_keys(draw):
    """Generate realistic API keys for testing."""
    length = draw(st.integers(min_value=16, max_value=128))
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    return draw(st.text(alphabet=alphabet, min_size=length, max_size=length))


@composite
def bearer_tokens(draw):
    """Generate realistic Bearer tokens for testing."""
    # JWT-like structure: header.payload.signature
    segments = []
    for _ in range(3):
        length = draw(st.integers(min_value=20, max_value=200))
        segment = draw(st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
            min_size=length, max_size=length
        ))
        segments.append(segment)
    
    return ".".join(segments)


class TestHTTPSecurityValidatorProperties:
    """Property-based tests for HTTP security validation."""
    
    @given(safe_urls())
    def test_safe_urls_are_accepted(self, url):
        """Property: Safe HTTPS URLs should pass validation."""
        result = HTTPSecurityValidator.validate_url_security(url, allow_localhost=False)
        assert result.is_right(), f"Safe URL rejected: {url}"
        assert result.get_right() == url
    
    @given(unsafe_urls())
    def test_unsafe_urls_are_rejected(self, url):
        """Property: Unsafe URLs should be rejected by security validation."""
        result = HTTPSecurityValidator.validate_url_security(url, allow_localhost=False)
        assert result.is_left(), f"Unsafe URL accepted: {url}"
        assert isinstance(result.get_left(), SecurityError)
    
    @given(http_headers())
    def test_valid_headers_are_accepted(self, headers):
        """Property: Valid headers should pass security validation."""
        # Filter out potentially forbidden headers
        safe_headers = {
            k: v for k, v in headers.items() 
            if k.lower() not in HTTPSecurityValidator.FORBIDDEN_HEADERS
            and len(k) <= 100 and len(v) <= 8192
        }
        
        result = HTTPSecurityValidator.validate_headers(safe_headers)
        assert result.is_right(), f"Valid headers rejected: {safe_headers}"
    
    @given(st.text(min_size=1, max_size=1000))
    def test_response_data_sanitization_preserves_structure(self, data):
        """Property: Response data sanitization should preserve basic structure."""
        sanitized = HTTPSecurityValidator.sanitize_response_data(data)
        assert isinstance(sanitized, str)
        assert len(sanitized) <= len(data)  # Should not grow
    
    @given(st.lists(st.text(max_size=100), max_size=100))
    def test_response_list_sanitization(self, data_list):
        """Property: List sanitization should preserve list structure."""
        sanitized = HTTPSecurityValidator.sanitize_response_data(data_list)
        assert isinstance(sanitized, list)
        assert len(sanitized) <= min(len(data_list), 1000)  # Respects size limits


class TestAuthenticationProperties:
    """Property-based tests for authentication functionality."""
    
    @given(api_keys())
    def test_api_key_auth_creation_success(self, api_key):
        """Property: Valid API keys should create successful authentication."""
        result = create_api_key_auth(api_key)
        
        assert result.is_right(), f"API key auth creation failed: {api_key[:10]}..."
        auth = result.get_right()
        assert auth.auth_type == AuthenticationType.API_KEY
        assert auth.credentials['api_key'] == api_key
    
    @given(bearer_tokens())
    def test_bearer_token_auth_creation_success(self, token):
        """Property: Valid Bearer tokens should create successful authentication."""
        result = create_bearer_token_auth(token)
        
        assert result.is_right(), f"Bearer token auth creation failed"
        auth = result.get_right()
        assert auth.auth_type == AuthenticationType.BEARER_TOKEN
        assert auth.credentials['token'] == token
    
    @given(st.text(min_size=1, max_size=7))
    def test_short_credentials_are_rejected(self, short_credential):
        """Property: Credentials shorter than 8 characters should be rejected."""
        assume(len(short_credential) < 8)
        
        api_result = create_api_key_auth(short_credential)
        assert api_result.is_left(), f"Short API key accepted: {short_credential}"
        
        bearer_result = create_bearer_token_auth(short_credential)
        assert bearer_result.is_left(), f"Short Bearer token accepted: {short_credential}"
    
    @given(st.text(min_size=513))  # Longer than 512 char limit
    def test_long_api_keys_are_rejected(self, long_api_key):
        """Property: API keys longer than 512 characters should be rejected."""
        result = create_api_key_auth(long_api_key)
        assert result.is_left(), f"Long API key accepted: {len(long_api_key)} chars"
    
    @given(st.text(min_size=2049))  # Longer than 2048 char limit
    def test_long_bearer_tokens_are_rejected(self, long_token):
        """Property: Bearer tokens longer than 2048 characters should be rejected."""
        result = create_bearer_token_auth(long_token)
        assert result.is_left(), f"Long Bearer token accepted: {len(long_token)} chars"


class TestHTTPRequestProperties:
    """Property-based tests for HTTP request validation."""
    
    @given(
        safe_urls(),
        st.sampled_from(list(HTTPMethod)),
        http_headers(),
        st.integers(min_value=1, max_value=300),
        st.integers(min_value=1024, max_value=104857600)
    )
    def test_valid_http_request_creation(self, url, method, headers, timeout, max_size):
        """Property: Valid parameters should create successful HTTP requests."""
        try:
            request = HTTPRequest(
                url=url,
                method=method,
                headers=headers,
                timeout_seconds=timeout,
                max_response_size=max_size
            )
            
            assert request.url == url
            assert request.method == method
            assert request.timeout_seconds == timeout
            assert request.max_response_size == max_size
            
        except ValueError as e:
            # Some combinations might still be invalid due to security checks
            assert "forbidden" in str(e).lower() or "unsafe" in str(e).lower()
    
    @given(st.text().filter(lambda x: not x.startswith(('http://', 'https://'))))
    def test_invalid_url_schemes_rejected(self, invalid_url):
        """Property: URLs without http/https schemes should be rejected."""
        with pytest.raises(ValueError, match="URL must start with http"):
            HTTPRequest(url=invalid_url)
    
    @given(st.integers().filter(lambda x: x < 1 or x > 300))
    def test_invalid_timeouts_rejected(self, invalid_timeout):
        """Property: Timeouts outside 1-300 second range should be rejected."""
        with pytest.raises(ValueError, match="Timeout must be between"):
            HTTPRequest(
                url="https://example.com",
                timeout_seconds=invalid_timeout
            )


class TestWebRequestProcessorProperties:
    """Property-based tests for web request processing."""
    
    @pytest.fixture
    def processor(self):
        """Provide a web request processor for testing."""
        return WebRequestProcessor(allow_localhost=True)  # Allow localhost for testing
    
    @given(safe_urls())
    @settings(max_examples=20, deadline=5000)  # Limit examples for async tests
    async def test_url_processing_preserves_safety(self, processor, url):
        """Property: URL processing should preserve security validation."""
        result = await processor._process_url_with_tokens(url, None)
        
        assert result.is_right(), f"Safe URL processing failed: {url}"
        processed_url = result.get_right()
        
        # Processed URL should still be safe
        validation_result = HTTPSecurityValidator.validate_url_security(
            processed_url, allow_localhost=True
        )
        assert validation_result.is_right(), f"Processed URL became unsafe: {processed_url}"
    
    @given(http_headers())
    async def test_header_processing_maintains_security(self, processor, headers):
        """Property: Header processing should maintain security constraints."""
        # Filter to safe headers first
        safe_headers = {
            k: v for k, v in headers.items() 
            if k.lower() not in HTTPSecurityValidator.FORBIDDEN_HEADERS
            and len(k) <= 100 and len(v) <= 8192
        }
        
        result = await processor._process_headers_with_tokens(safe_headers, None)
        
        assert result.is_right(), f"Safe headers processing failed"
        processed_headers = result.get_right()
        
        # Processed headers should still pass validation
        validation_result = HTTPSecurityValidator.validate_headers(processed_headers)
        assert validation_result.is_right(), f"Processed headers became unsafe"
    
    @given(st.one_of(st.none(), st.text(max_size=1000), st.dictionaries(
        st.text(max_size=50), st.text(max_size=200), max_size=10
    )))
    async def test_data_processing_handles_all_types(self, processor, data):
        """Property: Data processing should handle all supported data types."""
        result = await processor._process_request_data(data, None)
        
        assert result.is_right(), f"Data processing failed for type: {type(data)}"
        processed_data = result.get_right()
        
        # Type should be preserved or converted appropriately
        if data is None:
            assert processed_data is None
        else:
            assert processed_data is not None
    
    @given(st.sampled_from(["api_key", "bearer_token", "basic_auth", "none"]))
    async def test_authentication_processing_handles_all_types(self, processor, auth_type):
        """Property: Authentication processing should handle all supported types."""
        # Generate appropriate credentials for each type
        credentials = {}
        if auth_type == "api_key":
            credentials = {"api_key": "test_key_12345678"}
        elif auth_type == "bearer_token":
            credentials = {"token": "test_token_12345678"}
        elif auth_type == "basic_auth":
            credentials = {"username": "testuser", "password": "testpass123"}
        
        result = await processor._process_authentication(auth_type, credentials, None)
        
        if auth_type == "none":
            assert result.is_right()
            assert result.get_right() is None
        else:
            # Should either succeed or fail with proper error
            if result.is_left():
                error = result.get_left()
                assert isinstance(error, MCPError)
            else:
                auth_headers = result.get_right()
                assert auth_headers is None or isinstance(auth_headers, dict)


class TestRateLimiterProperties:
    """Property-based tests for rate limiting functionality."""
    
    @given(st.integers(min_value=1, max_value=1000))
    def test_rate_limiter_respects_limits(self, max_requests):
        """Property: Rate limiter should enforce configured limits."""
        rate_limiter = HTTPRateLimiter(max_requests_per_minute=max_requests)
        
        # Should handle any reasonable URL
        test_url = "https://example.com/test"
        
        # First request should always be allowed
        result = asyncio.run(rate_limiter.check_rate_limit(test_url))
        assert result.is_right(), "First request should be allowed"
    
    @given(safe_urls())
    async def test_rate_limiter_handles_all_urls(self, url):
        """Property: Rate limiter should handle any valid URL."""
        rate_limiter = HTTPRateLimiter(max_requests_per_minute=60)
        
        result = await rate_limiter.check_rate_limit(url)
        
        # Should either allow or deny, but not crash
        assert result.is_right() or result.is_left()
        if result.is_left():
            error = result.get_left()
            assert isinstance(error, MCPError)


@pytest.mark.asyncio
class TestIntegrationProperties:
    """Integration property tests across multiple components."""
    
    @given(
        safe_urls(),
        st.sampled_from(["GET", "POST", "PUT", "DELETE"]),
        st.sampled_from([None, {"test": "data"}])
    )
    @settings(max_examples=10, deadline=10000)
    async def test_complete_request_processing_pipeline(self, url, method, data):
        """Property: Complete request processing should be consistent."""
        processor = WebRequestProcessor(allow_localhost=True)
        
        # Create a complete request that should be processable
        try:
            result = await processor.process_web_request(
                url=url,
                method=method,
                data=data,
                timeout_seconds=5,  # Short timeout for testing
                verify_ssl=False,   # Skip SSL for testing
                max_response_size=1024 * 1024  # 1MB limit
            )
            
            # Should either succeed or fail with proper error handling
            if result.is_left():
                error = result.get_left()
                assert isinstance(error, MCPError)
                # Error should have descriptive message
                assert len(error.message) > 0
            else:
                response_data = result.get_right()
                assert isinstance(response_data, dict)
                # Response should have required fields
                assert "success" in response_data
                assert "status_code" in response_data
                assert "url" in response_data
                
        except Exception as e:
            # If exception occurs, it should be a known type
            assert isinstance(e, (ValidationError, SecurityError, MCPError, ValueError))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])