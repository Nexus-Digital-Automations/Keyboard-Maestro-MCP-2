# TASK_27: km_web_request - HTTP/REST API Integration Tool

**Created By**: Agent_ADDER+ (Protocol Gap Analysis) | **Priority**: MEDIUM | **Duration**: 4 hours
**Technique Focus**: HTTP Security + API Integration + Response Processing + Authentication
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED ✅
**Assigned**: Agent_ADDER+
**Dependencies**: TASK_19 (token processor) for dynamic URL construction ✅
**Blocking**: Modern cloud service integration and webhook automation - UNBLOCKED ✅

## 📖 Required Reading (Complete before starting)
- [x] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Web request specification ✅
- [x] **KM Documentation**: development/protocols/KM_MCP.md - HTTP request actions and configuration ✅
- [x] **Token Integration**: development/tasks/TASK_19.md - Token processing for dynamic parameters ✅
- [x] **Security Standards**: HTTPS requirements, authentication patterns, input validation ✅
- [x] **Testing Framework**: tests/TESTING.md - HTTP integration testing requirements ✅

## 🎯 Problem Analysis
**Classification**: Missing Critical Integration Functionality
**Gap Identified**: No HTTP/REST API integration capabilities for cloud service automation
**Impact**: AI cannot integrate with modern web services, APIs, webhooks, or cloud platforms

<thinking>
Root Cause Analysis:
1. Current implementation is limited to local macOS automation only
2. Missing HTTP client capabilities for modern cloud service integration
3. Cannot interact with REST APIs, webhooks, or web services
4. No authentication support for API keys, OAuth, or token-based auth
5. Limited to applications installed locally - cannot leverage cloud services
6. Essential for modern automation workflows that integrate multiple platforms
7. Webhooks and API calls are fundamental to contemporary automation
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design ✅ COMPLETED
- [x] **HTTP client foundation**: Secure HTTP client with timeout and validation ✅
- [x] **Authentication framework**: API key, Bearer token, OAuth, basic auth support ✅
- [x] **Request/response types**: Type-safe HTTP operations with validation ✅

### Phase 2: Core HTTP Operations ✅ COMPLETED
- [x] **GET requests**: Data retrieval with query parameters and headers ✅
- [x] **POST requests**: Data submission with JSON, form data, and file uploads ✅
- [x] **PUT/PATCH requests**: Resource updates with proper content types ✅
- [x] **DELETE requests**: Resource deletion with confirmation patterns ✅

### Phase 3: Advanced Features ✅ COMPLETED
- [x] **Authentication integration**: Multiple auth methods with secure storage ✅
- [x] **Response processing**: JSON parsing, error handling, data extraction ✅
- [x] **Request templating**: Dynamic URLs with token substitution (TASK_19) ✅
- [x] **Token integration**: Full TASK_19 integration with wrapper method ✅

### Phase 4: Security & Integration ✅ COMPLETED
- [x] **URL validation**: Prevent SSRF attacks and validate target endpoints ✅
- [x] **Response sanitization**: Clean and validate response data ✅
- [x] **Property-based tests**: Hypothesis validation for HTTP operations ✅
- [x] **TESTING.md update**: HTTP integration test coverage and security validation ✅

## 🔧 Implementation Files & Specifications
```
src/server/tools/web_request_tools.py       # Main web request tool implementation
src/core/http_client.py                     # HTTP client and request types
src/web/authentication.py                   # Authentication method implementations
src/web/response_processor.py               # Response parsing and data extraction
src/security/url_validator.py               # URL security validation
tests/tools/test_web_request_tools.py       # Unit and integration tests
tests/property_tests/test_http_operations.py # Property-based HTTP validation
```

### km_web_request Tool Specification
```python
@mcp.tool()
async def km_web_request(
    url: str,                                # Target URL with validation
    method: str = "GET",                     # HTTP method (GET, POST, PUT, PATCH, DELETE)
    headers: Optional[Dict[str, str]] = None, # Request headers
    data: Optional[Union[str, Dict]] = None, # Request body data
    params: Optional[Dict[str, str]] = None, # Query parameters
    auth_type: str = "none",                 # Authentication type
    auth_credentials: Optional[Dict[str, str]] = None, # Auth credentials
    timeout_seconds: int = 30,               # Request timeout
    follow_redirects: bool = True,           # Follow HTTP redirects
    verify_ssl: bool = True,                 # SSL certificate verification
    max_response_size: int = 10485760,       # Max response size (10MB)
    response_format: str = "auto",           # Response parsing format
    save_response_to: Optional[str] = None,  # Save response to file/variable
    ctx = None
) -> Dict[str, Any]:
```

### HTTP Integration Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple
from enum import Enum
import httpx
from urllib.parse import urlparse

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
    BEARER_TOKEN = "bearer_token"
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
    headers: Dict[str, str] = field(default_factory=dict)
    data: Optional[Union[str, bytes, Dict]] = None
    params: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    
    @require(lambda self: len(self.url) > 0 and self.url.startswith(('http://', 'https://')))
    @require(lambda self: 1 <= self.timeout_seconds <= 300)
    def __post_init__(self):
        pass
    
    def to_httpx_kwargs(self) -> Dict[str, Any]:
        """Convert to httpx client kwargs."""
        kwargs = {
            "method": self.method.value,
            "url": self.url,
            "headers": self.headers,
            "params": self.params,
            "timeout": self.timeout_seconds
        }
        
        if self.data is not None:
            if isinstance(self.data, dict):
                kwargs["json"] = self.data
            else:
                kwargs["data"] = self.data
        
        return kwargs

@dataclass(frozen=True)
class HTTPResponse:
    """HTTP response container."""
    status_code: int
    headers: Dict[str, str]
    content: Union[str, bytes, Dict]
    url: str
    method: str
    duration_ms: float
    
    @require(lambda self: 100 <= self.status_code <= 599)
    @require(lambda self: self.duration_ms >= 0)
    def __post_init__(self):
        pass
    
    def is_success(self) -> bool:
        """Check if response indicates success."""
        return 200 <= self.status_code < 300
    
    def is_client_error(self) -> bool:
        """Check if response indicates client error."""
        return 400 <= self.status_code < 500
    
    def is_server_error(self) -> bool:
        """Check if response indicates server error."""
        return 500 <= self.status_code < 600

class HTTPClient:
    """Secure HTTP client with comprehensive validation."""
    
    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            verify=True  # SSL verification enabled by default
        )
    
    @require(lambda request: request.url.startswith(('http://', 'https://')))
    @ensure(lambda result: result.is_right() or result.get_left().is_http_error())
    async def execute_request(
        self,
        request: HTTPRequest,
        auth: Optional[Dict[str, str]] = None
    ) -> Either[HTTPError, HTTPResponse]:
        """Execute HTTP request with security validation."""
        # URL security validation
        url_validation = self._validate_url_security(request.url)
        if url_validation.is_left():
            return Either.left(url_validation.get_left())
        
        # Rate limiting check
        rate_limit_result = await self._check_rate_limit()
        if rate_limit_result.is_left():
            return Either.left(rate_limit_result.get_left())
        
        try:
            # Prepare request with authentication
            kwargs = request.to_httpx_kwargs()
            if auth:
                kwargs.update(self._apply_authentication(auth))
            
            # Execute request
            start_time = time.time()
            response = await self._client.request(**kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # Process response
            return await self._process_response(response, request, duration_ms)
            
        except httpx.TimeoutException:
            return Either.left(HTTPError("REQUEST_TIMEOUT", "Request timed out"))
        except httpx.NetworkError as e:
            return Either.left(HTTPError("NETWORK_ERROR", f"Network error: {str(e)}"))
        except Exception as e:
            return Either.left(HTTPError("REQUEST_ERROR", f"Request failed: {str(e)}"))
    
    def _validate_url_security(self, url: str) -> Either[HTTPError, None]:
        """Validate URL for security threats."""
        parsed = urlparse(url)
        
        # Require HTTPS for security (except localhost for testing)
        if parsed.scheme != 'https' and parsed.hostname not in ['localhost', '127.0.0.1']:
            return Either.left(HTTPError("INSECURE_URL", "HTTPS required for external URLs"))
        
        # Prevent SSRF attacks
        forbidden_hosts = [
            'metadata.google.internal',
            '169.254.169.254',  # AWS metadata
            '100.100.100.200',  # Alibaba metadata
            'metadata.digitalocean.com'
        ]
        
        if parsed.hostname in forbidden_hosts:
            return Either.left(HTTPError("FORBIDDEN_HOST", "Access to metadata services not allowed"))
        
        # Validate port ranges
        if parsed.port and (parsed.port < 80 or parsed.port > 65535):
            return Either.left(HTTPError("INVALID_PORT", "Invalid port number"))
        
        return Either.right(None)
    
    async def _process_response(
        self,
        response: httpx.Response,
        request: HTTPRequest,
        duration_ms: float
    ) -> Either[HTTPError, HTTPResponse]:
        """Process HTTP response with content parsing."""
        try:
            # Limit response size
            if len(response.content) > 10 * 1024 * 1024:  # 10MB limit
                return Either.left(HTTPError("RESPONSE_TOO_LARGE", "Response exceeds size limit"))
            
            # Parse content based on content type
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                try:
                    content = response.json()
                except Exception:
                    content = response.text
            elif 'text/' in content_type:
                content = response.text
            else:
                content = response.content
            
            return Either.right(HTTPResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=content,
                url=str(response.url),
                method=request.method.value,
                duration_ms=duration_ms
            ))
            
        except Exception as e:
            return Either.left(HTTPError("RESPONSE_PROCESSING_ERROR", f"Failed to process response: {str(e)}"))

class AuthenticationManager:
    """Secure authentication handling for HTTP requests."""
    
    @staticmethod
    def apply_api_key_auth(credentials: Dict[str, str]) -> Dict[str, Any]:
        """Apply API key authentication."""
        api_key = credentials.get('api_key')
        header_name = credentials.get('header_name', 'X-API-Key')
        
        if not api_key:
            raise ValueError("API key required")
        
        return {"headers": {header_name: api_key}}
    
    @staticmethod
    def apply_bearer_token_auth(credentials: Dict[str, str]) -> Dict[str, Any]:
        """Apply Bearer token authentication."""
        token = credentials.get('token')
        
        if not token:
            raise ValueError("Bearer token required")
        
        return {"headers": {"Authorization": f"Bearer {token}"}}
    
    @staticmethod
    def apply_basic_auth(credentials: Dict[str, str]) -> Dict[str, Any]:
        """Apply HTTP Basic authentication."""
        username = credentials.get('username')
        password = credentials.get('password')
        
        if not username or not password:
            raise ValueError("Username and password required")
        
        return {"auth": (username, password)}
```

## 🔒 Security Implementation
```python
class WebRequestSecurityManager:
    """Security-first web request validation."""
    
    ALLOWED_SCHEMES = ['http', 'https']
    FORBIDDEN_HOSTS = [
        'metadata.google.internal',
        '169.254.169.254',  # AWS metadata
        '100.100.100.200',  # Alibaba metadata
        'metadata.digitalocean.com',
        '169.254.169.254',  # Azure metadata
        'localhost',        # Prevent localhost access from external
        '127.0.0.1',       # Prevent loopback access
        '0.0.0.0',         # Prevent wildcard access
    ]
    
    @staticmethod
    def validate_url_safety(url: str, allow_localhost: bool = False) -> Either[SecurityError, None]:
        """Comprehensive URL security validation."""
        try:
            parsed = urlparse(url)
        except Exception:
            return Either.left(SecurityError("Invalid URL format"))
        
        # Validate scheme
        if parsed.scheme not in WebRequestSecurityManager.ALLOWED_SCHEMES:
            return Either.left(SecurityError(f"Scheme {parsed.scheme} not allowed"))
        
        # Check forbidden hosts
        forbidden_hosts = WebRequestSecurityManager.FORBIDDEN_HOSTS.copy()
        if not allow_localhost:
            forbidden_hosts.extend(['localhost', '127.0.0.1'])
        
        if parsed.hostname in forbidden_hosts:
            return Either.left(SecurityError(f"Host {parsed.hostname} is forbidden"))
        
        # Validate private IP ranges (prevent SSRF)
        if parsed.hostname:
            try:
                import ipaddress
                ip = ipaddress.ip_address(parsed.hostname)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return Either.left(SecurityError("Private IP addresses not allowed"))
            except ValueError:
                # Not an IP address, which is fine
                pass
        
        return Either.right(None)
    
    @staticmethod
    def sanitize_response_data(data: Any, max_depth: int = 10) -> Any:
        """Sanitize response data to prevent injection attacks."""
        if max_depth <= 0:
            return "[MAX_DEPTH_EXCEEDED]"
        
        if isinstance(data, dict):
            return {
                str(k)[:100]: WebRequestSecurityManager.sanitize_response_data(v, max_depth - 1)
                for k, v in list(data.items())[:100]  # Limit dict size
            }
        elif isinstance(data, list):
            return [
                WebRequestSecurityManager.sanitize_response_data(item, max_depth - 1)
                for item in data[:100]  # Limit list size
            ]
        elif isinstance(data, str):
            # Limit string length and remove dangerous characters
            return data[:10000].replace('\x00', '').replace('\x1b', '')
        else:
            return data
    
    @staticmethod
    def validate_request_headers(headers: Dict[str, str]) -> Either[SecurityError, None]:
        """Validate request headers for security."""
        dangerous_headers = [
            'x-forwarded-for',
            'x-real-ip',
            'x-forwarded-host',
            'host'  # Prevent host header injection
        ]
        
        for header_name in headers:
            if header_name.lower() in dangerous_headers:
                return Either.left(SecurityError(f"Header {header_name} not allowed"))
            
            # Check header value length
            if len(headers[header_name]) > 1000:
                return Either.left(SecurityError("Header value too long"))
        
        return Either.right(None)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=100))
def test_url_validation_properties(url_path):
    """Property: URL validation should handle all input safely."""
    test_url = f"https://api.example.com/{url_path}"
    validation = WebRequestSecurityManager.validate_url_safety(test_url)
    
    # Should either pass or fail with security error
    assert validation.is_right() or validation.get_left().code == "SECURITY_ERROR"

@given(st.dictionaries(st.text(min_size=1, max_size=50), st.text(min_size=1, max_size=100)))
def test_header_validation_properties(headers):
    """Property: Header validation should handle all header combinations."""
    validation = WebRequestSecurityManager.validate_request_headers(headers)
    
    # Should validate safely regardless of input
    assert validation.is_right() or validation.get_left().code == "SECURITY_ERROR"

@given(st.integers(min_value=100, max_value=599))
def test_response_status_properties(status_code):
    """Property: HTTP response should handle all valid status codes."""
    response = HTTPResponse(
        status_code=status_code,
        headers={},
        content="test",
        url="https://api.example.com",
        method="GET",
        duration_ms=100.0
    )
    
    assert response.status_code == status_code
    if 200 <= status_code < 300:
        assert response.is_success()
    elif 400 <= status_code < 500:
        assert response.is_client_error()
    elif 500 <= status_code < 600:
        assert response.is_server_error()
```

## 🏗️ Modularity Strategy
- **web_request_tools.py**: Main MCP tool interface (<250 lines)
- **http_client.py**: HTTP client and request handling (<300 lines)
- **authentication.py**: Authentication method implementations (<200 lines)
- **response_processor.py**: Response parsing and data extraction (<250 lines)
- **url_validator.py**: URL security validation (<150 lines)

## 📋 Advanced Web Request Examples

### API Data Retrieval
```python
# Example: Fetch data from REST API
result = await web_client.execute_request(
    HTTPRequest(
        url="https://api.github.com/user/repos",
        method=HTTPMethod.GET,
        headers={"User-Agent": "KM-MCP-Client/1.0"}
    ),
    auth={"type": "bearer_token", "token": "github_token"}
)
```

### JSON Data Submission
```python
# Example: Post data to API
result = await web_client.execute_request(
    HTTPRequest(
        url="https://api.example.com/data",
        method=HTTPMethod.POST,
        data={"name": "test", "value": 42},
        headers={"Content-Type": "application/json"}
    ),
    auth={"type": "api_key", "api_key": "secret_key", "header_name": "X-API-Key"}
)
```

### Webhook Integration
```python
# Example: Send webhook notification
result = await web_client.execute_request(
    HTTPRequest(
        url="https://hooks.slack.com/services/webhook_url",
        method=HTTPMethod.POST,
        data={"text": "Automation completed successfully"},
        timeout_seconds=10
    )
)
```

## ✅ Success Criteria
- Complete HTTP/REST API integration with all major HTTP methods and authentication types
- Comprehensive security validation prevents SSRF attacks and validates all inputs
- Property-based tests validate behavior across all HTTP scenarios and edge cases
- Integration with token processor (TASK_19) for dynamic URL and parameter construction
- Performance: <2s for typical API calls, <5s for complex requests with large responses
- Documentation: Complete API documentation with security considerations and examples
- TESTING.md shows 95%+ test coverage with all security and HTTP integration tests passing
- Tool enables AI to integrate with any REST API, webhook, or cloud service

## 🔄 Integration Points
- **TASK_19 (km_token_processor)**: Dynamic URL construction and parameter substitution
- **TASK_21/22 (conditions/control_flow)**: Conditional API calls based on response data
- **All Future Cloud Tasks**: Foundation for cloud service integration and automation
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- This enables integration with modern cloud services and web APIs
- Essential for contemporary automation workflows that span multiple platforms
- Security is critical - HTTP requests can be used for SSRF and injection attacks
- Must maintain functional programming patterns for testability and composability
- Success here transforms the platform from local-only to cloud-integrated automation
- Foundation for webhook automation, API integrations, and modern cloud workflows