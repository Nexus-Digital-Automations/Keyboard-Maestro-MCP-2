# TASK_33: km_web_automation - Advanced Web Integration & API Automation

**Created By**: Agent_1 (Platform Expansion) | **Priority**: HIGH | **Duration**: 4 hours
**Technique Focus**: Design by Contract + Type Safety + HTTP Security + API Integration + Async Operations
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: Foundation tasks (TASK_1-20), Communication integration (TASK_32)
**Blocking**: Advanced web automation workflows requiring HTTP/API integration

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/KM_MCP.md - Web request handling (lines 1076-1097)
- [ ] **Foundation Architecture**: src/server/tools/ - Existing tool patterns and HTTP integration
- [ ] **Security Framework**: src/core/contracts.py - HTTP security and validation
- [ ] **Action Integration**: development/tasks/TASK_14.md - Action builder integration patterns
- [ ] **Testing Requirements**: tests/TESTING.md - HTTP testing and security validation

## 🎯 Problem Analysis
**Classification**: Web Integration Infrastructure Gap
**Gap Identified**: No comprehensive web automation and API integration capabilities
**Impact**: AI cannot interact with web services, APIs, or perform web-based automation tasks

<thinking>
Root Cause Analysis:
1. Current platform focuses on local macOS automation but lacks web integration
2. No systematic HTTP request handling for API automation and web service interaction
3. Missing webhook support for external system integration
4. Cannot handle authentication, headers, and complex HTTP workflows
5. Essential for modern automation that integrates with cloud services and APIs
6. Should integrate with existing action builder for web-based automation sequences
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **HTTP types**: Define branded types for requests, responses, and authentication
- [ ] **Security framework**: Comprehensive HTTP security and input validation
- [ ] **Authentication system**: Support for various auth methods (API keys, OAuth, etc.)

### Phase 2: Core HTTP Operations
- [ ] **Request handling**: GET, POST, PUT, DELETE with comprehensive parameter support
- [ ] **Response processing**: JSON/XML parsing, header extraction, status handling
- [ ] **Error handling**: HTTP error codes, timeout handling, retry logic
- [ ] **Content types**: Support for JSON, XML, form data, file uploads

### Phase 3: Advanced Features
- [ ] **Authentication flows**: API key, Bearer token, Basic auth, custom headers
- [ ] **Webhook support**: Receive and process incoming HTTP requests
- [ ] **Rate limiting**: Respect API rate limits and implement backoff strategies
- [ ] **Response caching**: Intelligent caching for improved performance

### Phase 4: Integration & Automation
- [ ] **Action builder integration**: HTTP actions for macro sequences
- [ ] **Template system**: Reusable request templates and API configurations
- [ ] **Batch operations**: Multiple requests with dependency management
- [ ] **Monitoring**: Request logging, performance metrics, error tracking

### Phase 5: Testing & Security
- [ ] **TESTING.md update**: HTTP testing coverage and security validation
- [ ] **Security testing**: Prevent injection attacks and validate all HTTP content
- [ ] **Performance optimization**: Connection pooling and efficient request handling
- [ ] **Integration tests**: End-to-end web automation workflow validation

## 🔧 Implementation Files & Specifications
```
src/server/tools/web_automation_tools.py          # Main web automation tool implementation
src/core/web_requests.py                          # HTTP type definitions and client
src/web/http_client.py                            # HTTP request handling and security
src/web/auth_manager.py                           # Authentication and credential management
src/web/webhook_handler.py                       # Webhook receiving and processing
src/web/request_templates.py                     # Reusable request templates
tests/tools/test_web_automation_tools.py          # Unit and integration tests
tests/property_tests/test_web_automation.py       # Property-based HTTP validation
```

### km_web_automation Tool Specification
```python
@mcp.tool()
async def km_web_automation(
    operation: str,                             # request|webhook|template|batch
    method: str = "GET",                        # GET|POST|PUT|DELETE|PATCH
    url: str,                                   # Target URL for request
    headers: Optional[Dict[str, str]] = None,   # HTTP headers
    body: Optional[Union[str, Dict]] = None,    # Request body (JSON or string)
    auth_type: Optional[str] = None,            # none|api_key|bearer|basic|custom
    auth_credentials: Optional[Dict] = None,    # Authentication credentials
    timeout: int = 30,                          # Request timeout in seconds
    follow_redirects: bool = True,              # Follow HTTP redirects
    verify_ssl: bool = True,                    # Verify SSL certificates
    save_response_to: Optional[str] = None,     # Variable name to save response
    template_name: Optional[str] = None,        # Use predefined request template
    retry_count: int = 3,                       # Number of retry attempts
    rate_limit: Optional[Dict] = None,          # Rate limiting configuration
    ctx = None
) -> Dict[str, Any]:
```

### Web Request Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Set
from enum import Enum
import re
import json
from urllib.parse import urlparse

class HTTPMethod(Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class AuthenticationType(Enum):
    """HTTP authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer"
    BASIC_AUTH = "basic"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom"

class ContentType(Enum):
    """HTTP content types."""
    JSON = "application/json"
    XML = "application/xml"
    FORM_DATA = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"
    TEXT = "text/plain"
    HTML = "text/html"

@dataclass(frozen=True)
class HTTPHeaders:
    """Type-safe HTTP headers with validation."""
    headers: Dict[str, str] = field(default_factory=dict)
    
    @require(lambda self: all(self._is_valid_header_name(k) for k in self.headers.keys()))
    @require(lambda self: all(self._is_valid_header_value(v) for v in self.headers.values()))
    def __post_init__(self):
        pass
    
    def _is_valid_header_name(self, name: str) -> bool:
        """Validate HTTP header name."""
        # RFC 7230: header names are case-insensitive and contain only token characters
        return re.match(r'^[a-zA-Z0-9!#$%&\'*+\-.^_`|~]+$', name) is not None
    
    def _is_valid_header_value(self, value: str) -> bool:
        """Validate HTTP header value."""
        # Basic validation - no control characters except tab
        return not re.search(r'[\x00-\x08\x0A-\x1F\x7F]', value)
    
    def add_header(self, name: str, value: str) -> 'HTTPHeaders':
        """Add header with validation."""
        new_headers = self.headers.copy()
        new_headers[name] = value
        return HTTPHeaders(new_headers)
    
    def get_content_type(self) -> Optional[str]:
        """Get content type header."""
        for name, value in self.headers.items():
            if name.lower() == 'content-type':
                return value
        return None

@dataclass(frozen=True)
class HTTPAuthentication:
    """HTTP authentication configuration."""
    auth_type: AuthenticationType
    credentials: Dict[str, str] = field(default_factory=dict)
    
    @require(lambda self: self._validate_credentials())
    def __post_init__(self):
        pass
    
    def _validate_credentials(self) -> bool:
        """Validate authentication credentials."""
        if self.auth_type == AuthenticationType.API_KEY:
            return 'api_key' in self.credentials and 'header_name' in self.credentials
        elif self.auth_type == AuthenticationType.BEARER_TOKEN:
            return 'token' in self.credentials
        elif self.auth_type == AuthenticationType.BASIC_AUTH:
            return 'username' in self.credentials and 'password' in self.credentials
        elif self.auth_type == AuthenticationType.CUSTOM_HEADER:
            return 'header_name' in self.credentials and 'header_value' in self.credentials
        return True
    
    def apply_to_headers(self, headers: HTTPHeaders) -> HTTPHeaders:
        """Apply authentication to HTTP headers."""
        if self.auth_type == AuthenticationType.API_KEY:
            header_name = self.credentials['header_name']
            api_key = self.credentials['api_key']
            return headers.add_header(header_name, api_key)
        elif self.auth_type == AuthenticationType.BEARER_TOKEN:
            token = self.credentials['token']
            return headers.add_header('Authorization', f'Bearer {token}')
        elif self.auth_type == AuthenticationType.BASIC_AUTH:
            import base64
            username = self.credentials['username']
            password = self.credentials['password']
            auth_string = base64.b64encode(f'{username}:{password}'.encode()).decode()
            return headers.add_header('Authorization', f'Basic {auth_string}')
        elif self.auth_type == AuthenticationType.CUSTOM_HEADER:
            header_name = self.credentials['header_name']
            header_value = self.credentials['header_value']
            return headers.add_header(header_name, header_value)
        return headers

@dataclass(frozen=True)
class HTTPRequest:
    """Complete HTTP request specification."""
    method: HTTPMethod
    url: str
    headers: HTTPHeaders = field(default_factory=HTTPHeaders)
    body: Optional[Union[str, Dict[str, Any]]] = None
    authentication: Optional[HTTPAuthentication] = None
    timeout: int = 30
    follow_redirects: bool = True
    verify_ssl: bool = True
    
    @require(lambda self: self._is_valid_url(self.url))
    @require(lambda self: 1 <= self.timeout <= 300)
    def __post_init__(self):
        pass
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and security."""
        try:
            parsed = urlparse(url)
            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
            # Only allow HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False
            # Block localhost and internal IPs for security
            if self._is_internal_address(parsed.netloc):
                return False
            return True
        except:
            return False
    
    def _is_internal_address(self, netloc: str) -> bool:
        """Check if address is internal/localhost."""
        internal_patterns = [
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
            '::1',
            '10.',
            '192.168.',
            '172.16.',
            '172.17.',
            '172.18.',
            '172.19.',
            '172.20.',
            '172.21.',
            '172.22.',
            '172.23.',
            '172.24.',
            '172.25.',
            '172.26.',
            '172.27.',
            '172.28.',
            '172.29.',
            '172.30.',
            '172.31.'
        ]
        
        hostname = netloc.split(':')[0].lower()
        return any(hostname.startswith(pattern) for pattern in internal_patterns)
    
    def prepare_for_execution(self) -> 'HTTPRequest':
        """Prepare request with authentication and final headers."""
        final_headers = self.headers
        
        # Apply authentication
        if self.authentication:
            final_headers = self.authentication.apply_to_headers(final_headers)
        
        # Add default headers if not present
        if self.body and not final_headers.get_content_type():
            if isinstance(self.body, dict):
                final_headers = final_headers.add_header('Content-Type', ContentType.JSON.value)
            else:
                final_headers = final_headers.add_header('Content-Type', ContentType.TEXT.value)
        
        return HTTPRequest(
            method=self.method,
            url=self.url,
            headers=final_headers,
            body=self.body,
            authentication=self.authentication,
            timeout=self.timeout,
            follow_redirects=self.follow_redirects,
            verify_ssl=self.verify_ssl
        )

@dataclass(frozen=True)
class HTTPResponse:
    """HTTP response with comprehensive data."""
    status_code: int
    headers: HTTPHeaders
    body: str
    url: str
    execution_time: float
    error: Optional[str] = None
    
    def is_success(self) -> bool:
        """Check if response indicates success."""
        return 200 <= self.status_code < 300
    
    def is_client_error(self) -> bool:
        """Check if response indicates client error."""
        return 400 <= self.status_code < 500
    
    def is_server_error(self) -> bool:
        """Check if response indicates server error."""
        return 500 <= self.status_code < 600
    
    def get_json(self) -> Optional[Dict[str, Any]]:
        """Parse response body as JSON."""
        try:
            return json.loads(self.body)
        except (json.JSONDecodeError, TypeError):
            return None
    
    def get_content_type(self) -> Optional[str]:
        """Get response content type."""
        return self.headers.get_content_type()

@dataclass(frozen=True)
class RequestTemplate:
    """Reusable HTTP request template."""
    template_id: str
    name: str
    description: str
    base_request: HTTPRequest
    variable_placeholders: Set[str] = field(default_factory=set)
    
    @require(lambda self: len(self.template_id) > 0)
    @require(lambda self: len(self.name) > 0)
    def __post_init__(self):
        # Extract variable placeholders from URL and body
        url_vars = set(re.findall(r'\{(\w+)\}', self.base_request.url))
        body_vars = set()
        if isinstance(self.base_request.body, str):
            body_vars = set(re.findall(r'\{(\w+)\}', self.base_request.body))
        
        object.__setattr__(self, 'variable_placeholders', url_vars | body_vars)
    
    def render(self, variables: Dict[str, str]) -> HTTPRequest:
        """Render template with provided variables."""
        missing_vars = self.variable_placeholders - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing template variables: {missing_vars}")
        
        # Render URL
        rendered_url = self.base_request.url.format(**variables)
        
        # Render body
        rendered_body = self.base_request.body
        if isinstance(rendered_body, str):
            rendered_body = rendered_body.format(**variables)
        
        return HTTPRequest(
            method=self.base_request.method,
            url=rendered_url,
            headers=self.base_request.headers,
            body=rendered_body,
            authentication=self.base_request.authentication,
            timeout=self.base_request.timeout,
            follow_redirects=self.base_request.follow_redirects,
            verify_ssl=self.base_request.verify_ssl
        )

class HTTPClient:
    """Secure HTTP client with comprehensive validation."""
    
    def __init__(self):
        self.session = None  # Will be initialized with aiohttp
        self.request_cache = {}
        self.rate_limiters = {}
    
    async def execute_request(self, request: HTTPRequest) -> HTTPResponse:
        """Execute HTTP request with security validation."""
        import aiohttp
        import asyncio
        from datetime import datetime
        
        # Prepare request
        prepared_request = request.prepare_for_execution()
        
        # Check rate limits
        await self._check_rate_limits(prepared_request.url)
        
        start_time = datetime.utcnow()
        
        try:
            timeout = aiohttp.ClientTimeout(total=prepared_request.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Prepare request parameters
                kwargs = {
                    'method': prepared_request.method.value,
                    'url': prepared_request.url,
                    'headers': prepared_request.headers.headers,
                    'ssl': prepared_request.verify_ssl,
                    'allow_redirects': prepared_request.follow_redirects
                }
                
                # Add body if present
                if prepared_request.body:
                    if isinstance(prepared_request.body, dict):
                        kwargs['json'] = prepared_request.body
                    else:
                        kwargs['data'] = prepared_request.body
                
                # Execute request
                async with session.request(**kwargs) as response:
                    body = await response.text()
                    
                    # Build response headers
                    response_headers = HTTPHeaders({k: v for k, v in response.headers.items()})
                    
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    return HTTPResponse(
                        status_code=response.status,
                        headers=response_headers,
                        body=body,
                        url=str(response.url),
                        execution_time=execution_time
                    )
                    
        except asyncio.TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return HTTPResponse(
                status_code=408,
                headers=HTTPHeaders(),
                body="",
                url=prepared_request.url,
                execution_time=execution_time,
                error="Request timeout"
            )
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return HTTPResponse(
                status_code=0,
                headers=HTTPHeaders(),
                body="",
                url=prepared_request.url,
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _check_rate_limits(self, url: str):
        """Check and enforce rate limits."""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        # Simple rate limiting - can be enhanced
        current_time = time.time()
        if domain in self.rate_limiters:
            last_request, count = self.rate_limiters[domain]
            if current_time - last_request < 1.0:  # 1 second window
                if count >= 10:  # Max 10 requests per second
                    await asyncio.sleep(1.0 - (current_time - last_request))
                    self.rate_limiters[domain] = (time.time(), 1)
                else:
                    self.rate_limiters[domain] = (last_request, count + 1)
            else:
                self.rate_limiters[domain] = (current_time, 1)
        else:
            self.rate_limiters[domain] = (current_time, 1)

class WebAutomationManager:
    """Comprehensive web automation management."""
    
    def __init__(self):
        self.http_client = HTTPClient()
        self.template_manager = RequestTemplateManager()
        self.webhook_handler = WebhookHandler()
    
    async def execute_web_request(self, request: HTTPRequest) -> Either[WebError, HTTPResponse]:
        """Execute web request with comprehensive validation."""
        try:
            # Security validation
            security_result = self._validate_request_security(request)
            if security_result.is_left():
                return security_result
            
            # Execute request
            response = await self.http_client.execute_request(request)
            
            # Validate response
            if response.error:
                return Either.left(WebError.request_failed(response.error))
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(WebError.execution_error(str(e)))
    
    def _validate_request_security(self, request: HTTPRequest) -> Either[WebError, None]:
        """Validate request for security compliance."""
        # Check URL safety
        if not request._is_valid_url(request.url):
            return Either.left(WebError.invalid_url(request.url))
        
        # Check for dangerous headers
        dangerous_headers = ['x-forwarded-for', 'x-real-ip', 'host']
        for header_name in request.headers.headers:
            if header_name.lower() in dangerous_headers:
                return Either.left(WebError.dangerous_header(header_name))
        
        # Validate body content
        if request.body and isinstance(request.body, str):
            if self._contains_malicious_content(request.body):
                return Either.left(WebError.malicious_content())
        
        return Either.right(None)
    
    def _contains_malicious_content(self, content: str) -> bool:
        """Check for malicious patterns in request content."""
        malicious_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'data:text/html',
            r'eval\s*\(',
            r'exec\s*\(',
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in malicious_patterns)
```

## 🔒 Security Implementation
```python
class WebSecurityValidator:
    """Security-first web automation validation."""
    
    @staticmethod
    def validate_url_safety(url: str) -> Either[SecurityError, None]:
        """Validate URL for security."""
        try:
            parsed = urlparse(url)
            
            # Must use HTTPS for sensitive operations
            if parsed.scheme not in ['http', 'https']:
                return Either.left(SecurityError("Only HTTP/HTTPS protocols allowed"))
            
            # Block internal networks
            if WebSecurityValidator._is_internal_network(parsed.netloc):
                return Either.left(SecurityError("Internal network access not allowed"))
            
            # Check for suspicious patterns
            if WebSecurityValidator._contains_suspicious_patterns(url):
                return Either.left(SecurityError("URL contains suspicious patterns"))
            
            return Either.right(None)
            
        except Exception:
            return Either.left(SecurityError("Invalid URL format"))
    
    @staticmethod
    def _is_internal_network(netloc: str) -> bool:
        """Check if network location is internal."""
        hostname = netloc.split(':')[0].lower()
        
        # Block localhost variants
        localhost_patterns = ['localhost', '127.', '0.0.0.0', '::1']
        if any(hostname.startswith(pattern) for pattern in localhost_patterns):
            return True
        
        # Block private networks
        import ipaddress
        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_reserved
        except ValueError:
            # Not an IP address, check domain patterns
            internal_domains = ['.local', '.internal', '.corp']
            return any(hostname.endswith(domain) for domain in internal_domains)
    
    @staticmethod
    def _contains_suspicious_patterns(url: str) -> bool:
        """Check for suspicious URL patterns."""
        suspicious_patterns = [
            r'\.\./',  # Path traversal
            r'%2e%2e%2f',  # Encoded path traversal
            r'file://',  # File protocol
            r'ftp://',  # FTP protocol
        ]
        
        url_lower = url.lower()
        return any(re.search(pattern, url_lower) for pattern in suspicious_patterns)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st
from hypothesis.strategies import composite

@composite
def valid_urls(draw):
    """Generate valid HTTP/HTTPS URLs."""
    scheme = draw(st.sampled_from(['http', 'https']))
    domain = draw(st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz'))
    tld = draw(st.sampled_from(['com', 'org', 'net', 'edu']))
    path = draw(st.text(min_size=0, max_size=50, alphabet='abcdefghijklmnopqrstuvwxyz/'))
    
    return f'{scheme}://{domain}.{tld}/{path}'

@given(valid_urls())
def test_url_validation_properties(url):
    """Property: Valid URLs should pass security validation."""
    # Filter out localhost/internal URLs for this test
    if not any(pattern in url.lower() for pattern in ['localhost', '127.', '192.168.', '10.']):
        result = WebSecurityValidator.validate_url_safety(url)
        assert result.is_right()

@given(st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=100), min_size=0, max_size=10))
def test_headers_validation_properties(headers_dict):
    """Property: Valid headers should be accepted."""
    try:
        headers = HTTPHeaders(headers_dict)
        assert len(headers.headers) == len(headers_dict)
    except ValueError:
        # Some generated headers might be invalid, which is acceptable
        pass

@given(st.text(min_size=1, max_size=1000))
def test_request_body_properties(body_content):
    """Property: Request bodies should handle various content."""
    if not any(pattern in body_content.lower() for pattern in ['<script', 'javascript:', 'eval(']):
        request = HTTPRequest(
            method=HTTPMethod.POST,
            url="https://api.example.com/test",
            body=body_content
        )
        assert request.body == body_content
```

## 🏗️ Modularity Strategy
- **web_automation_tools.py**: Main MCP tool interface (<250 lines)
- **web_requests.py**: HTTP type definitions and core logic (<350 lines)
- **http_client.py**: HTTP client implementation with security (<250 lines)
- **auth_manager.py**: Authentication handling (<200 lines)
- **webhook_handler.py**: Webhook processing (<150 lines)
- **request_templates.py**: Template system for requests (<200 lines)

## ✅ Success Criteria
- Complete HTTP request support (GET, POST, PUT, DELETE, PATCH) with security validation
- Authentication support for API keys, Bearer tokens, Basic auth, and custom headers
- Webhook handling for incoming HTTP requests and automation triggers
- Request templating system for reusable API configurations
- Comprehensive security validation prevents SSRF and injection attacks
- Rate limiting and connection pooling for efficient API usage
- Property-based tests validate all HTTP scenarios and security boundaries
- Performance: <500ms simple requests, <2s complex API calls, <100ms template rendering
- Integration with action builder for web-based automation sequences
- Documentation: Complete web automation API with security guidelines and examples
- TESTING.md shows 95%+ test coverage with all HTTP security tests passing
- Tool enables AI to interact with web services and APIs securely

## 🔄 Integration Points
- **TASK_14 (km_action_builder)**: Include HTTP actions in macro sequences
- **TASK_32 (km_email_sms_integration)**: Web-based communication via APIs
- **TASK_10 (km_macro_manager)**: Web-triggered macro execution
- **All Existing Tools**: Integrate with external APIs and web services
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- Essential for modern automation that integrates with cloud services and APIs
- Security is critical - must prevent SSRF, injection, and other web vulnerabilities
- Integration with action builder enables web-based automation sequences
- Template system enables reusable API configurations and workflows
- Webhook support enables external systems to trigger macro execution
- Success here enables AI to interact with any web service or API securely