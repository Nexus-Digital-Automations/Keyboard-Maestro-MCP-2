# TASK_34: km_remote_triggers - Remote Execution & URL Scheme Integration

**Created By**: Agent_1 (Platform Expansion) | **Priority**: MEDIUM | **Duration**: 3 hours
**Technique Focus**: Design by Contract + Type Safety + URL Security + Remote Authentication + Event-Driven Architecture
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: Foundation tasks (TASK_1-20), Web automation (TASK_33)
**Blocking**: Remote macro execution and external system integration

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/KM_MCP.md - URL scheme handling (lines 900-920)
- [ ] **Web Integration**: development/tasks/TASK_33.md - HTTP security and validation patterns
- [ ] **Trigger Architecture**: development/tasks/TASK_23.md - Advanced trigger system integration
- [ ] **Security Framework**: src/core/contracts.py - Remote access security validation
- [ ] **Testing Requirements**: tests/TESTING.md - Remote execution testing patterns

## 🎯 Problem Analysis
**Classification**: Remote Access Infrastructure Gap
**Gap Identified**: No secure remote macro execution and URL scheme integration
**Impact**: AI cannot trigger macros from external systems or provide remote automation capabilities

<thinking>
Root Cause Analysis:
1. Current platform focuses on local automation but lacks remote execution capabilities
2. No URL scheme handling for `kmtrigger://` and `keyboardmaestro://` protocols
3. Missing remote HTTP trigger support for external system integration
4. Cannot handle authentication and authorization for remote access
5. Essential for integration with external services, webhooks, and remote automation
6. Should integrate with existing trigger system and web automation capabilities
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **URL scheme types**: Define branded types for URL schemes and remote triggers
- [ ] **Security framework**: Authentication, authorization, and access control for remote triggers
- [ ] **Protocol handlers**: Support for kmtrigger:// and keyboardmaestro:// URL schemes

### Phase 2: Core Remote Execution
- [ ] **Remote trigger handling**: Process incoming remote trigger requests
- [ ] **Authentication system**: API key, token-based, and IP-based access control
- [ ] **Parameter passing**: Secure parameter validation and macro variable injection
- [ ] **Response handling**: Return execution results and status information

### Phase 3: URL Scheme Integration
- [ ] **Protocol registration**: Register URL scheme handlers with macOS
- [ ] **URL parsing**: Parse and validate URL scheme parameters
- [ ] **Deep linking**: Support for editor control and macro execution URLs
- [ ] **Security validation**: Prevent malicious URL scheme exploitation

### Phase 4: HTTP Remote Triggers
- [ ] **HTTP endpoint**: Expose secure HTTP endpoint for remote triggers
- [ ] **Webhook integration**: Process incoming webhooks from external services
- [ ] **Rate limiting**: Prevent abuse and ensure system stability
- [ ] **Logging**: Comprehensive audit logging for remote access

### Phase 5: Integration & Testing
- [ ] **TESTING.md update**: Remote trigger testing coverage and security validation
- [ ] **Security testing**: Prevent unauthorized access and validate all remote requests
- [ ] **Performance optimization**: Efficient remote request handling and response
- [ ] **Integration tests**: End-to-end remote automation workflow validation

## 🔧 Implementation Files & Specifications
```
src/server/tools/remote_trigger_tools.py          # Main remote trigger tool implementation
src/core/remote_access.py                         # Remote access type definitions
src/remote/url_scheme_handler.py                  # URL scheme processing and validation
src/remote/http_trigger_server.py                 # HTTP trigger endpoint handling
src/remote/auth_manager.py                        # Remote authentication and authorization
src/remote/trigger_processor.py                   # Remote trigger execution logic
tests/tools/test_remote_trigger_tools.py          # Unit and integration tests
tests/property_tests/test_remote_triggers.py      # Property-based remote access validation
```

### km_remote_triggers Tool Specification
```python
@mcp.tool()
async def km_remote_triggers(
    operation: str,                             # setup|execute|status|config
    trigger_type: str = "url_scheme",           # url_scheme|http_endpoint|webhook
    macro_identifier: Optional[str] = None,     # Target macro for execution
    url_scheme: Optional[str] = None,           # URL scheme for registration
    trigger_value: Optional[str] = None,        # Parameter value for macro
    auth_token: Optional[str] = None,           # Authentication token
    allowed_origins: Optional[List[str]] = None, # Allowed IP addresses/domains
    enable_logging: bool = True,                # Enable audit logging
    response_format: str = "json",              # json|xml|text response format
    timeout: int = 30,                          # Execution timeout
    rate_limit: Optional[Dict] = None,          # Rate limiting configuration
    ctx = None
) -> Dict[str, Any]:
```

### Remote Trigger Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Set
from enum import Enum
import re
import hashlib
import secrets
from urllib.parse import urlparse, parse_qs

class TriggerType(Enum):
    """Remote trigger types."""
    URL_SCHEME = "url_scheme"
    HTTP_ENDPOINT = "http_endpoint"
    WEBHOOK = "webhook"
    REMOTE_API = "remote_api"

class AuthenticationMethod(Enum):
    """Remote authentication methods."""
    NONE = "none"
    API_KEY = "api_key"
    TOKEN = "token"
    IP_WHITELIST = "ip_whitelist"
    SIGNATURE = "signature"

@dataclass(frozen=True)
class URLSchemeRequest:
    """Type-safe URL scheme request with validation."""
    scheme: str
    action: str
    parameters: Dict[str, str] = field(default_factory=dict)
    
    @require(lambda self: self.scheme in ["kmtrigger", "keyboardmaestro"])
    @require(lambda self: len(self.action) > 0)
    def __post_init__(self):
        pass
    
    @classmethod
    def from_url(cls, url: str) -> 'URLSchemeRequest':
        """Parse URL scheme request from URL string."""
        parsed = urlparse(url)
        
        if parsed.scheme not in ["kmtrigger", "keyboardmaestro"]:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        
        # Extract action from hostname/path
        action = parsed.hostname or parsed.path.lstrip('/')
        if not action:
            raise ValueError("URL scheme requires action specification")
        
        # Parse query parameters
        parameters = {}
        if parsed.query:
            query_params = parse_qs(parsed.query)
            # Flatten single-value parameters
            parameters = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
        
        return cls(
            scheme=parsed.scheme,
            action=action,
            parameters=parameters
        )
    
    def get_macro_identifier(self) -> Optional[str]:
        """Extract macro identifier from request."""
        return self.parameters.get('macro') or self.parameters.get('name')
    
    def get_trigger_value(self) -> Optional[str]:
        """Extract trigger value from request."""
        return self.parameters.get('value') or self.parameters.get('parameter')

@dataclass(frozen=True)
class RemoteAuthentication:
    """Remote access authentication configuration."""
    method: AuthenticationMethod
    credentials: Dict[str, str] = field(default_factory=dict)
    allowed_origins: Set[str] = field(default_factory=set)
    
    @require(lambda self: self._validate_credentials())
    def __post_init__(self):
        pass
    
    def _validate_credentials(self) -> bool:
        """Validate authentication credentials."""
        if self.method == AuthenticationMethod.API_KEY:
            return 'api_key' in self.credentials and len(self.credentials['api_key']) >= 32
        elif self.method == AuthenticationMethod.TOKEN:
            return 'token' in self.credentials and len(self.credentials['token']) >= 32
        elif self.method == AuthenticationMethod.SIGNATURE:
            return 'secret' in self.credentials and len(self.credentials['secret']) >= 16
        return True
    
    def authenticate_request(self, headers: Dict[str, str], body: str = "", origin: str = "") -> bool:
        """Authenticate incoming request."""
        if self.method == AuthenticationMethod.NONE:
            return True
        
        # Check origin whitelist
        if self.allowed_origins and origin:
            if not any(self._matches_origin(origin, allowed) for allowed in self.allowed_origins):
                return False
        
        if self.method == AuthenticationMethod.API_KEY:
            provided_key = headers.get('X-API-Key') or headers.get('Authorization', '').replace('Bearer ', '')
            return provided_key == self.credentials['api_key']
        
        elif self.method == AuthenticationMethod.TOKEN:
            auth_header = headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                provided_token = auth_header[7:]
                return provided_token == self.credentials['token']
            return False
        
        elif self.method == AuthenticationMethod.SIGNATURE:
            signature = headers.get('X-Signature', '')
            expected_signature = self._calculate_signature(body)
            return signature == expected_signature
        
        elif self.method == AuthenticationMethod.IP_WHITELIST:
            client_ip = headers.get('X-Forwarded-For', '').split(',')[0].strip()
            if not client_ip:
                client_ip = origin
            return any(self._matches_ip(client_ip, allowed) for allowed in self.allowed_origins)
        
        return False
    
    def _matches_origin(self, origin: str, allowed: str) -> bool:
        """Check if origin matches allowed pattern."""
        # Support wildcards
        if allowed == "*":
            return True
        if allowed.startswith("*."):
            domain = allowed[2:]
            return origin.endswith(domain)
        return origin == allowed
    
    def _matches_ip(self, ip: str, allowed: str) -> bool:
        """Check if IP matches allowed pattern."""
        import ipaddress
        try:
            if '/' in allowed:  # CIDR notation
                network = ipaddress.ip_network(allowed)
                return ipaddress.ip_address(ip) in network
            else:
                return ip == allowed
        except ValueError:
            return False
    
    def _calculate_signature(self, body: str) -> str:
        """Calculate HMAC signature for request body."""
        import hmac
        secret = self.credentials['secret'].encode()
        return hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()

@dataclass(frozen=True)
class RemoteTriggerRequest:
    """Complete remote trigger request specification."""
    trigger_type: TriggerType
    macro_identifier: str
    trigger_value: Optional[str] = None
    parameters: Dict[str, str] = field(default_factory=dict)
    authentication: Optional[RemoteAuthentication] = None
    origin: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    
    @require(lambda self: len(self.macro_identifier) > 0)
    def __post_init__(self):
        pass
    
    def is_authenticated(self) -> bool:
        """Check if request is properly authenticated."""
        if not self.authentication:
            return True  # No authentication required
        
        return self.authentication.authenticate_request(
            self.headers,
            body=self.trigger_value or "",
            origin=self.origin
        )

@dataclass(frozen=True)
class RemoteTriggerResponse:
    """Remote trigger execution response."""
    success: bool
    execution_id: str
    macro_id: str
    execution_time: float
    output: Optional[str] = None
    error_message: Optional[str] = None
    status_code: int = 200
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        response = {
            "success": self.success,
            "execution_id": self.execution_id,
            "macro_id": self.macro_id,
            "execution_time": self.execution_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.output:
            response["output"] = self.output
        
        if self.error_message:
            response["error"] = self.error_message
        
        return response
    
    def to_json(self) -> str:
        """Convert response to JSON string."""
        import json
        return json.dumps(self.to_dict())

class URLSchemeHandler:
    """Handle URL scheme requests for Keyboard Maestro."""
    
    def __init__(self):
        self.registered_schemes = set()
        self.auth_manager = RemoteAuthManager()
    
    async def process_url_scheme(self, url: str) -> Either[RemoteError, RemoteTriggerResponse]:
        """Process URL scheme request."""
        try:
            # Parse URL scheme request
            scheme_request = URLSchemeRequest.from_url(url)
            
            # Convert to remote trigger request
            trigger_request = RemoteTriggerRequest(
                trigger_type=TriggerType.URL_SCHEME,
                macro_identifier=scheme_request.get_macro_identifier() or scheme_request.action,
                trigger_value=scheme_request.get_trigger_value(),
                parameters=scheme_request.parameters
            )
            
            # Execute trigger
            return await self._execute_remote_trigger(trigger_request)
            
        except Exception as e:
            return Either.left(RemoteError.url_scheme_error(str(e)))
    
    def register_url_scheme(self, scheme: str) -> Either[RemoteError, str]:
        """Register URL scheme with macOS."""
        if scheme not in ["kmtrigger", "keyboardmaestro"]:
            return Either.left(RemoteError.invalid_scheme(scheme))
        
        try:
            # Register with macOS Launch Services
            # This would typically involve modifying Info.plist and calling LSSetDefaultHandlerForURLScheme
            self.registered_schemes.add(scheme)
            return Either.right(f"URL scheme {scheme} registered successfully")
        except Exception as e:
            return Either.left(RemoteError.registration_failed(str(e)))

class HTTPTriggerServer:
    """HTTP endpoint server for remote triggers."""
    
    def __init__(self):
        self.auth_manager = RemoteAuthManager()
        self.rate_limiter = RateLimiter()
        self.server = None
    
    async def start_server(self, port: int = 4490, host: str = "localhost") -> Either[RemoteError, str]:
        """Start HTTP trigger server."""
        from aiohttp import web, ClientSession
        
        try:
            app = web.Application()
            app.router.add_post('/trigger', self.handle_trigger_request)
            app.router.add_get('/status', self.handle_status_request)
            app.router.add_get('/health', self.handle_health_request)
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, host, port)
            await site.start()
            
            self.server = runner
            return Either.right(f"HTTP trigger server started on {host}:{port}")
            
        except Exception as e:
            return Either.left(RemoteError.server_start_failed(str(e)))
    
    async def handle_trigger_request(self, request) -> web.Response:
        """Handle incoming trigger request."""
        try:
            # Rate limiting check
            client_ip = request.remote
            if not await self.rate_limiter.check_rate_limit(client_ip):
                return web.json_response(
                    {"error": "Rate limit exceeded"}, 
                    status=429
                )
            
            # Parse request
            headers = dict(request.headers)
            body = await request.text()
            
            # Extract macro identifier and parameters
            if request.content_type == 'application/json':
                import json
                data = json.loads(body)
                macro_identifier = data.get('macro', '')
                trigger_value = data.get('value', '')
                parameters = data.get('parameters', {})
            else:
                # Form data or query parameters
                data = await request.post()
                macro_identifier = data.get('macro', '')
                trigger_value = data.get('value', '')
                parameters = dict(data)
            
            # Create trigger request
            trigger_request = RemoteTriggerRequest(
                trigger_type=TriggerType.HTTP_ENDPOINT,
                macro_identifier=macro_identifier,
                trigger_value=trigger_value,
                parameters=parameters,
                origin=client_ip,
                headers=headers
            )
            
            # Execute trigger
            result = await self._execute_remote_trigger(trigger_request)
            
            if result.is_left():
                error = result.get_left()
                return web.json_response(
                    {"error": error.message}, 
                    status=error.status_code or 400
                )
            
            response = result.get_right()
            return web.json_response(
                response.to_dict(),
                status=response.status_code
            )
            
        except Exception as e:
            return web.json_response(
                {"error": f"Internal server error: {str(e)}"}, 
                status=500
            )

class RemoteTriggerManager:
    """Comprehensive remote trigger management."""
    
    def __init__(self):
        self.url_scheme_handler = URLSchemeHandler()
        self.http_server = HTTPTriggerServer()
        self.macro_executor = MacroExecutor()
        self.audit_logger = AuditLogger()
    
    async def execute_remote_trigger(self, request: RemoteTriggerRequest) -> Either[RemoteError, RemoteTriggerResponse]:
        """Execute remote trigger with comprehensive validation."""
        try:
            # Authentication check
            if not request.is_authenticated():
                await self.audit_logger.log_unauthorized_access(request)
                return Either.left(RemoteError.authentication_failed())
            
            # Validate macro identifier
            if not await self._validate_macro_exists(request.macro_identifier):
                return Either.left(RemoteError.macro_not_found(request.macro_identifier))
            
            # Log access
            await self.audit_logger.log_remote_access(request)
            
            # Execute macro
            execution_result = await self.macro_executor.execute_macro(
                request.macro_identifier,
                trigger_value=request.trigger_value,
                timeout=30
            )
            
            if execution_result.is_left():
                error = execution_result.get_left()
                return Either.left(RemoteError.execution_failed(error.message))
            
            result = execution_result.get_right()
            
            # Create response
            response = RemoteTriggerResponse(
                success=True,
                execution_id=result.execution_id,
                macro_id=result.macro_id,
                execution_time=result.execution_time,
                output=result.output
            )
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(RemoteError.execution_error(str(e)))
    
    async def _validate_macro_exists(self, identifier: str) -> bool:
        """Validate that macro exists and is accessible."""
        # This would integrate with the existing macro discovery system
        return True  # Placeholder implementation
```

## 🔒 Security Implementation
```python
class RemoteSecurityValidator:
    """Security-first remote access validation."""
    
    @staticmethod
    def validate_url_scheme_safety(url: str) -> Either[SecurityError, None]:
        """Validate URL scheme for security."""
        try:
            parsed = urlparse(url)
            
            # Only allow supported schemes
            if parsed.scheme not in ['kmtrigger', 'keyboardmaestro']:
                return Either.left(SecurityError("Unsupported URL scheme"))
            
            # Check for suspicious parameters
            if parsed.query:
                query_params = parse_qs(parsed.query)
                for key, values in query_params.items():
                    for value in values:
                        if RemoteSecurityValidator._contains_malicious_content(value):
                            return Either.left(SecurityError(f"Malicious content in parameter: {key}"))
            
            return Either.right(None)
            
        except Exception:
            return Either.left(SecurityError("Invalid URL format"))
    
    @staticmethod
    def _contains_malicious_content(content: str) -> bool:
        """Check for malicious patterns in content."""
        malicious_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'data:text/html',
            r'\.\./.*',  # Path traversal
            r'file://',
            r'eval\s*\(',
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in malicious_patterns)

class RateLimiter:
    """Rate limiting for remote requests."""
    
    def __init__(self):
        self.request_counts = {}
        self.blocked_ips = set()
    
    async def check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limits."""
        import time
        
        current_time = time.time()
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return False
        
        # Get current request count
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        # Clean old requests (older than 1 minute)
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check rate limit (max 60 requests per minute)
        if len(self.request_counts[client_ip]) >= 60:
            # Block IP temporarily
            self.blocked_ips.add(client_ip)
            return False
        
        # Add current request
        self.request_counts[client_ip].append(current_time)
        return True
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=50))
def test_url_scheme_parsing_properties(action):
    """Property: Valid URL schemes should parse correctly."""
    if action.replace('_', '').replace('-', '').isalnum():
        url = f"kmtrigger://{action}?macro=TestMacro&value=TestValue"
        try:
            request = URLSchemeRequest.from_url(url)
            assert request.scheme == "kmtrigger"
            assert request.action == action
        except ValueError:
            # Some actions might be invalid, which is acceptable
            pass

@given(st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=100), min_size=0, max_size=10))
def test_authentication_properties(credentials):
    """Property: Authentication should handle various credential sets."""
    if 'api_key' in credentials and len(credentials['api_key']) >= 32:
        auth = RemoteAuthentication(
            method=AuthenticationMethod.API_KEY,
            credentials=credentials
        )
        
        # Test authentication with correct key
        headers = {'X-API-Key': credentials['api_key']}
        assert auth.authenticate_request(headers)
        
        # Test authentication with incorrect key
        headers = {'X-API-Key': 'wrong_key'}
        assert not auth.authenticate_request(headers)

@given(st.text(min_size=1, max_size=100))
def test_macro_identifier_properties(identifier):
    """Property: Macro identifiers should be handled safely."""
    if not any(char in identifier for char in ['<', '>', '&', '"', "'"]):
        request = RemoteTriggerRequest(
            trigger_type=TriggerType.URL_SCHEME,
            macro_identifier=identifier
        )
        assert request.macro_identifier == identifier
```

## 🏗️ Modularity Strategy
- **remote_trigger_tools.py**: Main MCP tool interface (<250 lines)
- **remote_access.py**: Type definitions and core logic (<350 lines)
- **url_scheme_handler.py**: URL scheme processing (<200 lines)
- **http_trigger_server.py**: HTTP endpoint handling (<300 lines)
- **auth_manager.py**: Authentication and authorization (<200 lines)
- **trigger_processor.py**: Remote trigger execution (<150 lines)

## ✅ Success Criteria
- Complete URL scheme support for kmtrigger:// and keyboardmaestro:// protocols
- Secure HTTP endpoint for remote macro execution with authentication
- Webhook integration for external system triggers
- Comprehensive authentication (API keys, tokens, IP whitelisting, signatures)
- Rate limiting and abuse prevention for remote access
- Audit logging for all remote access attempts and executions
- Property-based tests validate all remote access scenarios and security
- Performance: <200ms URL scheme processing, <500ms HTTP trigger response
- Integration with existing macro system for seamless remote execution
- Documentation: Complete remote trigger API with security guidelines
- TESTING.md shows 95%+ test coverage with all remote security tests passing
- Tool enables secure remote macro execution from external systems

## 🔄 Integration Points
- **TASK_33 (km_web_automation)**: HTTP trigger endpoints and webhook processing
- **TASK_23 (km_create_trigger_advanced)**: Remote trigger type integration
- **TASK_10 (km_macro_manager)**: Remote macro execution capabilities
- **All Existing Tools**: Enable remote triggering of any macro or automation
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- Essential for integration with external services and remote automation
- Security is critical - must prevent unauthorized access and abuse
- URL scheme registration enables deep linking and system integration
- HTTP endpoints enable webhook and API-based automation triggers
- Audit logging ensures traceability of all remote access
- Success here enables external systems to securely trigger macro execution