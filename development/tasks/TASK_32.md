# TASK_32: km_email_sms_integration - Communication Automation Hub

**Created By**: Agent_1 (Platform Expansion) | **Priority**: HIGH | **Duration**: 5 hours
**Technique Focus**: Design by Contract + Type Safety + Communication Protocols + Security Validation + Async Operations
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: Foundation tasks (TASK_1-20), Intelligent Automation (TASK_21-23)
**Blocking**: Advanced communication workflows requiring email/SMS automation

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/KM_MCP.md - Email and SMS integration (lines 1032-1074)
- [ ] **Foundation Architecture**: src/server/tools/ - Existing tool patterns and security validation
- [ ] **Communication Patterns**: src/core/types.py - Message types and validation
- [ ] **Security Framework**: src/core/contracts.py - Input validation and sanitization
- [ ] **Testing Requirements**: tests/TESTING.md - Communication testing patterns

## 🎯 Problem Analysis
**Classification**: Communication Infrastructure Gap
**Gap Identified**: No comprehensive email/SMS automation capabilities for AI-driven communication workflows
**Impact**: AI cannot send notifications, alerts, or communicate with users through standard channels

<thinking>
Root Cause Analysis:
1. Current platform focuses on macro execution but lacks communication capabilities
2. No systematic email/SMS automation for user notifications and alerts
3. Missing integration with macOS Mail app and Messages for automated communication
4. Cannot handle contact management and message templating
5. Essential for complete automation platform that can communicate results and status
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **Communication types**: Define branded types for email, SMS, and contact management
- [ ] **Message templates**: Template system for reusable communication patterns
- [ ] **Validation framework**: Comprehensive input validation and security boundaries

### Phase 2: Email Integration
- [ ] **Mail app integration**: AppleScript integration with macOS Mail application
- [ ] **Email composition**: HTML and plain text email with attachment support
- [ ] **Account management**: Multiple email account support and selection
- [ ] **Recipient management**: Contact validation and group messaging

### Phase 3: SMS/iMessage Integration
- [ ] **Messages app integration**: AppleScript integration with macOS Messages
- [ ] **SMS sending**: Phone number validation and message composition
- [ ] **iMessage support**: Rich message formatting and delivery confirmation
- [ ] **Contact integration**: Address book integration and contact lookup

### Phase 4: Advanced Features
- [ ] **Message templating**: Dynamic template system with variable substitution
- [ ] **Delivery tracking**: Status monitoring and delivery confirmation
- [ ] **Rate limiting**: Anti-spam protection and throttling
- [ ] **Error handling**: Comprehensive error recovery and retry mechanisms

### Phase 5: Integration & Testing
- [ ] **TESTING.md update**: Communication testing coverage and validation
- [ ] **Security validation**: Prevent spam and validate all communication content
- [ ] **Performance optimization**: Efficient message queuing and delivery
- [ ] **Integration tests**: End-to-end communication workflow validation

## 🔧 Implementation Files & Specifications
```
src/server/tools/communication_tools.py          # Main communication tool implementation
src/core/communication.py                        # Communication type definitions
src/communication/email_manager.py               # Email composition and sending
src/communication/sms_manager.py                 # SMS/iMessage management
src/communication/contact_manager.py             # Contact validation and lookup
src/communication/message_templates.py           # Template system for messages
tests/tools/test_communication_tools.py          # Unit and integration tests
tests/property_tests/test_communication.py       # Property-based communication validation
```

### km_email_sms_integration Tool Specification
```python
@mcp.tool()
async def km_email_sms_integration(
    operation: str,                             # send_email|send_sms|manage_contacts|template
    communication_type: str,                   # email|sms|imessage
    recipient: str,                             # Email address or phone number
    subject: Optional[str] = None,              # Email subject (required for email)
    message: str,                               # Message content
    template_name: Optional[str] = None,        # Use predefined template
    template_variables: Optional[Dict] = None,  # Variables for template substitution
    attachments: Optional[List[str]] = None,    # File paths for email attachments
    from_account: Optional[str] = None,         # Specific account to send from
    delivery_receipt: bool = False,             # Request delivery confirmation
    priority: str = "normal",                   # normal|high|low message priority
    ctx = None
) -> Dict[str, Any]:
```

### Communication Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Set
from enum import Enum
import re

class CommunicationType(Enum):
    """Communication channel types."""
    EMAIL = "email"
    SMS = "sms"
    IMESSAGE = "imessage"
    NOTIFICATION = "notification"

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass(frozen=True)
class EmailAddress:
    """Type-safe email address with validation."""
    address: str
    name: Optional[str] = None
    
    @require(lambda self: self._is_valid_email(self.address))
    def __post_init__(self):
        pass
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email address format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def format_recipient(self) -> str:
        """Format for use in email clients."""
        if self.name:
            return f'"{self.name}" <{self.address}>'
        return self.address

@dataclass(frozen=True)
class PhoneNumber:
    """Type-safe phone number with validation."""
    number: str
    country_code: Optional[str] = None
    
    @require(lambda self: self._is_valid_phone(self.number))
    def __post_init__(self):
        pass
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Validate phone number format."""
        # Remove common formatting characters
        cleaned = re.sub(r'[^\d+]', '', phone)
        # Basic validation for length and format
        return len(cleaned) >= 10 and len(cleaned) <= 15
    
    def format_for_sms(self) -> str:
        """Format for SMS sending."""
        if self.country_code and not self.number.startswith('+'):
            return f'+{self.country_code}{self.number}'
        return self.number

@dataclass(frozen=True)
class MessageTemplate:
    """Reusable message template with variable substitution."""
    template_id: str
    name: str
    subject_template: Optional[str] = None
    body_template: str = ""
    variables: Set[str] = field(default_factory=set)
    communication_type: CommunicationType = CommunicationType.EMAIL
    
    @require(lambda self: len(self.template_id) > 0)
    @require(lambda self: len(self.body_template) > 0)
    def __post_init__(self):
        # Extract variables from templates
        subject_vars = set(re.findall(r'\{(\w+)\}', self.subject_template or ''))
        body_vars = set(re.findall(r'\{(\w+)\}', self.body_template))
        object.__setattr__(self, 'variables', subject_vars | body_vars)
    
    def render(self, variables: Dict[str, str]) -> Dict[str, str]:
        """Render template with provided variables."""
        missing_vars = self.variables - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing template variables: {missing_vars}")
        
        rendered = {
            "body": self.body_template.format(**variables)
        }
        
        if self.subject_template:
            rendered["subject"] = self.subject_template.format(**variables)
        
        return rendered

@dataclass(frozen=True)
class CommunicationRequest:
    """Complete communication request specification."""
    communication_type: CommunicationType
    recipients: List[Union[EmailAddress, PhoneNumber]]
    message_content: str
    subject: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    attachments: List[str] = field(default_factory=list)
    delivery_receipt: bool = False
    template_id: Optional[str] = None
    
    @require(lambda self: len(self.recipients) > 0)
    @require(lambda self: len(self.message_content) > 0)
    @require(lambda self: len(self.message_content) <= 10000)  # Reasonable message limit
    def __post_init__(self):
        # Validate email subjects are provided
        if self.communication_type == CommunicationType.EMAIL and not self.subject:
            if not self.template_id:  # Allow templates to provide subjects
                raise ValueError("Email communication requires subject")

class CommunicationManager:
    """Comprehensive communication management system."""
    
    def __init__(self):
        self.email_manager = EmailManager()
        self.sms_manager = SMSManager()
        self.template_manager = MessageTemplateManager()
        self.contact_manager = ContactManager()
    
    async def send_communication(self, request: CommunicationRequest) -> Either[CommunicationError, CommunicationResult]:
        """Send communication through appropriate channel."""
        try:
            # Validate and process request
            validated_request = await self._validate_request(request)
            
            # Route to appropriate handler
            if request.communication_type == CommunicationType.EMAIL:
                return await self.email_manager.send_email(validated_request)
            elif request.communication_type in [CommunicationType.SMS, CommunicationType.IMESSAGE]:
                return await self.sms_manager.send_message(validated_request)
            else:
                return Either.left(CommunicationError.unsupported_type(request.communication_type))
                
        except Exception as e:
            return Either.left(CommunicationError.execution_error(str(e)))
    
    async def _validate_request(self, request: CommunicationRequest) -> CommunicationRequest:
        """Validate communication request for security and compliance."""
        # Check for spam patterns
        if self._contains_spam_patterns(request.message_content):
            raise SecurityError("Message content appears to be spam")
        
        # Validate attachment paths
        for attachment in request.attachments:
            if not self._is_safe_attachment_path(attachment):
                raise SecurityError(f"Unsafe attachment path: {attachment}")
        
        # Rate limiting check
        if not await self._check_rate_limits(request):
            raise RateLimitError("Communication rate limit exceeded")
        
        return request
    
    def _contains_spam_patterns(self, content: str) -> bool:
        """Check for common spam patterns."""
        spam_patterns = [
            r'click here now',
            r'urgent.{0,20}action.{0,20}required',
            r'congratulations.{0,20}winner',
            r'free.{0,20}money',
            r'viagra|cialis',
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in spam_patterns)
    
    def _is_safe_attachment_path(self, path: str) -> bool:
        """Validate attachment file path for security."""
        # Only allow files in safe directories
        safe_prefixes = [
            '/Users/',
            '~/Documents/',
            '~/Downloads/',
            './attachments/',
        ]
        
        expanded_path = os.path.expanduser(path)
        return any(expanded_path.startswith(prefix) for prefix in safe_prefixes)

class EmailManager:
    """Email sending and management."""
    
    async def send_email(self, request: CommunicationRequest) -> Either[CommunicationError, CommunicationResult]:
        """Send email using macOS Mail application."""
        try:
            # Build AppleScript for email sending
            applescript = self._build_email_applescript(request)
            
            # Execute AppleScript
            result = await self._execute_applescript(applescript)
            
            if result.is_left():
                return Either.left(CommunicationError.email_send_failed(result.get_left().message))
            
            return Either.right(CommunicationResult(
                communication_type=CommunicationType.EMAIL,
                status="sent",
                message_id=str(uuid.uuid4()),
                recipients=[r.address if isinstance(r, EmailAddress) else str(r) for r in request.recipients],
                timestamp=datetime.utcnow().isoformat()
            ))
            
        except Exception as e:
            return Either.left(CommunicationError.execution_error(str(e)))
    
    def _build_email_applescript(self, request: CommunicationRequest) -> str:
        """Build AppleScript for email sending."""
        recipients = ', '.join([
            f'"{r.format_recipient()}"' if isinstance(r, EmailAddress) else f'"{r}"'
            for r in request.recipients
        ])
        
        script = f'''
        tell application "Mail"
            set newMessage to make new outgoing message with properties {{
                subject: "{request.subject}",
                content: "{request.message_content}",
                visible: false
            }}
            
            tell newMessage
                make new to recipient at end of to recipients with properties {{address: {recipients}}}
        '''
        
        # Add attachments if present
        for attachment in request.attachments:
            script += f'''
                make new attachment with properties {{file name: POSIX file "{attachment}"}}
            '''
        
        script += '''
                send
            end tell
        end tell
        '''
        
        return script

class SMSManager:
    """SMS and iMessage management."""
    
    async def send_message(self, request: CommunicationRequest) -> Either[CommunicationError, CommunicationResult]:
        """Send SMS/iMessage using macOS Messages application."""
        try:
            # Build AppleScript for message sending
            applescript = self._build_message_applescript(request)
            
            # Execute AppleScript
            result = await self._execute_applescript(applescript)
            
            if result.is_left():
                return Either.left(CommunicationError.sms_send_failed(result.get_left().message))
            
            return Either.right(CommunicationResult(
                communication_type=request.communication_type,
                status="sent",
                message_id=str(uuid.uuid4()),
                recipients=[r.format_for_sms() if isinstance(r, PhoneNumber) else str(r) for r in request.recipients],
                timestamp=datetime.utcnow().isoformat()
            ))
            
        except Exception as e:
            return Either.left(CommunicationError.execution_error(str(e)))
    
    def _build_message_applescript(self, request: CommunicationRequest) -> str:
        """Build AppleScript for SMS/iMessage sending."""
        recipient = request.recipients[0]  # SMS typically single recipient
        if isinstance(recipient, PhoneNumber):
            recipient_str = recipient.format_for_sms()
        else:
            recipient_str = str(recipient)
        
        service = "iMessage" if request.communication_type == CommunicationType.IMESSAGE else "SMS"
        
        script = f'''
        tell application "Messages"
            set targetService to 1st account whose service type = {service}
            set targetBuddy to participant "{recipient_str}" of targetService
            send "{request.message_content}" to targetBuddy
        end tell
        '''
        
        return script
```

## 🔒 Security Implementation
```python
class CommunicationSecurityValidator:
    """Security-first communication validation."""
    
    @staticmethod
    def validate_recipients(recipients: List[Union[EmailAddress, PhoneNumber]]) -> Either[SecurityError, None]:
        """Validate recipient list for security."""
        if len(recipients) > 100:  # Reasonable bulk limit
            return Either.left(SecurityError("Too many recipients"))
        
        # Check for suspicious patterns
        for recipient in recipients:
            if isinstance(recipient, EmailAddress):
                if not CommunicationSecurityValidator._is_safe_email_domain(recipient.address):
                    return Either.left(SecurityError(f"Suspicious email domain: {recipient.address}"))
            elif isinstance(recipient, PhoneNumber):
                if not CommunicationSecurityValidator._is_safe_phone_number(recipient.number):
                    return Either.left(SecurityError(f"Invalid phone number: {recipient.number}"))
        
        return Either.right(None)
    
    @staticmethod
    def _is_safe_email_domain(email: str) -> bool:
        """Check if email domain is safe."""
        domain = email.split('@')[1].lower()
        
        # Block known spam domains
        blocked_domains = ['tempmail.org', '10minutemail.com', 'guerrillamail.com']
        return domain not in blocked_domains
    
    @staticmethod
    def _is_safe_phone_number(phone: str) -> bool:
        """Check if phone number is valid format."""
        # Remove formatting
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Basic validation - must be reasonable length
        return 10 <= len(cleaned) <= 15
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.emails())
def test_email_validation_properties(email):
    """Property: Valid email addresses should be accepted."""
    try:
        email_addr = EmailAddress(email)
        assert email_addr.address == email
        assert email_addr._is_valid_email(email)
    except ValueError:
        # Some generated emails might be invalid, which is acceptable
        pass

@given(st.text(min_size=1, max_size=1000))
def test_message_content_properties(message_content):
    """Property: Message content should handle various text inputs."""
    if not CommunicationSecurityValidator()._contains_spam_patterns(message_content):
        request = CommunicationRequest(
            communication_type=CommunicationType.EMAIL,
            recipients=[EmailAddress("test@example.com")],
            message_content=message_content,
            subject="Test Subject"
        )
        assert request.message_content == message_content

@given(st.lists(st.text(min_size=1), min_size=1, max_size=5))
def test_template_variable_properties(variable_names):
    """Property: Templates should handle various variable sets."""
    template_body = "Hello " + " and ".join([f"{{{var}}}" for var in variable_names])
    template = MessageTemplate(
        template_id="test_template",
        name="Test Template",
        body_template=template_body
    )
    
    assert len(template.variables) == len(set(variable_names))
    
    # Test rendering with all variables
    variables = {var: f"value_{var}" for var in variable_names}
    rendered = template.render(variables)
    assert "Hello" in rendered["body"]
```

## 🏗️ Modularity Strategy
- **communication_tools.py**: Main MCP tool interface (<250 lines)
- **communication.py**: Type definitions and core logic (<300 lines)
- **email_manager.py**: Email handling and AppleScript integration (<250 lines)
- **sms_manager.py**: SMS/iMessage management (<200 lines)
- **contact_manager.py**: Contact validation and lookup (<150 lines)
- **message_templates.py**: Template system implementation (<200 lines)

## ✅ Success Criteria
- Complete email automation with HTML/plain text and attachment support
- SMS and iMessage integration with delivery confirmation
- Contact management and validation for all communication types
- Message templating system with variable substitution
- Comprehensive security validation prevents spam and malicious content
- Rate limiting and anti-abuse protection
- Property-based tests validate all communication scenarios
- Performance: <2s email composition, <1s SMS sending, <500ms template rendering
- Integration with existing macro system for automated notifications
- Documentation: Complete communication API with examples and security guidelines
- TESTING.md shows 95%+ test coverage with all communication tests passing
- Tool enables AI to communicate with users through standard channels

## 🔄 Integration Points
- **TASK_10 (km_macro_manager)**: Send notifications about macro execution status
- **TASK_14 (km_action_builder)**: Include communication actions in sequences
- **TASK_4 (comprehensive testing)**: Communication testing and validation
- **All Existing Tools**: Notify users of automation results and status
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- Essential for complete AI automation platform that can communicate results
- Security is critical - must prevent spam and validate all communications
- Must respect user privacy and communication preferences
- Integration with macOS native apps ensures reliable delivery
- Template system enables consistent, professional communication
- Success here enables AI to provide feedback and status updates to users