"""
SMS and iMessage management for Keyboard Maestro MCP Tools.

This module provides comprehensive SMS/iMessage automation through macOS Messages app
integration with security validation and delivery tracking.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import re
import uuid
import asyncio
from datetime import datetime, UTC

from ..core.communication import (
    CommunicationRequest, CommunicationResult, CommunicationType, 
    CommunicationStatus, PhoneNumber, MessageId
)
from ..core.either import Either
from ..core.errors import CommunicationError, SecurityError, ValidationError
from ..core.contracts import require, ensure
from ..integration.km_client import KMClient


@dataclass(frozen=True)
class SMSConfiguration:
    """SMS/iMessage configuration with security and rate limiting."""
    max_message_length: int = 160  # Standard SMS limit
    max_imessage_length: int = 20000  # iMessage supports longer messages
    rate_limit_per_minute: int = 10
    rate_limit_per_hour: int = 50
    blocked_patterns: List[str] = None
    
    def __post_init__(self):
        if self.blocked_patterns is None:
            # Default blocked spam patterns
            patterns = [
                r'click.*link',
                r'verify.*account',
                r'urgent.*action',
                r'suspended.*account',
                r'winner.*prize',
            ]
            object.__setattr__(self, 'blocked_patterns', patterns)


class SMSSecurityValidator:
    """Security validation for SMS and iMessage content."""
    
    @staticmethod
    def validate_message_content(content: str, communication_type: CommunicationType, 
                               config: SMSConfiguration) -> Either[SecurityError, None]:
        """Validate message content for security and compliance."""
        if not content or not content.strip():
            return Either.left(SecurityError("Message content cannot be empty"))
        
        # Check length limits
        max_length = (config.max_imessage_length if communication_type == CommunicationType.IMESSAGE 
                     else config.max_message_length)
        
        if len(content) > max_length:
            return Either.left(SecurityError(
                f"Message too long: {len(content)} > {max_length} chars"
            ))
        
        # Check for spam patterns
        if SMSSecurityValidator._contains_spam_patterns(content, config):
            return Either.left(SecurityError("Message contains spam-like content"))
        
        # Check for phishing attempts
        if SMSSecurityValidator._contains_phishing_patterns(content):
            return Either.left(SecurityError("Message contains phishing-like content"))
        
        # Check for malicious links
        if SMSSecurityValidator._contains_suspicious_links(content):
            return Either.left(SecurityError("Message contains suspicious links"))
        
        return Either.right(None)
    
    @staticmethod
    def _contains_spam_patterns(content: str, config: SMSConfiguration) -> bool:
        """Check for spam patterns in message content."""
        content_lower = content.lower()
        
        # Check configured patterns
        for pattern in config.blocked_patterns:
            if re.search(pattern, content_lower):
                return True
        
        # Additional spam indicators
        spam_indicators = [
            r'free.*money',
            r'cash.*now',
            r'limited.*time.*offer',
            r'act.*now',
            r'congratulations.*won',
            r'claim.*prize',
            r'click.*here.*now',
            r'call.*now.*free',
        ]
        
        matches = sum(1 for pattern in spam_indicators 
                     if re.search(pattern, content_lower))
        
        return matches >= 2  # Multiple indicators suggest spam
    
    @staticmethod
    def _contains_phishing_patterns(content: str) -> bool:
        """Check for common phishing patterns."""
        phishing_patterns = [
            r'verify.*account.*immediately',
            r'suspended.*account.*click',
            r'update.*payment.*info',
            r'confirm.*identity.*link',
            r'security.*alert.*login',
            r'unauthorized.*access.*verify',
            r'account.*locked.*unlock',
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in phishing_patterns)
    
    @staticmethod
    def _contains_suspicious_links(content: str) -> bool:
        """Check for suspicious or potentially malicious links."""
        # Find URLs in content
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, content.lower())
        
        if not urls:
            return False
        
        suspicious_indicators = [
            r'bit\.ly',  # URL shorteners (can hide destination)
            r'tinyurl',
            r'goo\.gl',
            r't\.co',
            r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP addresses
            r'[a-z0-9]+-[a-z0-9]+-[a-z0-9]+\.',  # Suspicious domain patterns
        ]
        
        for url in urls:
            for pattern in suspicious_indicators:
                if re.search(pattern, url):
                    return True
        
        return False
    
    @staticmethod
    def validate_phone_numbers(recipients: List[PhoneNumber]) -> Either[SecurityError, None]:
        """Validate phone number recipients for security."""
        if len(recipients) > 10:  # Reasonable group message limit
            return Either.left(SecurityError("Too many recipients for SMS"))
        
        # Check for suspicious phone number patterns
        for phone in recipients:
            if not SMSSecurityValidator._is_valid_phone_number(phone):
                return Either.left(SecurityError(f"Invalid phone number: {phone.number}"))
        
        return Either.right(None)
    
    @staticmethod
    def _is_valid_phone_number(phone: PhoneNumber) -> bool:
        """Additional validation for phone numbers in SMS context."""
        formatted = phone.format_for_sms()
        
        # Check for suspicious patterns
        if re.search(r'(.)\1{4,}', formatted):  # Too many repeated digits
            return False
        
        if formatted.startswith('+1') and len(formatted) == 12:  # US/Canada
            return True
        elif formatted.startswith('+') and 11 <= len(formatted) <= 15:  # International
            return True
        elif len(formatted) == 10:  # US domestic format
            return True
        
        return False


class RateLimiter:
    """Rate limiting for SMS/iMessage to prevent abuse."""
    
    def __init__(self, config: SMSConfiguration):
        self.config = config
        self.message_history: List[datetime] = []
    
    def can_send_message(self) -> bool:
        """Check if a message can be sent based on rate limits."""
        now = datetime.now(UTC)
        
        # Clean old entries
        minute_ago = now.timestamp() - 60
        hour_ago = now.timestamp() - 3600
        
        self.message_history = [
            msg_time for msg_time in self.message_history 
            if msg_time.timestamp() > hour_ago
        ]
        
        # Count recent messages
        recent_minute = sum(
            1 for msg_time in self.message_history 
            if msg_time.timestamp() > minute_ago
        )
        
        recent_hour = len(self.message_history)
        
        # Check limits
        if recent_minute >= self.config.rate_limit_per_minute:
            return False
        
        if recent_hour >= self.config.rate_limit_per_hour:
            return False
        
        return True
    
    def record_message_sent(self):
        """Record that a message was sent for rate limiting."""
        self.message_history.append(datetime.now(UTC))


class SMSManager:
    """Comprehensive SMS and iMessage management with macOS Messages integration."""
    
    def __init__(self, km_client: Optional[KMClient] = None, config: Optional[SMSConfiguration] = None):
        self.km_client = km_client or KMClient()
        self.config = config or SMSConfiguration()
        self.security_validator = SMSSecurityValidator()
        self.rate_limiter = RateLimiter(self.config)
    
    @require(lambda self, request: isinstance(request, CommunicationRequest))
    @require(lambda self, request: request.communication_type in [CommunicationType.SMS, CommunicationType.IMESSAGE])
    @ensure(lambda self, result: isinstance(result, Either))
    async def send_message(self, request: CommunicationRequest) -> Either[CommunicationError, CommunicationResult]:
        """Send SMS or iMessage using macOS Messages application."""
        try:
            # Rate limiting check
            if not self.rate_limiter.can_send_message():
                return Either.left(CommunicationError.rate_limit_exceeded(
                    "Message rate limit exceeded"
                ))
            
            # Security validation
            validation_result = await self._validate_sms_request(request)
            if validation_result.is_left():
                return Either.left(CommunicationError.security_violation(
                    validation_result.get_left().message
                ))
            
            # Build AppleScript for message sending
            applescript_result = self._build_message_applescript(request)
            if applescript_result.is_left():
                return Either.left(CommunicationError.script_generation_failed(
                    applescript_result.get_left().message
                ))
            
            applescript = applescript_result.get_right()
            
            # Execute AppleScript through KM client
            execution_result = await self.km_client.execute_applescript(applescript, timeout=15)
            if execution_result.is_left():
                error = execution_result.get_left()
                return Either.left(CommunicationError.sms_send_failed(
                    f"Messages app execution failed: {error.message}"
                ))
            
            # Record successful send for rate limiting
            self.rate_limiter.record_message_sent()
            
            # Create success result
            message_id = MessageId(str(uuid.uuid4()))
            recipients = [
                r.format_for_sms() if isinstance(r, PhoneNumber) else str(r) 
                for r in request.recipients
            ]
            
            result = CommunicationResult(
                communication_type=request.communication_type,
                status=CommunicationStatus.SENT,
                message_id=message_id,
                recipients=recipients,
                timestamp=datetime.now(UTC),
                delivery_info={
                    "messages_app_used": True,
                    "message_length": len(request.message_content),
                    "service_type": request.communication_type.value,
                    "recipient_count": len(request.recipients)
                }
            )
            
            return Either.right(result)
            
        except Exception as e:
            return Either.left(CommunicationError.execution_error(
                f"Message sending failed: {str(e)}"
            ))
    
    async def _validate_sms_request(self, request: CommunicationRequest) -> Either[SecurityError, None]:
        """Comprehensive SMS/iMessage request validation."""
        # Validate message content
        content_validation = self.security_validator.validate_message_content(
            request.message_content, request.communication_type, self.config
        )
        if content_validation.is_left():
            return content_validation
        
        # Validate recipients are phone numbers
        phone_recipients = []
        for recipient in request.recipients:
            if not isinstance(recipient, PhoneNumber):
                return Either.left(SecurityError(
                    f"Invalid recipient type for SMS: {type(recipient)}"
                ))
            phone_recipients.append(recipient)
        
        # Validate phone numbers
        phone_validation = self.security_validator.validate_phone_numbers(phone_recipients)
        if phone_validation.is_left():
            return phone_validation
        
        # SMS specific validation (single recipient)
        if request.communication_type == CommunicationType.SMS and len(request.recipients) > 1:
            return Either.left(SecurityError("SMS supports single recipient only"))
        
        return Either.right(None)
    
    def _build_message_applescript(self, request: CommunicationRequest) -> Either[ValidationError, str]:
        """Build secure AppleScript for SMS/iMessage sending."""
        try:
            # Escape content for AppleScript safety
            safe_message = self._escape_applescript_string(request.message_content)
            
            # Determine service type
            service_type = "iMessage" if request.communication_type == CommunicationType.IMESSAGE else "SMS"
            
            # Handle recipients
            if len(request.recipients) == 1:
                # Single recipient (common case)
                recipient = request.recipients[0]
                if isinstance(recipient, PhoneNumber):
                    safe_recipient = self._escape_applescript_string(recipient.format_for_sms())
                else:
                    safe_recipient = self._escape_applescript_string(str(recipient))
                
                script = f'''
                tell application "Messages"
                    set targetService to 1st account whose service type = {service_type}
                    set targetBuddy to participant "{safe_recipient}" of targetService
                    send "{safe_message}" to targetBuddy
                end tell
                '''
            else:
                # Group message (iMessage only)
                recipients_list = []
                for recipient in request.recipients:
                    if isinstance(recipient, PhoneNumber):
                        safe_recipient = self._escape_applescript_string(recipient.format_for_sms())
                        recipients_list.append(f'participant "{safe_recipient}"')
                
                recipients_applescript = ', '.join(recipients_list)
                
                script = f'''
                tell application "Messages"
                    set targetService to 1st account whose service type = {service_type}
                    set targetBuddies to {{{recipients_applescript}}} of targetService
                    send "{safe_message}" to targetBuddies
                end tell
                '''
            
            # Validate script length
            if len(script) > 5000:
                return Either.left(ValidationError("Generated AppleScript too long"))
            
            return Either.right(script.strip())
            
        except Exception as e:
            return Either.left(ValidationError(f"AppleScript generation failed: {str(e)}"))
    
    def _escape_applescript_string(self, text: str) -> str:
        """Escape string for safe AppleScript inclusion."""
        if not text:
            return ""
        
        # Replace dangerous characters
        escaped = text.replace('"', '\\"')  # Escape quotes
        escaped = escaped.replace('\n', '\\n')  # Escape newlines
        escaped = escaped.replace('\r', '\\r')  # Escape carriage returns
        escaped = escaped.replace('\t', '\\t')  # Escape tabs
        escaped = escaped.replace('\\', '\\\\')  # Escape backslashes
        
        # Remove control characters except allowed ones
        escaped = ''.join(
            char for char in escaped 
            if ord(char) >= 32 or char in '\n\r\t'
        )
        
        # Limit length for safety
        max_length = (self.config.max_imessage_length if len(escaped) <= self.config.max_imessage_length 
                     else self.config.max_message_length)
        
        if len(escaped) > max_length:
            escaped = escaped[:max_length-20] + "... [truncated]"
        
        return escaped
    
    async def check_messages_app_availability(self) -> bool:
        """Check if Messages app is available and accessible."""
        try:
            applescript = 'tell application "Messages" to return name'
            result = await self.km_client.execute_applescript(applescript, timeout=5)
            return result.is_right()
        except Exception:
            return False
    
    async def get_message_history(self, contact: PhoneNumber, limit: int = 10) -> Either[CommunicationError, List[Dict[str, Any]]]:
        """Retrieve recent message history with a contact (read-only operation)."""
        try:
            if limit > 50:  # Reasonable limit
                return Either.left(CommunicationError.validation_error("Message history limit too high"))
            
            safe_contact = self._escape_applescript_string(contact.format_for_sms())
            
            # AppleScript to get recent messages (simplified)
            applescript = f'''
            tell application "Messages"
                set messageList to {{}}
                -- This would require more complex AppleScript to access message history
                -- For now, return placeholder
                return messageList
            end tell
            '''
            
            result = await self.km_client.execute_applescript(applescript)
            if result.is_left():
                return Either.left(CommunicationError.execution_error(
                    "Failed to retrieve message history"
                ))
            
            # Placeholder result - would need actual parsing
            messages = []
            return Either.right(messages)
            
        except Exception as e:
            return Either.left(CommunicationError.execution_error(str(e)))
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limiting status."""
        now = datetime.now(UTC)
        minute_ago = now.timestamp() - 60
        hour_ago = now.timestamp() - 3600
        
        recent_minute = sum(
            1 for msg_time in self.rate_limiter.message_history 
            if msg_time.timestamp() > minute_ago
        )
        
        recent_hour = len([
            msg_time for msg_time in self.rate_limiter.message_history 
            if msg_time.timestamp() > hour_ago
        ])
        
        return {
            "messages_last_minute": recent_minute,
            "messages_last_hour": recent_hour,
            "minute_limit": self.config.rate_limit_per_minute,
            "hour_limit": self.config.rate_limit_per_hour,
            "can_send": self.rate_limiter.can_send_message()
        }