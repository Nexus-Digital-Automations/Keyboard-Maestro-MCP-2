"""
Email management and sending functionality for Keyboard Maestro MCP Tools.

This module provides comprehensive email automation through macOS Mail app integration
with enterprise-grade security validation and AppleScript execution.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re
import uuid
import os
import asyncio
from datetime import datetime, UTC

from ..core.communication import (
    CommunicationRequest, CommunicationResult, CommunicationType, 
    CommunicationStatus, EmailAddress, MessageId
)
from ..core.either import Either
from ..core.errors import CommunicationError, SecurityError, ValidationError
from ..core.contracts import require, ensure
from ..integration.km_client import KMClient


@dataclass(frozen=True)
class EmailConfiguration:
    """Email sending configuration with security boundaries."""
    max_recipients: int = 100
    max_attachment_size_mb: int = 25
    max_message_length: int = 10000
    max_subject_length: int = 998  # RFC 2822 limit
    allowed_attachment_types: List[str] = None
    
    def __post_init__(self):
        if self.allowed_attachment_types is None:
            # Safe attachment types by default
            safe_types = ['.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', 
                         '.ppt', '.pptx', '.jpg', '.jpeg', '.png', '.gif']
            object.__setattr__(self, 'allowed_attachment_types', safe_types)


class EmailSecurityValidator:
    """Security-first email validation with comprehensive threat detection."""
    
    @staticmethod
    def validate_email_content(subject: Optional[str], body: str) -> Either[SecurityError, None]:
        """Validate email content for security threats."""
        # Check subject security
        if subject:
            if len(subject) > 998:  # RFC limit
                return Either.left(SecurityError("Subject line too long"))
            
            if EmailSecurityValidator._contains_header_injection(subject):
                return Either.left(SecurityError("Subject contains header injection"))
        
        # Check body security
        if len(body) > 100000:  # Reasonable limit
            return Either.left(SecurityError("Email body too long"))
        
        if EmailSecurityValidator._contains_malicious_content(body):
            return Either.left(SecurityError("Email body contains malicious content"))
        
        if EmailSecurityValidator._is_likely_spam(subject or "", body):
            return Either.left(SecurityError("Email appears to be spam"))
        
        return Either.right(None)
    
    @staticmethod
    def _contains_header_injection(text: str) -> bool:
        """Check for email header injection attempts."""
        injection_patterns = [
            r'\r\n',  # CRLF injection
            r'\n[a-zA-Z-]+:',  # New header line
            r'(bcc|cc|to|from):\s*[^\s]',  # Header injection
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in injection_patterns)
    
    @staticmethod
    def _contains_malicious_content(text: str) -> bool:
        """Check for malicious content patterns."""
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL) 
                  for pattern in malicious_patterns)
    
    @staticmethod
    def _is_likely_spam(subject: str, body: str) -> bool:
        """Check for common spam indicators."""
        combined_text = f"{subject} {body}".lower()
        
        spam_indicators = [
            r'click here now',
            r'urgent.{0,30}action.{0,30}required',
            r'congratulations.{0,30}winner',
            r'you.{0,20}have.{0,20}won',
            r'free.{0,20}money',
            r'limited.{0,20}time.{0,20}offer',
            r'act.{0,20}now',
            r'no.{0,20}obligation',
            r'increase.{0,20}your.{0,20}income',
            r'work.{0,20}from.{0,20}home',
        ]
        
        matches = sum(1 for pattern in spam_indicators 
                     if re.search(pattern, combined_text))
        
        # If multiple spam indicators, likely spam
        return matches >= 2
    
    @staticmethod
    def validate_attachments(attachments: List[str], config: EmailConfiguration) -> Either[SecurityError, None]:
        """Validate email attachments for security."""
        if len(attachments) > 10:  # Reasonable limit
            return Either.left(SecurityError("Too many attachments"))
        
        total_size = 0
        for attachment_path in attachments:
            # Check if file exists and is accessible
            if not os.path.exists(attachment_path):
                return Either.left(SecurityError(f"Attachment not found: {attachment_path}"))
            
            # Check file extension
            _, ext = os.path.splitext(attachment_path.lower())
            if ext not in config.allowed_attachment_types:
                return Either.left(SecurityError(f"Attachment type not allowed: {ext}"))
            
            # Check file size
            try:
                file_size = os.path.getsize(attachment_path)
                total_size += file_size
                
                if file_size > config.max_attachment_size_mb * 1024 * 1024:
                    return Either.left(SecurityError(f"Attachment too large: {attachment_path}"))
                
            except OSError:
                return Either.left(SecurityError(f"Cannot access attachment: {attachment_path}"))
        
        # Check total size
        if total_size > config.max_attachment_size_mb * 1024 * 1024:
            return Either.left(SecurityError("Total attachment size too large"))
        
        return Either.right(None)


class EmailManager:
    """Comprehensive email management with macOS Mail integration."""
    
    def __init__(self, km_client: Optional[KMClient] = None, config: Optional[EmailConfiguration] = None):
        self.km_client = km_client or KMClient()
        self.config = config or EmailConfiguration()
        self.security_validator = EmailSecurityValidator()
    
    @require(lambda self, request: isinstance(request, CommunicationRequest))
    @require(lambda self, request: request.communication_type == CommunicationType.EMAIL)
    @ensure(lambda self, result: isinstance(result, Either))
    async def send_email(self, request: CommunicationRequest) -> Either[CommunicationError, CommunicationResult]:
        """Send email using macOS Mail application with comprehensive validation."""
        try:
            # Security validation
            validation_result = await self._validate_email_request(request)
            if validation_result.is_left():
                return Either.left(CommunicationError.security_violation(validation_result.get_left().message))
            
            # Build AppleScript for email sending
            applescript_result = self._build_email_applescript(request)
            if applescript_result.is_left():
                return Either.left(CommunicationError.script_generation_failed(applescript_result.get_left().message))
            
            applescript = applescript_result.get_right()
            
            # Execute AppleScript through KM client
            execution_result = await self.km_client.execute_applescript(applescript, timeout=30)
            if execution_result.is_left():
                error = execution_result.get_left()
                return Either.left(CommunicationError.email_send_failed(
                    f"Mail app execution failed: {error.message}"
                ))
            
            # Create success result
            message_id = MessageId(str(uuid.uuid4()))
            recipients = [
                r.format_recipient() if isinstance(r, EmailAddress) else str(r) 
                for r in request.recipients
            ]
            
            result = CommunicationResult(
                communication_type=CommunicationType.EMAIL,
                status=CommunicationStatus.SENT,
                message_id=message_id,
                recipients=recipients,
                timestamp=datetime.now(UTC),
                delivery_info={
                    "mail_app_used": True,
                    "attachment_count": len(request.attachments),
                    "subject_length": len(request.subject or ""),
                    "body_length": len(request.message_content)
                }
            )
            
            return Either.right(result)
            
        except Exception as e:
            return Either.left(CommunicationError.execution_error(
                f"Email sending failed: {str(e)}"
            ))
    
    async def _validate_email_request(self, request: CommunicationRequest) -> Either[SecurityError, None]:
        """Comprehensive email request validation."""
        # Validate recipients count
        if len(request.recipients) > self.config.max_recipients:
            return Either.left(SecurityError(
                f"Too many recipients: {len(request.recipients)} > {self.config.max_recipients}"
            ))
        
        # Validate content security
        content_validation = self.security_validator.validate_email_content(
            request.subject, request.message_content
        )
        if content_validation.is_left():
            return content_validation
        
        # Validate attachments
        if request.attachments:
            attachment_validation = self.security_validator.validate_attachments(
                request.attachments, self.config
            )
            if attachment_validation.is_left():
                return attachment_validation
        
        # Validate recipients are email addresses
        for recipient in request.recipients:
            if not isinstance(recipient, EmailAddress):
                return Either.left(SecurityError(
                    f"Invalid recipient type for email: {type(recipient)}"
                ))
        
        return Either.right(None)
    
    def _build_email_applescript(self, request: CommunicationRequest) -> Either[ValidationError, str]:
        """Build secure AppleScript for email sending."""
        try:
            # Escape content for AppleScript safety
            safe_subject = self._escape_applescript_string(request.subject or "")
            safe_body = self._escape_applescript_string(request.message_content)
            
            # Build recipient list
            recipients = []
            for recipient in request.recipients:
                if isinstance(recipient, EmailAddress):
                    safe_email = self._escape_applescript_string(recipient.address)
                    recipients.append(safe_email)
            
            recipients_applescript = ', '.join([f'"{email}"' for email in recipients])
            
            # Start building script
            script_parts = [
                'tell application "Mail"',
                '    activate',
                '    set newMessage to make new outgoing message with properties {',
                f'        subject:"{safe_subject}",',
                f'        content:"{safe_body}",',
                '        visible:false',
                '    }',
                '',
                '    tell newMessage'
            ]
            
            # Add recipients
            script_parts.extend([
                f'        make new to recipient at end of to recipients with properties {{address:"{email}"}}'
                for email in recipients
            ])
            
            # Add attachments if present
            if request.attachments:
                script_parts.append('')
                for attachment_path in request.attachments:
                    safe_path = self._escape_applescript_string(attachment_path)
                    script_parts.append(
                        f'        make new attachment with properties {{file name:POSIX file "{safe_path}"}}'
                    )
            
            # Complete script
            script_parts.extend([
                '',
                '        send',
                '    end tell',
                'end tell'
            ])
            
            applescript = '\n'.join(script_parts)
            
            # Validate script length
            if len(applescript) > 10000:
                return Either.left(ValidationError("Generated AppleScript too long"))
            
            return Either.right(applescript)
            
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
        
        # Remove any remaining control characters
        escaped = ''.join(char for char in escaped if ord(char) >= 32 or char in '\n\r\t')
        
        # Limit length for safety
        if len(escaped) > 5000:
            escaped = escaped[:5000] + "... [truncated for safety]"
        
        return escaped
    
    async def get_mail_accounts(self) -> Either[CommunicationError, List[Dict[str, str]]]:
        """Get available Mail app accounts."""
        try:
            applescript = '''
            tell application "Mail"
                set accountList to {}
                repeat with acc in accounts
                    set end of accountList to {name:(name of acc), email:(email address of acc)}
                end repeat
                return accountList
            end tell
            '''
            
            result = await self.km_client.execute_applescript(applescript)
            if result.is_left():
                return Either.left(CommunicationError.execution_error(
                    "Failed to retrieve Mail accounts"
                ))
            
            # Parse result (simplified - would need proper AppleScript result parsing)
            accounts = [{"name": "Default", "email": "user@example.com"}]  # Placeholder
            return Either.right(accounts)
            
        except Exception as e:
            return Either.left(CommunicationError.execution_error(str(e)))
    
    async def check_mail_app_availability(self) -> bool:
        """Check if Mail app is available and accessible."""
        try:
            applescript = 'tell application "Mail" to return name'
            result = await self.km_client.execute_applescript(applescript, timeout=5)
            return result.is_right()
        except Exception:
            return False