"""
Communication types and protocols for the Keyboard Maestro MCP macro engine.

This module defines comprehensive communication capabilities including email, SMS,
and messaging with type-safe validation and security boundaries.
"""

from __future__ import annotations
from typing import NewType, Union, Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, UTC
import re
import uuid
import os
from .types import Permission
from .contracts import require, ensure
from .either import Either
from .errors import SecurityError, ValidationError, CommunicationError


# Branded Types for Communication
EmailId = NewType('EmailId', str)
MessageId = NewType('MessageId', str)
ContactId = NewType('ContactId', str)
TemplateId = NewType('TemplateId', str)
AttachmentPath = NewType('AttachmentPath', str)


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


class CommunicationStatus(Enum):
    """Communication delivery status."""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class EmailAddress:
    """Type-safe email address with comprehensive validation."""
    address: str
    name: Optional[str] = None
    
    def __post_init__(self):
        """Validate email address on creation."""
        if not self._is_valid_email(self.address):
            raise ValidationError(
                field_name="address",
                value=self.address,
                constraint="must be a valid email address format"
            )
        
        if self.name and len(self.name) > 100:
            raise ValidationError(
                field_name="name",
                value=self.name,
                constraint="must be 100 characters or less"
            )
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email address format with security boundaries."""
        if not email or len(email) > 320:  # RFC 5321 limit
            return False
        
        # Basic format validation
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False
        
        # Security checks
        return not self._contains_suspicious_patterns(email)
    
    def _contains_suspicious_patterns(self, email: str) -> bool:
        """Check for suspicious email patterns."""
        suspicious_patterns = [
            r'\.{2,}',  # Multiple consecutive dots
            r'^\.|\.$',  # Starting or ending with dot
            r'[<>"\']',  # Potential injection characters
        ]
        
        return any(re.search(pattern, email) for pattern in suspicious_patterns)
    
    def format_recipient(self) -> str:
        """Format for use in email clients with security escaping."""
        if self.name:
            # Escape potential dangerous characters in name
            safe_name = re.sub(r'[<>"\']', '', self.name)
            return f'"{safe_name}" <{self.address}>'
        return self.address
    
    @property
    def domain(self) -> str:
        """Extract domain from email address."""
        return self.address.split('@')[1]


@dataclass(frozen=True)
class PhoneNumber:
    """Type-safe phone number with international support."""
    number: str
    country_code: Optional[str] = None
    
    def __post_init__(self):
        """Validate phone number on creation."""
        if not self._is_valid_phone(self.number):
            raise ValidationError(
                field_name="number",
                value=self.number,
                constraint="must be a valid phone number format"
            )
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Validate phone number format with security checks."""
        if not phone or len(phone) > 20:
            return False
        
        # Remove common formatting characters
        cleaned = re.sub(r'[^\d+\-\(\)\s]', '', phone)
        digits_only = re.sub(r'[^\d+]', '', cleaned)
        
        # Check length (reasonable international range)
        if len(digits_only) < 10 or len(digits_only) > 15:
            return False
        
        # Basic format validation
        if digits_only.startswith('+'):
            return len(digits_only) >= 11  # +1 + 10 digits minimum
        
        return True
    
    def format_for_sms(self) -> str:
        """Format for SMS sending with country code if needed."""
        cleaned = re.sub(r'[^\d+]', '', self.number)
        
        if self.country_code and not cleaned.startswith('+'):
            return f'+{self.country_code}{cleaned}'
        
        return cleaned if cleaned.startswith('+') else cleaned
    
    @property
    def national_format(self) -> str:
        """Get national format of phone number."""
        cleaned = self.format_for_sms()
        if cleaned.startswith('+1'):  # US/Canada
            return f"({cleaned[2:5]}) {cleaned[5:8]}-{cleaned[8:]}"
        return cleaned


@dataclass(frozen=True)
class MessageTemplate:
    """Reusable message template with secure variable substitution."""
    template_id: TemplateId
    name: str
    subject_template: Optional[str] = None
    body_template: str = ""
    variables: Set[str] = field(default_factory=set)
    communication_type: CommunicationType = CommunicationType.EMAIL
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Initialize template and extract variables."""
        if not self.template_id or len(self.template_id) == 0:
            raise ValidationError(
                field_name="template_id",
                value=self.template_id,
                constraint="cannot be empty"
            )
        
        if not self.body_template or len(self.body_template) == 0:
            raise ValidationError(
                field_name="body_template",
                value=self.body_template,
                constraint="cannot be empty"
            )
        
        if len(self.body_template) > 10000:
            raise ValidationError(
                field_name="body_template",
                value=self.body_template,
                constraint="must be 10000 characters or less"
            )
        
        # Extract variables from templates safely
        subject_vars = set(re.findall(r'\{(\w+)\}', self.subject_template or ''))
        body_vars = set(re.findall(r'\{(\w+)\}', self.body_template))
        all_vars = subject_vars | body_vars
        
        # Validate variable names
        for var in all_vars:
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', var):
                raise ValidationError(
                    field_name="variable_name",
                    value=var,
                    constraint="must start with letter and contain only letters, numbers, and underscores"
                )
        
        # Update variables field (frozen dataclass workaround)
        object.__setattr__(self, 'variables', all_vars)
    
    def render(self, variables: Dict[str, str]) -> Dict[str, str]:
        """Render template with provided variables and security validation."""
        # Check for missing variables
        missing_vars = self.variables - set(variables.keys())
        if missing_vars:
            raise ValidationError(
                field_name="variables",
                value=str(missing_vars),
                constraint="all template variables must be provided"
            )
        
        # Validate variable values for security
        for key, value in variables.items():
            if not isinstance(value, str):
                raise ValidationError(
                    field_name=key,
                    value=str(value),
                    constraint="must be a string value"
                )
            if len(value) > 1000:
                raise ValidationError(
                    field_name=key,
                    value=value,
                    constraint="must be 1000 characters or less"
                )
            if self._contains_injection_patterns(value):
                raise SecurityError(f"Variable {key} contains suspicious content")
        
        try:
            rendered = {
                "body": self.body_template.format(**variables)
            }
            
            if self.subject_template:
                rendered["subject"] = self.subject_template.format(**variables)
            
            return rendered
        except KeyError as e:
            raise ValidationError(
                field_name="template_variables",
                value=str(e),
                constraint="all template variables must be provided"
            )
        except Exception as e:
            raise ValidationError(
                field_name="template",
                value=str(e),
                constraint="template must be valid and renderable"
            )
    
    def _contains_injection_patterns(self, value: str) -> bool:
        """Check for potential injection patterns in variable values."""
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'data:text/html',
            r'eval\s*\(',
        ]
        
        value_lower = value.lower()
        return any(re.search(pattern, value_lower) for pattern in dangerous_patterns)


@dataclass(frozen=True)
class CommunicationRequest:
    """Complete communication request with comprehensive validation."""
    communication_type: CommunicationType
    recipients: List[Union[EmailAddress, PhoneNumber]]
    message_content: str
    subject: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    attachments: List[AttachmentPath] = field(default_factory=list)
    delivery_receipt: bool = False
    template_id: Optional[TemplateId] = None
    template_variables: Optional[Dict[str, str]] = None
    from_account: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Validate communication request on creation."""
        # Basic validation
        if not self.recipients:
            raise ValidationError(
                field_name="recipients",
                value=str(self.recipients),
                constraint="cannot be empty"
            )
        
        if len(self.recipients) > 100:
            raise ValidationError(
                field_name="recipients",
                value=str(len(self.recipients)),
                constraint="must have 100 recipients or fewer"
            )
        
        if not self.message_content or len(self.message_content.strip()) == 0:
            raise ValidationError(
                field_name="message_content",
                value=self.message_content,
                constraint="cannot be empty"
            )
        
        if len(self.message_content) > 10000:
            raise ValidationError(
                field_name="message_content",
                value=self.message_content,
                constraint="must be 10000 characters or less"
            )
        
        # Email-specific validation
        if self.communication_type == CommunicationType.EMAIL:
            if not self.subject and not self.template_id:
                raise ValidationError(
                    field_name="subject",
                    value=self.subject,
                    constraint="email communication requires subject"
                )
            
            # Validate email recipients
            for recipient in self.recipients:
                if not isinstance(recipient, EmailAddress):
                    raise ValidationError(
                        field_name="recipients",
                        value=str(type(recipient)),
                        constraint="email communication requires EmailAddress recipients"
                    )
        
        # SMS-specific validation
        elif self.communication_type in [CommunicationType.SMS, CommunicationType.IMESSAGE]:
            if len(self.recipients) > 1 and self.communication_type == CommunicationType.SMS:
                raise ValidationError(
                    field_name="recipients",
                    value=str(len(self.recipients)),
                    constraint="SMS supports single recipient only"
                )
            
            # Validate phone recipients
            for recipient in self.recipients:
                if not isinstance(recipient, PhoneNumber):
                    raise ValidationError(
                        field_name="recipients",
                        value=str(type(recipient)),
                        constraint="SMS/iMessage requires PhoneNumber recipients"
                    )
        
        # Attachment validation
        for attachment in self.attachments:
            if not self._is_safe_attachment_path(attachment):
                raise SecurityError(f"Unsafe attachment path: {attachment}")
    
    def _is_safe_attachment_path(self, path: str) -> bool:
        """Validate attachment file path for security."""
        if not path or len(path) > 500:
            return False
        
        # Expand user path safely
        try:
            expanded_path = os.path.expanduser(path)
            normalized_path = os.path.normpath(expanded_path)
        except Exception:
            return False
        
        # Only allow files in safe directories
        safe_prefixes = [
            '/Users/',
            '~/Documents/',
            '~/Downloads/',
            '~/Desktop/',
            './attachments/',
        ]
        
        return any(normalized_path.startswith(os.path.expanduser(prefix)) 
                  for prefix in safe_prefixes)


@dataclass(frozen=True)
class CommunicationResult:
    """Result of communication operation with tracking info."""
    communication_type: CommunicationType
    status: CommunicationStatus
    message_id: MessageId
    recipients: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    delivery_info: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None
    
    def was_successful(self) -> bool:
        """Check if communication was successful."""
        return self.status in [CommunicationStatus.SENT, CommunicationStatus.DELIVERED]
    
    def format_summary(self) -> str:
        """Format human-readable summary."""
        recipient_count = len(self.recipients)
        type_name = self.communication_type.value.upper()
        
        if self.was_successful():
            return f"{type_name} sent successfully to {recipient_count} recipient(s)"
        else:
            return f"{type_name} failed: {self.error_details or 'Unknown error'}"


# Required permissions for communication operations
COMMUNICATION_PERMISSIONS = {
    CommunicationType.EMAIL: [Permission.NETWORK_ACCESS, Permission.FILE_ACCESS],
    CommunicationType.SMS: [Permission.NETWORK_ACCESS],
    CommunicationType.IMESSAGE: [Permission.NETWORK_ACCESS],
    CommunicationType.NOTIFICATION: [Permission.SYSTEM_CONTROL],
}