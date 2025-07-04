"""
Comprehensive tests for communication tools functionality.

This module provides unit tests, integration tests, and property-based tests
for the communication automation system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from hypothesis import given, strategies as st, assume

from src.core.communication import (
    EmailAddress, PhoneNumber, MessageTemplate, CommunicationRequest,
    CommunicationType, MessagePriority, CommunicationStatus, TemplateId
)
from src.communication.email_manager import EmailManager, EmailConfiguration
from src.communication.sms_manager import SMSManager, SMSConfiguration  
from src.communication.message_templates import MessageTemplateManager
from src.communication.communication_security import (
    CommunicationSecurityManager, SecurityConfiguration, SecurityThreat
)
from src.core.either import Either
from src.core.errors import ValidationError, SecurityError, CommunicationError


class TestEmailAddress:
    """Test email address validation and formatting."""
    
    def test_valid_email_creation(self):
        """Test creating valid email addresses."""
        email = EmailAddress("test@example.com")
        assert email.address == "test@example.com"
        assert email.name is None
        assert email.domain == "example.com"
    
    def test_email_with_name(self):
        """Test email address with display name."""
        email = EmailAddress("test@example.com", "Test User")
        assert email.format_recipient() == '"Test User" <test@example.com>'
    
    def test_invalid_email_format(self):
        """Test invalid email format rejection."""
        with pytest.raises(ValidationError):
            EmailAddress("invalid-email")
        
        with pytest.raises(ValidationError):
            EmailAddress("test@")
        
        with pytest.raises(ValidationError):
            EmailAddress("@example.com")
    
    def test_email_security_validation(self):
        """Test email security pattern detection."""
        with pytest.raises(ValidationError):
            EmailAddress("test..double@example.com")  # Double dots
        
        with pytest.raises(ValidationError):
            EmailAddress(".test@example.com")  # Starting with dot
    
    def test_long_email_name(self):
        """Test email name length validation."""
        long_name = "x" * 101
        with pytest.raises(ValidationError):
            EmailAddress("test@example.com", long_name)


class TestPhoneNumber:
    """Test phone number validation and formatting."""
    
    def test_valid_phone_creation(self):
        """Test creating valid phone numbers."""
        phone = PhoneNumber("555-123-4567")
        assert phone.format_for_sms() == "5551234567"
    
    def test_international_phone(self):
        """Test international phone number formatting."""
        phone = PhoneNumber("+1-555-123-4567")
        assert phone.format_for_sms() == "+15551234567"
    
    def test_phone_with_country_code(self):
        """Test phone with separate country code."""
        phone = PhoneNumber("5551234567", "1")
        assert phone.format_for_sms() == "+15551234567"
    
    def test_invalid_phone_format(self):
        """Test invalid phone number rejection."""
        with pytest.raises(ValidationError):
            PhoneNumber("123")  # Too short
        
        with pytest.raises(ValidationError):
            PhoneNumber("123456789012345678")  # Too long
        
        with pytest.raises(ValidationError):
            PhoneNumber("abc-def-ghij")  # Invalid characters
    
    def test_national_format(self):
        """Test national format generation."""
        phone = PhoneNumber("+15551234567")
        assert phone.national_format == "(555) 123-4567"


class TestMessageTemplate:
    """Test message template functionality."""
    
    def test_template_creation(self):
        """Test creating message templates."""
        template = MessageTemplate(
            template_id=TemplateId("test_template"),
            name="Test Template",
            subject_template="Hello {name}",
            body_template="Welcome {name}, your code is {code}."
        )
        assert template.variables == {"name", "code"}
    
    def test_template_rendering(self):
        """Test template variable substitution."""
        template = MessageTemplate(
            template_id=TemplateId("test_template"),
            name="Test Template",
            subject_template="Hello {name}",
            body_template="Welcome {name}, your code is {code}."
        )
        
        rendered = template.render({"name": "John", "code": "12345"})
        assert rendered["subject"] == "Hello John"
        assert rendered["body"] == "Welcome John, your code is 12345."
    
    def test_missing_variables(self):
        """Test error on missing template variables."""
        template = MessageTemplate(
            template_id=TemplateId("test_template"),
            name="Test Template",
            body_template="Hello {name}, your code is {code}."
        )
        
        with pytest.raises(ValidationError):
            template.render({"name": "John"})  # Missing 'code'
    
    def test_template_security_validation(self):
        """Test template security validation."""
        with pytest.raises(ValidationError):
            MessageTemplate(
                template_id=TemplateId(""),  # Empty ID
                name="Test Template",
                body_template="Hello {name}"
            )
        
        with pytest.raises(ValidationError):
            MessageTemplate(
                template_id=TemplateId("test"),
                name="Test Template",
                body_template=""  # Empty body
            )


class TestCommunicationRequest:
    """Test communication request validation."""
    
    def test_email_request_creation(self):
        """Test creating valid email communication request."""
        email = EmailAddress("test@example.com")
        request = CommunicationRequest(
            communication_type=CommunicationType.EMAIL,
            recipients=[email],
            message_content="Test message",
            subject="Test Subject"
        )
        assert request.communication_type == CommunicationType.EMAIL
        assert len(request.recipients) == 1
    
    def test_sms_request_creation(self):
        """Test creating valid SMS communication request."""
        phone = PhoneNumber("555-123-4567")
        request = CommunicationRequest(
            communication_type=CommunicationType.SMS,
            recipients=[phone],
            message_content="Test SMS message"
        )
        assert request.communication_type == CommunicationType.SMS
    
    def test_email_requires_subject(self):
        """Test that email requires subject."""
        email = EmailAddress("test@example.com")
        with pytest.raises(ValidationError):
            CommunicationRequest(
                communication_type=CommunicationType.EMAIL,
                recipients=[email],
                message_content="Test message"
                # No subject provided
            )
    
    def test_sms_single_recipient(self):
        """Test that SMS requires single recipient."""
        phones = [PhoneNumber("555-123-4567"), PhoneNumber("555-987-6543")]
        with pytest.raises(ValidationError):
            CommunicationRequest(
                communication_type=CommunicationType.SMS,
                recipients=phones,
                message_content="Test message"
            )
    
    def test_too_many_recipients(self):
        """Test recipient limit validation."""
        emails = [EmailAddress(f"test{i}@example.com") for i in range(101)]
        with pytest.raises(ValidationError):
            CommunicationRequest(
                communication_type=CommunicationType.EMAIL,
                recipients=emails,
                message_content="Test message",
                subject="Test Subject"
            )
    
    def test_empty_message_content(self):
        """Test empty message content rejection."""
        email = EmailAddress("test@example.com")
        with pytest.raises(ValidationError):
            CommunicationRequest(
                communication_type=CommunicationType.EMAIL,
                recipients=[email],
                message_content="",
                subject="Test Subject"
            )


class TestEmailManager:
    """Test email manager functionality."""
    
    @pytest.fixture
    def mock_km_client(self):
        """Mock KM client for testing."""
        client = Mock()
        client.execute_applescript = AsyncMock(return_value=Either.right("success"))
        return client
    
    @pytest.fixture
    def email_manager(self, mock_km_client):
        """Email manager with mocked dependencies."""
        return EmailManager(km_client=mock_km_client)
    
    @pytest.mark.asyncio
    async def test_send_email_success(self, email_manager):
        """Test successful email sending."""
        email = EmailAddress("test@example.com")
        request = CommunicationRequest(
            communication_type=CommunicationType.EMAIL,
            recipients=[email],
            message_content="Test message",
            subject="Test Subject"
        )
        
        result = await email_manager.send_email(request)
        assert result.is_right()
        
        comm_result = result.get_right()
        assert comm_result.communication_type == CommunicationType.EMAIL
        assert comm_result.status == CommunicationStatus.SENT
    
    @pytest.mark.asyncio
    async def test_send_email_with_attachments(self, email_manager):
        """Test email sending with attachments."""
        email = EmailAddress("test@example.com")
        
        # Mock file existence
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024):
            
            request = CommunicationRequest(
                communication_type=CommunicationType.EMAIL,
                recipients=[email],
                message_content="Test message",
                subject="Test Subject",
                attachments=["/Users/test/document.pdf"]
            )
            
            result = await email_manager.send_email(request)
            assert result.is_right()
    
    @pytest.mark.asyncio
    async def test_send_email_applescript_failure(self, email_manager):
        """Test email sending with AppleScript failure."""
        email_manager.km_client.execute_applescript = AsyncMock(
            return_value=Either.left(Exception("AppleScript failed"))
        )
        
        email = EmailAddress("test@example.com")
        request = CommunicationRequest(
            communication_type=CommunicationType.EMAIL,
            recipients=[email],
            message_content="Test message",
            subject="Test Subject"
        )
        
        result = await email_manager.send_email(request)
        assert result.is_left()
        assert isinstance(result.get_left(), CommunicationError)
    
    def test_applescript_escaping(self, email_manager):
        """Test AppleScript string escaping."""
        dangerous_string = 'Test "quoted" string\nwith newlines'
        escaped = email_manager._escape_applescript_string(dangerous_string)
        
        assert '\\"' in escaped  # Quotes escaped
        assert '\\n' in escaped  # Newlines escaped
        assert '"' not in escaped.replace('\\"', '')  # No unescaped quotes


class TestSMSManager:
    """Test SMS manager functionality."""
    
    @pytest.fixture
    def mock_km_client(self):
        """Mock KM client for testing."""
        client = Mock()
        client.execute_applescript = AsyncMock(return_value=Either.right("success"))
        return client
    
    @pytest.fixture
    def sms_manager(self, mock_km_client):
        """SMS manager with mocked dependencies."""
        return SMSManager(km_client=mock_km_client)
    
    @pytest.mark.asyncio
    async def test_send_sms_success(self, sms_manager):
        """Test successful SMS sending."""
        phone = PhoneNumber("555-123-4567")
        request = CommunicationRequest(
            communication_type=CommunicationType.SMS,
            recipients=[phone],
            message_content="Test SMS message"
        )
        
        result = await sms_manager.send_message(request)
        assert result.is_right()
        
        comm_result = result.get_right()
        assert comm_result.communication_type == CommunicationType.SMS
        assert comm_result.status == CommunicationStatus.SENT
    
    @pytest.mark.asyncio
    async def test_send_imessage_group(self, sms_manager):
        """Test iMessage group messaging."""
        phones = [PhoneNumber("555-123-4567"), PhoneNumber("555-987-6543")]
        request = CommunicationRequest(
            communication_type=CommunicationType.IMESSAGE,
            recipients=phones,
            message_content="Test group message"
        )
        
        result = await sms_manager.send_message(request)
        assert result.is_right()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, sms_manager):
        """Test SMS rate limiting."""
        phone = PhoneNumber("555-123-4567")
        
        # Send messages up to rate limit
        for _ in range(sms_manager.config.rate_limit_per_minute):
            request = CommunicationRequest(
                communication_type=CommunicationType.SMS,
                recipients=[phone],
                message_content="Test message"
            )
            result = await sms_manager.send_message(request)
            assert result.is_right()
        
        # Next message should be rate limited
        request = CommunicationRequest(
            communication_type=CommunicationType.SMS,
            recipients=[phone],
            message_content="Rate limited message"
        )
        result = await sms_manager.send_message(request)
        assert result.is_left()
        assert isinstance(result.get_left(), CommunicationError)
    
    def test_get_rate_limit_status(self, sms_manager):
        """Test rate limit status reporting."""
        status = sms_manager.get_rate_limit_status()
        
        assert "messages_last_minute" in status
        assert "messages_last_hour" in status
        assert "can_send" in status
        assert isinstance(status["can_send"], bool)


class TestMessageTemplateManager:
    """Test message template management."""
    
    @pytest.fixture
    def template_manager(self):
        """Template manager instance."""
        return MessageTemplateManager()
    
    def test_default_templates_loaded(self, template_manager):
        """Test that default templates are loaded."""
        templates = template_manager.list_templates("default")
        assert "default" in templates
        assert len(templates["default"]) > 0
    
    def test_add_template(self, template_manager):
        """Test adding new template."""
        template = MessageTemplate(
            template_id=TemplateId("custom_template"),
            name="Custom Template",
            body_template="Hello {name}, welcome to {service}!"
        )
        
        result = template_manager.add_template(template)
        assert result.is_right()
        
        # Verify template was added
        retrieved = template_manager.get_template(TemplateId("custom_template"))
        assert retrieved.is_right()
        assert retrieved.get_right().name == "Custom Template"
    
    def test_render_template(self, template_manager):
        """Test template rendering with validation."""
        template = MessageTemplate(
            template_id=TemplateId("test_template"),
            name="Test Template",
            body_template="Hello {name}, your order {order_id} is ready!"
        )
        
        template_manager.add_template(template)
        
        rendered_result = template_manager.render_template(
            template, {"name": "John", "order_id": "12345"}
        )
        assert rendered_result.is_right()
        
        rendered = rendered_result.get_right()
        assert "Hello John" in rendered["body"]
        assert "order 12345" in rendered["body"]
    
    def test_template_security_validation(self, template_manager):
        """Test template security validation."""
        dangerous_template = MessageTemplate(
            template_id=TemplateId("dangerous_template"),
            name="Dangerous Template",
            body_template="<script>alert('xss')</script>Hello {name}"
        )
        
        result = template_manager.add_template(dangerous_template)
        assert result.is_left()
        assert isinstance(result.get_left(), ValidationError)
    
    def test_delete_template(self, template_manager):
        """Test template deletion."""
        template = MessageTemplate(
            template_id=TemplateId("delete_me"),
            name="Delete Me",
            body_template="This will be deleted"
        )
        
        template_manager.add_template(template)
        
        # Verify it exists
        result = template_manager.get_template(TemplateId("delete_me"))
        assert result.is_right()
        
        # Delete it
        delete_result = template_manager.delete_template(TemplateId("delete_me"))
        assert delete_result.is_right()
        
        # Verify it's gone
        result = template_manager.get_template(TemplateId("delete_me"))
        assert result.is_left()


class TestCommunicationSecurity:
    """Test communication security validation."""
    
    @pytest.fixture
    def security_manager(self):
        """Security manager instance."""
        return CommunicationSecurityManager()
    
    def test_spam_detection(self, security_manager):
        """Test spam content detection."""
        spam_content = "CONGRATULATIONS! You've WON $1000000! Click here NOW!"
        
        score, threats = security_manager.spam_detector.analyze_content(
            "URGENT ACTION REQUIRED", spam_content
        )
        
        assert score > 5.0  # High spam score
        assert len(threats) > 0
        assert any(t.threat_type == "spam_keywords" for t in threats)
    
    def test_legitimate_content(self, security_manager):
        """Test that legitimate content passes validation."""
        legitimate_content = "Hello John, your meeting is scheduled for tomorrow at 2 PM."
        
        score, threats = security_manager.spam_detector.analyze_content(
            "Meeting Reminder", legitimate_content
        )
        
        assert score < 3.0  # Low spam score
        assert len([t for t in threats if t.severity in ["high", "critical"]]) == 0
    
    @pytest.mark.asyncio
    async def test_security_validation_success(self, security_manager):
        """Test successful security validation."""
        email = EmailAddress("test@example.com")
        request = CommunicationRequest(
            communication_type=CommunicationType.EMAIL,
            recipients=[email],
            message_content="Hello, this is a legitimate business message.",
            subject="Business Update"
        )
        
        result = security_manager.validate_communication_security(request, "sender_123")
        assert result.is_right()
        
        validation_info = result.get_right()
        assert validation_info["security_level"] in ["safe", "caution"]
        assert validation_info["spam_score"] < 7.0
    
    @pytest.mark.asyncio
    async def test_security_validation_spam_rejection(self, security_manager):
        """Test spam content rejection."""
        email = EmailAddress("test@example.com")
        request = CommunicationRequest(
            communication_type=CommunicationType.EMAIL,
            recipients=[email],
            message_content="URGENT! Click here NOW to claim your FREE MONEY! Act fast!",
            subject="CONGRATULATIONS WINNER!!!"
        )
        
        result = security_manager.validate_communication_security(request, "sender_123")
        assert result.is_left()
        assert isinstance(result.get_left(), SecurityError)
    
    def test_security_statistics(self, security_manager):
        """Test security statistics generation."""
        stats = security_manager.get_security_statistics()
        
        assert "total_threats_logged" in stats
        assert "recent_threats_1h" in stats
        assert "rate_limit_status" in stats
        assert "security_config" in stats


# Property-based tests using Hypothesis
class TestCommunicationProperties:
    """Property-based tests for communication components."""
    
    @given(st.emails())
    def test_email_address_properties(self, email_string):
        """Property: Valid email strings should create valid EmailAddress objects."""
        try:
            email = EmailAddress(email_string)
            assert email.address == email_string
            assert "@" not in email.domain
            assert email.format_recipient() == email_string
        except ValidationError:
            # Some generated emails might be invalid, which is acceptable
            pass
    
    @given(st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
    def test_template_variable_names(self, var_name):
        """Property: Template variables should handle various valid names."""
        assume(var_name.isalpha())  # Only alphabetic characters
        assume(var_name[0].isalpha())  # Must start with letter
        
        try:
            template = MessageTemplate(
                template_id=TemplateId("test_template"),
                name="Test Template",
                body_template=f"Hello {{{var_name}}}, welcome!"
            )
            assert var_name in template.variables
            
            # Should be able to render with valid variable
            rendered = template.render({var_name: "TestValue"})
            assert "TestValue" in rendered["body"]
        except ValidationError:
            # Some generated names might be invalid
            pass
    
    @given(st.text(min_size=1, max_size=1000))
    def test_message_content_security(self, message_content):
        """Property: Message content should be analyzed for security threats."""
        security_manager = CommunicationSecurityManager()
        
        # Should not crash on any input
        score, threats = security_manager.spam_detector.analyze_content(
            "Test Subject", message_content
        )
        
        assert isinstance(score, (int, float))
        assert score >= 0.0
        assert isinstance(threats, list)
        
        # High spam scores should have corresponding threats
        if score > 7.0:
            assert len(threats) > 0
    
    @given(st.lists(st.emails(), min_size=1, max_size=10))
    def test_recipient_list_properties(self, email_list):
        """Property: Recipient lists should handle various valid email combinations."""
        try:
            recipients = [EmailAddress(email) for email in email_list]
            
            request = CommunicationRequest(
                communication_type=CommunicationType.EMAIL,
                recipients=recipients,
                message_content="Test message content",
                subject="Test Subject"
            )
            
            assert len(request.recipients) == len(email_list)
            assert all(isinstance(r, EmailAddress) for r in request.recipients)
            
        except ValidationError:
            # Some combinations might be invalid
            pass


# Integration tests
@pytest.mark.integration
class TestCommunicationIntegration:
    """Integration tests for complete communication workflows."""
    
    @pytest.mark.asyncio
    async def test_email_template_workflow(self):
        """Test complete email workflow with templates."""
        # Setup
        template_manager = MessageTemplateManager()
        security_manager = CommunicationSecurityManager()
        
        # Create template
        template = MessageTemplate(
            template_id=TemplateId("welcome_email"),
            name="Welcome Email",
            subject_template="Welcome to {service}, {name}!",
            body_template="Dear {name},\n\nWelcome to {service}! Your account has been created successfully.\n\nBest regards,\nThe Team"
        )
        
        template_manager.add_template(template)
        
        # Render template
        rendered_result = template_manager.render_template(
            template, {"name": "John Doe", "service": "TestApp"}
        )
        assert rendered_result.is_right()
        rendered = rendered_result.get_right()
        
        # Create communication request
        email = EmailAddress("john.doe@example.com")
        request = CommunicationRequest(
            communication_type=CommunicationType.EMAIL,
            recipients=[email],
            message_content=rendered["body"],
            subject=rendered["subject"]
        )
        
        # Security validation
        security_result = await security_manager.validate_communication_security(
            request, "system_sender"
        )
        assert security_result.is_right()
        
        validation_info = security_result.get_right()
        assert validation_info["security_level"] == "safe"
    
    @pytest.mark.asyncio
    async def test_sms_rate_limiting_workflow(self):
        """Test SMS workflow with rate limiting."""
        mock_km_client = Mock()
        mock_km_client.execute_applescript = AsyncMock(return_value=Either.right("success"))
        
        sms_manager = SMSManager(km_client=mock_km_client)
        phone = PhoneNumber("555-123-4567")
        
        # Send messages within rate limit
        for i in range(5):
            request = CommunicationRequest(
                communication_type=CommunicationType.SMS,
                recipients=[phone],
                message_content=f"Test message {i}"
            )
            
            result = await sms_manager.send_message(request)
            assert result.is_right()
        
        # Check rate limit status
        status = sms_manager.get_rate_limit_status()
        assert status["messages_last_minute"] == 5
        assert status["can_send"] == True  # Still within limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])