"""
Test suite for notification system implementation (TASK_17).

Comprehensive property-based testing for notification display, user interaction,
and multi-channel notification management with security validation.

Testing Categories:
- Notification type validation and channel selection
- Content security validation and injection prevention  
- User interaction tracking and button response handling
- Sound system integration and file validation
- HUD positioning and display duration management
- Priority-based notification ordering and state management
- AppleScript generation safety and parameter escaping
- Error handling and graceful degradation scenarios
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional

from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from src.notifications.notification_manager import (
    NotificationManager, NotificationSpec, NotificationType, 
    NotificationPosition, NotificationPriority, NotificationResult
)
from src.server.tools.notification_tools import (
    km_notifications, km_notification_status, km_dismiss_notifications
)
from src.core.errors import MacroEngineError, ValidationError, SecurityViolationError
from src.core.types import Duration


class TestNotificationManager:
    """Test core notification manager functionality."""
    
    @pytest.fixture
    def mock_km_client(self):
        """Create mock KM client for testing."""
        client = Mock()
        client.execute_applescript = AsyncMock()
        client.display_hud_text = AsyncMock()
        client.play_sound = AsyncMock()
        return client
    
    @pytest.fixture
    def notification_manager(self, mock_km_client):
        """Create notification manager with mocked client."""
        return NotificationManager(mock_km_client)
    
    def test_notification_type_enum_completeness(self):
        """Test that all notification types are properly defined."""
        expected_types = {"notification", "alert", "hud", "sound"}
        actual_types = {nt.value for nt in NotificationType}
        assert actual_types == expected_types
    
    def test_notification_position_enum_completeness(self):
        """Test that all HUD positions are properly defined."""
        expected_positions = {
            "center", "top", "bottom", "left", "right",
            "top_left", "top_right", "bottom_left", "bottom_right"
        }
        actual_positions = {np.value for np in NotificationPosition}
        assert actual_positions == expected_positions
    
    def test_notification_priority_enum_completeness(self):
        """Test that all priority levels are properly defined."""
        expected_priorities = {"low", "normal", "high", "urgent"}
        actual_priorities = {np.value for np in NotificationPriority}
        assert actual_priorities == expected_priorities
    
    def test_notification_spec_validation_title_length(self):
        """Test title length validation in notification spec."""
        # Valid title
        spec = NotificationSpec(
            notification_type=NotificationType.NOTIFICATION,
            title="Valid Title",
            message="Valid message"
        )
        assert spec.title == "Valid Title"
        
        # Empty title should fail
        with pytest.raises(ValueError, match="Title must be 1-100 characters"):
            NotificationSpec(
                notification_type=NotificationType.NOTIFICATION,
                title="",
                message="Valid message"
            )
        
        # Too long title should fail
        with pytest.raises(ValueError, match="Title must be 1-100 characters"):
            NotificationSpec(
                notification_type=NotificationType.NOTIFICATION,
                title="x" * 101,
                message="Valid message"
            )
    
    def test_notification_spec_validation_message_length(self):
        """Test message length validation in notification spec."""
        # Empty message should fail
        with pytest.raises(ValueError, match="Message must be 1-500 characters"):
            NotificationSpec(
                notification_type=NotificationType.NOTIFICATION,
                title="Valid Title",
                message=""
            )
        
        # Too long message should fail
        with pytest.raises(ValueError, match="Message must be 1-500 characters"):
            NotificationSpec(
                notification_type=NotificationType.NOTIFICATION,
                title="Valid Title",
                message="x" * 501
            )
    
    def test_notification_spec_validation_duration(self):
        """Test duration validation in notification spec."""
        # Too short duration
        with pytest.raises(ValueError, match="Duration must be 0.1-60.0 seconds"):
            NotificationSpec(
                notification_type=NotificationType.HUD,
                title="Valid Title",
                message="Valid message",
                duration=0.05
            )
        
        # Too long duration
        with pytest.raises(ValueError, match="Duration must be 0.1-60.0 seconds"):
            NotificationSpec(
                notification_type=NotificationType.HUD,
                title="Valid Title", 
                message="Valid message",
                duration=61.0
            )
    
    def test_notification_spec_validation_buttons(self):
        """Test button validation in notification spec."""
        # Too many buttons
        with pytest.raises(ValueError, match="Maximum 3 buttons allowed"):
            NotificationSpec(
                notification_type=NotificationType.ALERT,
                title="Valid Title",
                message="Valid message",
                buttons=["Button1", "Button2", "Button3", "Button4"]
            )
    
    def test_notification_spec_sound_validation(self):
        """Test sound file validation."""
        # Valid system sound
        spec = NotificationSpec(
            notification_type=NotificationType.NOTIFICATION,
            title="Valid Title",
            message="Valid message",
            sound="default"
        )
        assert spec.sound == "default"
        
        # Invalid sound should fail
        with pytest.raises(ValueError, match="Invalid sound specification"):
            NotificationSpec(
                notification_type=NotificationType.NOTIFICATION,
                title="Valid Title",
                message="Valid message",
                sound="invalid_sound_file"
            )
    
    @pytest.mark.asyncio
    async def test_display_system_notification_success(self, notification_manager, mock_km_client):
        """Test successful system notification display."""
        mock_km_client.execute_applescript.return_value = Mock()
        mock_km_client.execute_applescript.return_value.is_left.return_value = False
        mock_km_client.execute_applescript.return_value.get_right.return_value = "Success"
        
        spec = NotificationSpec(
            notification_type=NotificationType.NOTIFICATION,
            title="Test Title",
            message="Test message"
        )
        
        result = await notification_manager.display_notification(spec)
        
        assert result.is_right()
        notification_result = result.get_right()
        assert notification_result.success
        assert notification_result.notification_id is not None
        assert notification_result.display_time > 0
    
    @pytest.mark.asyncio
    async def test_display_alert_dialog_success(self, notification_manager, mock_km_client):
        """Test successful alert dialog display."""
        mock_km_client.execute_applescript.return_value = Mock()
        mock_km_client.execute_applescript.return_value.is_left.return_value = False
        mock_km_client.execute_applescript.return_value.get_right.return_value = 'button returned:"OK"'
        
        spec = NotificationSpec(
            notification_type=NotificationType.ALERT,
            title="Test Alert",
            message="Test alert message",
            buttons=["OK", "Cancel"]
        )
        
        result = await notification_manager.display_notification(spec)
        
        assert result.is_right()
        notification_result = result.get_right()
        assert notification_result.success
        assert notification_result.user_response == "OK"
        assert notification_result.get_button_clicked() == "OK"
    
    @pytest.mark.asyncio
    async def test_display_hud_success(self, notification_manager, mock_km_client):
        """Test successful HUD display."""
        mock_km_client.display_hud_text.return_value = Mock()
        mock_km_client.display_hud_text.return_value.is_left.return_value = False
        mock_km_client.display_hud_text.return_value.get_right.return_value = "HUD displayed"
        
        spec = NotificationSpec(
            notification_type=NotificationType.HUD,
            title="HUD Title",
            message="HUD message",
            position=NotificationPosition.TOP_RIGHT,
            duration=2.0
        )
        
        result = await notification_manager.display_notification(spec)
        
        assert result.is_right()
        notification_result = result.get_right()
        assert notification_result.success
        assert notification_result.display_time >= 2.0  # Should wait for duration
    
    @pytest.mark.asyncio
    async def test_display_sound_notification_success(self, notification_manager, mock_km_client):
        """Test successful sound notification."""
        mock_km_client.play_sound.return_value = Mock()
        mock_km_client.play_sound.return_value.is_left.return_value = False
        mock_km_client.play_sound.return_value.get_right.return_value = "Sound played"
        
        spec = NotificationSpec(
            notification_type=NotificationType.SOUND,
            title="Sound Alert",
            message="Sound message",
            sound="default"
        )
        
        result = await notification_manager.display_notification(spec)
        
        assert result.is_right()
        notification_result = result.get_right()
        assert notification_result.success
        assert "sound_file" in notification_result.interaction_data
    
    def test_content_validation_security(self, notification_manager):
        """Test content validation for security threats."""
        # Script injection attempt
        assert not notification_manager._validate_notification_content('<script>alert("hack")</script>')
        
        # JavaScript protocol
        assert not notification_manager._validate_notification_content('javascript:alert("hack")')
        
        # Command substitution
        assert not notification_manager._validate_notification_content('$(rm -rf /)')
        
        # Eval attempt
        assert not notification_manager._validate_notification_content('eval(malicious_code)')
        
        # Valid content
        assert notification_manager._validate_notification_content('This is a safe notification message')
    
    def test_applescript_string_escaping(self, notification_manager):
        """Test AppleScript string escaping for safety."""
        # Test quote escaping
        escaped = notification_manager._escape_applescript_string('Test "quoted" text')
        assert escaped == 'Test \\"quoted\\" text'
        
        # Test backslash escaping
        escaped = notification_manager._escape_applescript_string('Path\\to\\file')
        assert escaped == 'Path\\\\to\\\\file'
        
        # Test combined escaping
        escaped = notification_manager._escape_applescript_string('Test "path\\file" text')
        assert escaped == 'Test \\"path\\\\file\\" text'
    
    def test_hud_position_mapping(self, notification_manager):
        """Test HUD position enum to KM value mapping."""
        position_tests = [
            (NotificationPosition.CENTER, "Center"),
            (NotificationPosition.TOP_LEFT, "TopLeft"),
            (NotificationPosition.BOTTOM_RIGHT, "BottomRight"),
        ]
        
        for position_enum, expected_value in position_tests:
            actual_value = notification_manager._get_hud_position_value(position_enum)
            assert actual_value == expected_value
    
    def test_active_notification_tracking(self, notification_manager):
        """Test active notification state management."""
        # Initially empty
        assert len(notification_manager.get_active_notifications()) == 0
        
        # Add mock notification to tracking
        notification_id = "test_notification_1"
        notification_manager._active_notifications[notification_id] = {
            "type": NotificationType.NOTIFICATION,
            "start_time": time.time(),
            "spec": NotificationSpec(
                notification_type=NotificationType.NOTIFICATION,
                title="Test",
                message="Test message"
            )
        }
        
        # Should be tracked
        active = notification_manager.get_active_notifications()
        assert len(active) == 1
        assert notification_id in active
        
        # Clear specific notification
        success = notification_manager.clear_notification(notification_id)
        assert success
        assert len(notification_manager.get_active_notifications()) == 0
        
        # Clear non-existent notification
        success = notification_manager.clear_notification("non_existent")
        assert not success
    
    def test_clear_all_notifications(self, notification_manager):
        """Test clearing all active notifications."""
        # Add multiple mock notifications
        for i in range(3):
            notification_id = f"test_notification_{i}"
            notification_manager._active_notifications[notification_id] = {
                "type": NotificationType.NOTIFICATION,
                "start_time": time.time(),
                "spec": NotificationSpec(
                    notification_type=NotificationType.NOTIFICATION,
                    title=f"Test {i}",
                    message=f"Test message {i}"
                )
            }
        
        assert len(notification_manager.get_active_notifications()) == 3
        
        # Clear all
        count = notification_manager.clear_all_notifications()
        assert count == 3
        assert len(notification_manager.get_active_notifications()) == 0


class TestNotificationMCPTools:
    """Test MCP tool implementations for notifications."""
    
    @pytest.mark.asyncio
    async def test_km_notifications_system_notification_success(self):
        """Test km_notifications tool with system notification."""
        with patch('src.server.tools.notification_tools.get_notification_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = Mock(
                success=True,
                notification_id="test_123",
                display_time=0.5,
                user_response=None,
                was_dismissed_by_user=Mock(return_value=False),
                get_button_clicked=Mock(return_value=None)
            )
            mock_manager.display_notification = AsyncMock(return_value=mock_result)
            mock_manager.get_active_notifications.return_value = {}
            mock_get_manager.return_value = mock_manager
            
            result = await km_notifications(
                notification_type="notification",
                title="Test Notification",
                message="This is a test notification",
                sound="default"
            )
            
            assert result["success"] is True
            assert result["data"]["notification_type"] == "notification"
            assert result["data"]["title"] == "Test Notification"
            assert result["data"]["message"] == "This is a test notification"
            assert result["data"]["sound_played"] is True
    
    @pytest.mark.asyncio
    async def test_km_notifications_alert_with_buttons(self):
        """Test km_notifications tool with alert dialog and buttons."""
        with patch('src.server.tools.notification_tools.get_notification_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = Mock(
                success=True,
                notification_id="alert_456",
                display_time=1.2,
                user_response="Cancel",
                was_dismissed_by_user=Mock(return_value=True),
                get_button_clicked=Mock(return_value="Cancel")
            )
            mock_manager.display_notification = AsyncMock(return_value=mock_result)
            mock_manager.get_active_notifications.return_value = {}
            mock_get_manager.return_value = mock_manager
            
            result = await km_notifications(
                notification_type="alert",
                title="Confirm Action",
                message="Are you sure you want to proceed?",
                buttons=["OK", "Cancel"],
                priority="high"
            )
            
            assert result["success"] is True
            assert result["data"]["notification_type"] == "alert"
            assert result["data"]["user_response"] == "Cancel"
            assert result["data"]["button_clicked"] == "Cancel"
            assert result["data"]["dismissed_by_user"] is True
    
    @pytest.mark.asyncio
    async def test_km_notifications_hud_with_position(self):
        """Test km_notifications tool with HUD display and custom position."""
        with patch('src.server.tools.notification_tools.get_notification_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = Mock(
                success=True,
                notification_id="hud_789",
                display_time=3.0,
                user_response=None,
                was_dismissed_by_user=Mock(return_value=False),
                get_button_clicked=Mock(return_value=None)
            )
            mock_manager.display_notification = AsyncMock(return_value=mock_result)
            mock_manager.get_active_notifications.return_value = {}
            mock_get_manager.return_value = mock_manager
            
            result = await km_notifications(
                notification_type="hud",
                title="Status Update",
                message="Operation completed successfully",
                duration=3.0,
                position="top_right"
            )
            
            assert result["success"] is True
            assert result["data"]["notification_type"] == "hud"
            assert result["data"]["duration"] == 3.0
            assert result["data"]["position"] == "top_right"
    
    @pytest.mark.asyncio
    async def test_km_notifications_validation_errors(self):
        """Test km_notifications tool input validation."""
        # Invalid notification type
        result = await km_notifications(
            notification_type="invalid_type",
            title="Test",
            message="Test message"
        )
        
        assert result["success"] is False
        assert "INVALID_TYPE" in result["error"]["code"]
        
        # Invalid priority
        result = await km_notifications(
            notification_type="notification",
            title="Test",
            message="Test message",
            priority="invalid_priority"
        )
        
        assert result["success"] is False
        assert "INVALID_PRIORITY" in result["error"]["code"]
        
        # Invalid position
        result = await km_notifications(
            notification_type="hud",
            title="Test",
            message="Test message",
            position="invalid_position"
        )
        
        assert result["success"] is False
        assert "INVALID_POSITION" in result["error"]["code"]
    
    @pytest.mark.asyncio
    async def test_km_notification_status_specific_notification(self):
        """Test km_notification_status tool for specific notification."""
        with patch('src.server.tools.notification_tools.get_notification_manager') as mock_get_manager:
            mock_manager = Mock()
            test_spec = NotificationSpec(
                notification_type=NotificationType.NOTIFICATION,
                title="Test Notification",
                message="Test message",
                priority=NotificationPriority.NORMAL
            )
            mock_manager.get_active_notifications.return_value = {
                "test_123": {
                    "spec": test_spec,
                    "start_time": time.time(),
                    "type": NotificationType.NOTIFICATION
                }
            }
            mock_get_manager.return_value = mock_manager
            
            result = await km_notification_status(notification_id="test_123")
            
            assert result["success"] is True
            assert result["data"]["notification_id"] == "test_123"
            assert result["data"]["type"] == "notification"
            assert result["data"]["title"] == "Test Notification"
            assert result["data"]["priority"] == "normal"
            assert result["data"]["active"] is True
    
    @pytest.mark.asyncio
    async def test_km_notification_status_all_notifications(self):
        """Test km_notification_status tool for all active notifications."""
        with patch('src.server.tools.notification_tools.get_notification_manager') as mock_get_manager:
            mock_manager = Mock()
            test_specs = [
                NotificationSpec(
                    notification_type=NotificationType.NOTIFICATION,
                    title=f"Test {i}",
                    message=f"Message {i}",
                    priority=NotificationPriority.NORMAL
                ) for i in range(3)
            ]
            mock_manager.get_active_notifications.return_value = {
                f"test_{i}": {
                    "spec": spec,
                    "start_time": time.time(),
                    "type": NotificationType.NOTIFICATION
                } for i, spec in enumerate(test_specs)
            }
            mock_get_manager.return_value = mock_manager
            
            result = await km_notification_status()
            
            assert result["success"] is True
            assert result["data"]["active_count"] == 3
            assert len(result["data"]["notifications"]) == 3
    
    @pytest.mark.asyncio
    async def test_km_notification_status_not_found(self):
        """Test km_notification_status tool with non-existent notification."""
        with patch('src.server.tools.notification_tools.get_notification_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_active_notifications.return_value = {}
            mock_get_manager.return_value = mock_manager
            
            result = await km_notification_status(notification_id="nonexistent")
            
            assert result["success"] is False
            assert "NOT_FOUND" in result["error"]["code"]
    
    @pytest.mark.asyncio
    async def test_km_dismiss_notifications_specific(self):
        """Test km_dismiss_notifications tool for specific notification."""
        with patch('src.server.tools.notification_tools.get_notification_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.clear_notification.return_value = True
            mock_get_manager.return_value = mock_manager
            
            result = await km_dismiss_notifications(notification_id="test_123")
            
            assert result["success"] is True
            assert result["data"]["notification_id"] == "test_123"
            assert result["data"]["dismissed"] is True
    
    @pytest.mark.asyncio
    async def test_km_dismiss_notifications_all(self):
        """Test km_dismiss_notifications tool for all notifications."""
        with patch('src.server.tools.notification_tools.get_notification_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.clear_all_notifications.return_value = 5
            mock_get_manager.return_value = mock_manager
            
            result = await km_dismiss_notifications()
            
            assert result["success"] is True
            assert result["data"]["dismissed_count"] == 5
            assert result["data"]["dismissed_all"] is True


class TestNotificationPropertyBasedTesting:
    """Property-based testing for notification system."""
    
    @composite
    def notification_spec_strategy(draw):
        """Generate valid notification specifications."""
        notification_type = draw(st.sampled_from(list(NotificationType)))
        title = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
        message = draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
        duration = draw(st.one_of(
            st.none(),
            st.floats(min_value=0.1, max_value=60.0)
        ))
        sound = draw(st.one_of(
            st.none(),
            st.sampled_from(["default", "glass", "hero", "ping"])
        ))
        buttons = draw(st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            min_size=0,
            max_size=3
        ))
        position = draw(st.sampled_from(list(NotificationPosition)))
        priority = draw(st.sampled_from(list(NotificationPriority)))
        dismissible = draw(st.booleans())
        
        return NotificationSpec(
            notification_type=notification_type,
            title=title.strip(),
            message=message.strip(),
            duration=duration,
            sound=sound,
            buttons=buttons,
            position=position,
            priority=priority,
            dismissible=dismissible
        )
    
    @given(notification_spec_strategy())
    @settings(max_examples=50)
    def test_notification_spec_creation_properties(self, spec):
        """Property: Valid specifications should always be createable."""
        # If we get here, the spec was created successfully
        assert spec.title is not None
        assert spec.message is not None
        assert spec.notification_type in NotificationType
        assert spec.position in NotificationPosition
        assert spec.priority in NotificationPriority
        
        # Check consistency properties
        if spec.requires_user_interaction():
            assert spec.notification_type == NotificationType.ALERT
            assert len(spec.buttons) > 0
        
        if spec.duration is not None:
            assert 0.1 <= spec.duration <= 60.0
    
    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_content_validation_properties(self, content):
        """Property: Content validation should be consistent and safe."""
        from src.notifications.notification_manager import NotificationManager
        
        # Create mock client for manager
        mock_client = Mock()
        manager = NotificationManager(mock_client)
        
        is_valid = manager._validate_notification_content(content)
        
        # Property: Empty content should be invalid
        if not content.strip():
            assert not is_valid
        
        # Property: Extremely long content should be invalid
        if len(content) > 1000:
            assert not is_valid
        
        # Property: Content with dangerous patterns should be invalid
        dangerous_patterns = ['<script', 'javascript:', 'eval(', 'exec(', 'system(']
        has_dangerous_pattern = any(pattern in content.lower() for pattern in dangerous_patterns)
        if has_dangerous_pattern:
            assert not is_valid
    
    @given(st.text(max_size=200))
    @settings(max_examples=50)
    def test_applescript_escaping_properties(self, text):
        """Property: AppleScript escaping should be safe and reversible."""
        from src.notifications.notification_manager import NotificationManager
        
        mock_client = Mock()
        manager = NotificationManager(mock_client)
        
        escaped = manager._escape_applescript_string(text)
        
        # Property: All quotes should be escaped
        quote_count_original = text.count('"')
        escaped_quote_count = escaped.count('\\"')
        assert escaped_quote_count >= quote_count_original
        
        # Property: All backslashes should be escaped
        backslash_count_original = text.count('\\')
        # In escaped string, original backslashes become \\\\
        # But we need to account for the escaping of quotes too
        assert '\\\\' in escaped or backslash_count_original == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])