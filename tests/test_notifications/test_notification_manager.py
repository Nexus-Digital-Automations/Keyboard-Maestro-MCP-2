"""Property-based tests for the notification system.

import logging

logging.basicConfig(level=logging.DEBUG)
Tests comprehensive notification capabilities including system notifications,
alerts, HUD displays, and sound notifications with security validation and
user interaction tracking.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from src.core.errors import ErrorCategory, MacroEngineError
from src.notifications.notification_manager import (
    NotificationManager,
    NotificationPosition,
    NotificationResult,
    NotificationSpec,
    NotificationType,
)


class TestNotificationManager:
    """Test suite for NotificationManager with comprehensive validation."""

    @pytest.fixture
    def mock_km_client(self) -> Mock:
        """Mock KM client for testing."""
        km_client = Mock()
        km_client.execute_applescript = AsyncMock()
        km_client.display_hud_text = AsyncMock()
        km_client.play_sound = AsyncMock()
        return km_client

    @pytest.fixture
    def notification_manager(self, mock_km_client: Any) -> NotificationManager:
        """NotificationManager instance with mocked dependencies."""
        return NotificationManager(mock_km_client)

    @given(
        title=st.text(min_size=1, max_size=100),
        message=st.text(min_size=1, max_size=500),
        duration=st.floats(min_value=0.1, max_value=60.0) | st.none(),
    )
    async def test_notification_spec_creation_properties(
        self,
        title: str,
        message: str,
        duration: float,
    ) -> None:
        """Property test: NotificationSpec creation with valid inputs."""
        assume(len(title.strip()) > 0 and len(message.strip()) > 0)

        spec = NotificationSpec(
            notification_type=NotificationType.NOTIFICATION,
            title=title,
            message=message,
            duration=duration,
        )

        assert spec.title == title
        assert spec.message == message
        assert spec.duration == duration
        assert spec.notification_type == NotificationType.NOTIFICATION

    @given(
        title=st.text(min_size=101) | st.text(max_size=0),
        message=st.text(min_size=501) | st.text(max_size=0),
    )
    def test_notification_spec_validation_failures(
        self,
        title: str,
        message: str,
    ) -> None:
        """Property test: NotificationSpec validation with invalid inputs."""
        with pytest.raises(ValueError):
            NotificationSpec(
                notification_type=NotificationType.NOTIFICATION,
                title=title,
                message=message,
            )

    @pytest.mark.asyncio
    async def test_system_notification_display(
        self,
        notification_manager: Any,
        mock_km_client: Any,
    ) -> None:
        """Test system notification display with AppleScript execution."""
        # Setup mock response
        mock_km_client.execute_applescript.return_value = Mock(
            is_left=lambda: False,
            get_right=lambda: "notification displayed successfully",
        )

        spec = NotificationSpec(
            notification_type=NotificationType.NOTIFICATION,
            title="Test Title",
            message="Test message",
            sound="default",
        )

        result = await notification_manager.display_notification(spec)

        assert result.is_right()
        notification_result = result.get_right()
        assert notification_result.success
        assert notification_result.notification_id.startswith("notification_")
        assert notification_result.display_time > 0

        # Verify AppleScript was called
        mock_km_client.execute_applescript.assert_called_once()

    @pytest.mark.asyncio
    async def test_alert_dialog_with_buttons(
        self,
        notification_manager: Any,
        mock_km_client: Any,
    ) -> None:
        """Test alert dialog with user interaction buttons."""
        # Setup mock response with button click
        mock_km_client.execute_applescript.return_value = Mock(
            is_left=lambda: False,
            get_right=lambda: 'button returned:"Yes"',
        )

        spec = NotificationSpec(
            notification_type=NotificationType.ALERT,
            title="Confirmation",
            message="Do you want to proceed?",
            buttons=["Yes", "No"],
        )

        result = await notification_manager.display_notification(spec)

        assert result.is_right()
        notification_result = result.get_right()
        assert notification_result.success
        assert notification_result.get_button_clicked() == "Yes"
        assert notification_result.was_dismissed_by_user()

    @pytest.mark.asyncio
    async def test_hud_display_with_positioning(
        self,
        notification_manager: Any,
        mock_km_client: Any,
    ) -> None:
        """Test HUD display with custom positioning and duration."""
        # Setup mock response
        mock_km_client.display_hud_text.return_value = Mock(
            is_left=lambda: False,
            get_right=lambda: "hud displayed",
        )

        spec = NotificationSpec(
            notification_type=NotificationType.HUD,
            title="Processing",
            message="Please wait...",
            duration=2.0,
            position=NotificationPosition.TOP,
        )

        with patch("asyncio.sleep") as mock_sleep:
            result = await notification_manager.display_notification(spec)

        assert result.is_right()
        notification_result = result.get_right()
        assert notification_result.success

        # Verify HUD display was called
        mock_km_client.display_hud_text.assert_called_once_with(
            text="Processing\nPlease wait...",
            duration=2.0,
            position="Top",
        )

        # Verify sleep was called for duration
        mock_sleep.assert_called_once_with(2.0)

    @pytest.mark.asyncio
    async def test_sound_notification(
        self,
        notification_manager: Any,
        mock_km_client: Any,
    ) -> None:
        """Test sound notification with custom audio file."""
        # Setup mock response
        mock_km_client.play_sound.return_value = Mock(
            is_left=lambda: False,
            get_right=lambda: "sound played",
        )

        spec = NotificationSpec(
            notification_type=NotificationType.SOUND,
            title="Alert",
            message="System backup completed",
            sound="/System/Library/Sounds/Glass.aiff",
        )

        result = await notification_manager.display_notification(spec)

        assert result.is_right()
        notification_result = result.get_right()
        assert notification_result.success

        # Verify sound play was called
        mock_km_client.play_sound.assert_called_once_with(
            "/System/Library/Sounds/Glass.aiff",
        )

    @given(
        content=st.text().filter(
            lambda x: any(
                pattern in x.lower()
                for pattern in [
                    "<script",
                    "javascript:",
                    "eval(",
                    "system(",
                    "`",
                    "$(",
                ]
            ),
        ),
    )
    @settings(
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.filter_too_much,
        ]
    )
    def test_content_validation_security(
        self,
        notification_manager: Any,
        content: str,
    ) -> None:
        """Property test: Content validation prevents dangerous patterns."""
        is_valid = notification_manager._validate_notification_content(content)

        # Content with dangerous patterns should be rejected
        assert not is_valid

    @given(
        content=st.text(min_size=1, max_size=500).filter(
            lambda x: not any(
                pattern in x.lower()
                for pattern in ["<script", "javascript:", "eval(", "system(", "`", "$("]
            ),
        ),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_content_validation_safe_content(
        self,
        notification_manager: Any,
        content: str,
    ) -> None:
        """Property test: Safe content passes validation."""
        assume(len(content.strip()) > 0)

        is_valid = notification_manager._validate_notification_content(content)
        assert is_valid

    def test_applescript_string_escaping(self, notification_manager: Any) -> None:
        """Test AppleScript string escaping for security."""
        dangerous_text = 'Text with "quotes" and \\backslashes'
        escaped = notification_manager._escape_applescript_string(dangerous_text)

        assert '\\"' in escaped  # Quotes should be escaped
        assert "\\\\" in escaped  # Backslashes should be escaped

    def test_hud_position_mapping(self, notification_manager: Any) -> None:
        """Test HUD position enum to KM value mapping."""
        position_tests = [
            (NotificationPosition.CENTER, "Center"),
            (NotificationPosition.TOP_LEFT, "TopLeft"),
            (NotificationPosition.BOTTOM_RIGHT, "BottomRight"),
        ]

        for position, expected in position_tests:
            km_value = notification_manager._get_hud_position_value(position)
            assert km_value == expected

    @pytest.mark.asyncio
    async def test_notification_tracking(
        self,
        notification_manager: Any,
        mock_km_client: Any,
    ) -> None:
        """Test active notification tracking and management."""
        # Setup mock response
        mock_km_client.execute_applescript.return_value = Mock(
            is_left=lambda: False,
            get_right=lambda: "notification displayed",
        )

        spec = NotificationSpec(
            notification_type=NotificationType.NOTIFICATION,
            title="Test",
            message="Test message",
        )

        # Display notification
        result = await notification_manager.display_notification(spec)
        notification_result = result.get_right()
        notification_id = notification_result.notification_id

        # Check active notifications
        active = notification_manager.get_active_notifications()
        assert notification_id in active
        assert active[notification_id]["spec"] == spec

        # Clear specific notification
        cleared = notification_manager.clear_notification(notification_id)
        assert cleared

        # Verify it's removed
        active_after = notification_manager.get_active_notifications()
        assert notification_id not in active_after

    @pytest.mark.asyncio
    async def test_error_handling_km_client_failure(
        self,
        notification_manager: Any,
        mock_km_client: Any,
    ) -> None:
        """Test error handling when KM client operations fail."""
        # Setup mock to return error
        error = MacroEngineError(
            message="AppleScript execution failed",
            category=ErrorCategory.EXECUTION,
        )
        mock_km_client.execute_applescript.return_value = Mock(
            is_left=lambda: True,
            get_left=lambda: error,
        )

        spec = NotificationSpec(
            notification_type=NotificationType.NOTIFICATION,
            title="Test",
            message="Test message",
        )

        result = await notification_manager.display_notification(spec)

        assert result.is_left()
        returned_error = result.get_left()
        # The error is passed through from KM client
        assert returned_error.message == "AppleScript execution failed"
        assert returned_error.category == ErrorCategory.EXECUTION

    @given(sound_name=st.text(min_size=1))
    def test_sound_validation_properties(self, sound_name: str) -> None:
        """Property test: Sound validation for various inputs."""
        # Try to create spec with the sound - it will fail if invalid
        try:
            spec = NotificationSpec(
                notification_type=NotificationType.SOUND,
                title="Test",
                message="Test",
                sound=sound_name,
            )
            # If we got here, the sound was valid
            is_valid = True
            # Verify the spec was created with the correct sound
            assert spec.sound == sound_name
        except ValueError as e:
            # Sound was invalid
            is_valid = False
            # Make sure the error message is appropriate
            assert "Invalid sound specification" in str(e)

        # System sounds should be valid
        system_sounds = {
            "default",
            "glass",
            "hero",
            "morse",
            "ping",
            "pop",
            "purr",
            "sosumi",
            "submarine",
            "tink",
            "bottle",
            "basso",
            "blow",
            "frog",
            "funk",
            "temple",
        }

        if sound_name.lower() in system_sounds:
            assert is_valid
        elif sound_name.startswith("/") and any(
            sound_name.endswith(ext) for ext in [".aiff", ".wav", ".mp3", ".m4a"]
        ):
            assert is_valid
        else:
            # Other patterns may or may not be valid depending on implementation
            pass

    def test_notification_result_helper_methods(self) -> None:
        """Test NotificationResult helper methods."""
        # Test with user response
        result_with_response = NotificationResult(
            success=True,
            notification_id="test_1",
            display_time=1.5,
            user_response="Yes",
            interaction_data={"button_clicked": "Yes"},
        )

        assert result_with_response.was_dismissed_by_user()
        assert result_with_response.get_button_clicked() == "Yes"

        # Test without user response
        result_no_response = NotificationResult(
            success=True,
            notification_id="test_2",
            display_time=1.0,
        )

        assert not result_no_response.was_dismissed_by_user()
        assert result_no_response.get_button_clicked() is None


# Integration tests for notification workflow
class TestNotificationIntegration:
    """Integration tests for complete notification workflows."""

    @pytest.mark.asyncio
    async def test_complete_notification_workflow(self) -> None:
        """Test complete notification workflow from creation to cleanup."""
        with patch("src.integration.km_client.KMClient") as mock_client_class:
            # Setup mock client
            mock_client = Mock()
            mock_client.execute_applescript = AsyncMock(
                return_value=Mock(is_left=lambda: False, get_right=lambda: "success"),
            )
            mock_client_class.return_value = mock_client

            # Create manager
            manager = NotificationManager(mock_client)

            # Create and display notification
            spec = NotificationSpec(
                notification_type=NotificationType.NOTIFICATION,
                title="Workflow Test",
                message="Testing complete workflow",
            )

            result = await manager.display_notification(spec)

            # Verify success
            assert result.is_right()
            notification_result = result.get_right()
            assert notification_result.success

            # Verify tracking
            active = manager.get_active_notifications()
            assert len(active) == 1

            # Clear all notifications
            cleared_count = manager.clear_all_notifications()
            assert cleared_count == 1

            # Verify cleanup
            active_after = manager.get_active_notifications()
            assert len(active_after) == 0
