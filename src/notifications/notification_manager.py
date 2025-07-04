"""
Comprehensive notification system for Keyboard Maestro MCP Tools.

This module implements a multi-channel notification system that provides user feedback
through system notifications, modal alerts, HUD displays, and sound notifications
with proper timing, user interaction tracking, and security validation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import asyncio
import os
import re
import time
import json
import logging

from ..core.types import Duration, Permission
from ..core.contracts import require, ensure
from ..core.errors import MacroEngineError
from ..core.either import Either
from ..integration.km_client import KMClient


logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Supported notification types with comprehensive display options."""
    NOTIFICATION = "notification"  # System notification center
    ALERT = "alert"                # Modal dialog with interaction
    HUD = "hud"                    # Heads-up display overlay
    SOUND = "sound"                # Audio notification only


class NotificationPosition(Enum):
    """HUD display positions on screen."""
    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass(frozen=True)
class NotificationResult:
    """Result of notification display operation."""
    success: bool
    notification_id: str
    display_time: float
    user_response: Optional[str] = None
    interaction_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def was_dismissed_by_user(self) -> bool:
        """Check if notification was dismissed by user interaction."""
        return self.user_response is not None
    
    def get_button_clicked(self) -> Optional[str]:
        """Get the button clicked for alert notifications."""
        return self.interaction_data.get("button_clicked")


@dataclass(frozen=True)
class NotificationSpec:
    """Type-safe notification specification with comprehensive validation."""
    notification_type: NotificationType
    title: str
    message: str
    duration: Optional[float] = None
    sound: Optional[str] = None
    icon: Optional[str] = None
    buttons: List[str] = field(default_factory=list)
    position: NotificationPosition = NotificationPosition.CENTER
    priority: NotificationPriority = NotificationPriority.NORMAL
    dismissible: bool = True
    
    def __post_init__(self):
        """Validate notification specification."""
        # Title validation
        if not self.title or len(self.title) > 100:
            raise ValueError(f"Title must be 1-100 characters, got {len(self.title)}")
        
        # Message validation
        if not self.message or len(self.message) > 500:
            raise ValueError(f"Message must be 1-500 characters, got {len(self.message)}")
        
        # Duration validation
        if self.duration is not None:
            if self.duration < 0.1 or self.duration > 60.0:
                raise ValueError(f"Duration must be 0.1-60.0 seconds, got {self.duration}")
        
        # Button validation
        if len(self.buttons) > 3:
            raise ValueError(f"Maximum 3 buttons allowed, got {len(self.buttons)}")
        
        # Sound file validation
        if self.sound and not self._is_valid_sound(self.sound):
            raise ValueError(f"Invalid sound specification: {self.sound}")
    
    def _is_valid_sound(self, sound: str) -> bool:
        """Validate sound file or system sound name."""
        # System sounds (common macOS sounds)
        system_sounds = {
            "default", "glass", "hero", "morse", "ping", "pop", "purr",
            "sosumi", "submarine", "tink", "bottle", "basso", "blow",
            "frog", "funk", "temple"
        }
        
        if sound.lower() in system_sounds:
            return True
        
        # File path validation (basic check)
        if sound.startswith("/") and sound.endswith((".aiff", ".wav", ".mp3", ".m4a")):
            return True
        
        return False
    
    def is_dismissible(self) -> bool:
        """Check if notification can be dismissed by user."""
        return self.notification_type in [NotificationType.NOTIFICATION, NotificationType.HUD]
    
    def requires_user_interaction(self) -> bool:
        """Check if notification requires user interaction."""
        return self.notification_type == NotificationType.ALERT and bool(self.buttons)


class NotificationManager:
    """
    Manage user notifications with multiple display channels.
    
    Provides comprehensive notification capabilities including:
    - System notifications through macOS Notification Center
    - Modal alert dialogs with user interaction
    - HUD overlays with positioning control
    - Sound notifications with system and custom audio
    """
    
    def __init__(self, km_client: KMClient):
        self.km_client = km_client
        self._active_notifications: Dict[str, Dict[str, Any]] = {}
        self._notification_counter = 0
    
    def _generate_notification_id(self) -> str:
        """Generate unique notification ID."""
        self._notification_counter += 1
        return f"notification_{self._notification_counter}_{int(time.time())}"
    
    @require(lambda self, spec: isinstance(spec, NotificationSpec))
    @ensure(lambda result: isinstance(result, Either))
    async def display_notification(self, spec: NotificationSpec) -> Either[MacroEngineError, NotificationResult]:
        """
        Display notification with comprehensive validation and error handling.
        
        Args:
            spec: Complete notification specification
            
        Returns:
            Either notification result or error details
        """
        try:
            # Validate content safety
            if not self._validate_notification_content(spec.title) or \
               not self._validate_notification_content(spec.message):
                return Either.left(MacroEngineError(
                    code="CONTENT_VALIDATION_ERROR",
                    message="Notification content failed safety validation",
                    details={"title_length": len(spec.title), "message_length": len(spec.message)}
                ))
            
            # Route to appropriate display method
            if spec.notification_type == NotificationType.NOTIFICATION:
                return await self._display_system_notification(spec)
            elif spec.notification_type == NotificationType.ALERT:
                return await self._display_alert_dialog(spec)
            elif spec.notification_type == NotificationType.HUD:
                return await self._display_hud(spec)
            elif spec.notification_type == NotificationType.SOUND:
                return await self._display_sound_notification(spec)
            else:
                return Either.left(MacroEngineError(
                    code="INVALID_NOTIFICATION_TYPE",
                    message=f"Unsupported notification type: {spec.notification_type}",
                    details={"type": spec.notification_type.value}
                ))
        
        except Exception as e:
            logger.error(f"Failed to display notification: {e}")
            return Either.left(MacroEngineError(
                code="DISPLAY_ERROR",
                message=f"Notification display failed: {str(e)}",
                details={"error_type": type(e).__name__}
            ))
    
    async def _display_system_notification(self, spec: NotificationSpec) -> Either[MacroEngineError, NotificationResult]:
        """Display macOS system notification."""
        notification_id = self._generate_notification_id()
        start_time = time.time()
        
        try:
            # Build AppleScript for system notification
            script_parts = [
                f'display notification "{self._escape_applescript_string(spec.message)}"',
                f'with title "{self._escape_applescript_string(spec.title)}"'
            ]
            
            if spec.sound:
                if spec.sound.lower() in ["default", "glass", "hero", "morse"]:
                    script_parts.append(f'sound name "{spec.sound}"')
                elif os.path.exists(spec.sound):
                    script_parts.append(f'sound name (POSIX file "{spec.sound}")')
            
            applescript = " ".join(script_parts)
            
            # Execute through KM client
            result = await self.km_client.execute_applescript(applescript)
            
            if result.is_left():
                return Either.left(result.get_left())
            
            display_time = time.time() - start_time
            
            # Track active notification
            self._active_notifications[notification_id] = {
                "type": NotificationType.NOTIFICATION,
                "start_time": start_time,
                "spec": spec
            }
            
            return Either.right(NotificationResult(
                success=True,
                notification_id=notification_id,
                display_time=display_time,
                interaction_data={"applescript_result": result.get_right()}
            ))
        
        except Exception as e:
            return Either.left(MacroEngineError(
                code="SYSTEM_NOTIFICATION_ERROR",
                message=f"System notification failed: {str(e)}",
                details={"notification_id": notification_id}
            ))
    
    async def _display_alert_dialog(self, spec: NotificationSpec) -> Either[MacroEngineError, NotificationResult]:
        """Display modal alert dialog with user interaction."""
        notification_id = self._generate_notification_id()
        start_time = time.time()
        
        try:
            # Build AppleScript for alert dialog
            if spec.buttons:
                buttons_str = "{" + ", ".join(f'"{btn}"' for btn in spec.buttons) + "}"
                script = f'''
                display alert "{self._escape_applescript_string(spec.title)}" ¬
                message "{self._escape_applescript_string(spec.message)}" ¬
                buttons {buttons_str} ¬
                default button 1
                '''
            else:
                script = f'''
                display alert "{self._escape_applescript_string(spec.title)}" ¬
                message "{self._escape_applescript_string(spec.message)}"
                '''
            
            # Execute through KM client
            result = await self.km_client.execute_applescript(script)
            
            if result.is_left():
                return Either.left(result.get_left())
            
            display_time = time.time() - start_time
            
            # Parse user response
            user_response = None
            button_clicked = None
            applescript_result = result.get_right()
            
            if "button returned:" in applescript_result:
                button_clicked = applescript_result.split("button returned:")[1].strip().strip('"')
                user_response = button_clicked
            
            return Either.right(NotificationResult(
                success=True,
                notification_id=notification_id,
                display_time=display_time,
                user_response=user_response,
                interaction_data={
                    "button_clicked": button_clicked,
                    "applescript_result": applescript_result
                }
            ))
        
        except Exception as e:
            return Either.left(MacroEngineError(
                code="ALERT_DIALOG_ERROR",
                message=f"Alert dialog failed: {str(e)}",
                details={"notification_id": notification_id}
            ))
    
    async def _display_hud(self, spec: NotificationSpec) -> Either[MacroEngineError, NotificationResult]:
        """Display heads-up display overlay."""
        notification_id = self._generate_notification_id()
        start_time = time.time()
        
        try:
            # Use Keyboard Maestro's HUD display action
            duration = spec.duration or 3.0
            
            # Create HUD display through KM client
            hud_result = await self.km_client.display_hud_text(
                text=f"{spec.title}\n{spec.message}",
                duration=duration,
                position=self._get_hud_position_value(spec.position)
            )
            
            if hud_result.is_left():
                return Either.left(hud_result.get_left())
            
            # Wait for display duration
            if duration > 0:
                await asyncio.sleep(min(duration, 10.0))  # Cap at 10 seconds
            
            display_time = time.time() - start_time
            
            return Either.right(NotificationResult(
                success=True,
                notification_id=notification_id,
                display_time=display_time,
                interaction_data={
                    "position": spec.position.value,
                    "duration": duration
                }
            ))
        
        except Exception as e:
            return Either.left(MacroEngineError(
                code="HUD_DISPLAY_ERROR", 
                message=f"HUD display failed: {str(e)}",
                details={"notification_id": notification_id}
            ))
    
    async def _display_sound_notification(self, spec: NotificationSpec) -> Either[MacroEngineError, NotificationResult]:
        """Display sound notification."""
        notification_id = self._generate_notification_id()
        start_time = time.time()
        
        try:
            sound_file = spec.sound or "default"
            
            # Play sound through KM client
            sound_result = await self.km_client.play_sound(sound_file)
            
            if sound_result.is_left():
                return Either.left(sound_result.get_left())
            
            display_time = time.time() - start_time
            
            return Either.right(NotificationResult(
                success=True,
                notification_id=notification_id,
                display_time=display_time,
                interaction_data={"sound_file": sound_file}
            ))
        
        except Exception as e:
            return Either.left(MacroEngineError(
                code="SOUND_NOTIFICATION_ERROR",
                message=f"Sound notification failed: {str(e)}",
                details={"notification_id": notification_id}
            ))
    
    def _validate_notification_content(self, content: str) -> bool:
        """
        Validate notification content for safety and appropriateness.
        
        Args:
            content: Text content to validate
            
        Returns:
            True if content is safe, False otherwise
        """
        if not content or len(content.strip()) == 0:
            return False
        
        # Length limits
        if len(content) > 1000:  # Extended limit for safety check
            return False
        
        # Basic safety patterns (prevent script injection)
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'`[^`]*`',  # Command substitution
            r'\$\([^)]*\)',  # Command substitution
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning(f"Potentially dangerous content detected: {pattern}")
                return False
        
        return True
    
    def _escape_applescript_string(self, text: str) -> str:
        """Escape string for safe AppleScript usage."""
        # Escape quotes and backslashes
        escaped = text.replace('\\', '\\\\').replace('"', '\\"')
        return escaped
    
    def _get_hud_position_value(self, position: NotificationPosition) -> str:
        """Convert position enum to KM HUD position value."""
        position_map = {
            NotificationPosition.CENTER: "Center",
            NotificationPosition.TOP: "Top",
            NotificationPosition.BOTTOM: "Bottom",
            NotificationPosition.LEFT: "Left",
            NotificationPosition.RIGHT: "Right",
            NotificationPosition.TOP_LEFT: "TopLeft",
            NotificationPosition.TOP_RIGHT: "TopRight",
            NotificationPosition.BOTTOM_LEFT: "BottomLeft",
            NotificationPosition.BOTTOM_RIGHT: "BottomRight"
        }
        return position_map.get(position, "Center")
    
    def get_active_notifications(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active notifications."""
        return self._active_notifications.copy()
    
    def clear_notification(self, notification_id: str) -> bool:
        """Clear specific notification from active tracking."""
        return self._active_notifications.pop(notification_id, None) is not None
    
    def clear_all_notifications(self) -> int:
        """Clear all active notifications."""
        count = len(self._active_notifications)
        self._active_notifications.clear()
        return count