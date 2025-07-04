"""
Notification MCP Tools

Provides comprehensive user feedback capabilities through multiple notification channels
with validation, user interaction tracking, and platform integration.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import Context
from pydantic import Field
from typing_extensions import Annotated

from ...notifications.notification_manager import (
    NotificationManager, NotificationSpec, NotificationType, 
    NotificationPosition, NotificationPriority
)
from ...core.errors import ValidationError, SecurityViolationError

logger = logging.getLogger(__name__)

# Global notification manager instance
_notification_manager = None


def get_notification_manager() -> NotificationManager:
    """Get or create global notification manager instance."""
    global _notification_manager
    if _notification_manager is None:
        from ...server.initialization import get_km_client
        km_client = get_km_client()
        _notification_manager = NotificationManager(km_client)
    return _notification_manager


async def km_notifications(
    notification_type: Annotated[str, Field(
        description="Notification type: notification, alert, hud, sound",
        pattern=r"^(notification|alert|hud|sound)$"
    )],
    title: Annotated[str, Field(
        description="Notification title (1-100 characters)",
        min_length=1,
        max_length=100
    )],
    message: Annotated[str, Field(
        description="Notification message content (1-500 characters)",
        min_length=1,
        max_length=500
    )],
    sound: Annotated[Optional[str], Field(
        default=None,
        description="Sound name (system sound) or file path",
        max_length=255
    )] = None,
    duration: Annotated[Optional[float], Field(
        default=None,
        description="Display duration in seconds (0.1-60.0)",
        ge=0.1,
        le=60.0
    )] = None,
    buttons: Annotated[List[str], Field(
        default_factory=list,
        description="Button labels for alert dialogs (max 3)",
        max_items=3
    )] = [],
    position: Annotated[str, Field(
        default="center",
        description="HUD position: center, top, bottom, left, right, top_left, top_right, bottom_left, bottom_right",
        pattern=r"^(center|top|bottom|left|right|top_left|top_right|bottom_left|bottom_right)$"
    )] = "center",
    priority: Annotated[str, Field(
        default="normal",
        description="Notification priority: low, normal, high, urgent",
        pattern=r"^(low|normal|high|urgent)$"
    )] = "normal",
    dismissible: Annotated[bool, Field(
        default=True,
        description="Whether notification can be dismissed by user"
    )] = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Display user notifications with comprehensive formatting and interaction support.
    
    Architecture:
    - Pattern: Command Pattern with State Tracking and User Interaction
    - Security: Content validation, injection prevention, safe AppleScript execution
    - Performance: O(1) display with sound file caching and validation
    
    Notification Types:
    - notification: macOS Notification Center with optional sound and icon
    - alert: Modal dialog with customizable buttons and user interaction tracking
    - hud: On-screen heads-up display with positioning and duration control
    - sound: Audio notification with system sounds or custom audio files
    
    Features:
    - Content validation and length limits for safety
    - User interaction tracking for alerts with button response capture
    - Customizable timing and positioning for optimal user experience
    - Sound integration with system sounds and custom audio file support
    - Rich formatting support with title, message, and icon display
    - Priority-based notification ordering and management
    
    Security Features:
    - Input sanitization and content validation against injection attacks
    - Safe AppleScript string escaping and parameterization
    - Sound file validation and access checking
    - Character encoding validation for cross-platform compatibility
    - Length limits and pattern validation for all user inputs
    
    Performance Features:
    - Sound file existence caching for repeated notifications
    - Async execution with timeout protection for hanging operations
    - Lightweight state tracking with minimal memory footprint
    - Efficient AppleScript generation with optimized parameter passing
    
    Args:
        notification_type: Type of notification (notification, alert, hud, sound)
        title: Notification title (displayed prominently)
        message: Main notification message content
        sound: System sound name or custom audio file path
        duration: Display duration in seconds (auto-calculated if not provided)
        buttons: Button labels for alert dialogs (up to 3 buttons)
        position: HUD position on screen for overlay notifications
        priority: Notification priority for ordering and urgency
        dismissible: Whether user can manually dismiss the notification
        ctx: MCP context for progress reporting and logging
        
    Returns:
        Dictionary containing:
        - success: Boolean indicating notification display success
        - data: Notification details including ID, timing, and user interactions
        - error: Error information with recovery suggestions on failure
        - metadata: Performance metrics, execution time, and system information
        
    Raises:
        ValidationError: Input validation failed
        SecurityViolationError: Content security validation failed
    """
    correlation_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        if ctx:
            await ctx.info(f"Displaying {notification_type} notification: '{title}'")
        
        logger.info(f"Notification request: {notification_type} - '{title}' [correlation_id: {correlation_id}]")
        
        # Phase 1: Input validation and sanitization
        try:
            # Parse and validate notification type
            try:
                notif_type = NotificationType(notification_type.lower())
            except ValueError:
                return _create_error_response(
                    correlation_id, "INVALID_TYPE", 
                    f"Invalid notification type: {notification_type}",
                    "Supported types: notification, alert, hud, sound",
                    (datetime.now() - start_time).total_seconds()
                )
            
            # Parse and validate priority
            try:
                notif_priority = NotificationPriority(priority.lower())
            except ValueError:
                return _create_error_response(
                    correlation_id, "INVALID_PRIORITY",
                    f"Invalid priority: {priority}",
                    "Supported priorities: low, normal, high, urgent",
                    (datetime.now() - start_time).total_seconds()
                )
            
            # Parse and validate HUD position
            try:
                hud_pos = NotificationPosition(position.lower())
            except ValueError:
                return _create_error_response(
                    correlation_id, "INVALID_POSITION",
                    f"Invalid HUD position: {position}",
                    "Supported positions: center, top, bottom, left, right, top_left, top_right, bottom_left, bottom_right",
                    (datetime.now() - start_time).total_seconds()
                )
            
            # Validate button requirements for alert type
            if notif_type == NotificationType.ALERT and not buttons:
                buttons = ["OK"]  # Default button for alerts
            
            # Validate sound requirements for sound type
            if notif_type == NotificationType.SOUND and not sound:
                sound = "default"  # Default system sound
            
        except Exception as e:
            logger.warning(f"Input validation failed: {e} [correlation_id: {correlation_id}]")
            return _create_error_response(
                correlation_id, "VALIDATION_ERROR", str(e),
                f"Input validation failed: {e}",
                "Review input parameters and ensure they meet validation requirements",
                (datetime.now() - start_time).total_seconds()
            )
        
        if ctx:
            await ctx.report_progress(25, 100, "Creating notification specification")
        
        # Phase 2: Create notification specification
        try:
            notification_spec = NotificationSpec(
                notification_type=notif_type,
                title=title.strip(),
                message=message.strip(),
                duration=duration,
                sound=sound,
                buttons=buttons,
                priority=notif_priority,
                position=hud_pos,
                dismissible=dismissible
            )
        except ValueError as e:
            logger.warning(f"Notification spec creation failed: {e} [correlation_id: {correlation_id}]")
            return _create_error_response(
                correlation_id, "SPEC_ERROR", str(e),
                f"Notification specification error: {e}",
                "Check input values for length and format requirements",
                (datetime.now() - start_time).total_seconds()
            )
        
        if ctx:
            await ctx.report_progress(50, 100, f"Displaying {notification_type} notification")
        
        # Phase 3: Display notification
        notification_manager = get_notification_manager()
        result = await notification_manager.display_notification(notification_spec)
        
        if ctx:
            await ctx.report_progress(90, 100, "Processing notification result")
        
        # Phase 4: Process and return result
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if result.is_left():
            error = result.get_left()
            logger.error(f"Notification display failed: {error.message} [correlation_id: {correlation_id}]")
            
            if ctx:
                await ctx.error(f"Notification failed: {error.message}")
            
            return _create_error_response(
                correlation_id, error.code, error.message,
                f"Notification display failed: {error.message}",
                "Check system permissions and notification settings",
                execution_time
            )
        
        notification_result = result.get_right()
        
        if ctx:
            await ctx.report_progress(100, 100, "Notification displayed successfully")
            await ctx.info(f"Successfully displayed {notification_type} notification")
        
        logger.info(f"Notification success: {notification_type} - '{title}' [correlation_id: {correlation_id}]")
        
        return {
            "success": True,
            "data": {
                "notification_id": notification_result.notification_id,
                "notification_type": notification_type,
                "title": title,
                "message": message,
                "display_time": notification_result.display_time,
                "user_response": notification_result.user_response,
                "button_clicked": notification_result.get_button_clicked(),
                "dismissed_by_user": notification_result.was_dismissed_by_user(),
                "duration": duration,
                "sound_played": sound is not None,
                "priority": priority,
                "position": position if notif_type == NotificationType.HUD else None
            },
            "metadata": {
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat(),
                "server_version": "1.0.0",
                "execution_time": execution_time,
                "operation": "display_notification",
                "notification_manager_active_count": len(notification_manager.get_active_notifications())
            }
        }
        
    except ValidationError as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.warning(f"Validation error for notification '{title}': {e} [correlation_id: {correlation_id}]")
        
        if ctx:
            await ctx.error(f"Validation failed: {e}")
        
        return _create_error_response(
            correlation_id, "VALIDATION_ERROR", str(e),
            f"Input validation failed: {e}",
            "Review input parameters and ensure they meet validation requirements",
            execution_time
        )
        
    except SecurityViolationError as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Security violation for notification '{title}': {e} [correlation_id: {correlation_id}]")
        
        if ctx:
            await ctx.error(f"Security violation: {e}")
        
        return _create_error_response(
            correlation_id, "SECURITY_VIOLATION", "Security validation failed",
            f"Security requirements not met: {e}",
            "Ensure all inputs meet security requirements and try again",
            execution_time
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.exception(f"Unexpected error in notification '{title}' [correlation_id: {correlation_id}]")
        
        if ctx:
            await ctx.error(f"Unexpected error: {str(e)}")
        
        return _create_error_response(
            correlation_id, "SYSTEM_ERROR", "Unexpected system error",
            str(e),
            "Check system status and try again. Contact support if problem persists.",
            execution_time
        )


async def km_notification_status(
    notification_id: Annotated[Optional[str], Field(
        default=None,
        description="Notification ID to check status for (optional)"
    )] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get status of active notifications with detailed information.
    
    Args:
        notification_id: Specific notification ID to check (if provided)
        ctx: MCP context for logging
        
    Returns:
        Dictionary containing notification status information
    """
    correlation_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        if ctx:
            await ctx.info("Retrieving notification status")
        
        notification_manager = get_notification_manager()
        active_notifications = notification_manager.get_active_notifications()
        
        if notification_id:
            # Get specific notification status
            if notification_id in active_notifications:
                notification_data = active_notifications[notification_id]
                spec = notification_data["spec"]
                return {
                    "success": True,
                    "data": {
                        "notification_id": notification_id,
                        "type": spec.notification_type.value,
                        "title": spec.title,
                        "message": spec.message,
                        "priority": spec.priority.value,
                        "dismissible": spec.dismissible,
                        "active": True,
                        "start_time": notification_data["start_time"]
                    },
                    "metadata": {
                        "correlation_id": correlation_id,
                        "timestamp": datetime.now().isoformat(),
                        "execution_time": (datetime.now() - start_time).total_seconds()
                    }
                }
            else:
                return _create_error_response(
                    correlation_id, "NOT_FOUND", 
                    f"Notification not found: {notification_id}",
                    "The specified notification ID is not in the active notifications list",
                    (datetime.now() - start_time).total_seconds()
                )
        else:
            # Get all active notifications status
            notifications_data = []
            for notif_id, notification_data in active_notifications.items():
                spec = notification_data["spec"]
                notifications_data.append({
                    "notification_id": notif_id,
                    "type": spec.notification_type.value,
                    "title": spec.title,
                    "priority": spec.priority.value,
                    "dismissible": spec.dismissible,
                    "start_time": notification_data["start_time"]
                })
            
            return {
                "success": True,
                "data": {
                    "active_count": len(active_notifications),
                    "notifications": notifications_data
                },
                "metadata": {
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": (datetime.now() - start_time).total_seconds()
                }
            }
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.exception(f"Error retrieving notification status [correlation_id: {correlation_id}]")
        
        if ctx:
            await ctx.error(f"Status retrieval error: {str(e)}")
        
        return _create_error_response(
            correlation_id, "STATUS_ERROR", "Failed to retrieve notification status",
            str(e),
            "Check system status and try again",
            execution_time
        )


async def km_dismiss_notifications(
    notification_id: Annotated[Optional[str], Field(
        default=None,
        description="Specific notification ID to dismiss (optional - dismisses all if not provided)"
    )] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Dismiss active notifications with optional ID filtering.
    
    Args:
        notification_id: Specific notification to dismiss (if provided)
        ctx: MCP context for logging
        
    Returns:
        Dictionary containing dismissal results
    """
    correlation_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        if ctx:
            await ctx.info(f"Dismissing notifications{f' (ID: {notification_id})' if notification_id else ' (all)'}")
        
        notification_manager = get_notification_manager()
        
        if notification_id:
            # Dismiss specific notification
            success = notification_manager.clear_notification(notification_id)
            
            return {
                "success": success,
                "data": {
                    "notification_id": notification_id,
                    "dismissed": success
                },
                "metadata": {
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": (datetime.now() - start_time).total_seconds()
                }
            }
        else:
            # Dismiss all notifications
            count = notification_manager.clear_all_notifications()
            
            return {
                "success": True,
                "data": {
                    "dismissed_count": count,
                    "dismissed_all": True
                },
                "metadata": {
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": (datetime.now() - start_time).total_seconds()
                }
            }
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.exception(f"Error dismissing notifications [correlation_id: {correlation_id}]")
        
        if ctx:
            await ctx.error(f"Dismissal error: {str(e)}")
        
        return _create_error_response(
            correlation_id, "DISMISSAL_ERROR", "Failed to dismiss notifications",
            str(e),
            "Check system status and try again",
            execution_time
        )


def _create_error_response(
    correlation_id: str, error_code: str, error_message: str,
    details: str, recovery_suggestion: str, execution_time: float
) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "success": False,
        "error": {
            "code": error_code,
            "message": error_message,
            "details": details,
            "recovery_suggestion": recovery_suggestion
        },
        "metadata": {
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "operation": "display_notification",
            "failure_stage": error_code.lower()
        }
    }