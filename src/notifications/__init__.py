"""Notification system for Keyboard Maestro MCP Tools.

This module provides comprehensive user feedback capabilities through multiple
notification channels including system notifications, alerts, HUD displays,
and sound notifications with proper timing and user experience considerations.
"""

from .notification_manager import (
    NotificationManager,
    NotificationResult,
    NotificationSpec,
    NotificationType,
)

__all__ = [
    "NotificationManager",
    "NotificationResult",
    "NotificationSpec",
    "NotificationType",
]
