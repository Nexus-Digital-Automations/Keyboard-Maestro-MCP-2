"""
Performance Alert System - TASK_54 Phase 2 Implementation

Advanced alerting system for performance threshold monitoring, alert management,
and automated notification delivery with escalation policies.

Architecture: Alert engine + Type Safety + Notification system
Performance: <50ms alert evaluation, real-time notifications
Security: Alert data protection, notification channel validation
"""

from __future__ import annotations
import asyncio
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.performance_monitoring import (
    MonitoringSessionID, AlertID, MetricType, AlertSeverity,
    PerformanceAlert, PerformanceThreshold, MetricValue,
    ThresholdOperator, generate_alert_id,
    PerformanceMonitoringError, ThresholdViolationError
)

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Available notification channels for alerts."""
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DESKTOP = "desktop"
    CONSOLE = "console"


class AlertStatus(Enum):
    """Alert status lifecycle."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


@dataclass
class NotificationConfig:
    """Configuration for alert notifications."""
    channel: NotificationChannel
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        require(lambda: isinstance(self.config, dict), "Config must be dictionary")


@dataclass
class EscalationPolicy:
    """Alert escalation policy configuration."""
    policy_id: str
    escalation_levels: List[Dict[str, Any]] = field(default_factory=list)
    escalation_delay_minutes: int = 30
    max_escalations: int = 3
    
    def __post_init__(self):
        require(lambda: self.escalation_delay_minutes > 0, "Escalation delay must be positive")
        require(lambda: self.max_escalations > 0, "Max escalations must be positive")


@dataclass
class AlertRule:
    """Complete alert rule with threshold and notification configuration."""
    rule_id: str
    name: str
    threshold: PerformanceThreshold
    notification_channels: List[NotificationConfig] = field(default_factory=list)
    escalation_policy: Optional[EscalationPolicy] = None
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        require(lambda: len(self.name.strip()) > 0, "Alert rule name required")
        require(lambda: len(self.notification_channels) > 0, "At least one notification channel required")


@dataclass
class ActiveAlert:
    """Active alert with status tracking and history."""
    alert: PerformanceAlert
    rule: AlertRule
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    last_escalation: Optional[datetime] = None
    notification_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def acknowledge(self, acknowledged_by: str) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now(UTC)
        self.acknowledged_by = acknowledged_by
    
    def resolve(self) -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now(UTC)
    
    def suppress(self) -> None:
        """Suppress the alert."""
        self.status = AlertStatus.SUPPRESSED


class AlertSystem:
    """
    Advanced performance alert system with threshold monitoring and notifications.
    
    Provides real-time alert evaluation, escalation policies, and multi-channel
    notification delivery with comprehensive alert lifecycle management.
    """
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[AlertID, ActiveAlert] = {}
        self.alert_history: List[ActiveAlert] = []
        self.max_history_size = 1000
        
        # Notification handlers
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}
        self._initialize_notification_handlers()
        
        # Alert evaluation cache
        self.last_evaluations: Dict[str, datetime] = {}
        self.cooldown_cache: Dict[str, datetime] = {}
        
        logger.info("AlertSystem initialized")
    
    def _initialize_notification_handlers(self) -> None:
        """Initialize notification handlers for each channel type."""
        self.notification_handlers = {
            NotificationChannel.LOG: self._send_log_notification,
            NotificationChannel.EMAIL: self._send_email_notification,
            NotificationChannel.SMS: self._send_sms_notification,
            NotificationChannel.WEBHOOK: self._send_webhook_notification,
            NotificationChannel.SLACK: self._send_slack_notification,
            NotificationChannel.DESKTOP: self._send_desktop_notification,
            NotificationChannel.CONSOLE: self._send_console_notification
        }
    
    @require(lambda rule: len(rule.name.strip()) > 0, "Alert rule name required")
    def add_alert_rule(self, rule: AlertRule) -> Either[str, str]:
        """Add a new alert rule to the system."""
        try:
            if rule.rule_id in self.alert_rules:
                return Either.left(f"Alert rule {rule.rule_id} already exists")
            
            # Validate notification channels
            for notification in rule.notification_channels:
                if notification.channel not in self.notification_handlers:
                    return Either.left(f"Unsupported notification channel: {notification.channel}")
            
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")
            
            return Either.right(rule.rule_id)
            
        except Exception as e:
            error_msg = f"Failed to add alert rule: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    def remove_alert_rule(self, rule_id: str) -> Either[str, str]:
        """Remove an alert rule from the system."""
        try:
            if rule_id not in self.alert_rules:
                return Either.left(f"Alert rule {rule_id} not found")
            
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            
            return Either.right(rule_id)
            
        except Exception as e:
            error_msg = f"Failed to remove alert rule: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    @require(lambda metric: metric.value is not None, "Metric value required")
    async def evaluate_metric(self, metric: MetricValue, session_id: Optional[MonitoringSessionID] = None) -> List[PerformanceAlert]:
        """Evaluate a metric against all applicable alert rules."""
        alerts_generated = []
        
        try:
            current_time = datetime.now(UTC)
            
            # Find matching alert rules
            matching_rules = [
                rule for rule in self.alert_rules.values()
                if rule.enabled and rule.threshold.metric_type == metric.metric_type
            ]
            
            for rule in matching_rules:
                try:
                    # Check cooldown period
                    cooldown_key = f"{rule.rule_id}_{metric.source}"
                    if cooldown_key in self.cooldown_cache:
                        last_alert_time = self.cooldown_cache[cooldown_key]
                        if current_time - last_alert_time < timedelta(seconds=rule.threshold.cooldown_period):
                            continue  # Still in cooldown
                    
                    # Evaluate threshold
                    if rule.threshold.evaluate(metric.value):
                        # Generate alert
                        alert = PerformanceAlert(
                            alert_id=generate_alert_id(),
                            metric_type=metric.metric_type,
                            current_value=metric.value,
                            threshold=rule.threshold,
                            triggered_at=current_time,
                            source=metric.source,
                            message=self._generate_alert_message(rule, metric)
                        )
                        
                        # Create active alert
                        active_alert = ActiveAlert(
                            alert=alert,
                            rule=rule
                        )
                        
                        self.active_alerts[alert.alert_id] = active_alert
                        alerts_generated.append(alert)
                        
                        # Update cooldown
                        self.cooldown_cache[cooldown_key] = current_time
                        
                        # Send notifications
                        await self._send_alert_notifications(active_alert)
                        
                        logger.warning(f"Alert generated: {alert.message}")
                        
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
                    continue
            
            return alerts_generated
            
        except Exception as e:
            logger.error(f"Metric evaluation failed: {e}")
            return []
    
    async def _send_alert_notifications(self, active_alert: ActiveAlert) -> None:
        """Send notifications for an alert through all configured channels."""
        try:
            notification_tasks = []
            
            for notification_config in active_alert.rule.notification_channels:
                if not notification_config.enabled:
                    continue
                
                handler = self.notification_handlers.get(notification_config.channel)
                if handler:
                    task = asyncio.create_task(
                        handler(active_alert, notification_config)
                    )
                    notification_tasks.append(task)
            
            # Wait for all notifications to complete
            if notification_tasks:
                results = await asyncio.gather(*notification_tasks, return_exceptions=True)
                
                # Log notification results
                for i, result in enumerate(results):
                    channel = active_alert.rule.notification_channels[i].channel
                    if isinstance(result, Exception):
                        logger.error(f"Notification failed for {channel}: {result}")
                        active_alert.notification_history.append({
                            "channel": channel.value,
                            "status": "failed",
                            "error": str(result),
                            "timestamp": datetime.now(UTC).isoformat()
                        })
                    else:
                        active_alert.notification_history.append({
                            "channel": channel.value,
                            "status": "sent",
                            "timestamp": datetime.now(UTC).isoformat()
                        })
            
        except Exception as e:
            logger.error(f"Alert notification failed: {e}")
    
    async def _send_log_notification(self, active_alert: ActiveAlert, config: NotificationConfig) -> None:
        """Send alert notification to log."""
        logger.warning(f"ALERT: {active_alert.alert.message}")
    
    async def _send_email_notification(self, active_alert: ActiveAlert, config: NotificationConfig) -> None:
        """Send alert notification via email (placeholder)."""
        # This would integrate with an email service
        logger.info(f"EMAIL ALERT: {active_alert.alert.message}")
    
    async def _send_sms_notification(self, active_alert: ActiveAlert, config: NotificationConfig) -> None:
        """Send alert notification via SMS (placeholder)."""
        # This would integrate with an SMS service
        logger.info(f"SMS ALERT: {active_alert.alert.message}")
    
    async def _send_webhook_notification(self, active_alert: ActiveAlert, config: NotificationConfig) -> None:
        """Send alert notification via webhook (placeholder)."""
        # This would make HTTP POST to configured webhook URL
        logger.info(f"WEBHOOK ALERT: {active_alert.alert.message}")
    
    async def _send_slack_notification(self, active_alert: ActiveAlert, config: NotificationConfig) -> None:
        """Send alert notification to Slack (placeholder)."""
        # This would integrate with Slack API
        logger.info(f"SLACK ALERT: {active_alert.alert.message}")
    
    async def _send_desktop_notification(self, active_alert: ActiveAlert, config: NotificationConfig) -> None:
        """Send desktop notification (placeholder)."""
        # This would show OS-level notification
        logger.info(f"DESKTOP ALERT: {active_alert.alert.message}")
    
    async def _send_console_notification(self, active_alert: ActiveAlert, config: NotificationConfig) -> None:
        """Send alert to console output."""
        print(f"🚨 PERFORMANCE ALERT: {active_alert.alert.message}")
    
    def _generate_alert_message(self, rule: AlertRule, metric: MetricValue) -> str:
        """Generate a descriptive alert message."""
        threshold = rule.threshold
        return (f"{rule.name}: {metric.metric_type.value} {threshold.operator.value} "
                f"{threshold.threshold_value} (current: {metric.value}, "
                f"severity: {threshold.severity.value})")
    
    def acknowledge_alert(self, alert_id: AlertID, acknowledged_by: str) -> Either[str, str]:
        """Acknowledge an active alert."""
        try:
            if alert_id not in self.active_alerts:
                return Either.left(f"Alert {alert_id} not found")
            
            active_alert = self.active_alerts[alert_id]
            active_alert.acknowledge(acknowledged_by)
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return Either.right(alert_id)
            
        except Exception as e:
            error_msg = f"Failed to acknowledge alert: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    def resolve_alert(self, alert_id: AlertID) -> Either[str, str]:
        """Resolve an active alert."""
        try:
            if alert_id not in self.active_alerts:
                return Either.left(f"Alert {alert_id} not found")
            
            active_alert = self.active_alerts[alert_id]
            active_alert.resolve()
            
            # Move to history
            self.alert_history.append(active_alert)
            del self.active_alerts[alert_id]
            
            # Trim history if needed
            if len(self.alert_history) > self.max_history_size:
                self.alert_history = self.alert_history[-self.max_history_size:]
            
            logger.info(f"Alert {alert_id} resolved")
            return Either.right(alert_id)
            
        except Exception as e:
            error_msg = f"Failed to resolve alert: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[ActiveAlert]:
        """Get all active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [a for a in alerts if a.alert.threshold.severity == severity_filter]
        
        # Sort by severity and timestamp
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3
        }
        
        alerts.sort(key=lambda a: (
            severity_order.get(a.alert.threshold.severity, 4),
            a.alert.triggered_at
        ))
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        active_count = len(self.active_alerts)
        total_history = len(self.alert_history)
        
        # Count by severity
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in self.active_alerts.values():
            severity_counts[alert.alert.threshold.severity.value] += 1
        
        # Count by status
        status_counts = {status.value: 0 for status in AlertStatus}
        for alert in self.active_alerts.values():
            status_counts[alert.status.value] += 1
        
        return {
            "active_alerts": active_count,
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "alert_history_size": total_history,
            "severity_breakdown": severity_counts,
            "status_breakdown": status_counts,
            "notification_channels": list(set(
                nc.channel.value 
                for rule in self.alert_rules.values()
                for nc in rule.notification_channels
            ))
        }
    
    async def check_escalations(self) -> None:
        """Check and process alert escalations."""
        try:
            current_time = datetime.now(UTC)
            
            for active_alert in self.active_alerts.values():
                if (active_alert.status == AlertStatus.ACTIVE and 
                    active_alert.rule.escalation_policy and
                    active_alert.escalation_level < active_alert.rule.escalation_policy.max_escalations):
                    
                    # Check if escalation is due
                    escalation_delay = timedelta(minutes=active_alert.rule.escalation_policy.escalation_delay_minutes)
                    last_time = active_alert.last_escalation or active_alert.alert.triggered_at
                    
                    if current_time - last_time >= escalation_delay:
                        await self._escalate_alert(active_alert)
            
        except Exception as e:
            logger.error(f"Escalation check failed: {e}")
    
    async def _escalate_alert(self, active_alert: ActiveAlert) -> None:
        """Escalate an alert to the next level."""
        try:
            active_alert.escalation_level += 1
            active_alert.last_escalation = datetime.now(UTC)
            active_alert.status = AlertStatus.ESCALATED
            
            logger.warning(f"Alert {active_alert.alert.alert_id} escalated to level {active_alert.escalation_level}")
            
            # Send escalation notifications
            await self._send_alert_notifications(active_alert)
            
        except Exception as e:
            logger.error(f"Alert escalation failed: {e}")
    
    def cleanup_resolved_alerts(self, max_age_hours: int = 24) -> int:
        """Clean up old resolved alerts from history."""
        try:
            cutoff_time = datetime.now(UTC) - timedelta(hours=max_age_hours)
            
            original_count = len(self.alert_history)
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.resolved_at is None or alert.resolved_at > cutoff_time
            ]
            
            cleaned_count = original_count - len(self.alert_history)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old resolved alerts")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Alert cleanup failed: {e}")
            return 0


# Global instance
_alert_system: Optional[AlertSystem] = None


def get_alert_system() -> AlertSystem:
    """Get or create the global alert system instance."""
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertSystem()
    return _alert_system


# Convenience functions for common alert configurations
def create_cpu_alert_rule(
    rule_id: str,
    threshold_percent: float = 80.0,
    severity: AlertSeverity = AlertSeverity.HIGH,
    notification_channels: Optional[List[NotificationChannel]] = None
) -> AlertRule:
    """Create a CPU usage alert rule."""
    if notification_channels is None:
        notification_channels = [NotificationChannel.LOG]
    
    from ..core.performance_monitoring import create_cpu_threshold
    
    return AlertRule(
        rule_id=rule_id,
        name=f"High CPU Usage (>{threshold_percent}%)",
        threshold=create_cpu_threshold(threshold_percent, severity),
        notification_channels=[
            NotificationConfig(channel=channel) for channel in notification_channels
        ]
    )


def create_memory_alert_rule(
    rule_id: str,
    threshold_percent: float = 85.0,
    severity: AlertSeverity = AlertSeverity.HIGH,
    notification_channels: Optional[List[NotificationChannel]] = None
) -> AlertRule:
    """Create a memory usage alert rule."""
    if notification_channels is None:
        notification_channels = [NotificationChannel.LOG]
    
    from ..core.performance_monitoring import create_memory_threshold
    
    return AlertRule(
        rule_id=rule_id,
        name=f"High Memory Usage (>{threshold_percent}%)",
        threshold=create_memory_threshold(threshold_percent, severity),
        notification_channels=[
            NotificationConfig(channel=channel) for channel in notification_channels
        ]
    )