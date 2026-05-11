"""Security Monitor - TASK_62 Phase 2 Core Security Engine.

Real-time security monitoring and threat detection for zero trust security.
Provides continuous security monitoring, threat detection, incident response, and security analytics.

Architecture: Zero Trust Principles + Real-time Monitoring + Threat Detection + Incident Response
Performance: <300ms security monitoring, <100ms threat detection, <500ms incident response
Security: Continuous monitoring, comprehensive threat detection, automated incident response
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from src.core.contracts import ensure, require
from src.core.either import Either
from src.core.zero_trust_architecture import (
    RiskScore,
    SecurityContext,
    SecurityMetrics,
    SecurityMonitoringError,
    SecurityOperation,
    ThreatSeverity,
    create_risk_score,
)

logger = logging.getLogger(__name__)


class MonitoringStatus(Enum):
    """Security monitoring status."""

    ACTIVE = "active"  # Monitoring is active
    INACTIVE = "inactive"  # Monitoring is inactive
    SUSPENDED = "suspended"  # Monitoring temporarily suspended
    ERROR = "error"  # Monitoring has errors
    MAINTENANCE = "maintenance"  # Monitoring in maintenance mode
    DEGRADED = "degraded"  # Monitoring with reduced capability


class AlertSeverity(Enum):
    """Security alert severity levels."""

    INFO = "info"  # Informational alert
    LOW = "low"  # Low severity alert
    MEDIUM = "medium"  # Medium severity alert
    HIGH = "high"  # High severity alert
    CRITICAL = "critical"  # Critical severity alert
    EMERGENCY = "emergency"  # Emergency response required


class IncidentStatus(Enum):
    """Security incident status."""

    DETECTED = "detected"  # Incident detected
    INVESTIGATING = "investigating"  # Under investigation
    CONFIRMED = "confirmed"  # Incident confirmed
    CONTAINED = "contained"  # Incident contained
    RESOLVED = "resolved"  # Incident resolved
    CLOSED = "closed"  # Incident closed


class MonitoringScope(Enum):
    """Security monitoring scope."""

    USER_ACTIVITY = "user_activity"  # User behavior monitoring
    SYSTEM_ACCESS = "system_access"  # System access monitoring
    NETWORK_TRAFFIC = "network_traffic"  # Network traffic monitoring
    DATA_ACCESS = "data_access"  # Data access monitoring
    AUTHENTICATION = "authentication"  # Authentication monitoring
    POLICY_VIOLATIONS = "policy_violations"  # Policy violation monitoring
    THREAT_INDICATORS = "threat_indicators"  # Threat indicator monitoring
    COMPLIANCE = "compliance"  # Compliance monitoring


@dataclass(frozen=True)
class SecurityAlert:
    """Security alert specification."""

    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    source: str
    detected_at: datetime
    context: SecurityContext | None = None
    indicators: list[str] = field(default_factory=list)
    affected_resources: list[str] = field(default_factory=list)
    confidence: float = 1.0  # 0.0 to 1.0
    risk_score: RiskScore = RiskScore(0.5)
    remediation_steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.alert_id or not self.alert_type or not self.title:
            raise ValueError("Alert ID, type, and title are required")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class SecurityIncident:
    """Security incident specification."""

    incident_id: str
    incident_type: str
    severity: ThreatSeverity
    title: str
    description: str
    status: IncidentStatus
    detected_at: datetime
    updated_at: datetime
    related_alerts: list[str] = field(default_factory=list)
    affected_systems: list[str] = field(default_factory=list)
    impact_assessment: dict[str, Any] = field(default_factory=dict)
    response_actions: list[str] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    assigned_to: str | None = None
    resolution_notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.incident_id or not self.incident_type or not self.title:
            raise ValueError("Incident ID, type, and title are required")


@dataclass(frozen=True)
class MonitoringRule:
    """Security monitoring rule specification."""

    rule_id: str
    rule_name: str
    scope: MonitoringScope
    description: str
    conditions: dict[str, Any]
    actions: dict[str, Any]
    enabled: bool = True
    priority: int = 50  # 1-100, higher = more important
    alert_threshold: int = 1  # Number of matches to trigger alert
    time_window: int = 300  # Time window in seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_id or not self.rule_name or not self.description:
            raise ValueError("Rule ID, name, and description are required")
        if not (1 <= self.priority <= 100):
            raise ValueError("Priority must be between 1 and 100")
        if self.alert_threshold < 1:
            raise ValueError("Alert threshold must be at least 1")
        if self.time_window < 1:
            raise ValueError("Time window must be at least 1 second")


@dataclass(frozen=True)
class SecurityEvent:
    """Security event for monitoring."""

    event_id: str
    event_type: str
    source: str
    timestamp: datetime
    context: SecurityContext | None = None
    data: dict[str, Any] = field(default_factory=dict)
    risk_indicators: list[str] = field(default_factory=list)
    severity: AlertSeverity = AlertSeverity.INFO
    processed: bool = False

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_type or not self.source:
            raise ValueError("Event ID, type, and source are required")


class SecurityMonitor:
    """Real-time security monitoring and threat detection system."""

    def __init__(self) -> None:
        self.monitoring_rules: dict[str, MonitoringRule] = {}
        self.active_alerts: dict[str, SecurityAlert] = {}
        self.active_incidents: dict[str, SecurityIncident] = {}
        self.security_events: list[SecurityEvent] = []
        self.event_buffer: list[SecurityEvent] = []
        self.monitoring_status: dict[MonitoringScope, MonitoringStatus] = {}

        # Analytics and metrics
        self.threat_indicators: dict[str, list[str]] = {}
        self.alert_statistics: dict[str, int] = {}
        self.incident_statistics: dict[str, int] = {}
        self.performance_metrics: dict[str, float] = {}

        # Real-time monitoring
        self.monitoring_tasks: dict[MonitoringScope, asyncio.Task] = {}
        self.alert_callbacks: list[callable] = []
        self.incident_callbacks: list[callable] = []

        # Initialize monitoring status
        for scope in MonitoringScope:
            self.monitoring_status[scope] = MonitoringStatus.INACTIVE

    @require(lambda __self, rule: isinstance(rule, MonitoringRule))
    @ensure(
        lambda _self, result: result.is_right()
        or isinstance(result.get_left(), SecurityMonitoringError),
    )
    async def register_monitoring_rule(
        self,
        rule: MonitoringRule,
    ) -> Either[SecurityMonitoringError, str]:
        """Register a new security monitoring rule."""
        try:
            # Validate rule
            rule_validation = self._validate_monitoring_rule(rule)
            if rule_validation.is_left():
                return rule_validation

            # Check for rule conflicts
            conflict_check = self._check_rule_conflicts(rule)
            if conflict_check.is_left():
                return conflict_check

            # Register rule
            self.monitoring_rules[rule.rule_id] = rule

            # Start monitoring if rule is enabled
            if rule.enabled:
                await self._start_rule_monitoring(rule)

            return Either.right(
                f"Monitoring rule {rule.rule_id} registered successfully",
            )

        except Exception as e:
            return Either.left(
                SecurityMonitoringError(
                    f"Failed to register monitoring rule: {e!s}",
                    "RULE_REGISTRATION_ERROR",
                    SecurityOperation.MONITOR,
                    {"rule_id": rule.rule_id},
                ),
            )

    @require(lambda __self, event: isinstance(event, SecurityEvent | dict))
    async def process_security_event(
        self,
        event: SecurityEvent | dict[str, Any],
    ) -> Either[SecurityMonitoringError, list[SecurityAlert]]:
        """Process security event and generate alerts if needed."""
        try:
            start_time = datetime.now(UTC)

            # Convert dictionary to SecurityEvent if needed
            if isinstance(event, dict):
                event_obj = self._dict_to_security_event(event)
            else:
                event_obj = event

            # Add event to buffer and history
            self.event_buffer.append(event_obj)
            self.security_events.append(event_obj)

            # Evaluate against monitoring rules
            generated_alerts = []

            for rule in self.monitoring_rules.values():
                if rule.enabled and self._event_matches_rule(event_obj, rule):
                    # Check if alert threshold is met
                    matching_events = self._get_matching_events_in_window(rule)

                    if len(matching_events) >= rule.alert_threshold:
                        alert = await self._generate_alert_from_rule(
                            rule,
                            matching_events,
                        )
                        if alert.is_right():
                            generated_alerts.append(alert.get_right())

            # Process threat indicators
            threat_check = await self._check_threat_indicators(event_obj)
            if threat_check.is_right():
                threat_alerts = threat_check.get_right()
                generated_alerts.extend(threat_alerts)

            # Store generated alerts
            for alert in generated_alerts:
                self.active_alerts[alert.alert_id] = alert
                await self._handle_new_alert(alert)

            # Mark event as processed (create a new SecurityEvent since it's frozen)
            processed_event = SecurityEvent(
                event_id=event_obj.event_id,
                event_type=event_obj.event_type,
                source=event_obj.source,
                timestamp=event_obj.timestamp,
                context=event_obj.context,
                data=event_obj.data,
                risk_indicators=event_obj.risk_indicators,
                severity=event_obj.severity,
                processed=True,
            )
            # Update the event in our collections
            if (
                self.event_buffer
                and self.event_buffer[-1].event_id == event_obj.event_id
            ):
                self.event_buffer[-1] = processed_event
            if (
                self.security_events
                and self.security_events[-1].event_id == event_obj.event_id
            ):
                self.security_events[-1] = processed_event

            # Update performance metrics
            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            self._update_monitoring_metrics(processing_time)

            return Either.right(generated_alerts)

        except Exception as e:
            return Either.left(
                SecurityMonitoringError(
                    f"Failed to process security event: {e!s}",
                    "EVENT_PROCESSING_ERROR",
                    SecurityOperation.MONITOR,
                    {
                        "event_id": getattr(event_obj, "event_id", "unknown")
                        if "event_obj" in locals()
                        else "unknown"
                    },
                ),
            )

    async def start_monitoring(
        self,
        scope: MonitoringScope,
    ) -> Either[SecurityMonitoringError, str]:
        """Start security monitoring for specified scope."""
        try:
            if (
                scope in self.monitoring_tasks
                and not self.monitoring_tasks[scope].done()
            ):
                return Either.left(
                    SecurityMonitoringError(
                        f"Monitoring already active for scope {scope.value}",
                        "MONITORING_ALREADY_ACTIVE",
                        SecurityOperation.MONITOR,
                    ),
                )

            # Start monitoring task
            self.monitoring_tasks[scope] = asyncio.create_task(
                self._run_continuous_monitoring(scope),
            )

            self.monitoring_status[scope] = MonitoringStatus.ACTIVE

            return Either.right(f"Monitoring started for scope {scope.value}")

        except Exception as e:
            return Either.left(
                SecurityMonitoringError(
                    f"Failed to start monitoring: {e!s}",
                    "MONITORING_START_ERROR",
                    SecurityOperation.MONITOR,
                    {"scope": scope.value},
                ),
            )

    async def stop_monitoring(
        self,
        scope: MonitoringScope,
    ) -> Either[SecurityMonitoringError, str]:
        """Stop security monitoring for specified scope."""
        try:
            if scope in self.monitoring_tasks:
                task = self.monitoring_tasks[scope]
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                del self.monitoring_tasks[scope]

            self.monitoring_status[scope] = MonitoringStatus.INACTIVE

            return Either.right(f"Monitoring stopped for scope {scope.value}")

        except Exception as e:
            return Either.left(
                SecurityMonitoringError(
                    f"Failed to stop monitoring: {e!s}",
                    "MONITORING_STOP_ERROR",
                    SecurityOperation.MONITOR,
                    {"scope": scope.value},
                ),
            )

    async def create_incident(
        self,
        incident_type: str,
        severity: ThreatSeverity,
        title: str,
        description: str,
        related_alerts: list[str] = None,
    ) -> Either[SecurityMonitoringError, SecurityIncident]:
        """Create a new security incident."""
        try:
            incident_id = f"incident_{int(datetime.now(UTC).timestamp())}"

            if related_alerts is None:
                related_alerts = []

            incident = SecurityIncident(
                incident_id=incident_id,
                incident_type=incident_type,
                severity=severity,
                title=title,
                description=description,
                status=IncidentStatus.DETECTED,
                detected_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                related_alerts=related_alerts,
                timeline=[
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "action": "incident_created",
                        "description": "Security incident created",
                        "actor": "security_monitor",
                    },
                ],
            )

            # Store incident
            self.active_incidents[incident_id] = incident

            # Update statistics
            self.incident_statistics[incident_type] = (
                self.incident_statistics.get(incident_type, 0) + 1
            )

            # Notify incident handlers
            await self._handle_new_incident(incident)

            return Either.right(incident)

        except Exception as e:
            return Either.left(
                SecurityMonitoringError(
                    f"Failed to create incident: {e!s}",
                    "INCIDENT_CREATION_ERROR",
                    SecurityOperation.RESPOND,
                ),
            )

    async def update_incident_status(
        self,
        incident_id: str,
        new_status: IncidentStatus,
        notes: str | None = None,
    ) -> Either[SecurityMonitoringError, SecurityIncident]:
        """Update security incident status."""
        try:
            if incident_id not in self.active_incidents:
                return Either.left(
                    SecurityMonitoringError(
                        f"Incident {incident_id} not found",
                        "INCIDENT_NOT_FOUND",
                        SecurityOperation.RESPOND,
                        {"incident_id": incident_id},
                    ),
                )

            old_incident = self.active_incidents[incident_id]

            # Create timeline entry
            timeline_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "action": "status_change",
                "description": f"Status changed from {old_incident.status.value} to {new_status.value}",
                "actor": "security_monitor",
                "notes": notes,
            }

            # Update incident
            updated_timeline = old_incident.timeline + [timeline_entry]

            updated_incident = SecurityIncident(
                incident_id=old_incident.incident_id,
                incident_type=old_incident.incident_type,
                severity=old_incident.severity,
                title=old_incident.title,
                description=old_incident.description,
                status=new_status,
                detected_at=old_incident.detected_at,
                updated_at=datetime.now(UTC),
                related_alerts=old_incident.related_alerts,
                affected_systems=old_incident.affected_systems,
                impact_assessment=old_incident.impact_assessment,
                response_actions=old_incident.response_actions,
                timeline=updated_timeline,
                assigned_to=old_incident.assigned_to,
                resolution_notes=notes
                if new_status == IncidentStatus.RESOLVED
                else old_incident.resolution_notes,
                metadata=old_incident.metadata,
            )

            # Store updated incident
            self.active_incidents[incident_id] = updated_incident

            # Move to history if closed
            if new_status == IncidentStatus.CLOSED:
                del self.active_incidents[incident_id]

            return Either.right(updated_incident)

        except Exception as e:
            return Either.left(
                SecurityMonitoringError(
                    f"Failed to update incident status: {e!s}",
                    "INCIDENT_UPDATE_ERROR",
                    SecurityOperation.RESPOND,
                    {"incident_id": incident_id},
                ),
            )

    async def get_security_metrics(
        self,
        time_range: timedelta = timedelta(hours=24),
    ) -> Either[SecurityMonitoringError, SecurityMetrics]:
        """Get security monitoring metrics."""
        try:
            end_time = datetime.now(UTC)
            start_time = end_time - time_range

            # Filter events in time range
            recent_events = [
                event
                for event in self.security_events
                if start_time <= event.timestamp <= end_time
            ]

            # Filter alerts in time range
            recent_alerts = [
                alert
                for alert in self.active_alerts.values()
                if start_time <= alert.detected_at <= end_time
            ]

            # Calculate metrics
            total_events = len(recent_events)
            total_alerts = len(recent_alerts)
            total_incidents = len(
                [
                    incident
                    for incident in self.active_incidents.values()
                    if start_time <= incident.detected_at <= end_time
                ],
            )

            # Calculate average response times
            response_times = {}
            if total_alerts > 0:
                alert_processing_times = [
                    alert.metadata.get("processing_time_ms", 0)
                    for alert in recent_alerts
                    if "processing_time_ms" in alert.metadata
                ]

                if alert_processing_times:
                    response_times["alert_processing"] = statistics.mean(
                        alert_processing_times,
                    )

            # Calculate compliance scores (placeholder)
            compliance_scores = {
                "overall": 0.95,
                "threat_detection": 0.98,
                "incident_response": 0.92,
            }

            metrics = SecurityMetrics(
                metrics_id=f"metrics_{int(end_time.timestamp())}",
                period_start=start_time,
                period_end=end_time,
                trust_validations=0,  # Would be integrated with trust validator
                policy_violations=len(
                    [
                        event
                        for event in recent_events
                        if event.event_type == "policy_violation"
                    ],
                ),
                threats_detected=total_alerts,
                incidents_resolved=len(
                    [
                        incident
                        for incident in self.active_incidents.values()
                        if incident.status == IncidentStatus.RESOLVED
                    ],
                ),
                average_trust_score=0.8,  # Would be calculated from trust scores
                average_risk_score=0.3,  # Would be calculated from risk assessments
                compliance_scores=compliance_scores,
                response_times=response_times,
                metadata={
                    "total_events": total_events,
                    "total_alerts": total_alerts,
                    "total_incidents": total_incidents,
                    "monitoring_scopes_active": len(
                        [
                            scope
                            for scope, status in self.monitoring_status.items()
                            if status == MonitoringStatus.ACTIVE
                        ],
                    ),
                },
            )

            return Either.right(metrics)

        except Exception as e:
            return Either.left(
                SecurityMonitoringError(
                    f"Failed to get security metrics: {e!s}",
                    "METRICS_ERROR",
                    SecurityOperation.MONITOR,
                ),
            )

    def register_alert_callback(self, callback: callable) -> None:
        """Register callback for new alerts."""
        self.alert_callbacks.append(callback)

    def register_incident_callback(self, callback: callable) -> None:
        """Register callback for new incidents."""
        self.incident_callbacks.append(callback)

    def _validate_monitoring_rule(
        self,
        rule: MonitoringRule,
    ) -> Either[SecurityMonitoringError, None]:
        """Validate monitoring rule configuration."""
        # Check required fields
        if not rule.conditions:
            return Either.left(
                SecurityMonitoringError(
                    "Monitoring rule must have conditions",
                    "MISSING_CONDITIONS",
                    SecurityOperation.MONITOR,
                ),
            )

        if not rule.actions:
            return Either.left(
                SecurityMonitoringError(
                    "Monitoring rule must have actions",
                    "MISSING_ACTIONS",
                    SecurityOperation.MONITOR,
                ),
            )

        return Either.right(None)

    def _check_rule_conflicts(
        self,
        new_rule: MonitoringRule,
    ) -> Either[SecurityMonitoringError, None]:
        """Check for conflicts with existing monitoring rules."""
        for existing_rule in self.monitoring_rules.values():
            # SIM102 fix: Combine nested if statements
            if (
                existing_rule.scope == new_rule.scope
                and existing_rule.priority == new_rule.priority
                and existing_rule.enabled
                and new_rule.enabled
                and self._rules_have_overlapping_conditions(existing_rule, new_rule)
            ):
                return Either.left(
                    SecurityMonitoringError(
                        f"Rule conflict with existing rule {existing_rule.rule_id}",
                        "RULE_CONFLICT",
                        SecurityOperation.MONITOR,
                        {"conflicting_rule": existing_rule.rule_id},
                    ),
                )

        return Either.right(None)

    def _rules_have_overlapping_conditions(
        self,
        rule1: MonitoringRule,
        rule2: MonitoringRule,
    ) -> bool:
        """Check if two rules have overlapping conditions."""
        # Simplified overlap detection
        conditions1 = set(rule1.conditions.keys())
        conditions2 = set(rule2.conditions.keys())

        return len(conditions1 & conditions2) > 0

    async def _start_rule_monitoring(self, rule: MonitoringRule) -> None:
        """Start monitoring for a specific rule."""
        # This would start rule-specific monitoring logic
        # For now, just mark as active

    def _event_matches_rule(self, event: SecurityEvent, rule: MonitoringRule) -> bool:
        """Check if event matches monitoring rule conditions."""
        try:
            # Check event type
            if "event_types" in rule.conditions:
                allowed_types = rule.conditions["event_types"]
                if event.event_type not in allowed_types:
                    return False

            # Check severity
            if "min_severity" in rule.conditions:
                min_severity = AlertSeverity(rule.conditions["min_severity"])
                severity_levels = {
                    AlertSeverity.INFO: 0,
                    AlertSeverity.LOW: 1,
                    AlertSeverity.MEDIUM: 2,
                    AlertSeverity.HIGH: 3,
                    AlertSeverity.CRITICAL: 4,
                    AlertSeverity.EMERGENCY: 5,
                }

                if severity_levels.get(event.severity, 0) < severity_levels.get(
                    min_severity,
                    0,
                ):
                    return False

            # Check source
            if "sources" in rule.conditions:
                allowed_sources = rule.conditions["sources"]
                if event.source not in allowed_sources:
                    return False

            # Check risk indicators
            if "risk_indicators" in rule.conditions:
                required_indicators = rule.conditions["risk_indicators"]
                if not any(
                    indicator in event.risk_indicators
                    for indicator in required_indicators
                ):
                    return False

            return True

        except Exception:
            return False  # Fail safe

    def _get_matching_events_in_window(
        self,
        rule: MonitoringRule,
    ) -> list[SecurityEvent]:
        """Get events matching rule within time window."""
        current_time = datetime.now(UTC)
        window_start = current_time - timedelta(seconds=rule.time_window)

        matching_events = [
            event
            for event in self.event_buffer
            if (
                window_start <= event.timestamp <= current_time
                and self._event_matches_rule(event, rule)
            )
        ]

        return matching_events

    async def _generate_alert_from_rule(
        self,
        rule: MonitoringRule,
        matching_events: list[SecurityEvent],
    ) -> Either[SecurityMonitoringError, SecurityAlert]:
        """Generate security alert from monitoring rule and events."""
        try:
            alert_id = f"alert_{rule.rule_id}_{int(datetime.now(UTC).timestamp())}"

            # Determine alert severity
            severity = AlertSeverity.MEDIUM  # Default
            if "alert_severity" in rule.actions:
                severity = AlertSeverity(rule.actions["alert_severity"])

            # Calculate risk score
            event_risk_scores = [
                event.data.get("risk_score", 0.5) for event in matching_events
            ]
            avg_risk_score = (
                statistics.mean(event_risk_scores) if event_risk_scores else 0.5
            )

            # Create alert
            alert = SecurityAlert(
                alert_id=alert_id,
                alert_type=rule.scope.value,
                severity=severity,
                title=f"Security Alert: {rule.rule_name}",
                description=f"Rule '{rule.rule_name}' triggered by {len(matching_events)} events",
                source="security_monitor",
                detected_at=datetime.now(UTC),
                context=matching_events[0].context if matching_events else None,
                indicators=[
                    indicator
                    for event in matching_events
                    for indicator in event.risk_indicators
                ],
                affected_resources=list(
                    {
                        event.data.get("resource", "unknown")
                        for event in matching_events
                    },
                ),
                confidence=min(1.0, len(matching_events) / rule.alert_threshold),
                risk_score=create_risk_score(avg_risk_score),
                remediation_steps=rule.actions.get("remediation_steps", []),
                metadata={
                    "rule_id": rule.rule_id,
                    "triggering_events": len(matching_events),
                    "time_window": rule.time_window,
                    "processing_time_ms": 0,  # Will be updated
                },
            )

            return Either.right(alert)

        except Exception as e:
            return Either.left(
                SecurityMonitoringError(
                    f"Failed to generate alert: {e!s}",
                    "ALERT_GENERATION_ERROR",
                    SecurityOperation.MONITOR,
                    {"rule_id": rule.rule_id},
                ),
            )

    async def _check_threat_indicators(
        self,
        event: SecurityEvent,
    ) -> Either[SecurityMonitoringError, list[SecurityAlert]]:
        """Check event against known threat indicators."""
        try:
            threat_alerts = []

            # Check for known threat patterns
            for indicator in event.risk_indicators:
                if self._is_known_threat_indicator(indicator):
                    threat_alert = await self._create_threat_alert(event, indicator)
                    if threat_alert.is_right():
                        threat_alerts.append(threat_alert.get_right())

            return Either.right(threat_alerts)

        except Exception as e:
            return Either.left(
                SecurityMonitoringError(
                    f"Failed to check threat indicators: {e!s}",
                    "THREAT_CHECK_ERROR",
                    SecurityOperation.DETECT,
                ),
            )

    def _is_known_threat_indicator(self, indicator: str) -> bool:
        """Check if indicator is a known threat."""
        # Placeholder - would check threat intelligence feeds
        known_threats = [
            "suspicious_login_pattern",
            "privilege_escalation_attempt",
            "data_exfiltration_pattern",
            "malware_signature",
            "brute_force_attack",
        ]

        return indicator in known_threats

    async def _create_threat_alert(
        self,
        event: SecurityEvent,
        threat_indicator: str,
    ) -> Either[SecurityMonitoringError, SecurityAlert]:
        """Create alert for detected threat."""
        try:
            alert_id = f"threat_alert_{int(datetime.now(UTC).timestamp())}"

            alert = SecurityAlert(
                alert_id=alert_id,
                alert_type="threat_detection",
                severity=AlertSeverity.HIGH,
                title=f"Threat Detected: {threat_indicator}",
                description=f"Known threat indicator '{threat_indicator}' detected in event {event.event_id}",
                source="threat_intelligence",
                detected_at=datetime.now(UTC),
                context=event.context,
                indicators=[threat_indicator],
                affected_resources=[event.data.get("resource", "unknown")],
                confidence=0.9,
                risk_score=create_risk_score(0.8),
                remediation_steps=[
                    "Investigate affected systems",
                    "Check for lateral movement",
                    "Review access logs",
                    "Consider containment measures",
                ],
                metadata={
                    "source_event": event.event_id,
                    "threat_type": threat_indicator,
                    "detection_method": "threat_intelligence",
                },
            )

            return Either.right(alert)

        except Exception as e:
            return Either.left(
                SecurityMonitoringError(
                    f"Failed to create threat alert: {e!s}",
                    "THREAT_ALERT_ERROR",
                    SecurityOperation.DETECT,
                ),
            )

    async def _handle_new_alert(self, alert: SecurityAlert) -> None:
        """Handle new security alert."""
        try:
            # Update statistics
            self.alert_statistics[alert.alert_type] = (
                self.alert_statistics.get(alert.alert_type, 0) + 1
            )

            # Check if alert should trigger incident
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                await self.create_incident(
                    incident_type=alert.alert_type,
                    severity=ThreatSeverity.HIGH,
                    title=f"High-severity alert: {alert.title}",
                    description=f"Alert {alert.alert_id} requires immediate attention",
                    related_alerts=[alert.alert_id],
                )

            # Notify alert callbacks
            for callback in self.alert_callbacks:
                # SIM105 fix: Use contextlib.suppress instead of try-except-pass
                with contextlib.suppress(Exception):
                    await callback(alert)

        except Exception as e:
            logger.error(f"Alert handling failed: {e!s}", exc_info=True)

    async def _handle_new_incident(self, incident: SecurityIncident) -> None:
        """Handle new security incident."""
        try:
            # Notify incident callbacks
            for callback in self.incident_callbacks:
                with contextlib.suppress(Exception):
                    await callback(incident)

        except Exception as e:
            logger.error(f"Incident handling failed: {e!s}", exc_info=True)

    async def _run_continuous_monitoring(self, scope: MonitoringScope) -> None:
        """Run continuous monitoring for scope."""
        try:
            while self.monitoring_status.get(scope) == MonitoringStatus.ACTIVE:
                # Perform scope-specific monitoring
                await self._perform_scope_monitoring(scope)

                # Wait before next monitoring cycle
                await asyncio.sleep(10)  # 10 second intervals

        except asyncio.CancelledError:
            self.monitoring_status[scope] = MonitoringStatus.INACTIVE
        except Exception:
            self.monitoring_status[scope] = MonitoringStatus.ERROR

    async def _perform_scope_monitoring(self, scope: MonitoringScope) -> None:
        """Perform monitoring for specific scope."""
        try:
            # Scope-specific monitoring logic
            if scope == MonitoringScope.USER_ACTIVITY:
                await self._monitor_user_activity()
            elif scope == MonitoringScope.SYSTEM_ACCESS:
                await self._monitor_system_access()
            elif scope == MonitoringScope.NETWORK_TRAFFIC:
                await self._monitor_network_traffic()
            elif scope == MonitoringScope.DATA_ACCESS:
                await self._monitor_data_access()
            elif scope == MonitoringScope.AUTHENTICATION:
                await self._monitor_authentication()
            elif scope == MonitoringScope.POLICY_VIOLATIONS:
                await self._monitor_policy_violations()
            elif scope == MonitoringScope.THREAT_INDICATORS:
                await self._monitor_threat_indicators()
            elif scope == MonitoringScope.COMPLIANCE:
                await self._monitor_compliance()

        except Exception as e:
            # Log error but continue monitoring
            logger.error(
                f"Scope monitoring failed for {scope}: {e!s}",
                exc_info=True,
            )

    async def _monitor_user_activity(self) -> None:
        """Monitor user activity for anomalies."""
        # Placeholder for user activity monitoring

    async def _monitor_system_access(self) -> None:
        """Monitor system access patterns."""
        # Placeholder for system access monitoring

    async def _monitor_network_traffic(self) -> None:
        """Monitor network traffic for threats."""
        # Placeholder for network traffic monitoring

    async def _monitor_data_access(self) -> None:
        """Monitor data access patterns."""
        # Placeholder for data access monitoring

    async def _monitor_authentication(self) -> None:
        """Monitor authentication events."""
        # Placeholder for authentication monitoring

    async def _monitor_policy_violations(self) -> None:
        """Monitor for policy violations."""
        # Placeholder for policy violation monitoring

    async def _monitor_threat_indicators(self) -> None:
        """Monitor for threat indicators."""
        # Placeholder for threat indicator monitoring

    async def _monitor_compliance(self) -> None:
        """Monitor compliance status."""
        # Placeholder for compliance monitoring

    def _update_monitoring_metrics(self, processing_time: float) -> None:
        """Update monitoring performance metrics."""
        if "avg_processing_time" not in self.performance_metrics:
            self.performance_metrics["avg_processing_time"] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.performance_metrics["avg_processing_time"] = (
                alpha * processing_time
                + (1 - alpha) * self.performance_metrics["avg_processing_time"]
            )

        # Update event count
        self.performance_metrics["total_events"] = (
            self.performance_metrics.get("total_events", 0) + 1
        )

    def _analyze_event(self, event: SecurityEvent | dict[str, Any]) -> dict[str, Any]:
        """Analyze security event and return threat analysis.

        This method analyzes a security event and returns a dictionary containing
        threat analysis information including threat level.

        Args:
            event: Security event to analyze (can be SecurityEvent object or dict)

        Returns:
            Dictionary containing threat analysis with 'threat_level' key
        """
        # Convert dict to SecurityEvent if needed
        if isinstance(event, dict):
            event_obj = self._dict_to_security_event(event)
        else:
            event_obj = event

        # Analyze event for threats
        threat_level = "low"  # Default threat level

        # Check event type for threat indicators
        if event_obj.event_type in ["authentication_failure", "access_denied"]:
            threat_level = "medium"
        elif event_obj.event_type in ["privilege_escalation", "data_exfiltration"]:
            threat_level = "high"
        elif event_obj.event_type in ["malware_detected", "brute_force_detected"]:
            threat_level = "critical"

        # Check severity
        if event_obj.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            threat_level = "high"
        elif event_obj.severity == AlertSeverity.EMERGENCY:
            threat_level = "critical"

        # Check risk indicators
        high_risk_indicators = [
            "brute_force_attack",
            "privilege_escalation_attempt",
            "data_exfiltration_pattern",
            "malware_signature",
        ]

        if any(
            indicator in event_obj.risk_indicators for indicator in high_risk_indicators
        ):
            threat_level = "high"

        return {
            "threat_level": threat_level,
            "event_type": event_obj.event_type,
            "severity": event_obj.severity.value,
            "risk_indicators": event_obj.risk_indicators,
            "analysis_timestamp": datetime.now(UTC).isoformat(),
        }

    def _dict_to_security_event(self, event_dict: dict[str, Any]) -> SecurityEvent:
        """Convert dictionary to SecurityEvent object.

        Args:
            event_dict: Dictionary containing event data

        Returns:
            SecurityEvent object created from dictionary
        """
        # Generate event ID if not provided
        event_id = event_dict.get(
            "event_id", f"event_{int(datetime.now(UTC).timestamp())}"
        )

        # Map common dictionary keys to SecurityEvent fields
        event_type = event_dict.get("type", event_dict.get("event_type", "unknown"))
        source = event_dict.get("source", "unknown")

        # Handle timestamp
        timestamp_str = event_dict.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now(UTC)
        else:
            timestamp = datetime.now(UTC)

        # Create event data
        data = {
            "user_id": event_dict.get("user_id"),
            "source_ip": event_dict.get("source_ip"),
            "resource": event_dict.get("resource"),
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        # Add any additional fields from the dict
        for key, value in event_dict.items():
            if key not in [
                "type",
                "event_type",
                "source",
                "timestamp",
                "user_id",
                "source_ip",
                "resource",
            ]:
                data[key] = value

        return SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            source=source,
            timestamp=timestamp,
            data=data,
            risk_indicators=event_dict.get("risk_indicators", []),
            severity=AlertSeverity.INFO,  # Default severity
        )

    def detect_threat(self, event: SecurityEvent | dict[str, Any]) -> dict[str, Any]:
        """Detect threat in security event.

        This method provides a simple interface for threat detection that can be
        used by tests. It analyzes the event and returns threat information.

        Args:
            event: Security event to analyze

        Returns:
            Dictionary containing threat analysis
        """
        return self._analyze_event(event)

    # Simple interface methods for test compatibility
    def log_event(self, event: SecurityEvent) -> None:
        """Log security event (simple interface for test compatibility)."""
        self.security_events.append(event)

        # Trigger real-time alerts for critical events
        if (
            event.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            and self.alert_callbacks
        ):
            for callback in self.alert_callbacks:
                with contextlib.suppress(Exception):
                    callback(event)

    def get_events(self, limit: int = 100) -> list[SecurityEvent]:
        """Get recent security events (simple interface for test compatibility)."""
        return self.security_events[-limit:] if limit > 0 else self.security_events

    def detect_threats(
        self,
        user_id: str = None,
        time_window: timedelta = timedelta(hours=1),
    ) -> list[SecurityEvent]:
        """Detect threats for user (simple interface for test compatibility)."""
        current_time = datetime.now(UTC)
        start_time = current_time - time_window

        # Filter events for user and time window
        filtered_events = [
            event
            for event in self.security_events
            if (
                start_time <= event.timestamp <= current_time
                and (user_id is None or event.data.get("user_id") == user_id)
            )
        ]

        # Detect patterns - multiple failures indicate threat
        threat_events = []
        if user_id:
            failed_attempts = [
                event
                for event in filtered_events
                if event.event_type in ["authentication_failure", "access_denied"]
            ]

            if len(failed_attempts) >= 3:  # Threshold for threat detection
                # Create high-severity threat event
                threat_event = SecurityEvent(
                    event_id=f"threat_{int(current_time.timestamp())}",
                    event_type="brute_force_detected",
                    source="threat_detector",
                    timestamp=current_time,
                    data={"user_id": user_id, "failed_attempts": len(failed_attempts)},
                    risk_indicators=["brute_force_attack", "multiple_failures"],
                    severity=AlertSeverity.HIGH,
                )
                threat_events.append(threat_event)

        return threat_events

    def detect_anomalies(
        self,
        user_id: str = None,
        time_window: timedelta = timedelta(hours=1),
    ) -> list[SecurityEvent]:
        """Detect anomalies for user (simple interface for test compatibility)."""
        current_time = datetime.now(UTC)
        start_time = current_time - time_window

        # Simple anomaly detection - unusual resource access
        filtered_events = [
            event
            for event in self.security_events
            if (
                start_time <= event.timestamp <= current_time
                and (user_id is None or event.data.get("user_id") == user_id)
            )
        ]

        # Look for unusual patterns
        anomalies = []
        if user_id:
            resources = [event.data.get("resource", "") for event in filtered_events]
            resource_counts = {}
            for resource in resources:
                if resource:
                    resource_counts[resource] = resource_counts.get(resource, 0) + 1

            # Check for access to admin resources by regular users
            for resource in resources:
                if resource and "admin" in resource.lower():
                    anomaly_event = SecurityEvent(
                        event_id=f"anomaly_{int(current_time.timestamp())}",
                        event_type="anomalous_access",
                        source="anomaly_detector",
                        timestamp=current_time,
                        data={"user_id": user_id, "unusual_resource": resource},
                        risk_indicators=["unusual_access_pattern"],
                        severity=AlertSeverity.MEDIUM,
                    )
                    anomalies.append(anomaly_event)

        return anomalies

    def set_alert_handler(self, handler: callable) -> None:
        """Set alert handler (simple interface for test compatibility)."""
        if handler not in self.alert_callbacks:
            self.alert_callbacks.append(handler)

    def supports_real_time_alerts(self) -> bool:
        """Check if real-time alerts are supported (simple interface for test compatibility)."""
        return len(self.alert_callbacks) > 0

    def get_threat_summary(
        self,
        user_id: str = None,
        time_window: timedelta = timedelta(hours=1),
    ) -> ThreatSummary:
        """Get threat summary for user (simple interface for test compatibility)."""
        current_time = datetime.now(UTC)
        start_time = current_time - time_window

        # Filter events for user and time window
        filtered_events = [
            event
            for event in self.security_events
            if (
                start_time <= event.timestamp <= current_time
                and (user_id is None or event.data.get("user_id") == user_id)
            )
        ]

        # Calculate threat level
        max_severity = AlertSeverity.INFO
        for event in filtered_events:
            if event.severity.value == "critical":
                max_severity = AlertSeverity.CRITICAL
            elif event.severity.value == "high" and max_severity.value not in [
                "critical",
            ]:
                max_severity = AlertSeverity.HIGH
            elif event.severity.value == "medium" and max_severity.value not in [
                "critical",
                "high",
            ]:
                max_severity = AlertSeverity.MEDIUM

        return ThreatSummary(
            total_events=len(filtered_events),
            max_threat_level=ThreatSeverity.HIGH
            if max_severity == AlertSeverity.HIGH
            else ThreatSeverity.MEDIUM,
            user_id=user_id,
            time_window=time_window,
        )

    def get_events_for_user(
        self,
        user_id: str,
        time_window: timedelta = timedelta(hours=1),
    ) -> list[SecurityEvent]:
        """Get events for specific user (simple interface for test compatibility)."""
        current_time = datetime.now(UTC)
        start_time = current_time - time_window

        return [
            event
            for event in self.security_events
            if (
                start_time <= event.timestamp <= current_time
                and event.data.get("user_id") == user_id
            )
        ]


# Simple result class for test compatibility
@dataclass
class ThreatSummary:
    """Simple threat summary for test compatibility."""

    total_events: int
    max_threat_level: ThreatSeverity
    user_id: str | None = None
    time_window: timedelta | None = None


# Utility functions for security monitoring
def create_monitoring_rule(
    rule_name: str,
    scope: MonitoringScope,
    description: str,
    conditions: dict[str, Any],
    actions: dict[str, Any],
    priority: int = 50,
    alert_threshold: int = 1,
    time_window: int = 300,
) -> MonitoringRule:
    """Create a monitoring rule with validation."""
    rule_id = f"rule_{rule_name.lower().replace(' ', '_')}"

    return MonitoringRule(
        rule_id=rule_id,
        rule_name=rule_name,
        scope=scope,
        description=description,
        conditions=conditions,
        actions=actions,
        priority=priority,
        alert_threshold=alert_threshold,
        time_window=time_window,
    )


def create_security_event(
    event_type: str,
    source: str,
    data: dict[str, Any],
    context: SecurityContext | None = None,
    risk_indicators: list[str] = None,
    severity: AlertSeverity = AlertSeverity.INFO,
) -> SecurityEvent:
    """Create a security event with validation."""
    event_id = f"event_{int(datetime.now(UTC).timestamp())}"

    if risk_indicators is None:
        risk_indicators = []

    return SecurityEvent(
        event_id=event_id,
        event_type=event_type,
        source=source,
        timestamp=datetime.now(UTC),
        context=context,
        data=data,
        risk_indicators=risk_indicators,
        severity=severity,
    )
