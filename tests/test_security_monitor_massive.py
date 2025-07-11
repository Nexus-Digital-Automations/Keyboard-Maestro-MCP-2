"""Comprehensive tests for src/security/security_monitor.py - MASSIVE 504 statements coverage.

🚨 CRITICAL COVERAGE ENFORCEMENT: Phase 8 targeting highest-impact zero-coverage modules.
This test covers src/security/security_monitor.py (504 statements - 4th HIGHEST IMPACT) to achieve
significant progress toward mandatory 95% coverage threshold.

Coverage Focus: SecurityMonitor class, real-time monitoring, threat detection, incident response,
security analytics, alert generation, monitoring rules, and all security monitoring functionality.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.core.zero_trust_architecture import (
    RiskScore,
    SecurityContext,
    SecurityMetrics,
    SecurityMonitoringError,
    SecurityOperation,
    ThreatSeverity,
    TrustLevel,
)
from src.security.security_monitor import (
    AlertSeverity,
    IncidentStatus,
    MonitoringRule,
    MonitoringScope,
    MonitoringStatus,
    SecurityAlert,
    SecurityEvent,
    SecurityIncident,
    SecurityMonitor,
    ThreatSummary,
    create_monitoring_rule,
    create_security_event,
)


class TestMonitoringEnums:
    """Test enumeration values for security monitoring."""

    def test_monitoring_status_values(self):
        """Test MonitoringStatus enum values."""
        assert MonitoringStatus.ACTIVE.value == "active"
        assert MonitoringStatus.INACTIVE.value == "inactive"
        assert MonitoringStatus.SUSPENDED.value == "suspended"
        assert MonitoringStatus.ERROR.value == "error"
        assert MonitoringStatus.MAINTENANCE.value == "maintenance"
        assert MonitoringStatus.DEGRADED.value == "degraded"

    def test_alert_severity_values(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.EMERGENCY.value == "emergency"

    def test_incident_status_values(self):
        """Test IncidentStatus enum values."""
        assert IncidentStatus.DETECTED.value == "detected"
        assert IncidentStatus.INVESTIGATING.value == "investigating"
        assert IncidentStatus.CONFIRMED.value == "confirmed"
        assert IncidentStatus.CONTAINED.value == "contained"
        assert IncidentStatus.RESOLVED.value == "resolved"
        assert IncidentStatus.CLOSED.value == "closed"

    def test_monitoring_scope_values(self):
        """Test MonitoringScope enum values."""
        assert MonitoringScope.USER_ACTIVITY.value == "user_activity"
        assert MonitoringScope.SYSTEM_ACCESS.value == "system_access"
        assert MonitoringScope.NETWORK_TRAFFIC.value == "network_traffic"
        assert MonitoringScope.DATA_ACCESS.value == "data_access"
        assert MonitoringScope.AUTHENTICATION.value == "authentication"
        assert MonitoringScope.POLICY_VIOLATIONS.value == "policy_violations"
        assert MonitoringScope.THREAT_INDICATORS.value == "threat_indicators"
        assert MonitoringScope.COMPLIANCE.value == "compliance"


class TestSecurityAlert:
    """Comprehensive tests for SecurityAlert class."""

    def test_security_alert_creation_success(self):
        """Test successful SecurityAlert creation."""
        alert = SecurityAlert(
            alert_id="alert_001",
            alert_type="authentication_failure",
            severity=AlertSeverity.HIGH,
            title="Failed Login Attempt",
            description="Multiple failed login attempts detected",
            source="auth_monitor",
            detected_at=datetime.now(UTC),
            indicators=["brute_force_pattern", "suspicious_source"],
            affected_resources=["user_account_001", "login_service"],
            confidence=0.9,
            risk_score=RiskScore(0.8),
            remediation_steps=["Lock account", "Investigate source"],
            metadata={"source_ip": "192.168.1.100", "attempt_count": 5},
        )

        assert alert.alert_id == "alert_001"
        assert alert.alert_type == "authentication_failure"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.title == "Failed Login Attempt"
        assert alert.description == "Multiple failed login attempts detected"
        assert alert.source == "auth_monitor"
        assert len(alert.indicators) == 2
        assert len(alert.affected_resources) == 2
        assert alert.confidence == 0.9
        assert alert.risk_score.value == 0.8
        assert len(alert.remediation_steps) == 2
        assert alert.metadata["source_ip"] == "192.168.1.100"

    def test_security_alert_validation_errors(self):
        """Test SecurityAlert validation errors."""
        # Empty alert ID
        with pytest.raises(ValueError, match="Alert ID, type, and title are required"):
            SecurityAlert(
                alert_id="",
                alert_type="test_type",
                severity=AlertSeverity.LOW,
                title="Test Alert",
                description="Test description",
                source="test_source",
                detected_at=datetime.now(UTC),
            )

        # Empty alert type
        with pytest.raises(ValueError, match="Alert ID, type, and title are required"):
            SecurityAlert(
                alert_id="alert_001",
                alert_type="",
                severity=AlertSeverity.LOW,
                title="Test Alert",
                description="Test description",
                source="test_source",
                detected_at=datetime.now(UTC),
            )

        # Empty title
        with pytest.raises(ValueError, match="Alert ID, type, and title are required"):
            SecurityAlert(
                alert_id="alert_001",
                alert_type="test_type",
                severity=AlertSeverity.LOW,
                title="",
                description="Test description",
                source="test_source",
                detected_at=datetime.now(UTC),
            )

    def test_security_alert_confidence_validation_too_low(self):
        """Test SecurityAlert confidence validation - too low."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SecurityAlert(
                alert_id="alert_001",
                alert_type="test_type",
                severity=AlertSeverity.LOW,
                title="Test Alert",
                description="Test description",
                source="test_source",
                detected_at=datetime.now(UTC),
                confidence=-0.1,  # Invalid confidence
            )

    def test_security_alert_confidence_validation_too_high(self):
        """Test SecurityAlert confidence validation - too high."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SecurityAlert(
                alert_id="alert_001",
                alert_type="test_type",
                severity=AlertSeverity.LOW,
                title="Test Alert",
                description="Test description",
                source="test_source",
                detected_at=datetime.now(UTC),
                confidence=1.1,  # Invalid confidence
            )


class TestSecurityIncident:
    """Comprehensive tests for SecurityIncident class."""

    def test_security_incident_creation_success(self):
        """Test successful SecurityIncident creation."""
        current_time = datetime.now(UTC)
        incident = SecurityIncident(
            incident_id="incident_001",
            incident_type="data_breach",
            severity=ThreatSeverity.HIGH,
            title="Data Access Incident",
            description="Unauthorized access to sensitive data detected",
            status=IncidentStatus.DETECTED,
            detected_at=current_time,
            updated_at=current_time,
            related_alerts=["alert_001", "alert_002"],
            affected_systems=["database_server", "web_server"],
            impact_assessment={"data_exposed": True, "users_affected": 1000},
            response_actions=["Isolate systems", "Reset credentials"],
            timeline=[{"timestamp": current_time.isoformat(), "action": "detected"}],
            assigned_to="security_team",
            resolution_notes="Investigation ongoing",
            metadata={"source": "automated_detection"},
        )

        assert incident.incident_id == "incident_001"
        assert incident.incident_type == "data_breach"
        assert incident.severity == ThreatSeverity.HIGH
        assert incident.title == "Data Access Incident"
        assert incident.description == "Unauthorized access to sensitive data detected"
        assert incident.status == IncidentStatus.DETECTED
        assert incident.detected_at == current_time
        assert incident.updated_at == current_time
        assert len(incident.related_alerts) == 2
        assert len(incident.affected_systems) == 2
        assert incident.impact_assessment["users_affected"] == 1000
        assert len(incident.response_actions) == 2
        assert len(incident.timeline) == 1
        assert incident.assigned_to == "security_team"
        assert incident.resolution_notes == "Investigation ongoing"

    def test_security_incident_validation_errors(self):
        """Test SecurityIncident validation errors."""
        current_time = datetime.now(UTC)

        # Empty incident ID
        with pytest.raises(ValueError, match="Incident ID, type, and title are required"):
            SecurityIncident(
                incident_id="",
                incident_type="test_type",
                severity=ThreatSeverity.LOW,
                title="Test Incident",
                description="Test description",
                status=IncidentStatus.DETECTED,
                detected_at=current_time,
                updated_at=current_time,
            )

        # Empty incident type
        with pytest.raises(ValueError, match="Incident ID, type, and title are required"):
            SecurityIncident(
                incident_id="incident_001",
                incident_type="",
                severity=ThreatSeverity.LOW,
                title="Test Incident",
                description="Test description",
                status=IncidentStatus.DETECTED,
                detected_at=current_time,
                updated_at=current_time,
            )

        # Empty title
        with pytest.raises(ValueError, match="Incident ID, type, and title are required"):
            SecurityIncident(
                incident_id="incident_001",
                incident_type="test_type",
                severity=ThreatSeverity.LOW,
                title="",
                description="Test description",
                status=IncidentStatus.DETECTED,
                detected_at=current_time,
                updated_at=current_time,
            )


class TestMonitoringRule:
    """Comprehensive tests for MonitoringRule class."""

    def test_monitoring_rule_creation_success(self):
        """Test successful MonitoringRule creation."""
        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Authentication Failure Monitor",
            scope=MonitoringScope.AUTHENTICATION,
            description="Monitor for authentication failures",
            conditions={"event_types": ["auth_failure"], "threshold": 3},
            actions={"alert": True, "escalate": False},
            enabled=True,
            priority=75,
            alert_threshold=3,
            time_window=300,
            metadata={"created_by": "security_admin"},
        )

        assert rule.rule_id == "rule_001"
        assert rule.rule_name == "Authentication Failure Monitor"
        assert rule.scope == MonitoringScope.AUTHENTICATION
        assert rule.description == "Monitor for authentication failures"
        assert rule.conditions["threshold"] == 3
        assert rule.actions["alert"] is True
        assert rule.enabled is True
        assert rule.priority == 75
        assert rule.alert_threshold == 3
        assert rule.time_window == 300
        assert rule.metadata["created_by"] == "security_admin"

    def test_monitoring_rule_validation_errors(self):
        """Test MonitoringRule validation errors."""
        # Empty rule ID
        with pytest.raises(ValueError, match="Rule ID, name, and description are required"):
            MonitoringRule(
                rule_id="",
                rule_name="Test Rule",
                scope=MonitoringScope.USER_ACTIVITY,
                description="Test description",
                conditions={"test": True},
                actions={"alert": True},
            )

        # Empty rule name
        with pytest.raises(ValueError, match="Rule ID, name, and description are required"):
            MonitoringRule(
                rule_id="rule_001",
                rule_name="",
                scope=MonitoringScope.USER_ACTIVITY,
                description="Test description",
                conditions={"test": True},
                actions={"alert": True},
            )

        # Empty description
        with pytest.raises(ValueError, match="Rule ID, name, and description are required"):
            MonitoringRule(
                rule_id="rule_001",
                rule_name="Test Rule",
                scope=MonitoringScope.USER_ACTIVITY,
                description="",
                conditions={"test": True},
                actions={"alert": True},
            )

    def test_monitoring_rule_priority_validation_too_low(self):
        """Test MonitoringRule priority validation - too low."""
        with pytest.raises(ValueError, match="Priority must be between 1 and 100"):
            MonitoringRule(
                rule_id="rule_001",
                rule_name="Test Rule",
                scope=MonitoringScope.USER_ACTIVITY,
                description="Test description",
                conditions={"test": True},
                actions={"alert": True},
                priority=0,  # Invalid priority
            )

    def test_monitoring_rule_priority_validation_too_high(self):
        """Test MonitoringRule priority validation - too high."""
        with pytest.raises(ValueError, match="Priority must be between 1 and 100"):
            MonitoringRule(
                rule_id="rule_001",
                rule_name="Test Rule",
                scope=MonitoringScope.USER_ACTIVITY,
                description="Test description",
                conditions={"test": True},
                actions={"alert": True},
                priority=101,  # Invalid priority
            )

    def test_monitoring_rule_alert_threshold_validation(self):
        """Test MonitoringRule alert threshold validation."""
        with pytest.raises(ValueError, match="Alert threshold must be at least 1"):
            MonitoringRule(
                rule_id="rule_001",
                rule_name="Test Rule",
                scope=MonitoringScope.USER_ACTIVITY,
                description="Test description",
                conditions={"test": True},
                actions={"alert": True},
                alert_threshold=0,  # Invalid threshold
            )

    def test_monitoring_rule_time_window_validation(self):
        """Test MonitoringRule time window validation."""
        with pytest.raises(ValueError, match="Time window must be at least 1 second"):
            MonitoringRule(
                rule_id="rule_001",
                rule_name="Test Rule",
                scope=MonitoringScope.USER_ACTIVITY,
                description="Test description",
                conditions={"test": True},
                actions={"alert": True},
                time_window=0,  # Invalid time window
            )


class TestSecurityEvent:
    """Comprehensive tests for SecurityEvent class."""

    def test_security_event_creation_success(self):
        """Test successful SecurityEvent creation."""
        current_time = datetime.now(UTC)
        context = SecurityContext(
            user_id="user_001",
            device_id="device_001",
            trust_level=TrustLevel.HIGH,
            risk_score=0.2,
            location="office",
            session_id="session_001",
            timestamp=current_time,
        )

        event = SecurityEvent(
            event_id="event_001",
            event_type="user_login",
            source="auth_service",
            timestamp=current_time,
            context=context,
            data={"user_id": "user_001", "source_ip": "192.168.1.100"},
            risk_indicators=["new_location", "unusual_time"],
            severity=AlertSeverity.MEDIUM,
            processed=False,
        )

        assert event.event_id == "event_001"
        assert event.event_type == "user_login"
        assert event.source == "auth_service"
        assert event.timestamp == current_time
        assert event.context == context
        assert event.data["user_id"] == "user_001"
        assert len(event.risk_indicators) == 2
        assert event.severity == AlertSeverity.MEDIUM
        assert event.processed is False

    def test_security_event_validation_errors(self):
        """Test SecurityEvent validation errors."""
        current_time = datetime.now(UTC)

        # Empty event ID
        with pytest.raises(ValueError, match="Event ID, type, and source are required"):
            SecurityEvent(
                event_id="",
                event_type="test_type",
                source="test_source",
                timestamp=current_time,
            )

        # Empty event type
        with pytest.raises(ValueError, match="Event ID, type, and source are required"):
            SecurityEvent(
                event_id="event_001",
                event_type="",
                source="test_source",
                timestamp=current_time,
            )

        # Empty source
        with pytest.raises(ValueError, match="Event ID, type, and source are required"):
            SecurityEvent(
                event_id="event_001",
                event_type="test_type",
                source="",
                timestamp=current_time,
            )


class TestSecurityMonitor:
    """Comprehensive tests for SecurityMonitor core functionality."""

    @pytest.fixture
    def security_monitor(self):
        """Create SecurityMonitor instance for testing."""
        return SecurityMonitor()

    @pytest.fixture
    def sample_monitoring_rule(self):
        """Create sample MonitoringRule for testing."""
        return MonitoringRule(
            rule_id="test_rule_001",
            rule_name="Test Authentication Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Monitor authentication events",
            conditions={
                "event_types": ["authentication_failure"],
                "min_severity": "medium",
                "sources": ["auth_service"],
            },
            actions={
                "alert": True,
                "alert_severity": "high",
                "remediation_steps": ["Lock account", "Investigate"],
            },
            enabled=True,
            priority=80,
            alert_threshold=3,
            time_window=300,
        )

    @pytest.fixture
    def sample_security_event(self):
        """Create sample SecurityEvent for testing."""
        return SecurityEvent(
            event_id="test_event_001",
            event_type="authentication_failure",
            source="auth_service",
            timestamp=datetime.now(UTC),
            data={"user_id": "user_001", "source_ip": "192.168.1.100"},
            risk_indicators=["brute_force_pattern"],
            severity=AlertSeverity.MEDIUM,
        )

    def test_security_monitor_initialization(self, security_monitor):
        """Test SecurityMonitor initialization and default state."""
        assert security_monitor.monitoring_rules == {}
        assert security_monitor.active_alerts == {}
        assert security_monitor.active_incidents == {}
        assert security_monitor.security_events == []
        assert security_monitor.event_buffer == []
        assert len(security_monitor.monitoring_status) == len(MonitoringScope)

        # Check all monitoring scopes are initialized as INACTIVE
        for scope in MonitoringScope:
            assert security_monitor.monitoring_status[scope] == MonitoringStatus.INACTIVE

        # Analytics and metrics initialization
        assert security_monitor.threat_indicators == {}
        assert security_monitor.alert_statistics == {}
        assert security_monitor.incident_statistics == {}
        assert security_monitor.performance_metrics == {}

        # Real-time monitoring initialization
        assert security_monitor.monitoring_tasks == {}
        assert security_monitor.alert_callbacks == []
        assert security_monitor.incident_callbacks == []

    @pytest.mark.asyncio
    async def test_register_monitoring_rule_success(self, security_monitor, sample_monitoring_rule):
        """Test successful monitoring rule registration."""
        with patch.object(security_monitor, '_validate_monitoring_rule') as mock_validate, \
             patch.object(security_monitor, '_check_rule_conflicts') as mock_conflicts, \
             patch.object(security_monitor, '_start_rule_monitoring') as mock_start:

            # Setup mocks
            from src.core.either import Either
            mock_validate.return_value = Either.right(None)
            mock_conflicts.return_value = Either.right(None)
            mock_start.return_value = None

            result = await security_monitor.register_monitoring_rule(sample_monitoring_rule)

            assert result.is_right()
            assert "registered successfully" in result.get_right()
            assert sample_monitoring_rule.rule_id in security_monitor.monitoring_rules
            assert security_monitor.monitoring_rules[sample_monitoring_rule.rule_id] == sample_monitoring_rule

            # Verify mocks were called
            mock_validate.assert_called_once_with(sample_monitoring_rule)
            mock_conflicts.assert_called_once_with(sample_monitoring_rule)
            mock_start.assert_called_once_with(sample_monitoring_rule)

    @pytest.mark.asyncio
    async def test_register_monitoring_rule_validation_failure(self, security_monitor, sample_monitoring_rule):
        """Test monitoring rule registration validation failure."""
        with patch.object(security_monitor, '_validate_monitoring_rule') as mock_validate:
            from src.core.either import Either
            mock_validate.return_value = Either.left(
                SecurityMonitoringError("Invalid rule", "VALIDATION_ERROR", SecurityOperation.MONITOR)
            )

            result = await security_monitor.register_monitoring_rule(sample_monitoring_rule)

            assert result.is_left()
            assert "Invalid rule" in str(result.get_left())
            assert sample_monitoring_rule.rule_id not in security_monitor.monitoring_rules

    @pytest.mark.asyncio
    async def test_register_monitoring_rule_conflict_failure(self, security_monitor, sample_monitoring_rule):
        """Test monitoring rule registration conflict failure."""
        with patch.object(security_monitor, '_validate_monitoring_rule') as mock_validate, \
             patch.object(security_monitor, '_check_rule_conflicts') as mock_conflicts:

            from src.core.either import Either
            mock_validate.return_value = Either.right(None)
            mock_conflicts.return_value = Either.left(
                SecurityMonitoringError("Rule conflict", "RULE_CONFLICT", SecurityOperation.MONITOR)
            )

            result = await security_monitor.register_monitoring_rule(sample_monitoring_rule)

            assert result.is_left()
            assert "Rule conflict" in str(result.get_left())
            assert sample_monitoring_rule.rule_id not in security_monitor.monitoring_rules

    @pytest.mark.asyncio
    async def test_register_monitoring_rule_exception_handling(self, security_monitor, sample_monitoring_rule):
        """Test monitoring rule registration exception handling."""
        with patch.object(security_monitor, '_validate_monitoring_rule') as mock_validate:
            mock_validate.side_effect = Exception("Validation error")

            result = await security_monitor.register_monitoring_rule(sample_monitoring_rule)

            assert result.is_left()
            error = result.get_left()
            assert "Failed to register monitoring rule" in error.message
            assert "RULE_REGISTRATION_ERROR" == error.error_code

    @pytest.mark.asyncio
    async def test_process_security_event_dict_input(self, security_monitor):
        """Test processing security event from dictionary input."""
        event_dict = {
            "event_id": "test_event_001",
            "type": "authentication_failure",
            "source": "auth_service",
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": "user_001",
            "source_ip": "192.168.1.100",
            "risk_indicators": ["brute_force_pattern"],
        }

        with patch.object(security_monitor, '_dict_to_security_event') as mock_convert:
            mock_event = SecurityEvent(
                event_id="test_event_001",
                event_type="authentication_failure",
                source="auth_service",
                timestamp=datetime.now(UTC),
            )
            mock_convert.return_value = mock_event

            result = await security_monitor.process_security_event(event_dict)

            assert result.is_right()
            alerts = result.get_right()
            assert isinstance(alerts, list)

            # Verify event was added to collections
            assert len(security_monitor.event_buffer) == 1
            assert len(security_monitor.security_events) == 1

            mock_convert.assert_called_once_with(event_dict)

    @pytest.mark.asyncio
    async def test_process_security_event_object_input(self, security_monitor, sample_security_event):
        """Test processing security event from SecurityEvent object."""
        result = await security_monitor.process_security_event(sample_security_event)

        assert result.is_right()
        alerts = result.get_right()
        assert isinstance(alerts, list)

        # Verify event was added to collections
        assert len(security_monitor.event_buffer) == 1
        assert len(security_monitor.security_events) == 1
        assert security_monitor.event_buffer[0].event_id == sample_security_event.event_id

    @pytest.mark.asyncio
    async def test_process_security_event_with_matching_rule(self, security_monitor, sample_monitoring_rule, sample_security_event):
        """Test processing security event that matches monitoring rule."""
        # Register the monitoring rule
        security_monitor.monitoring_rules[sample_monitoring_rule.rule_id] = sample_monitoring_rule

        with patch.object(security_monitor, '_event_matches_rule') as mock_matches, \
             patch.object(security_monitor, '_get_matching_events_in_window') as mock_window, \
             patch.object(security_monitor, '_generate_alert_from_rule') as mock_generate, \
             patch.object(security_monitor, '_handle_new_alert') as mock_handle:

            from src.core.either import Either
            mock_matches.return_value = True
            mock_window.return_value = [sample_security_event] * 3  # Meet threshold

            mock_alert = SecurityAlert(
                alert_id="alert_001",
                alert_type="authentication",
                severity=AlertSeverity.HIGH,
                title="Authentication Alert",
                description="Alert generated from rule",
                source="security_monitor",
                detected_at=datetime.now(UTC),
            )
            mock_generate.return_value = Either.right(mock_alert)
            mock_handle.return_value = None

            result = await security_monitor.process_security_event(sample_security_event)

            assert result.is_right()
            alerts = result.get_right()
            assert len(alerts) == 1
            assert alerts[0].alert_id == "alert_001"

            # Verify alert was stored
            assert "alert_001" in security_monitor.active_alerts

            mock_matches.assert_called()
            mock_window.assert_called_once_with(sample_monitoring_rule)
            mock_generate.assert_called_once()
            mock_handle.assert_called_once_with(mock_alert)

    @pytest.mark.asyncio
    async def test_process_security_event_exception_handling(self, security_monitor):
        """Test security event processing exception handling."""
        # Create invalid event that will cause exception
        invalid_event = {"invalid": "data"}

        result = await security_monitor.process_security_event(invalid_event)

        assert result.is_left()
        error = result.get_left()
        assert "Failed to process security event" in error.message
        assert "EVENT_PROCESSING_ERROR" == error.error_code

    @pytest.mark.asyncio
    async def test_start_monitoring_success(self, security_monitor):
        """Test successful monitoring start."""
        scope = MonitoringScope.AUTHENTICATION

        with patch.object(security_monitor, '_run_continuous_monitoring') as mock_run:
            # Mock asyncio.create_task
            mock_task = AsyncMock()
            with patch('asyncio.create_task', return_value=mock_task) as mock_create_task:
                result = await security_monitor.start_monitoring(scope)

                assert result.is_right()
                assert f"Monitoring started for scope {scope.value}" in result.get_right()
                assert security_monitor.monitoring_status[scope] == MonitoringStatus.ACTIVE
                assert scope in security_monitor.monitoring_tasks

                mock_create_task.assert_called_once()
                mock_run.assert_not_called()  # Called inside task

    @pytest.mark.asyncio
    async def test_start_monitoring_already_active(self, security_monitor):
        """Test starting monitoring when already active."""
        scope = MonitoringScope.AUTHENTICATION

        # Create a mock task that's not done
        mock_task = Mock()
        mock_task.done.return_value = False
        security_monitor.monitoring_tasks[scope] = mock_task

        result = await security_monitor.start_monitoring(scope)

        assert result.is_left()
        error = result.get_left()
        assert "Monitoring already active" in error.message
        assert "MONITORING_ALREADY_ACTIVE" == error.error_code

    @pytest.mark.asyncio
    async def test_start_monitoring_exception_handling(self, security_monitor):
        """Test monitoring start exception handling."""
        scope = MonitoringScope.AUTHENTICATION

        with patch('asyncio.create_task') as mock_create_task:
            mock_create_task.side_effect = Exception("Task creation failed")

            result = await security_monitor.start_monitoring(scope)

            assert result.is_left()
            error = result.get_left()
            assert "Failed to start monitoring" in error.message
            assert "MONITORING_START_ERROR" == error.error_code

    @pytest.mark.asyncio
    async def test_stop_monitoring_success(self, security_monitor):
        """Test successful monitoring stop."""
        scope = MonitoringScope.AUTHENTICATION

        # Create a mock task
        mock_task = Mock()
        mock_task.done.return_value = False
        security_monitor.monitoring_tasks[scope] = mock_task
        security_monitor.monitoring_status[scope] = MonitoringStatus.ACTIVE

        result = await security_monitor.stop_monitoring(scope)

        assert result.is_right()
        assert f"Monitoring stopped for scope {scope.value}" in result.get_right()
        assert security_monitor.monitoring_status[scope] == MonitoringStatus.INACTIVE
        assert scope not in security_monitor.monitoring_tasks

        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_monitoring_no_task(self, security_monitor):
        """Test stopping monitoring when no task exists."""
        scope = MonitoringScope.AUTHENTICATION

        result = await security_monitor.stop_monitoring(scope)

        assert result.is_right()
        assert f"Monitoring stopped for scope {scope.value}" in result.get_right()
        assert security_monitor.monitoring_status[scope] == MonitoringStatus.INACTIVE

    @pytest.mark.asyncio
    async def test_stop_monitoring_exception_handling(self, security_monitor):
        """Test monitoring stop exception handling."""
        scope = MonitoringScope.AUTHENTICATION

        # Create a mock task that will raise exception on cancel
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel.side_effect = Exception("Cancel failed")
        security_monitor.monitoring_tasks[scope] = mock_task

        result = await security_monitor.stop_monitoring(scope)

        assert result.is_left()
        error = result.get_left()
        assert "Failed to stop monitoring" in error.message
        assert "MONITORING_STOP_ERROR" == error.error_code

    @pytest.mark.asyncio
    async def test_create_incident_success(self, security_monitor):
        """Test successful incident creation."""
        incident_type = "data_breach"
        severity = ThreatSeverity.HIGH
        title = "Data Access Violation"
        description = "Unauthorized access to sensitive data"
        related_alerts = ["alert_001", "alert_002"]

        with patch.object(security_monitor, '_handle_new_incident') as mock_handle:
            mock_handle.return_value = None

            result = await security_monitor.create_incident(
                incident_type, severity, title, description, related_alerts
            )

            assert result.is_right()
            incident = result.get_right()
            assert incident.incident_type == incident_type
            assert incident.severity == severity
            assert incident.title == title
            assert incident.description == description
            assert incident.status == IncidentStatus.DETECTED
            assert incident.related_alerts == related_alerts
            assert len(incident.timeline) == 1

            # Verify incident was stored
            assert incident.incident_id in security_monitor.active_incidents

            # Verify statistics were updated
            assert security_monitor.incident_statistics[incident_type] == 1

            mock_handle.assert_called_once_with(incident)

    @pytest.mark.asyncio
    async def test_create_incident_none_alerts(self, security_monitor):
        """Test incident creation with None alerts."""
        result = await security_monitor.create_incident(
            "test_incident", ThreatSeverity.LOW, "Test", "Test description", None
        )

        assert result.is_right()
        incident = result.get_right()
        assert incident.related_alerts == []

    @pytest.mark.asyncio
    async def test_create_incident_exception_handling(self, security_monitor):
        """Test incident creation exception handling."""
        with patch.object(security_monitor, '_handle_new_incident') as mock_handle:
            mock_handle.side_effect = Exception("Handler failed")

            result = await security_monitor.create_incident(
                "test_incident", ThreatSeverity.LOW, "Test", "Test description"
            )

            assert result.is_left()
            error = result.get_left()
            assert "Failed to create incident" in error.message
            assert "INCIDENT_CREATION_ERROR" == error.error_code

    @pytest.mark.asyncio
    async def test_update_incident_status_success(self, security_monitor):
        """Test successful incident status update."""
        # Create initial incident
        incident_id = "incident_001"
        initial_incident = SecurityIncident(
            incident_id=incident_id,
            incident_type="test_incident",
            severity=ThreatSeverity.MEDIUM,
            title="Test Incident",
            description="Test description",
            status=IncidentStatus.DETECTED,
            detected_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            timeline=[{"timestamp": datetime.now(UTC).isoformat(), "action": "created"}],
        )
        security_monitor.active_incidents[incident_id] = initial_incident

        new_status = IncidentStatus.INVESTIGATING
        notes = "Investigation started"

        result = await security_monitor.update_incident_status(incident_id, new_status, notes)

        assert result.is_right()
        updated_incident = result.get_right()
        assert updated_incident.status == new_status
        assert len(updated_incident.timeline) == 2
        assert updated_incident.timeline[-1]["action"] == "status_change"
        assert updated_incident.timeline[-1]["notes"] == notes

        # Verify incident was updated in storage
        assert security_monitor.active_incidents[incident_id].status == new_status

    @pytest.mark.asyncio
    async def test_update_incident_status_not_found(self, security_monitor):
        """Test incident status update when incident not found."""
        result = await security_monitor.update_incident_status(
            "nonexistent_incident", IncidentStatus.INVESTIGATING
        )

        assert result.is_left()
        error = result.get_left()
        assert "Incident nonexistent_incident not found" in error.message
        assert "INCIDENT_NOT_FOUND" == error.error_code

    @pytest.mark.asyncio
    async def test_update_incident_status_resolved(self, security_monitor):
        """Test incident status update to resolved with resolution notes."""
        # Create initial incident
        incident_id = "incident_001"
        initial_incident = SecurityIncident(
            incident_id=incident_id,
            incident_type="test_incident",
            severity=ThreatSeverity.MEDIUM,
            title="Test Incident",
            description="Test description",
            status=IncidentStatus.INVESTIGATING,
            detected_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            timeline=[{"timestamp": datetime.now(UTC).isoformat(), "action": "created"}],
        )
        security_monitor.active_incidents[incident_id] = initial_incident

        resolution_notes = "Issue resolved by patching vulnerability"

        result = await security_monitor.update_incident_status(
            incident_id, IncidentStatus.RESOLVED, resolution_notes
        )

        assert result.is_right()
        updated_incident = result.get_right()
        assert updated_incident.status == IncidentStatus.RESOLVED
        assert updated_incident.resolution_notes == resolution_notes

    @pytest.mark.asyncio
    async def test_update_incident_status_closed(self, security_monitor):
        """Test incident status update to closed removes from active incidents."""
        # Create initial incident
        incident_id = "incident_001"
        initial_incident = SecurityIncident(
            incident_id=incident_id,
            incident_type="test_incident",
            severity=ThreatSeverity.MEDIUM,
            title="Test Incident",
            description="Test description",
            status=IncidentStatus.RESOLVED,
            detected_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            timeline=[{"timestamp": datetime.now(UTC).isoformat(), "action": "created"}],
        )
        security_monitor.active_incidents[incident_id] = initial_incident

        result = await security_monitor.update_incident_status(incident_id, IncidentStatus.CLOSED)

        assert result.is_right()
        updated_incident = result.get_right()
        assert updated_incident.status == IncidentStatus.CLOSED

        # Verify incident was removed from active incidents
        assert incident_id not in security_monitor.active_incidents

    @pytest.mark.asyncio
    async def test_update_incident_status_exception_handling(self, security_monitor):
        """Test incident status update exception handling."""
        # Create malformed incident that will cause exception
        incident_id = "incident_001"
        security_monitor.active_incidents[incident_id] = "invalid_incident"  # Not a SecurityIncident

        result = await security_monitor.update_incident_status(incident_id, IncidentStatus.INVESTIGATING)

        assert result.is_left()
        error = result.get_left()
        assert "Failed to update incident status" in error.message
        assert "INCIDENT_UPDATE_ERROR" == error.error_code

    @pytest.mark.asyncio
    async def test_get_security_metrics_success(self, security_monitor):
        """Test successful security metrics retrieval."""
        # Add some sample data
        current_time = datetime.now(UTC)

        # Add events
        event1 = SecurityEvent(
            event_id="event_001",
            event_type="authentication_failure",
            source="auth_service",
            timestamp=current_time - timedelta(hours=1),
        )
        event2 = SecurityEvent(
            event_id="event_002",
            event_type="policy_violation",
            source="policy_engine",
            timestamp=current_time - timedelta(minutes=30),
        )
        security_monitor.security_events = [event1, event2]

        # Add alerts
        alert1 = SecurityAlert(
            alert_id="alert_001",
            alert_type="authentication",
            severity=AlertSeverity.HIGH,
            title="Auth Alert",
            description="Authentication alert",
            source="auth_monitor",
            detected_at=current_time - timedelta(hours=1),
            metadata={"processing_time_ms": 150},
        )
        security_monitor.active_alerts["alert_001"] = alert1

        # Add incidents
        incident1 = SecurityIncident(
            incident_id="incident_001",
            incident_type="data_breach",
            severity=ThreatSeverity.HIGH,
            title="Data Breach",
            description="Data breach incident",
            status=IncidentStatus.RESOLVED,
            detected_at=current_time - timedelta(hours=2),
            updated_at=current_time - timedelta(hours=1),
        )
        security_monitor.active_incidents["incident_001"] = incident1

        # Set some monitoring scopes as active
        security_monitor.monitoring_status[MonitoringScope.AUTHENTICATION] = MonitoringStatus.ACTIVE
        security_monitor.monitoring_status[MonitoringScope.DATA_ACCESS] = MonitoringStatus.ACTIVE

        result = await security_monitor.get_security_metrics()

        assert result.is_right()
        metrics = result.get_right()
        assert isinstance(metrics, SecurityMetrics)
        assert metrics.threats_detected == 1  # One alert in time range
        assert metrics.policy_violations == 1  # One policy violation event
        assert metrics.incidents_resolved == 1  # One resolved incident
        assert metrics.metadata["total_events"] == 2
        assert metrics.metadata["total_alerts"] == 1
        assert metrics.metadata["monitoring_scopes_active"] == 2
        assert "alert_processing" in metrics.response_times

    @pytest.mark.asyncio
    async def test_get_security_metrics_custom_time_range(self, security_monitor):
        """Test security metrics with custom time range."""
        time_range = timedelta(hours=6)

        result = await security_monitor.get_security_metrics(time_range)

        assert result.is_right()
        metrics = result.get_right()
        assert isinstance(metrics, SecurityMetrics)
        # Verify time range is properly used
        assert metrics.period_end - metrics.period_start == time_range

    @pytest.mark.asyncio
    async def test_get_security_metrics_exception_handling(self, security_monitor):
        """Test security metrics exception handling."""
        # Cause exception by making security_events non-iterable
        security_monitor.security_events = "invalid"

        result = await security_monitor.get_security_metrics()

        assert result.is_left()
        error = result.get_left()
        assert "Failed to get security metrics" in error.message
        assert "METRICS_ERROR" == error.error_code

    def test_register_alert_callback(self, security_monitor):
        """Test alert callback registration."""
        def test_callback(alert):
            pass

        security_monitor.register_alert_callback(test_callback)

        assert test_callback in security_monitor.alert_callbacks

    def test_register_incident_callback(self, security_monitor):
        """Test incident callback registration."""
        def test_callback(incident):
            pass

        security_monitor.register_incident_callback(test_callback)

        assert test_callback in security_monitor.incident_callbacks


class TestSecurityMonitorValidation:
    """Test SecurityMonitor validation methods."""

    @pytest.fixture
    def security_monitor(self):
        """Create SecurityMonitor instance for testing."""
        return SecurityMonitor()

    def test_validate_monitoring_rule_success(self, security_monitor):
        """Test successful monitoring rule validation."""
        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Test Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Test rule",
            conditions={"event_types": ["auth_failure"]},
            actions={"alert": True},
        )

        result = security_monitor._validate_monitoring_rule(rule)

        assert result.is_right()

    def test_validate_monitoring_rule_missing_conditions(self, security_monitor):
        """Test monitoring rule validation with missing conditions."""
        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Test Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Test rule",
            conditions={},  # Empty conditions
            actions={"alert": True},
        )

        result = security_monitor._validate_monitoring_rule(rule)

        assert result.is_left()
        error = result.get_left()
        assert "must have conditions" in error.message
        assert "MISSING_CONDITIONS" == error.error_code

    def test_validate_monitoring_rule_missing_actions(self, security_monitor):
        """Test monitoring rule validation with missing actions."""
        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Test Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Test rule",
            conditions={"event_types": ["auth_failure"]},
            actions={},  # Empty actions
        )

        result = security_monitor._validate_monitoring_rule(rule)

        assert result.is_left()
        error = result.get_left()
        assert "must have actions" in error.message
        assert "MISSING_ACTIONS" == error.error_code

    def test_check_rule_conflicts_no_conflict(self, security_monitor):
        """Test rule conflict check with no conflicts."""
        new_rule = MonitoringRule(
            rule_id="rule_002",
            rule_name="New Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="New rule",
            conditions={"event_types": ["auth_success"]},
            actions={"alert": True},
            priority=60,
        )

        result = security_monitor._check_rule_conflicts(new_rule)

        assert result.is_right()

    def test_check_rule_conflicts_with_conflict(self, security_monitor):
        """Test rule conflict check with conflicts."""
        # Add existing rule
        existing_rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Existing Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Existing rule",
            conditions={"event_types": ["auth_failure"]},
            actions={"alert": True},
            enabled=True,
            priority=60,
        )
        security_monitor.monitoring_rules["rule_001"] = existing_rule

        # New rule with same scope, priority, and overlapping conditions
        new_rule = MonitoringRule(
            rule_id="rule_002",
            rule_name="New Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="New rule",
            conditions={"event_types": ["auth_failure"]},  # Same condition key
            actions={"alert": True},
            enabled=True,
            priority=60,  # Same priority
        )

        result = security_monitor._check_rule_conflicts(new_rule)

        assert result.is_left()
        error = result.get_left()
        assert "Rule conflict" in error.message
        assert "RULE_CONFLICT" == error.error_code

    def test_rules_have_overlapping_conditions_true(self, security_monitor):
        """Test rules overlap detection - overlapping conditions."""
        rule1 = MonitoringRule(
            rule_id="rule_001",
            rule_name="Rule 1",
            scope=MonitoringScope.AUTHENTICATION,
            description="Rule 1",
            conditions={"event_types": ["auth_failure"], "severity": "high"},
            actions={"alert": True},
        )

        rule2 = MonitoringRule(
            rule_id="rule_002",
            rule_name="Rule 2",
            scope=MonitoringScope.AUTHENTICATION,
            description="Rule 2",
            conditions={"event_types": ["auth_success"], "priority": "medium"},  # Share event_types key
            actions={"alert": True},
        )

        result = security_monitor._rules_have_overlapping_conditions(rule1, rule2)

        assert result is True

    def test_rules_have_overlapping_conditions_false(self, security_monitor):
        """Test rules overlap detection - no overlapping conditions."""
        rule1 = MonitoringRule(
            rule_id="rule_001",
            rule_name="Rule 1",
            scope=MonitoringScope.AUTHENTICATION,
            description="Rule 1",
            conditions={"event_types": ["auth_failure"]},
            actions={"alert": True},
        )

        rule2 = MonitoringRule(
            rule_id="rule_002",
            rule_name="Rule 2",
            scope=MonitoringScope.AUTHENTICATION,
            description="Rule 2",
            conditions={"severity": "high", "priority": "medium"},  # No shared keys
            actions={"alert": True},
        )

        result = security_monitor._rules_have_overlapping_conditions(rule1, rule2)

        assert result is False


class TestSecurityMonitorEventMatching:
    """Test SecurityMonitor event matching methods."""

    @pytest.fixture
    def security_monitor(self):
        """Create SecurityMonitor instance for testing."""
        return SecurityMonitor()

    def test_event_matches_rule_event_type_match(self, security_monitor):
        """Test event matching rule - event type match."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="authentication_failure",
            source="auth_service",
            timestamp=datetime.now(UTC),
        )

        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Auth Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Auth monitoring",
            conditions={"event_types": ["authentication_failure", "authentication_success"]},
            actions={"alert": True},
        )

        result = security_monitor._event_matches_rule(event, rule)

        assert result is True

    def test_event_matches_rule_event_type_no_match(self, security_monitor):
        """Test event matching rule - event type no match."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="user_login",
            source="auth_service",
            timestamp=datetime.now(UTC),
        )

        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Auth Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Auth monitoring",
            conditions={"event_types": ["authentication_failure"]},
            actions={"alert": True},
        )

        result = security_monitor._event_matches_rule(event, rule)

        assert result is False

    def test_event_matches_rule_severity_match(self, security_monitor):
        """Test event matching rule - severity match."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="test_event",
            source="test_service",
            timestamp=datetime.now(UTC),
            severity=AlertSeverity.HIGH,
        )

        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Severity Rule",
            scope=MonitoringScope.USER_ACTIVITY,
            description="Severity monitoring",
            conditions={"min_severity": "medium"},
            actions={"alert": True},
        )

        result = security_monitor._event_matches_rule(event, rule)

        assert result is True

    def test_event_matches_rule_severity_no_match(self, security_monitor):
        """Test event matching rule - severity no match."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="test_event",
            source="test_service",
            timestamp=datetime.now(UTC),
            severity=AlertSeverity.LOW,
        )

        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Severity Rule",
            scope=MonitoringScope.USER_ACTIVITY,
            description="Severity monitoring",
            conditions={"min_severity": "high"},
            actions={"alert": True},
        )

        result = security_monitor._event_matches_rule(event, rule)

        assert result is False

    def test_event_matches_rule_source_match(self, security_monitor):
        """Test event matching rule - source match."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="test_event",
            source="auth_service",
            timestamp=datetime.now(UTC),
        )

        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Source Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Source monitoring",
            conditions={"sources": ["auth_service", "user_service"]},
            actions={"alert": True},
        )

        result = security_monitor._event_matches_rule(event, rule)

        assert result is True

    def test_event_matches_rule_source_no_match(self, security_monitor):
        """Test event matching rule - source no match."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="test_event",
            source="file_service",
            timestamp=datetime.now(UTC),
        )

        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Source Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Source monitoring",
            conditions={"sources": ["auth_service", "user_service"]},
            actions={"alert": True},
        )

        result = security_monitor._event_matches_rule(event, rule)

        assert result is False

    def test_event_matches_rule_risk_indicators_match(self, security_monitor):
        """Test event matching rule - risk indicators match."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="test_event",
            source="test_service",
            timestamp=datetime.now(UTC),
            risk_indicators=["brute_force_attack", "suspicious_location"],
        )

        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Risk Rule",
            scope=MonitoringScope.THREAT_INDICATORS,
            description="Risk monitoring",
            conditions={"risk_indicators": ["brute_force_attack"]},
            actions={"alert": True},
        )

        result = security_monitor._event_matches_rule(event, rule)

        assert result is True

    def test_event_matches_rule_risk_indicators_no_match(self, security_monitor):
        """Test event matching rule - risk indicators no match."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="test_event",
            source="test_service",
            timestamp=datetime.now(UTC),
            risk_indicators=["normal_activity"],
        )

        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Risk Rule",
            scope=MonitoringScope.THREAT_INDICATORS,
            description="Risk monitoring",
            conditions={"risk_indicators": ["brute_force_attack"]},
            actions={"alert": True},
        )

        result = security_monitor._event_matches_rule(event, rule)

        assert result is False

    def test_event_matches_rule_exception_handling(self, security_monitor):
        """Test event matching rule - exception handling."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="test_event",
            source="test_service",
            timestamp=datetime.now(UTC),
        )

        # Create rule with invalid severity condition that will cause exception
        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Invalid Rule",
            scope=MonitoringScope.USER_ACTIVITY,
            description="Invalid monitoring",
            conditions={"min_severity": "invalid_severity"},
            actions={"alert": True},
        )

        result = security_monitor._event_matches_rule(event, rule)

        assert result is False  # Fail safe

    def test_get_matching_events_in_window(self, security_monitor):
        """Test getting matching events within time window."""
        current_time = datetime.now(UTC)

        # Create events - some within window, some outside
        event1 = SecurityEvent(
            event_id="event_001",
            event_type="authentication_failure",
            source="auth_service",
            timestamp=current_time - timedelta(seconds=100),  # Within window
        )
        event2 = SecurityEvent(
            event_id="event_002",
            event_type="authentication_failure",
            source="auth_service",
            timestamp=current_time - timedelta(seconds=400),  # Outside window
        )
        event3 = SecurityEvent(
            event_id="event_003",
            event_type="authentication_failure",
            source="auth_service",
            timestamp=current_time - timedelta(seconds=50),   # Within window
        )

        security_monitor.event_buffer = [event1, event2, event3]

        rule = MonitoringRule(
            rule_id="rule_001",
            rule_name="Auth Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Auth monitoring",
            conditions={"event_types": ["authentication_failure"]},
            actions={"alert": True},
            time_window=300,  # 5 minutes
        )

        matching_events = security_monitor._get_matching_events_in_window(rule)

        # Should only return events within 300 seconds (event1 and event3)
        assert len(matching_events) == 2
        assert event1 in matching_events
        assert event3 in matching_events
        assert event2 not in matching_events


class TestUtilityFunctions:
    """Test utility functions for security monitoring."""

    def test_create_monitoring_rule_success(self):
        """Test successful monitoring rule creation."""
        rule = create_monitoring_rule(
            rule_name="Test Rule",
            scope=MonitoringScope.AUTHENTICATION,
            description="Test monitoring rule",
            conditions={"event_types": ["auth_failure"]},
            actions={"alert": True},
            priority=80,
            alert_threshold=3,
            time_window=600,
        )

        assert rule.rule_id == "rule_test_rule"
        assert rule.rule_name == "Test Rule"
        assert rule.scope == MonitoringScope.AUTHENTICATION
        assert rule.description == "Test monitoring rule"
        assert rule.conditions["event_types"] == ["auth_failure"]
        assert rule.actions["alert"] is True
        assert rule.priority == 80
        assert rule.alert_threshold == 3
        assert rule.time_window == 600

    def test_create_monitoring_rule_defaults(self):
        """Test monitoring rule creation with default values."""
        rule = create_monitoring_rule(
            rule_name="Default Rule",
            scope=MonitoringScope.USER_ACTIVITY,
            description="Default rule",
            conditions={"test": True},
            actions={"alert": True},
        )

        assert rule.priority == 50  # Default
        assert rule.alert_threshold == 1  # Default
        assert rule.time_window == 300  # Default

    def test_create_security_event_success(self):
        """Test successful security event creation."""
        context = SecurityContext(
            user_id="user_001",
            device_id="device_001",
            trust_level=TrustLevel.MEDIUM,
            risk_score=0.3,
            location="office",
            session_id="session_001",
            timestamp=datetime.now(UTC),
        )

        event = create_security_event(
            event_type="user_login",
            source="auth_service",
            data={"user_id": "user_001", "ip": "192.168.1.100"},
            context=context,
            risk_indicators=["new_location"],
            severity=AlertSeverity.MEDIUM,
        )

        assert event.event_type == "user_login"
        assert event.source == "auth_service"
        assert event.data["user_id"] == "user_001"
        assert event.context == context
        assert event.risk_indicators == ["new_location"]
        assert event.severity == AlertSeverity.MEDIUM
        assert event.event_id.startswith("event_")

    def test_create_security_event_defaults(self):
        """Test security event creation with default values."""
        event = create_security_event(
            event_type="test_event",
            source="test_source",
            data={"test": True},
        )

        assert event.context is None  # Default
        assert event.risk_indicators == []  # Default
        assert event.severity == AlertSeverity.INFO  # Default


class TestThreatSummary:
    """Test ThreatSummary class."""

    def test_threat_summary_creation(self):
        """Test ThreatSummary creation."""
        summary = ThreatSummary(
            total_events=50,
            max_threat_level=ThreatSeverity.HIGH,
            user_id="user_001",
            time_window=timedelta(hours=24),
        )

        assert summary.total_events == 50
        assert summary.max_threat_level == ThreatSeverity.HIGH
        assert summary.user_id == "user_001"
        assert summary.time_window == timedelta(hours=24)

    def test_threat_summary_optional_fields(self):
        """Test ThreatSummary with optional fields."""
        summary = ThreatSummary(
            total_events=10,
            max_threat_level=ThreatSeverity.LOW,
        )

        assert summary.total_events == 10
        assert summary.max_threat_level == ThreatSeverity.LOW
        assert summary.user_id is None
        assert summary.time_window is None


class TestSecurityMonitorSimpleInterface:
    """Test SecurityMonitor simple interface methods for compatibility."""

    @pytest.fixture
    def security_monitor(self):
        """Create SecurityMonitor instance for testing."""
        return SecurityMonitor()

    def test_log_event(self, security_monitor):
        """Test logging security event."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="test_event",
            source="test_source",
            timestamp=datetime.now(UTC),
        )

        security_monitor.log_event(event)

        assert len(security_monitor.security_events) == 1
        assert security_monitor.security_events[0] == event

    def test_log_event_critical_with_callbacks(self, security_monitor):
        """Test logging critical event triggers callbacks."""
        callback_called = []

        def test_callback(event):
            callback_called.append(event)

        security_monitor.register_alert_callback(test_callback)

        critical_event = SecurityEvent(
            event_id="event_001",
            event_type="critical_event",
            source="test_source",
            timestamp=datetime.now(UTC),
            severity=AlertSeverity.CRITICAL,
        )

        security_monitor.log_event(critical_event)

        assert len(callback_called) == 1
        assert callback_called[0] == critical_event

    def test_get_events_default_limit(self, security_monitor):
        """Test getting events with default limit."""
        # Add more than 100 events
        events = []
        for i in range(150):
            event = SecurityEvent(
                event_id=f"event_{i:03d}",
                event_type="test_event",
                source="test_source",
                timestamp=datetime.now(UTC),
            )
            events.append(event)

        security_monitor.security_events = events

        result = security_monitor.get_events()

        assert len(result) == 100  # Default limit
        # Should return last 100 events
        assert result[0].event_id == "event_050"
        assert result[-1].event_id == "event_149"

    def test_get_events_custom_limit(self, security_monitor):
        """Test getting events with custom limit."""
        events = []
        for i in range(20):
            event = SecurityEvent(
                event_id=f"event_{i:03d}",
                event_type="test_event",
                source="test_source",
                timestamp=datetime.now(UTC),
            )
            events.append(event)

        security_monitor.security_events = events

        result = security_monitor.get_events(limit=5)

        assert len(result) == 5
        assert result[0].event_id == "event_015"
        assert result[-1].event_id == "event_019"

    def test_get_events_zero_limit(self, security_monitor):
        """Test getting events with zero limit returns all."""
        events = []
        for i in range(10):
            event = SecurityEvent(
                event_id=f"event_{i:03d}",
                event_type="test_event",
                source="test_source",
                timestamp=datetime.now(UTC),
            )
            events.append(event)

        security_monitor.security_events = events

        result = security_monitor.get_events(limit=0)

        assert len(result) == 10  # All events

    def test_detect_threats_no_threats(self, security_monitor):
        """Test threat detection with no threats detected."""
        # Add normal events
        current_time = datetime.now(UTC)
        event = SecurityEvent(
            event_id="event_001",
            event_type="user_login",
            source="auth_service",
            timestamp=current_time,
            data={"user_id": "user_001"},
        )
        security_monitor.security_events = [event]

        threats = security_monitor.detect_threats("user_001")

        assert len(threats) == 0

    def test_detect_threats_brute_force_detected(self, security_monitor):
        """Test threat detection - brute force attack."""
        current_time = datetime.now(UTC)
        user_id = "user_001"

        # Add multiple authentication failures
        events = []
        for i in range(5):  # Above threshold of 3
            event = SecurityEvent(
                event_id=f"event_{i:03d}",
                event_type="authentication_failure",
                source="auth_service",
                timestamp=current_time - timedelta(minutes=i * 5),
                data={"user_id": user_id},
            )
            events.append(event)

        security_monitor.security_events = events

        threats = security_monitor.detect_threats(user_id)

        assert len(threats) == 1
        threat = threats[0]
        assert threat.event_type == "brute_force_detected"
        assert threat.data["user_id"] == user_id
        assert threat.data["failed_attempts"] == 5
        assert "brute_force_attack" in threat.risk_indicators
        assert threat.severity == AlertSeverity.HIGH

    def test_detect_threats_custom_time_window(self, security_monitor):
        """Test threat detection with custom time window."""
        current_time = datetime.now(UTC)
        user_id = "user_001"

        # Add events outside custom time window
        event = SecurityEvent(
            event_id="event_001",
            event_type="authentication_failure",
            source="auth_service",
            timestamp=current_time - timedelta(hours=2),  # Outside 30-minute window
            data={"user_id": user_id},
        )
        security_monitor.security_events = [event]

        threats = security_monitor.detect_threats(user_id, time_window=timedelta(minutes=30))

        assert len(threats) == 0  # Event outside time window

    def test_detect_anomalies_no_anomalies(self, security_monitor):
        """Test anomaly detection with no anomalies."""
        current_time = datetime.now(UTC)
        event = SecurityEvent(
            event_id="event_001",
            event_type="file_access",
            source="file_service",
            timestamp=current_time,
            data={"user_id": "user_001", "resource": "/documents/report.pdf"},
        )
        security_monitor.security_events = [event]

        anomalies = security_monitor.detect_anomalies("user_001")

        assert len(anomalies) == 0

    def test_detect_anomalies_admin_access(self, security_monitor):
        """Test anomaly detection - admin resource access."""
        current_time = datetime.now(UTC)
        user_id = "user_001"

        event = SecurityEvent(
            event_id="event_001",
            event_type="file_access",
            source="file_service",
            timestamp=current_time,
            data={"user_id": user_id, "resource": "/admin/system_config.json"},
        )
        security_monitor.security_events = [event]

        anomalies = security_monitor.detect_anomalies(user_id)

        assert len(anomalies) == 1
        anomaly = anomalies[0]
        assert anomaly.event_type == "anomalous_access"
        assert anomaly.data["user_id"] == user_id
        assert anomaly.data["unusual_resource"] == "/admin/system_config.json"
        assert "unusual_access_pattern" in anomaly.risk_indicators
        assert anomaly.severity == AlertSeverity.MEDIUM

    def test_set_alert_handler(self, security_monitor):
        """Test setting alert handler."""
        def test_handler(alert):
            pass

        security_monitor.set_alert_handler(test_handler)

        assert test_handler in security_monitor.alert_callbacks

    def test_set_alert_handler_duplicate(self, security_monitor):
        """Test setting alert handler - no duplicates."""
        def test_handler(alert):
            pass

        security_monitor.set_alert_handler(test_handler)
        security_monitor.set_alert_handler(test_handler)  # Add again

        # Should only appear once
        callback_count = security_monitor.alert_callbacks.count(test_handler)
        assert callback_count == 1

    def test_supports_real_time_alerts_true(self, security_monitor):
        """Test real-time alerts support check - supported."""
        def test_handler(alert):
            pass

        security_monitor.register_alert_callback(test_handler)

        assert security_monitor.supports_real_time_alerts() is True

    def test_supports_real_time_alerts_false(self, security_monitor):
        """Test real-time alerts support check - not supported."""
        assert security_monitor.supports_real_time_alerts() is False

    def test_get_threat_summary(self, security_monitor):
        """Test getting threat summary."""
        current_time = datetime.now(UTC)
        user_id = "user_001"

        # Add events with different severities
        events = [
            SecurityEvent(
                event_id="event_001",
                event_type="test_event",
                source="test_source",
                timestamp=current_time - timedelta(minutes=30),
                data={"user_id": user_id},
                severity=AlertSeverity.HIGH,
            ),
            SecurityEvent(
                event_id="event_002",
                event_type="test_event",
                source="test_source",
                timestamp=current_time - timedelta(minutes=15),
                data={"user_id": user_id},
                severity=AlertSeverity.MEDIUM,
            ),
        ]
        security_monitor.security_events = events

        summary = security_monitor.get_threat_summary(user_id)

        assert summary.total_events == 2
        assert summary.max_threat_level == ThreatSeverity.HIGH
        assert summary.user_id == user_id
        assert summary.time_window == timedelta(hours=1)

    def test_get_threat_summary_critical_severity(self, security_monitor):
        """Test threat summary with critical severity."""
        current_time = datetime.now(UTC)
        user_id = "user_001"

        event = SecurityEvent(
            event_id="event_001",
            event_type="critical_event",
            source="test_source",
            timestamp=current_time,
            data={"user_id": user_id},
            severity=AlertSeverity.CRITICAL,
        )
        security_monitor.security_events = [event]

        summary = security_monitor.get_threat_summary(user_id)

        assert summary.max_threat_level == ThreatSeverity.HIGH  # Critical maps to HIGH

    def test_get_events_for_user(self, security_monitor):
        """Test getting events for specific user."""
        current_time = datetime.now(UTC)

        # Add events for different users
        events = [
            SecurityEvent(
                event_id="event_001",
                event_type="test_event",
                source="test_source",
                timestamp=current_time - timedelta(minutes=30),
                data={"user_id": "user_001"},
            ),
            SecurityEvent(
                event_id="event_002",
                event_type="test_event",
                source="test_source",
                timestamp=current_time - timedelta(minutes=15),
                data={"user_id": "user_002"},
            ),
            SecurityEvent(
                event_id="event_003",
                event_type="test_event",
                source="test_source",
                timestamp=current_time - timedelta(hours=2),  # Outside time window
                data={"user_id": "user_001"},
            ),
        ]
        security_monitor.security_events = events

        user_events = security_monitor.get_events_for_user("user_001")

        assert len(user_events) == 1  # Only one event within time window for user_001
        assert user_events[0].event_id == "event_001"

    def test_get_events_for_user_custom_time_window(self, security_monitor):
        """Test getting events for user with custom time window."""
        current_time = datetime.now(UTC)

        event = SecurityEvent(
            event_id="event_001",
            event_type="test_event",
            source="test_source",
            timestamp=current_time - timedelta(hours=2),
            data={"user_id": "user_001"},
        )
        security_monitor.security_events = [event]

        # Use larger time window
        user_events = security_monitor.get_events_for_user("user_001", timedelta(hours=4))

        assert len(user_events) == 1
        assert user_events[0].event_id == "event_001"


class TestSecurityMonitorDictConversion:
    """Test SecurityMonitor dictionary conversion methods."""

    @pytest.fixture
    def security_monitor(self):
        """Create SecurityMonitor instance for testing."""
        return SecurityMonitor()

    def test_dict_to_security_event_basic(self, security_monitor):
        """Test converting basic dictionary to SecurityEvent."""
        event_dict = {
            "event_id": "test_event_001",
            "type": "user_login",
            "source": "auth_service",
            "timestamp": "2024-01-01T12:00:00Z",
        }

        event = security_monitor._dict_to_security_event(event_dict)

        assert event.event_id == "test_event_001"
        assert event.event_type == "user_login"
        assert event.source == "auth_service"
        assert event.timestamp.year == 2024
        assert event.timestamp.month == 1
        assert event.timestamp.day == 1

    def test_dict_to_security_event_event_type_fallback(self, security_monitor):
        """Test dictionary conversion with event_type field."""
        event_dict = {
            "event_type": "authentication_failure",  # Use event_type instead of type
            "source": "auth_service",
        }

        event = security_monitor._dict_to_security_event(event_dict)

        assert event.event_type == "authentication_failure"

    def test_dict_to_security_event_generated_id(self, security_monitor):
        """Test dictionary conversion with generated event ID."""
        event_dict = {
            "type": "test_event",
            "source": "test_source",
        }

        event = security_monitor._dict_to_security_event(event_dict)

        assert event.event_id.startswith("event_")
        assert event.event_type == "test_event"
        assert event.source == "test_source"

    def test_dict_to_security_event_unknown_defaults(self, security_monitor):
        """Test dictionary conversion with unknown defaults."""
        event_dict = {}  # Empty dict

        event = security_monitor._dict_to_security_event(event_dict)

        assert event.event_type == "unknown"
        assert event.source == "unknown"
        assert event.severity == AlertSeverity.INFO

    def test_dict_to_security_event_invalid_timestamp(self, security_monitor):
        """Test dictionary conversion with invalid timestamp."""
        event_dict = {
            "type": "test_event",
            "source": "test_source",
            "timestamp": "invalid_timestamp",
        }

        event = security_monitor._dict_to_security_event(event_dict)

        # Should use current time when timestamp is invalid
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_dict_to_security_event_with_data_fields(self, security_monitor):
        """Test dictionary conversion with data fields."""
        event_dict = {
            "type": "user_activity",
            "source": "user_service",
            "user_id": "user_001",
            "source_ip": "192.168.1.100",
            "resource": "/api/users",
            "custom_field": "custom_value",
            "risk_indicators": ["unusual_time", "new_location"],
        }

        event = security_monitor._dict_to_security_event(event_dict)

        assert event.event_type == "user_activity"
        assert event.source == "user_service"
        assert event.data["user_id"] == "user_001"
        assert event.data["source_ip"] == "192.168.1.100"
        assert event.data["resource"] == "/api/users"
        assert event.data["custom_field"] == "custom_value"
        assert event.risk_indicators == ["unusual_time", "new_location"]

    def test_dict_to_security_event_none_values_filtered(self, security_monitor):
        """Test dictionary conversion filters None values from data."""
        event_dict = {
            "type": "test_event",
            "source": "test_source",
            "user_id": "user_001",
            "source_ip": None,  # Should be filtered out
            "resource": "/test",
        }

        event = security_monitor._dict_to_security_event(event_dict)

        assert "user_id" in event.data
        assert "source_ip" not in event.data  # None value filtered
        assert "resource" in event.data

    def test_analyze_event_dict_input(self, security_monitor):
        """Test analyzing event from dictionary input."""
        event_dict = {
            "type": "authentication_failure",
            "source": "auth_service",
            "user_id": "user_001",
        }

        analysis = security_monitor._analyze_event(event_dict)

        assert analysis["threat_level"] == "medium"  # auth_failure -> medium
        assert analysis["event_type"] == "authentication_failure"
        assert analysis["severity"] == "info"  # Default severity

    def test_analyze_event_object_input(self, security_monitor):
        """Test analyzing event from SecurityEvent object."""
        event = SecurityEvent(
            event_id="event_001",
            event_type="privilege_escalation",
            source="system_service",
            timestamp=datetime.now(UTC),
            severity=AlertSeverity.HIGH,
            risk_indicators=["privilege_escalation_attempt"],
        )

        analysis = security_monitor._analyze_event(event)

        assert analysis["threat_level"] == "high"  # privilege_escalation -> high, but HIGH severity overrides
        assert analysis["event_type"] == "privilege_escalation"
        assert analysis["severity"] == "high"
        assert analysis["risk_indicators"] == ["privilege_escalation_attempt"]

    def test_analyze_event_threat_levels(self, security_monitor):
        """Test threat level analysis for different event types."""
        test_cases = [
            ("user_activity", AlertSeverity.INFO, [], "low"),
            ("authentication_failure", AlertSeverity.INFO, [], "medium"),
            ("privilege_escalation", AlertSeverity.INFO, [], "high"),
            ("malware_detected", AlertSeverity.INFO, [], "critical"),
            ("normal_event", AlertSeverity.CRITICAL, [], "high"),  # Severity overrides
            ("normal_event", AlertSeverity.EMERGENCY, [], "critical"),  # Emergency -> critical
            ("normal_event", AlertSeverity.INFO, ["brute_force_attack"], "high"),  # Risk indicator overrides
        ]

        for event_type, severity, risk_indicators, expected_threat in test_cases:
            event = SecurityEvent(
                event_id="test_event",
                event_type=event_type,
                source="test_source",
                timestamp=datetime.now(UTC),
                severity=severity,
                risk_indicators=risk_indicators,
            )

            analysis = security_monitor._analyze_event(event)
            assert analysis["threat_level"] == expected_threat, f"Failed for {event_type}"

    def test_detect_threat_simple_interface(self, security_monitor):
        """Test detect_threat simple interface method."""
        event_dict = {
            "type": "brute_force_detected",
            "source": "threat_detector",
        }

        result = security_monitor.detect_threat(event_dict)

        assert result["threat_level"] == "critical"
        assert result["event_type"] == "brute_force_detected"
        assert "analysis_timestamp" in result
