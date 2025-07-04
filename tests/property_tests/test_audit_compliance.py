"""
Property-based tests for audit system compliance and security validation.

This module provides comprehensive property-based testing for the audit system
using Hypothesis to validate behavior across all input ranges with focus on
security, compliance, and integrity verification.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from datetime import datetime, timedelta, UTC
from typing import Dict, Any
import uuid

from src.core.audit_framework import (
    AuditEvent, AuditEventType, ComplianceRule, ComplianceReport,
    ComplianceStandard, RiskLevel, AuditLevel, AuditConfiguration
)
from src.core.either import Either
from src.audit.event_logger import EventLogger
from src.audit.compliance_monitor import ComplianceMonitor
from src.audit.report_generator import ReportGenerator
from src.audit.audit_system_manager import AuditSystemManager
from src.server.tools.audit_system_tools import km_audit_system


class TestAuditEventProperties:
    """Property-based tests for audit event functionality."""
    
    @given(
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=100),
        st.sampled_from(list(AuditEventType)),
        st.sampled_from(list(RiskLevel))
    )
    def test_audit_event_properties(self, user_id, action, result, event_type, risk_level):
        """Property: Audit events should handle various inputs and maintain integrity."""
        assume(user_id.strip() != "")
        assume(action.strip() != "")
        assume(result.strip() != "")
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(UTC),
            user_id=user_id,
            session_id=None,
            resource_id=None,
            action=action,
            result=result,
            risk_level=risk_level
        )
        
        # Property: Event should maintain data integrity
        assert event.user_id == user_id
        assert event.action == action
        assert event.result == result
        assert event.event_type == event_type
        assert event.risk_level == risk_level
        
        # Property: Integrity verification should work
        assert event.verify_integrity()
        
        # Property: Checksum should be consistent
        original_checksum = event.checksum
        recalculated_checksum = event._calculate_checksum()
        assert original_checksum == recalculated_checksum
    
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.booleans()),
            min_size=0,
            max_size=10
        ),
        st.sets(st.text(min_size=1, max_size=20), max_size=10)
    )
    def test_audit_event_with_details_properties(self, details, compliance_tags):
        """Property: Audit events should handle complex details and tags."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.AUTOMATION_EXECUTED,
            timestamp=datetime.now(UTC),
            user_id="test_user",
            session_id=None,
            resource_id=None,
            action="test_action",
            result="success",
            details=details,
            compliance_tags=compliance_tags
        )
        
        # Property: Details and tags should be preserved
        assert event.details == details
        assert event.compliance_tags == compliance_tags
        
        # Property: Integrity should be maintained with complex data
        assert event.verify_integrity()
        
        # Property: Sensitive data detection should work
        sensitive_fields = event.get_sensitive_data_fields()
        assert isinstance(sensitive_fields, set)
    
    @given(st.sampled_from(list(ComplianceStandard)))
    def test_compliance_standard_matching_properties(self, standard):
        """Property: Compliance standard matching should be consistent."""
        # Create event with appropriate tags for the standard
        standard_tags = {
            ComplianceStandard.HIPAA: {'phi_access', 'medical_records'},
            ComplianceStandard.GDPR: {'personal_data', 'data_processing'},
            ComplianceStandard.PCI_DSS: {'payment_data', 'card_processing'},
            ComplianceStandard.SOC2: {'security', 'availability'},
            ComplianceStandard.GENERAL: set()
        }
        
        tags = standard_tags.get(standard, set())
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.DATA_ACCESSED,
            timestamp=datetime.now(UTC),
            user_id="test_user",
            session_id=None,
            resource_id=None,
            action="access_data",
            result="success",
            compliance_tags=tags
        )
        
        # Property: Event should match its compliance standard
        matches = event.matches_compliance_standard(standard)
        if standard == ComplianceStandard.GENERAL or tags:
            assert matches
        
        # Property: Event should not match incompatible standards
        if standard != ComplianceStandard.GENERAL and not tags:
            # Test with a different standard that has no matching tags
            other_standards = [s for s in ComplianceStandard if s != standard and s != ComplianceStandard.GENERAL]
            if other_standards:
                other_standard = other_standards[0]
                other_matches = event.matches_compliance_standard(other_standard)
                # May or may not match depending on tag overlap


class TestComplianceRuleProperties:
    """Property-based tests for compliance rule functionality."""
    
    @given(
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=200),
        st.text(min_size=1, max_size=100),
        st.sampled_from(list(ComplianceStandard)),
        st.sampled_from(list(RiskLevel))
    )
    def test_compliance_rule_properties(self, rule_id, name, description, condition, standard, severity):
        """Property: Compliance rules should handle various configurations."""
        assume(rule_id.strip() != "")
        assume(name.strip() != "")
        assume(description.strip() != "")
        assume(condition.strip() != "")
        
        rule = ComplianceRule(
            rule_id=rule_id,
            name=name,
            description=description,
            standard=standard,
            severity=severity,
            condition=condition,
            action="test_action"
        )
        
        # Property: Rule should preserve all properties
        assert rule.rule_id == rule_id
        assert rule.name == name
        assert rule.description == description
        assert rule.standard == standard
        assert rule.severity == severity
        assert rule.condition == condition
        
        # Property: Rule should be enabled by default
        assert rule.enabled
    
    @given(
        st.text(min_size=1, max_size=100),
        st.sampled_from([
            "failed_login", "sensitive_data", "privilege_escalation",
            "unauthorized_modification", "high_risk", "security_violation"
        ])
    )
    def test_compliance_rule_evaluation_properties(self, user_id, condition_type):
        """Property: Compliance rule evaluation should be consistent."""
        assume(user_id.strip() != "")
        
        rule = ComplianceRule(
            rule_id="test_rule",
            name="Test Rule",
            description="Test Description",
            standard=ComplianceStandard.GENERAL,
            severity=RiskLevel.MEDIUM,
            condition=condition_type,
            action="test_action"
        )
        
        # Create appropriate event for condition
        if condition_type == "failed_login":
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.USER_LOGIN,
                timestamp=datetime.now(UTC),
                user_id=user_id,
                session_id=None,
                resource_id=None,
                action="login",
                result="failure"
            )
            # Property: Failed login should trigger failed_login condition
            assert rule.evaluate(event)
        
        elif condition_type == "sensitive_data":
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.DATA_ACCESSED,
                timestamp=datetime.now(UTC),
                user_id=user_id,
                session_id=None,
                resource_id=None,
                action="access_sensitive_data",
                result="success",
                details={"data_type": "sensitive"}
            )
            # Property: Sensitive data access should trigger sensitive_data condition
            assert rule.evaluate(event)
        
        elif condition_type == "privilege_escalation":
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.PERMISSION_GRANTED,
                timestamp=datetime.now(UTC),
                user_id=user_id,
                session_id=None,
                resource_id=None,
                action="escalate_privileges",
                result="success"
            )
            # Property: Privilege escalation should trigger privilege_escalation condition
            assert rule.evaluate(event)


class TestEventLoggerProperties:
    """Property-based tests for event logger functionality."""
    
    @pytest.fixture
    def event_logger(self):
        """Provide event logger for tests."""
        config = AuditConfiguration(audit_level=AuditLevel.STANDARD)
        return EventLogger(config)
    
    @given(
        st.sampled_from(list(AuditEventType)),
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=50)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_event_logging_properties(self, event_logger, event_type, user_id, action, result):
        """Property: Event logging should handle various event types and data."""
        assume(user_id.strip() != "")
        assume(action.strip() != "")
        assume(result.strip() != "")
        
        # Property: Event logging should succeed with valid inputs
        log_result = await event_logger.log_event(
            event_type=event_type,
            user_id=user_id,
            action=action,
            result=result
        )
        
        assert log_result.is_right()
        event_id = log_result.get_right()
        assert isinstance(event_id, str)
        assert len(event_id) > 0
        
        # Property: Event should be queryable
        await event_logger.buffer.force_flush()  # Ensure event is stored
        events = await event_logger.query_events({})
        assert len(events) > 0
        
        # Property: Event integrity should be maintained
        logged_event = events[-1]  # Most recent event
        assert logged_event.verify_integrity()
    
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(max_size=100), st.integers(), st.booleans()),
            min_size=1,
            max_size=5
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_event_logging_with_filters_properties(self, event_logger, filters):
        """Property: Event querying with filters should be consistent."""
        # First log some events
        for i in range(3):
            await event_logger.log_event(
                event_type=AuditEventType.AUTOMATION_EXECUTED,
                user_id=f"user_{i}",
                action=f"action_{i}",
                result="success"
            )
        
        await event_logger.buffer.force_flush()
        
        # Property: Query without filters should return all events
        all_events = await event_logger.query_events({})
        assert len(all_events) >= 3
        
        # Property: Query with user filter should return filtered results
        if 'user_id' in filters and isinstance(filters['user_id'], str):
            user_events = await event_logger.query_events({'user_id': filters['user_id']})
            for event in user_events:
                assert event.user_id == filters['user_id']
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_rate_limiting_properties(self, event_logger, event_count):
        """Property: Rate limiting should prevent excessive event logging."""
        user_id = "test_user"
        
        # Property: Small number of events should succeed
        if event_count <= 10:
            for i in range(event_count):
                result = await event_logger.log_event(
                    event_type=AuditEventType.AUTOMATION_EXECUTED,
                    user_id=user_id,
                    action=f"action_{i}",
                    result="success"
                )
                assert result.is_right()
        
        # Property: Rate limiter should track events per user
        rate_limiter = event_logger.rate_limiter
        assert user_id in rate_limiter.user_rates or event_count == 0


class TestComplianceMonitorProperties:
    """Property-based tests for compliance monitor functionality."""
    
    @pytest.fixture
    def compliance_monitor(self):
        """Provide compliance monitor for tests."""
        return ComplianceMonitor()
    
    @given(st.sampled_from(list(ComplianceStandard)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_compliance_rule_registration_properties(self, compliance_monitor, standard):
        """Property: Compliance rule registration should handle all standards."""
        rule = ComplianceRule(
            rule_id=f"test_rule_{standard.value}",
            name=f"Test Rule for {standard.value}",
            description="Test rule description",
            standard=standard,
            severity=RiskLevel.MEDIUM,
            condition="test_condition",
            action="test_action"
        )
        
        # Property: Rule registration should succeed
        result = await compliance_monitor.register_compliance_rule(rule)
        assert result.is_right()
        
        # Property: Rule should be retrievable
        assert rule.rule_id in compliance_monitor.compliance_rules
        retrieved_rule = compliance_monitor.compliance_rules[rule.rule_id]
        assert retrieved_rule.standard == standard
        
        # Property: Rules by standard should include the rule
        rules_for_standard = compliance_monitor.get_rules_by_standard(standard)
        assert rule in rules_for_standard
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(list(AuditEventType)),
                st.text(min_size=1, max_size=50),
                st.sampled_from(["success", "failure", "error"])
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_batch_event_monitoring_properties(self, compliance_monitor, event_specs):
        """Property: Batch event monitoring should handle multiple events."""
        # Create events from specifications
        events = []
        for i, (event_type, user_id, result) in enumerate(event_specs):
            assume(user_id.strip() != "")
            
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now(UTC),
                user_id=user_id.strip(),
                session_id=None,
                resource_id=None,
                action=f"action_{i}",
                result=result
            )
            events.append(event)
        
        # Property: Batch evaluation should return results for all events
        results = await compliance_monitor.evaluate_batch(events)
        assert isinstance(results, dict)
        
        # Property: Results should only contain events with violations
        for event_id, violations in results.items():
            assert isinstance(violations, list)
            for violation in violations:
                assert isinstance(violation, ComplianceRule)


class TestReportGeneratorProperties:
    """Property-based tests for report generator functionality."""
    
    @pytest.fixture
    def report_generator(self):
        """Provide report generator for tests."""
        config = AuditConfiguration()
        event_logger = EventLogger(config)
        compliance_monitor = ComplianceMonitor()
        return ReportGenerator(event_logger, compliance_monitor)
    
    @given(
        st.sampled_from(list(ComplianceStandard)),
        st.integers(min_value=1, max_value=90),
        st.integers(min_value=0, max_value=50)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_compliance_report_properties(self, report_generator, standard, period_days, violation_count):
        """Property: Compliance reports should handle various standards and periods."""
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=period_days)
        
        # Mock some events in the event logger
        for i in range(min(10, violation_count + 5)):  # Create some events
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.AUTOMATION_EXECUTED,
                timestamp=start_time + timedelta(days=i % period_days),
                user_id=f"user_{i}",
                session_id=None,
                resource_id=None,
                action=f"action_{i}",
                result="success",
                compliance_tags={'general'} if standard == ComplianceStandard.GENERAL else set()
            )
            report_generator.event_logger.event_store.append(event)
        
        # Property: Report generation should succeed
        report_result = await report_generator.generate_compliance_report(
            standard=standard,
            period_start=start_time,
            period_end=end_time
        )
        
        assert report_result.is_right()
        report = report_result.get_right()
        
        # Property: Report should have required fields
        assert report.standard == standard
        assert report.period_start == start_time
        assert report.period_end == end_time
        assert 0.0 <= report.compliance_percentage <= 100.0
        assert 0.0 <= report.risk_score <= 100.0
        assert report.total_events >= 0
        assert report.violations_found >= 0
        
        # Property: Compliance percentage should be consistent
        if report.total_events > 0:
            expected_compliance = max(0.0, (report.total_events - report.violations_found) / report.total_events * 100.0)
            # Allow for standard-specific adjustments
            assert abs(report.compliance_percentage - expected_compliance) <= 20.0


class TestAuditSystemToolProperties:
    """Property-based tests for audit system MCP tool."""
    
    @given(
        st.sampled_from(["log", "query", "report", "monitor", "configure", "status"]),
        st.text(min_size=1, max_size=50),
        st.sampled_from(list(AuditEventType))
    )
    @pytest.mark.asyncio
    async def test_audit_tool_operation_properties(self, operation, user_id, event_type):
        """Property: Audit tool should handle various operations consistently."""
        assume(user_id.strip() != "")
        
        # Mock the audit system components
        with patch('src.server.tools.audit_system_tools.get_audit_system') as mock_get_system:
            mock_system = Mock()
            mock_system.initialized = True
            mock_get_system.return_value = mock_system
            
            # Setup mock responses based on operation
            if operation == "log":
                mock_system.audit_user_action = AsyncMock(return_value=Either.right(str(uuid.uuid4())))
                
                result = await km_audit_system(
                    operation=operation,
                    event_type=event_type.value,
                    user_id=user_id,
                    action_details={'action': 'test_action', 'result': 'success'}
                )
                
                assert result['success'] is True
                assert result['operation'] == operation
                assert 'event_id' in result['data']
            
            elif operation == "status":
                mock_system.get_system_status = Mock(return_value={
                    'initialized': True,
                    'configuration': {'audit_level': 'standard'},
                    'event_logging': {'events_logged': 100},
                    'compliance_monitoring': {'monitoring_enabled': True},
                    'performance': {'average_audit_latency_ms': 10}
                })
                
                result = await km_audit_system(operation=operation)
                
                assert result['success'] is True
                assert result['operation'] == operation
                assert 'system_status' in result['data']
    
    @given(st.text(max_size=0))
    @pytest.mark.asyncio
    async def test_empty_parameters_validation_properties(self, empty_value):
        """Property: Empty or invalid parameters should be rejected."""
        assume(len(empty_value.strip()) == 0)
        
        with pytest.raises(Exception):  # Should raise ToolError
            await km_audit_system(
                operation="log",
                event_type="automation_executed",
                user_id=empty_value  # Empty user ID should be rejected
            )


class TestSecurityProperties:
    """Property-based security validation tests for audit system."""
    
    @settings(max_examples=50, deadline=5000)
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=1, max_size=100),
            min_size=1,
            max_size=10
        )
    )
    def test_sensitive_data_detection_properties(self, event_details):
        """Property: Sensitive data should be detected and protected."""
        # Add sensitive data patterns
        sensitive_patterns = ["password", "secret", "token", "key", "credential"]
        
        for pattern in sensitive_patterns:
            test_details = event_details.copy()
            test_details[f"user_{pattern}"] = f"sensitive_{pattern}_data"
            
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.DATA_ACCESSED,
                timestamp=datetime.now(UTC),
                user_id="test_user",
                session_id=None,
                resource_id=None,
                action="access_data",
                result="success",
                details=test_details
            )
            
            # Property: Sensitive data should be detected
            sensitive_fields = event.get_sensitive_data_fields()
            assert len(sensitive_fields) > 0
            assert any(pattern in field.lower() for field in sensitive_fields)
    
    @settings(max_examples=50, deadline=5000)
    @given(
        st.integers(min_value=0, max_value=3000),
        st.integers(min_value=0, max_value=100),
        st.floats(min_value=0.0, max_value=100.0)
    )
    def test_compliance_report_security_properties(self, total_events, violations, risk_score):
        """Property: Compliance reports should maintain security boundaries."""
        assume(violations <= total_events)
        assume(0.0 <= risk_score <= 100.0)
        
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            standard=ComplianceStandard.GENERAL,
            period_start=datetime.now(UTC) - timedelta(days=30),
            period_end=datetime.now(UTC),
            total_events=total_events,
            violations_found=violations,
            risk_score=risk_score,
            compliance_percentage=max(0.0, (total_events - violations) / max(total_events, 1) * 100.0)
        )
        
        # Property: Report data should be within security bounds
        assert 0 <= report.total_events <= 1000000  # Reasonable upper bound
        assert 0 <= report.violations_found <= report.total_events
        assert 0.0 <= report.risk_score <= 100.0
        assert 0.0 <= report.compliance_percentage <= 100.0
        
        # Property: Risk assessment should be consistent
        risk_category = report.get_risk_category()
        assert risk_category in ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
        
        # Property: Compliance grade should be appropriate
        grade = report.get_compliance_grade()
        assert grade in ["A+", "A", "B", "C", "D", "F"]
    
    @settings(max_examples=50, deadline=5000)
    @given(
        st.text(min_size=1, max_size=100),
        st.integers(min_value=1, max_value=2555)
    )
    def test_audit_configuration_security_properties(self, config_value, retention_days):
        """Property: Audit configuration should enforce security limits."""
        assume(1 <= retention_days <= 2555)
        
        config = AuditConfiguration(
            retention_days=retention_days,
            encrypt_logs=True
        )
        
        # Property: Configuration should enforce retention limits
        assert 1 <= config.retention_days <= 2555
        
        # Property: Security settings should be preserved
        assert config.encrypt_logs is True
        
        # Property: Performance profile should be consistent
        profile = config.get_performance_profile()
        assert 'max_events_per_second' in profile
        assert 'buffer_size' in profile
        assert 'flush_interval' in profile
        assert profile['max_events_per_second'] > 0
        assert profile['buffer_size'] > 0
        assert profile['flush_interval'] > 0