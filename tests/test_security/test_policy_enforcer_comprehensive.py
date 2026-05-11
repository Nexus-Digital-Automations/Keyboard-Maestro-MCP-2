"""Comprehensive tests for src/security/policy_enforcer.py.

This module provides targeted tests for the policy_enforcer module to achieve high coverage
toward the mandatory 95% threshold.
"""


import pytest
from src.core.zero_trust_architecture import (
    ComplianceFramework,
    ThreatSeverity,
)
from src.security.policy_enforcer import (
    ComplianceRule,
    ComplianceStatus,
    EnforcementResult,
    PolicyEnforcer,
    PolicyStatus,
    PolicyValidationResult,
)


class TestPolicyStatus:
    """Test PolicyStatus enum values."""

    def test_policy_status_enum_values(self) -> None:
        """Test PolicyStatus enum has expected values."""
        assert PolicyStatus.ACTIVE.value == "active"
        assert PolicyStatus.INACTIVE.value == "inactive"
        assert PolicyStatus.PENDING.value == "pending"
        assert PolicyStatus.DEPRECATED.value == "deprecated"
        assert PolicyStatus.SUSPENDED.value == "suspended"
        assert PolicyStatus.DRAFT.value == "draft"


class TestEnforcementResult:
    """Test EnforcementResult enum values."""

    def test_enforcement_result_enum_values(self) -> None:
        """Test EnforcementResult enum has expected values."""
        assert EnforcementResult.ALLOWED.value == "allowed"
        assert EnforcementResult.DENIED.value == "denied"
        assert EnforcementResult.CONDITIONAL.value == "conditional"
        assert EnforcementResult.REQUIRES_APPROVAL.value == "requires_approval"
        assert EnforcementResult.REMEDIATED.value == "remediated"
        assert EnforcementResult.ESCALATED.value == "escalated"


class TestComplianceStatus:
    """Test ComplianceStatus enum values."""

    def test_compliance_status_enum_values(self) -> None:
        """Test ComplianceStatus enum has expected values."""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceStatus.UNKNOWN.value == "unknown"
        assert ComplianceStatus.PENDING_REVIEW.value == "pending_review"
        assert ComplianceStatus.REMEDIATION_REQUIRED.value == "remediation_required"


















class TestComplianceRule:
    """Test ComplianceRule dataclass."""

    def test_compliance_rule_creation(self) -> None:
        """Test ComplianceRule creation."""
        rule = ComplianceRule(
            rule_id="rule-001",
            framework=ComplianceFramework.SOC2,
            requirement_id="CC6.1",
            description="Access controls to protect system resources",
            validation_criteria={"access_logs": True, "role_based_access": True},
            severity=ThreatSeverity.MEDIUM,
            automated_check=True,
        )

        assert rule.rule_id == "rule-001"
        assert rule.framework == ComplianceFramework.SOC2
        assert rule.requirement_id == "CC6.1"
        assert rule.description == "Access controls to protect system resources"
        assert rule.validation_criteria["access_logs"] is True
        assert rule.severity == ThreatSeverity.MEDIUM
        assert rule.automated_check is True



class TestPolicyEnforcer:
    """Test PolicyEnforcer class."""

    @pytest.fixture
    def enforcer(self) -> PolicyEnforcer:
        """Create PolicyEnforcer instance for testing."""
        return PolicyEnforcer()

    def test_policy_enforcer_initialization(self, enforcer: PolicyEnforcer) -> None:
        """Test PolicyEnforcer initialization."""
        assert isinstance(enforcer.active_policies, dict)
        assert isinstance(enforcer.policy_status, dict)
        assert isinstance(enforcer.enforcement_history, list)
        assert isinstance(enforcer.violation_history, list)
        assert isinstance(enforcer.compliance_rules, dict)
        assert isinstance(enforcer.enforcement_cache, dict)

        # Check initial metrics
        assert enforcer.evaluation_count == 0
        assert enforcer.average_evaluation_time == 0.0
        assert enforcer.cache_hit_rate == 0.0
        assert enforcer.violation_rate == 0.0














    def test_get_cached_evaluation_miss(self, enforcer: PolicyEnforcer) -> None:
        """Test cache miss for evaluation."""
        result = enforcer._get_cached_evaluation("non-existent-key")
        assert result is None


    def test_update_evaluation_metrics(self, enforcer: PolicyEnforcer) -> None:
        """Test evaluation metrics update."""
        initial_count = enforcer.evaluation_count
        enforcer._update_evaluation_metrics(100.5, True)

        assert enforcer.evaluation_count == initial_count + 1

    def test_add_policy_dict_format(self, enforcer: PolicyEnforcer) -> None:
        """Test adding policy in dictionary format."""
        policy_dict = {
            "name": "Test Policy",
            "description": "Test policy from dict",
            "rules": [{"resource": "*.txt", "action": "read"}],
            "priority": 100,
        }

        enforcer.add_policy(policy_dict)
        # Should not raise error for valid policy dict

    def test_validate_against_policies_no_policies(self, enforcer: PolicyEnforcer) -> None:
        """Test validation against policies when no policies exist."""
        data = {"resource": "test.txt", "action": "read"}
        result = enforcer.validate_against_policies(data)

        assert isinstance(result, PolicyValidationResult)
        assert result.is_valid is True  # No policies to violate

    def test_enforce_policies_no_policies(self, enforcer: PolicyEnforcer) -> None:
        """Test policy enforcement when no policies exist."""
        request_data = {"resource": "test.txt", "action": "read"}
        # Should not raise error when no policies exist
        enforcer.enforce_policies(request_data)

    def test_list_policies_empty(self, enforcer: PolicyEnforcer) -> None:
        """Test listing policies when none exist."""
        policies = enforcer.list_policies()
        assert isinstance(policies, list)
        assert len(policies) == 0








class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    @pytest.fixture
    def configured_enforcer(self) -> PolicyEnforcer:
        """Create configured PolicyEnforcer with policies and rules."""
        enforcer = PolicyEnforcer()
        return enforcer



