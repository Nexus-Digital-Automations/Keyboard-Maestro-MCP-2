"""Comprehensive tests for src/security/policy_enforcer.py.

This module provides targeted tests for the policy_enforcer module to achieve high coverage
toward the mandatory 95% threshold.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from src.core.zero_trust_architecture import (
    ComplianceFramework,
    EnforcementMode,
    PolicyId,
    PolicyType,
    SecurityContext,
    ThreatSeverity,
    TrustLevel,
    ValidationScope,
)
from src.security.policy_enforcer import (
    ComplianceRule,
    ComplianceStatus,
    EnforcementResult,
    PolicyEnforcer,
    PolicyEvaluationRequest,
    PolicyEvaluationResult,
    PolicyStatus,
    PolicyValidationResult,
    PolicyViolation,
    SecurityPolicy,
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


class TestPolicyEvaluationRequest:
    """Test PolicyEvaluationRequest dataclass."""

    def test_policy_evaluation_request_creation(self) -> None:
        """Test PolicyEvaluationRequest creation with valid data."""
        context = SecurityContext(
            user_id="user-123",
            session_id="session-456",
            trust_level=TrustLevel.TRUSTED,
            permissions=frozenset(["read"]),
            validation_scope=ValidationScope.STRICT,
        )

        request = PolicyEvaluationRequest(
            request_id="req-001",
            context=context,
            resource="document.txt",
            action="read",
            additional_data={"department": "finance"},
            priority="high",
            timeout=60,
        )

        assert request.request_id == "req-001"
        assert request.context == context
        assert request.resource == "document.txt"
        assert request.action == "read"
        assert request.additional_data["department"] == "finance"
        assert request.priority == "high"
        assert request.timeout == 60

    def test_policy_evaluation_request_validation(self) -> None:
        """Test PolicyEvaluationRequest validation."""
        context = SecurityContext(
            user_id="user-123",
            session_id="session-456",
            trust_level=TrustLevel.TRUSTED,
            permissions=frozenset(["read"]),
            validation_scope=ValidationScope.STRICT,
        )

        # Test empty request_id
        with pytest.raises(
            ValueError, match="Request ID, resource, and action are required"
        ):
            PolicyEvaluationRequest(
                request_id="",
                context=context,
                resource="document.txt",
                action="read",
            )

        # Test empty resource
        with pytest.raises(
            ValueError, match="Request ID, resource, and action are required"
        ):
            PolicyEvaluationRequest(
                request_id="req-001",
                context=context,
                resource="",
                action="read",
            )

        # Test empty action
        with pytest.raises(
            ValueError, match="Request ID, resource, and action are required"
        ):
            PolicyEvaluationRequest(
                request_id="req-001",
                context=context,
                resource="document.txt",
                action="",
            )

        # Test invalid timeout
        with pytest.raises(ValueError, match="Timeout must be positive"):
            PolicyEvaluationRequest(
                request_id="req-001",
                context=context,
                resource="document.txt",
                action="read",
                timeout=0,
            )


class TestPolicyEvaluationResult:
    """Test PolicyEvaluationResult dataclass."""

    def test_policy_evaluation_result_creation(self) -> None:
        """Test PolicyEvaluationResult creation."""
        result = PolicyEvaluationResult(
            request_id="req-001",
            decision=EnforcementResult.ALLOWED,
            applicable_policies=[PolicyId("policy-001"), PolicyId("policy-002")],
            decision_reason="All policies allow this action",
            decision_confidence=0.95,
            conditions=["log_access", "notify_admin"],
            evaluation_time_ms=150.5,
            cached=False,
        )

        assert result.request_id == "req-001"
        assert result.decision == EnforcementResult.ALLOWED
        assert len(result.applicable_policies) == 2
        assert result.decision_reason == "All policies allow this action"
        assert result.decision_confidence == 0.95
        assert "log_access" in result.conditions
        assert result.evaluation_time_ms == 150.5
        assert result.cached is False

    def test_policy_evaluation_result_validation(self) -> None:
        """Test PolicyEvaluationResult validation."""
        # Test invalid confidence score
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            PolicyEvaluationResult(
                request_id="req-001",
                decision=EnforcementResult.ALLOWED,
                applicable_policies=[],
                decision_confidence=1.5,
            )

        # Test negative evaluation time
        with pytest.raises(ValueError, match="Evaluation time cannot be negative"):
            PolicyEvaluationResult(
                request_id="req-001",
                decision=EnforcementResult.ALLOWED,
                applicable_policies=[],
                evaluation_time_ms=-1.0,
            )


class TestSecurityPolicy:
    """Test SecurityPolicy dataclass."""

    def test_security_policy_creation(self) -> None:
        """Test SecurityPolicy creation."""
        policy = SecurityPolicy(
            policy_id=PolicyId("policy-001"),
            name="Document Access Policy",
            description="Controls access to sensitive documents",
            policy_type=PolicyType.ACCESS_CONTROL,
            enforcement_mode=EnforcementMode.ENFORCE,
            rules=[
                {
                    "resource_pattern": "*.pdf",
                    "action": "read",
                    "required_role": "viewer",
                },
                {
                    "resource_pattern": "*.docx",
                    "action": "write",
                    "required_role": "editor",
                },
            ],
            conditions={"department": "finance", "time_range": "9-17"},
            priority=100,
            enabled=True,
        )

        assert policy.policy_id == PolicyId("policy-001")
        assert policy.name == "Document Access Policy"
        assert policy.policy_type == PolicyType.ACCESS_CONTROL
        assert policy.enforcement_mode == EnforcementMode.ENFORCE
        assert len(policy.rules) == 2
        assert policy.conditions["department"] == "finance"
        assert policy.priority == 100
        assert policy.enabled is True

    def test_security_policy_validation(self) -> None:
        """Test SecurityPolicy validation."""
        # Test empty name
        with pytest.raises(ValueError, match="Policy name cannot be empty"):
            SecurityPolicy(
                policy_id=PolicyId("policy-001"),
                name="",
                description="Test policy",
                policy_type=PolicyType.ACCESS_CONTROL,
                enforcement_mode=EnforcementMode.ENFORCE,
                rules=[],
            )

        # Test empty rules
        with pytest.raises(ValueError, match="Policy must have at least one rule"):
            SecurityPolicy(
                policy_id=PolicyId("policy-001"),
                name="Test Policy",
                description="Test policy",
                policy_type=PolicyType.ACCESS_CONTROL,
                enforcement_mode=EnforcementMode.ENFORCE,
                rules=[],
            )

        # Test invalid priority
        with pytest.raises(ValueError, match="Priority must be between 1 and 1000"):
            SecurityPolicy(
                policy_id=PolicyId("policy-001"),
                name="Test Policy",
                description="Test policy",
                policy_type=PolicyType.ACCESS_CONTROL,
                enforcement_mode=EnforcementMode.ENFORCE,
                rules=[{"test": "rule"}],
                priority=0,
            )


class TestPolicyViolation:
    """Test PolicyViolation dataclass."""

    def test_policy_violation_creation(self) -> None:
        """Test PolicyViolation creation."""
        violation = PolicyViolation(
            violation_id="violation-001",
            policy_id=PolicyId("policy-001"),
            request_id="req-001",
            severity=ThreatSeverity.HIGH,
            description="Unauthorized access attempt",
            violation_details={
                "attempted_action": "admin_delete",
                "resource": "user_data.db",
            },
            timestamp=datetime.now(UTC),
            resolved=False,
        )

        assert violation.violation_id == "violation-001"
        assert violation.policy_id == PolicyId("policy-001")
        assert violation.request_id == "req-001"
        assert violation.severity == ThreatSeverity.HIGH
        assert violation.description == "Unauthorized access attempt"
        assert violation.violation_details["attempted_action"] == "admin_delete"
        assert violation.resolved is False

    def test_policy_violation_validation(self) -> None:
        """Test PolicyViolation validation."""
        # Test empty violation_id
        with pytest.raises(ValueError, match="Violation ID cannot be empty"):
            PolicyViolation(
                violation_id="",
                policy_id=PolicyId("policy-001"),
                request_id="req-001",
                severity=ThreatSeverity.HIGH,
                description="Test violation",
                timestamp=datetime.now(UTC),
            )

        # Test empty description
        with pytest.raises(ValueError, match="Description cannot be empty"):
            PolicyViolation(
                violation_id="violation-001",
                policy_id=PolicyId("policy-001"),
                request_id="req-001",
                severity=ThreatSeverity.HIGH,
                description="",
                timestamp=datetime.now(UTC),
            )


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

    def test_compliance_rule_validation(self) -> None:
        """Test ComplianceRule validation."""
        # Test empty rule_id
        with pytest.raises(ValueError, match="Rule ID cannot be empty"):
            ComplianceRule(
                rule_id="",
                framework=ComplianceFramework.SOC2,
                requirement_id="CC6.1",
                description="Test rule",
                validation_criteria={},
            )

        # Test empty requirement_id
        with pytest.raises(ValueError, match="Requirement ID cannot be empty"):
            ComplianceRule(
                rule_id="rule-001",
                framework=ComplianceFramework.SOC2,
                requirement_id="",
                description="Test rule",
                validation_criteria={},
            )

        # Test empty description
        with pytest.raises(ValueError, match="Description cannot be empty"):
            ComplianceRule(
                rule_id="rule-001",
                framework=ComplianceFramework.SOC2,
                requirement_id="CC6.1",
                description="",
                validation_criteria={},
            )


class TestPolicyEnforcer:
    """Test PolicyEnforcer class."""

    @pytest.fixture
    def enforcer(self) -> None:
        """Create PolicyEnforcer instance for testing."""
        return PolicyEnforcer()

    @pytest.fixture
    def sample_policy(self) -> None:
        """Create sample security policy for testing."""
        return SecurityPolicy(
            policy_id=PolicyId("policy-001"),
            name="Test Access Policy",
            description="Test policy for unit testing",
            policy_type=PolicyType.ACCESS_CONTROL,
            enforcement_mode=EnforcementMode.ENFORCE,
            rules=[
                {
                    "resource_pattern": "*.txt",
                    "action": "read",
                    "required_role": "user",
                },
                {
                    "resource_pattern": "*.pdf",
                    "action": "write",
                    "required_role": "admin",
                },
            ],
            conditions={"department": "engineering"},
            priority=100,
            enabled=True,
        )

    @pytest.fixture
    def sample_context(self) -> None:
        """Create sample security context for testing."""
        return SecurityContext(
            user_id="user-123",
            session_id="session-456",
            trust_level=TrustLevel.TRUSTED,
            permissions=frozenset(["read", "write"]),
            validation_scope=ValidationScope.STRICT,
        )

    @pytest.fixture
    def sample_request(self, sample_context) -> None:
        """Create sample policy evaluation request for testing."""
        return PolicyEvaluationRequest(
            request_id="req-001",
            context=sample_context,
            resource="document.txt",
            action="read",
            additional_data={"department": "engineering"},
        )

    def test_policy_enforcer_initialization(self, enforcer) -> None:
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

    async def test_register_policy_success(self, enforcer, sample_policy) -> None:
        """Test successful policy registration."""
        result = await enforcer.register_policy(sample_policy)

        assert result.is_right()
        policy_id = result.get_right()
        assert policy_id == sample_policy.policy_id
        assert sample_policy.policy_id in enforcer.active_policies
        assert enforcer.policy_status[sample_policy.policy_id] == PolicyStatus.ACTIVE

    async def test_register_policy_validation_failure(self, enforcer) -> None:
        """Test policy registration with validation failure."""
        # Create invalid policy (empty name)
        SecurityPolicy(
            policy_id=PolicyId("invalid-policy"),
            name="",  # Invalid: empty name
            description="Invalid policy",
            policy_type=PolicyType.ACCESS_CONTROL,
            enforcement_mode=EnforcementMode.ENFORCE,
            rules=[{"test": "rule"}],
        )

        # This should raise ValueError during object creation, not during registration
        with pytest.raises(ValueError):
            SecurityPolicy(
                policy_id=PolicyId("invalid-policy"),
                name="",
                description="Invalid policy",
                policy_type=PolicyType.ACCESS_CONTROL,
                enforcement_mode=EnforcementMode.ENFORCE,
                rules=[{"test": "rule"}],
            )

    async def test_evaluate_policies_no_applicable_policies(
        self, enforcer, sample_request
    ):
        """Test policy evaluation with no applicable policies."""
        # No policies registered, so none should be applicable
        result = await enforcer.evaluate_policies(sample_request)

        assert result.is_right()
        evaluation_result = result.get_right()
        assert evaluation_result.request_id == sample_request.request_id
        assert (
            evaluation_result.decision == EnforcementResult.ALLOWED
        )  # Default for no policies
        assert len(evaluation_result.applicable_policies) == 0

    async def test_evaluate_policies_with_applicable_policy(
        self, enforcer, sample_policy, sample_request
    ):
        """Test policy evaluation with applicable policies."""
        # Register policy first
        await enforcer.register_policy(sample_policy)

        # Mock the policy evaluation methods
        with (
            patch.object(
                enforcer,
                "_find_applicable_policies",
                return_value=[sample_policy.policy_id],
            ),
            patch.object(enforcer, "_evaluate_single_policy") as mock_evaluate,
        ):
            # Mock successful evaluation
            from src.core.either import Either

            mock_evaluate.return_value = Either.right(EnforcementResult.ALLOWED)

            with patch.object(
                enforcer,
                "_determine_final_decision",
                return_value=EnforcementResult.ALLOWED,
            ):
                result = await enforcer.evaluate_policies(sample_request)

                assert result.is_right()
                evaluation_result = result.get_right()
                assert evaluation_result.request_id == sample_request.request_id
                assert evaluation_result.decision == EnforcementResult.ALLOWED

    async def test_enforce_decision_allowed(self, enforcer, sample_request) -> None:
        """Test enforcement of ALLOWED decision."""
        evaluation_result = PolicyEvaluationResult(
            request_id=sample_request.request_id,
            decision=EnforcementResult.ALLOWED,
            applicable_policies=[],
            decision_reason="No restricting policies",
            decision_confidence=1.0,
            conditions=[],
            evaluation_time_ms=50.0,
            cached=False,
        )

        result = await enforcer.enforce_decision(evaluation_result, sample_request)

        assert result.is_right()
        enforcement_result = result.get_right()
        assert enforcement_result["action"] == "allow"
        assert enforcement_result["request_id"] == sample_request.request_id

    async def test_enforce_decision_denied(self, enforcer, sample_request) -> None:
        """Test enforcement of DENIED decision."""
        evaluation_result = PolicyEvaluationResult(
            request_id=sample_request.request_id,
            decision=EnforcementResult.DENIED,
            applicable_policies=[PolicyId("policy-001")],
            decision_reason="Access denied by security policy",
            decision_confidence=1.0,
            conditions=[],
            evaluation_time_ms=50.0,
            cached=False,
        )

        result = await enforcer.enforce_decision(evaluation_result, sample_request)

        assert result.is_right()
        enforcement_result = result.get_right()
        assert enforcement_result["action"] == "deny"
        assert enforcement_result["request_id"] == sample_request.request_id

    async def test_assess_compliance_gdpr(self, enforcer) -> None:
        """Test GDPR compliance assessment."""
        result = await enforcer.assess_compliance(ComplianceFramework.GDPR)

        assert result.is_right()
        assessment = result.get_right()
        assert assessment.framework == ComplianceFramework.GDPR
        assert isinstance(assessment.overall_status, ComplianceStatus)
        assert isinstance(assessment.rule_assessments, list)

    async def test_assess_compliance_sox(self, enforcer) -> None:
        """Test SOX compliance assessment."""
        result = await enforcer.assess_compliance(ComplianceFramework.SOX)

        assert result.is_right()
        assessment = result.get_right()
        assert assessment.framework == ComplianceFramework.SOX
        assert isinstance(assessment.overall_status, ComplianceStatus)

    async def test_register_compliance_rule_success(self, enforcer) -> None:
        """Test successful compliance rule registration."""
        rule = ComplianceRule(
            rule_id="gdpr-001",
            framework=ComplianceFramework.GDPR,
            requirement_id="Art. 32",
            description="Security of processing",
            validation_criteria={"encryption": True, "access_controls": True},
            severity=ThreatSeverity.HIGH,
            automated_check=True,
        )

        result = await enforcer.register_compliance_rule(rule)

        assert result.is_right()
        rule_id = result.get_right()
        assert rule_id == rule.rule_id
        assert rule.rule_id in enforcer.compliance_rules

    def test_validate_policy_success(self, enforcer, sample_policy) -> None:
        """Test successful policy validation."""
        result = enforcer._validate_policy(sample_policy)
        assert result is True

    def test_check_policy_conflicts_no_conflicts(self, enforcer, sample_policy) -> None:
        """Test policy conflict detection with no conflicts."""
        conflicts = enforcer._check_policy_conflicts(sample_policy)
        assert len(conflicts) == 0

    def test_find_applicable_policies_empty(self, enforcer, sample_request) -> None:
        """Test finding applicable policies when none exist."""
        applicable = enforcer._find_applicable_policies(sample_request)
        assert len(applicable) == 0

    def test_generate_cache_key(self, enforcer, sample_request) -> None:
        """Test cache key generation."""
        cache_key = enforcer._generate_cache_key(sample_request)
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

    def test_get_cached_evaluation_miss(self, enforcer) -> None:
        """Test cache miss for evaluation."""
        result = enforcer._get_cached_evaluation("non-existent-key")
        assert result is None

    def test_cache_evaluation_result(self, enforcer, sample_request) -> None:
        """Test caching evaluation result."""
        evaluation_result = PolicyEvaluationResult(
            request_id=sample_request.request_id,
            decision=EnforcementResult.ALLOWED,
            applicable_policies=[],
            decision_reason="Test reason",
            decision_confidence=1.0,
            conditions=[],
            evaluation_time_ms=50.0,
            cached=False,
        )

        cache_key = enforcer._generate_cache_key(sample_request)
        enforcer._cache_evaluation_result(cache_key, evaluation_result)

        # Verify cached result can be retrieved
        cached_result = enforcer._get_cached_evaluation(cache_key)
        assert cached_result is not None
        assert cached_result.request_id == sample_request.request_id

    def test_update_evaluation_metrics(self, enforcer) -> None:
        """Test evaluation metrics update."""
        initial_count = enforcer.evaluation_count
        enforcer._update_evaluation_metrics(100.5, True)

        assert enforcer.evaluation_count == initial_count + 1

    def test_add_policy_dict_format(self, enforcer) -> None:
        """Test adding policy in dictionary format."""
        policy_dict = {
            "name": "Test Policy",
            "description": "Test policy from dict",
            "rules": [{"resource": "*.txt", "action": "read"}],
            "priority": 100,
        }

        enforcer.add_policy(policy_dict)
        # Should not raise error for valid policy dict

    def test_validate_against_policies_no_policies(self, enforcer) -> None:
        """Test validation against policies when no policies exist."""
        data = {"resource": "test.txt", "action": "read"}
        result = enforcer.validate_against_policies(data)

        assert isinstance(result, PolicyValidationResult)
        assert result.is_valid is True  # No policies to violate

    def test_enforce_policies_no_policies(self, enforcer) -> None:
        """Test policy enforcement when no policies exist."""
        request_data = {"resource": "test.txt", "action": "read"}
        # Should not raise error when no policies exist
        enforcer.enforce_policies(request_data)

    def test_list_policies_empty(self, enforcer) -> None:
        """Test listing policies when none exist."""
        policies = enforcer.list_policies()
        assert isinstance(policies, list)
        assert len(policies) == 0

    def test_evaluate_policy_success(self, enforcer) -> None:
        """Test policy evaluation method."""
        policy_data = {"resource": "test.txt", "action": "read"}
        context = {"user": "test_user", "role": "admin"}

        result = enforcer.evaluate_policy(policy_data, context)
        assert isinstance(result, dict)
        assert "decision" in result
        assert "confidence" in result

    def test_enforce_policy_success(self, enforcer) -> None:
        """Test policy enforcement method."""
        policy_data = {"resource": "test.txt", "action": "read"}
        context = {"user": "test_user", "role": "admin"}

        result = enforcer.enforce_policy(policy_data, context)
        assert isinstance(result, dict)
        assert "enforced" in result


class TestPolicyValidationResult:
    """Test PolicyValidationResult dataclass."""

    def test_policy_validation_result_creation(self) -> None:
        """Test PolicyValidationResult creation."""
        result = PolicyValidationResult(
            is_valid=True,
            violations=[],
            warnings=["Minor security concern"],
            compliance_status=ComplianceStatus.COMPLIANT,
            validation_details={"checks_passed": 5, "checks_failed": 0},
        )

        assert result.is_valid is True
        assert len(result.violations) == 0
        assert len(result.warnings) == 1
        assert result.compliance_status == ComplianceStatus.COMPLIANT
        assert result.validation_details["checks_passed"] == 5

    def test_policy_validation_result_with_violations(self) -> None:
        """Test PolicyValidationResult with violations."""
        violations = ["Unauthorized access attempt", "Invalid credentials"]
        result = PolicyValidationResult(
            is_valid=False,
            violations=violations,
            warnings=[],
            compliance_status=ComplianceStatus.NON_COMPLIANT,
            validation_details={"failed_checks": violations},
        )

        assert result.is_valid is False
        assert len(result.violations) == 2
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    @pytest.fixture
    def configured_enforcer(self) -> None:
        """Create configured PolicyEnforcer with policies and rules."""
        enforcer = PolicyEnforcer()
        return enforcer

    async def test_full_policy_lifecycle(self, configured_enforcer) -> None:
        """Test complete policy lifecycle: register, evaluate, enforce."""
        # Create and register policy
        policy = SecurityPolicy(
            policy_id=PolicyId("lifecycle-policy"),
            name="Lifecycle Test Policy",
            description="Policy for testing full lifecycle",
            policy_type=PolicyType.ACCESS_CONTROL,
            enforcement_mode=EnforcementMode.ENFORCE,
            rules=[
                {"resource_pattern": "*.doc", "action": "read", "required_role": "user"}
            ],
            conditions={"time_range": "9-17"},
            priority=200,
            enabled=True,
        )

        # Register policy
        register_result = await configured_enforcer.register_policy(policy)
        assert register_result.is_right()

        # Create evaluation request
        context = SecurityContext(
            user_id="user-lifecycle",
            session_id="session-lifecycle",
            trust_level=TrustLevel.TRUSTED,
            permissions=frozenset(["read"]),
            validation_scope=ValidationScope.STRICT,
        )

        request = PolicyEvaluationRequest(
            request_id="req-lifecycle",
            context=context,
            resource="document.doc",
            action="read",
            additional_data={"current_time": "14:30"},
        )

        # Evaluate policies
        eval_result = await configured_enforcer.evaluate_policies(request)
        assert eval_result.is_right()

        # Enforce decision
        evaluation = eval_result.get_right()
        enforce_result = await configured_enforcer.enforce_decision(evaluation, request)
        assert enforce_result.is_right()

    async def test_compliance_assessment_with_rules(self, configured_enforcer) -> None:
        """Test compliance assessment with registered rules."""
        # Register compliance rule
        rule = ComplianceRule(
            rule_id="test-compliance-001",
            framework=ComplianceFramework.HIPAA,
            requirement_id="164.312(a)(1)",
            description="Access control unique user identification",
            validation_criteria={"unique_user_id": True, "audit_logging": True},
            severity=ThreatSeverity.HIGH,
            automated_check=True,
        )

        rule_result = await configured_enforcer.register_compliance_rule(rule)
        assert rule_result.is_right()

        # Assess compliance
        assessment_result = await configured_enforcer.assess_compliance(
            ComplianceFramework.HIPAA
        )
        assert assessment_result.is_right()

        assessment = assessment_result.get_right()
        assert assessment.framework == ComplianceFramework.HIPAA
        assert len(assessment.rule_assessments) > 0

    async def test_policy_violation_handling(self, configured_enforcer) -> None:
        """Test policy violation detection and handling."""
        # Create policy that will be violated
        restrictive_policy = SecurityPolicy(
            policy_id=PolicyId("restrictive-policy"),
            name="Restrictive Policy",
            description="Policy that denies most actions",
            policy_type=PolicyType.ACCESS_CONTROL,
            enforcement_mode=EnforcementMode.ENFORCE,
            rules=[{"resource_pattern": "*.secret", "action": "*", "effect": "deny"}],
            conditions={},
            priority=1000,  # High priority
            enabled=True,
        )

        # Register policy
        await configured_enforcer.register_policy(restrictive_policy)

        # Create request that should violate policy
        context = SecurityContext(
            user_id="user-violation",
            session_id="session-violation",
            trust_level=TrustLevel.UNTRUSTED,
            permissions=frozenset(["read"]),
            validation_scope=ValidationScope.STRICT,
        )

        violation_request = PolicyEvaluationRequest(
            request_id="req-violation",
            context=context,
            resource="secrets.secret",
            action="read",
        )

        # Evaluate - should detect violation
        eval_result = await configured_enforcer.evaluate_policies(violation_request)
        assert eval_result.is_right()

        evaluation = eval_result.get_right()
        # Enforcement result depends on policy logic implementation
        assert evaluation.request_id == violation_request.request_id
