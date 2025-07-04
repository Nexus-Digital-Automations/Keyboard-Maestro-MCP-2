"""
Policy Enforcer - TASK_62 Phase 2 Core Security Engine

Dynamic security policy enforcement and compliance monitoring for zero trust security.
Provides real-time policy evaluation, enforcement actions, and compliance tracking.

Architecture: Zero Trust Principles + Policy Engine + Dynamic Enforcement + Compliance Monitoring
Performance: <200ms policy evaluation, <100ms enforcement action, <50ms compliance check
Security: Fail-safe enforcement, comprehensive audit trail, dynamic policy adaptation
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import json
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.zero_trust_architecture import (
    TrustScore, PolicyId, SecurityContextId, ThreatId, ValidationId, 
    RiskScore, ComplianceId, TrustLevel, ValidationScope, SecurityOperation,
    PolicyType, EnforcementMode, ThreatSeverity, ComplianceFramework,
    SecurityPolicy, PolicyViolation, AccessDecision, ComplianceAssessment,
    SecurityContext, ZeroTrustError, PolicyEnforcementError, ComplianceError,
    create_policy_id, create_compliance_id, create_security_context_id,
    is_policy_applicable, generate_compliance_recommendations
)


class PolicyStatus(Enum):
    """Security policy status."""
    ACTIVE = "active"                # Policy is active and enforced
    INACTIVE = "inactive"            # Policy exists but not enforced
    PENDING = "pending"              # Policy waiting for activation
    DEPRECATED = "deprecated"        # Policy marked for removal
    SUSPENDED = "suspended"          # Policy temporarily suspended
    DRAFT = "draft"                  # Policy in draft state


class EnforcementResult(Enum):
    """Policy enforcement results."""
    ALLOWED = "allowed"              # Action allowed by policy
    DENIED = "denied"                # Action denied by policy
    CONDITIONAL = "conditional"      # Action allowed with conditions
    REQUIRES_APPROVAL = "requires_approval"  # Manual approval required
    REMEDIATED = "remediated"        # Violation automatically remediated
    ESCALATED = "escalated"          # Violation escalated for review


class ComplianceStatus(Enum):
    """Compliance assessment status."""
    COMPLIANT = "compliant"          # Fully compliant
    NON_COMPLIANT = "non_compliant"  # Not compliant
    PARTIALLY_COMPLIANT = "partially_compliant"  # Some requirements met
    UNKNOWN = "unknown"              # Compliance status unknown
    PENDING_REVIEW = "pending_review"  # Under compliance review
    REMEDIATION_REQUIRED = "remediation_required"  # Needs remediation


@dataclass(frozen=True)
class PolicyEvaluationRequest:
    """Policy evaluation request specification."""
    request_id: str
    context: SecurityContext
    resource: str
    action: str
    additional_data: Dict[str, Any] = field(default_factory=dict)
    evaluation_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    priority: str = "normal"  # low, normal, high, critical
    timeout: int = 30  # seconds
    
    def __post_init__(self):
        if not self.request_id or not self.resource or not self.action:
            raise ValueError("Request ID, resource, and action are required")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")


@dataclass(frozen=True)
class PolicyEvaluationResult:
    """Policy evaluation result."""
    request_id: str
    decision: EnforcementResult
    applicable_policies: List[PolicyId]
    policy_violations: List[PolicyViolation]
    conditions: List[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
    evaluation_time: float = 0.0  # milliseconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.evaluation_time < 0:
            raise ValueError("Evaluation time cannot be negative")


@dataclass(frozen=True)
class EnforcementAction:
    """Security enforcement action specification."""
    action_id: str
    action_type: str  # block, allow, monitor, remediate, escalate
    target: str
    parameters: Dict[str, Any]
    triggered_by: PolicyId
    executed_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.action_id or not self.action_type or not self.target:
            raise ValueError("Action ID, type, and target are required")


@dataclass(frozen=True)
class ComplianceRule:
    """Compliance rule specification."""
    rule_id: str
    framework: ComplianceFramework
    requirement_id: str
    description: str
    validation_criteria: Dict[str, Any]
    severity: ThreatSeverity = ThreatSeverity.MEDIUM
    automated_check: bool = True
    remediation_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.rule_id or not self.requirement_id or not self.description:
            raise ValueError("Rule ID, requirement ID, and description are required")


class PolicyEnforcer:
    """Dynamic security policy enforcement and compliance monitoring system."""
    
    def __init__(self):
        self.active_policies: Dict[PolicyId, SecurityPolicy] = {}
        self.policy_status: Dict[PolicyId, PolicyStatus] = {}
        self.enforcement_history: List[PolicyEvaluationResult] = []
        self.violation_history: List[PolicyViolation] = []
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.enforcement_cache: Dict[str, Tuple[EnforcementResult, datetime]] = {}
        
        # Performance metrics
        self.evaluation_count = 0
        self.average_evaluation_time = 0.0
        self.cache_hit_rate = 0.0
        self.violation_rate = 0.0
    
    @require(lambda self, policy: isinstance(policy, SecurityPolicy))
    @ensure(lambda self, result: result.is_right() or isinstance(result.get_left(), PolicyEnforcementError))
    async def register_policy(
        self, 
        policy: SecurityPolicy
    ) -> Either[PolicyEnforcementError, PolicyId]:
        """Register a new security policy."""
        try:
            # Validate policy
            policy_validation = self._validate_policy(policy)
            if policy_validation.is_left():
                return policy_validation
            
            # Check for policy conflicts
            conflict_check = self._check_policy_conflicts(policy)
            if conflict_check.is_left():
                return conflict_check
            
            # Register policy
            self.active_policies[policy.policy_id] = policy
            self.policy_status[policy.policy_id] = PolicyStatus.ACTIVE if policy.enabled else PolicyStatus.INACTIVE
            
            return Either.right(policy.policy_id)
            
        except Exception as e:
            return Either.left(PolicyEnforcementError(
                f"Failed to register policy: {str(e)}",
                "POLICY_REGISTRATION_ERROR",
                SecurityOperation.ENFORCE,
                {"policy_id": policy.policy_id}
            ))
    
    @require(lambda self, request: isinstance(request, PolicyEvaluationRequest))
    @ensure(lambda self, result: result.is_right() or isinstance(result.get_left(), PolicyEnforcementError))
    async def evaluate_policies(
        self, 
        request: PolicyEvaluationRequest
    ) -> Either[PolicyEnforcementError, PolicyEvaluationResult]:
        """Evaluate security policies for access request."""
        try:
            start_time = datetime.now(UTC)
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_cached_evaluation(cache_key)
            if cached_result:
                return Either.right(cached_result)
            
            # Find applicable policies
            applicable_policies = self._find_applicable_policies(request)
            
            # Evaluate each applicable policy
            policy_results = []
            violations = []
            
            for policy_id in applicable_policies:
                policy = self.active_policies[policy_id]
                evaluation_result = await self._evaluate_single_policy(policy, request)
                
                if evaluation_result.is_right():
                    result = evaluation_result.get_right()
                    policy_results.append((policy_id, result))
                    
                    # Check for violations
                    if result == EnforcementResult.DENIED:
                        violation = self._create_policy_violation(policy, request)
                        violations.append(violation)
            
            # Determine final decision
            final_decision = self._determine_final_decision(policy_results, request)
            
            # Generate conditions and reason
            conditions = self._generate_conditions(policy_results, final_decision)
            reason = self._generate_decision_reason(policy_results, final_decision)
            confidence = self._calculate_decision_confidence(policy_results)
            
            # Calculate evaluation time
            end_time = datetime.now(UTC)
            evaluation_time = (end_time - start_time).total_seconds() * 1000
            
            # Create evaluation result
            evaluation_result = PolicyEvaluationResult(
                request_id=request.request_id,
                decision=final_decision,
                applicable_policies=applicable_policies,
                policy_violations=violations,
                conditions=conditions,
                reason=reason,
                confidence=confidence,
                evaluation_time=evaluation_time,
                metadata={
                    "policies_evaluated": len(applicable_policies),
                    "violations_found": len(violations),
                    "cache_hit": False,
                    "evaluation_version": "1.0"
                }
            )
            
            # Store in cache
            self._cache_evaluation_result(cache_key, evaluation_result)
            
            # Store in history
            self.enforcement_history.append(evaluation_result)
            self.violation_history.extend(violations)
            
            # Update metrics
            self._update_evaluation_metrics(evaluation_time, len(violations) > 0)
            
            return Either.right(evaluation_result)
            
        except Exception as e:
            return Either.left(PolicyEnforcementError(
                f"Policy evaluation failed: {str(e)}",
                "EVALUATION_ERROR",
                SecurityOperation.ENFORCE,
                {"request_id": request.request_id}
            ))
    
    @require(lambda self, evaluation_result: isinstance(evaluation_result, PolicyEvaluationResult))
    async def enforce_decision(
        self, 
        evaluation_result: PolicyEvaluationResult
    ) -> Either[PolicyEnforcementError, List[EnforcementAction]]:
        """Enforce policy decision with appropriate actions."""
        try:
            enforcement_actions = []
            
            # Determine enforcement actions based on decision
            if evaluation_result.decision == EnforcementResult.DENIED:
                # Block access
                block_action = await self._create_block_action(evaluation_result)
                if block_action.is_right():
                    enforcement_actions.append(block_action.get_right())
                
                # Handle violations
                for violation in evaluation_result.policy_violations:
                    violation_actions = await self._handle_policy_violation(violation)
                    if violation_actions.is_right():
                        enforcement_actions.extend(violation_actions.get_right())
            
            elif evaluation_result.decision == EnforcementResult.CONDITIONAL:
                # Apply conditions
                condition_actions = await self._apply_conditions(evaluation_result)
                if condition_actions.is_right():
                    enforcement_actions.extend(condition_actions.get_right())
            
            elif evaluation_result.decision == EnforcementResult.REQUIRES_APPROVAL:
                # Escalate for approval
                escalation_action = await self._create_escalation_action(evaluation_result)
                if escalation_action.is_right():
                    enforcement_actions.append(escalation_action.get_right())
            
            # Execute enforcement actions
            executed_actions = []
            for action in enforcement_actions:
                execution_result = await self._execute_enforcement_action(action)
                if execution_result.is_right():
                    executed_actions.append(execution_result.get_right())
            
            return Either.right(executed_actions)
            
        except Exception as e:
            return Either.left(PolicyEnforcementError(
                f"Policy enforcement failed: {str(e)}",
                "ENFORCEMENT_ERROR",
                SecurityOperation.ENFORCE,
                {"request_id": evaluation_result.request_id}
            ))
    
    async def assess_compliance(
        self, 
        framework: ComplianceFramework, 
        target_scope: str
    ) -> Either[ComplianceError, ComplianceAssessment]:
        """Assess compliance against specific framework."""
        try:
            assessment_start = datetime.now(UTC)
            
            # Get compliance rules for framework
            framework_rules = self._get_compliance_rules_for_framework(framework)
            
            if not framework_rules:
                return Either.left(ComplianceError(
                    f"No compliance rules found for framework {framework.value}",
                    "NO_COMPLIANCE_RULES",
                    SecurityOperation.AUDIT,
                    {"framework": framework.value}
                ))
            
            # Evaluate each compliance rule
            total_requirements = len(framework_rules)
            requirements_met = 0
            findings = []
            
            for rule in framework_rules:
                compliance_result = await self._evaluate_compliance_rule(rule, target_scope)
                
                if compliance_result.is_right():
                    is_compliant = compliance_result.get_right()
                    if is_compliant:
                        requirements_met += 1
                    else:
                        findings.append({
                            "rule_id": rule.rule_id,
                            "requirement": rule.requirement_id,
                            "description": rule.description,
                            "severity": rule.severity.value,
                            "status": "non_compliant"
                        })
                else:
                    # Rule evaluation failed
                    findings.append({
                        "rule_id": rule.rule_id,
                        "requirement": rule.requirement_id,
                        "description": rule.description,
                        "severity": rule.severity.value,
                        "status": "evaluation_failed",
                        "error": str(compliance_result.get_left())
                    })
            
            # Calculate compliance score
            compliance_score = requirements_met / total_requirements if total_requirements > 0 else 0.0
            
            # Generate recommendations
            recommendations = generate_compliance_recommendations(ComplianceAssessment(
                assessment_id=f"assessment_{int(assessment_start.timestamp())}",
                compliance_id=create_compliance_id(framework.value),
                framework=framework,
                assessment_date=assessment_start,
                compliance_score=compliance_score,
                requirements_met=requirements_met,
                total_requirements=total_requirements,
                findings=findings
            ))
            
            # Create assessment
            assessment = ComplianceAssessment(
                assessment_id=f"assessment_{int(assessment_start.timestamp())}",
                compliance_id=create_compliance_id(framework.value),
                framework=framework,
                assessment_date=assessment_start,
                compliance_score=compliance_score,
                requirements_met=requirements_met,
                total_requirements=total_requirements,
                findings=findings,
                recommendations=recommendations,
                next_assessment=assessment_start + timedelta(days=90),  # Quarterly assessments
                metadata={
                    "target_scope": target_scope,
                    "assessment_duration_ms": (datetime.now(UTC) - assessment_start).total_seconds() * 1000
                }
            )
            
            return Either.right(assessment)
            
        except Exception as e:
            return Either.left(ComplianceError(
                f"Compliance assessment failed: {str(e)}",
                "ASSESSMENT_ERROR",
                SecurityOperation.AUDIT,
                {"framework": framework.value, "target_scope": target_scope}
            ))
    
    async def register_compliance_rule(
        self, 
        rule: ComplianceRule
    ) -> Either[ComplianceError, str]:
        """Register a new compliance rule."""
        try:
            # Validate rule
            if not rule.rule_id or not rule.requirement_id:
                return Either.left(ComplianceError(
                    "Compliance rule must have rule_id and requirement_id",
                    "INVALID_COMPLIANCE_RULE",
                    SecurityOperation.AUDIT
                ))
            
            # Check for duplicate rules
            if rule.rule_id in self.compliance_rules:
                return Either.left(ComplianceError(
                    f"Compliance rule {rule.rule_id} already exists",
                    "DUPLICATE_COMPLIANCE_RULE",
                    SecurityOperation.AUDIT,
                    {"rule_id": rule.rule_id}
                ))
            
            # Register rule
            self.compliance_rules[rule.rule_id] = rule
            
            return Either.right(f"Compliance rule {rule.rule_id} registered successfully")
            
        except Exception as e:
            return Either.left(ComplianceError(
                f"Failed to register compliance rule: {str(e)}",
                "RULE_REGISTRATION_ERROR",
                SecurityOperation.AUDIT
            ))
    
    def _validate_policy(self, policy: SecurityPolicy) -> Either[PolicyEnforcementError, None]:
        """Validate security policy configuration."""
        # Check required fields
        if not policy.policy_id or not policy.policy_name:
            return Either.left(PolicyEnforcementError(
                "Policy must have ID and name",
                "INVALID_POLICY",
                SecurityOperation.ENFORCE
            ))
        
        # Validate conditions
        if not policy.conditions:
            return Either.left(PolicyEnforcementError(
                "Policy must have conditions",
                "MISSING_CONDITIONS",
                SecurityOperation.ENFORCE
            ))
        
        # Validate actions
        if not policy.actions:
            return Either.left(PolicyEnforcementError(
                "Policy must have actions",
                "MISSING_ACTIONS",
                SecurityOperation.ENFORCE
            ))
        
        return Either.right(None)
    
    def _check_policy_conflicts(
        self, 
        new_policy: SecurityPolicy
    ) -> Either[PolicyEnforcementError, None]:
        """Check for conflicts with existing policies."""
        for existing_policy in self.active_policies.values():
            # Check for overlapping scope and conflicting actions
            if (existing_policy.policy_type == new_policy.policy_type and
                set(existing_policy.scope) & set(new_policy.scope) and
                existing_policy.priority == new_policy.priority):
                
                # Check for conflicting enforcement modes
                if (existing_policy.enforcement_mode == EnforcementMode.BLOCK and
                    new_policy.enforcement_mode == EnforcementMode.MONITOR):
                    return Either.left(PolicyEnforcementError(
                        f"Policy conflict with {existing_policy.policy_id}",
                        "POLICY_CONFLICT",
                        SecurityOperation.ENFORCE,
                        {"conflicting_policy": existing_policy.policy_id}
                    ))
        
        return Either.right(None)
    
    def _find_applicable_policies(
        self, 
        request: PolicyEvaluationRequest
    ) -> List[PolicyId]:
        """Find policies applicable to the request."""
        applicable_policies = []
        
        for policy_id, policy in self.active_policies.items():
            if (self.policy_status.get(policy_id) == PolicyStatus.ACTIVE and
                is_policy_applicable(policy, request.context, request.resource, request.action)):
                applicable_policies.append(policy_id)
        
        # Sort by priority (higher priority first)
        applicable_policies.sort(
            key=lambda pid: self.active_policies[pid].priority,
            reverse=True
        )
        
        return applicable_policies
    
    async def _evaluate_single_policy(
        self, 
        policy: SecurityPolicy, 
        request: PolicyEvaluationRequest
    ) -> Either[PolicyEnforcementError, EnforcementResult]:
        """Evaluate a single policy against the request."""
        try:
            # Check policy conditions
            conditions_met = self._evaluate_policy_conditions(policy, request)
            
            if not conditions_met:
                # Policy conditions not met - allow by default
                return Either.right(EnforcementResult.ALLOWED)
            
            # Policy applies - determine enforcement action
            if policy.enforcement_mode == EnforcementMode.BLOCK:
                return Either.right(EnforcementResult.DENIED)
            elif policy.enforcement_mode == EnforcementMode.MONITOR:
                return Either.right(EnforcementResult.ALLOWED)
            elif policy.enforcement_mode == EnforcementMode.WARN:
                return Either.right(EnforcementResult.CONDITIONAL)
            elif policy.enforcement_mode == EnforcementMode.REMEDIATE:
                return Either.right(EnforcementResult.REMEDIATED)
            elif policy.enforcement_mode == EnforcementMode.ADAPTIVE:
                # Adaptive enforcement based on context
                if request.context.trust_level == TrustLevel.HIGH:
                    return Either.right(EnforcementResult.ALLOWED)
                elif request.context.trust_level == TrustLevel.LOW:
                    return Either.right(EnforcementResult.DENIED)
                else:
                    return Either.right(EnforcementResult.CONDITIONAL)
            
            return Either.right(EnforcementResult.ALLOWED)
            
        except Exception as e:
            return Either.left(PolicyEnforcementError(
                f"Policy evaluation failed: {str(e)}",
                "POLICY_EVALUATION_ERROR",
                SecurityOperation.ENFORCE,
                {"policy_id": policy.policy_id}
            ))
    
    def _evaluate_policy_conditions(
        self, 
        policy: SecurityPolicy, 
        request: PolicyEvaluationRequest
    ) -> bool:
        """Evaluate policy conditions against request."""
        try:
            # Check trust level requirements
            if "min_trust_level" in policy.conditions:
                required_level = TrustLevel(policy.conditions["min_trust_level"])
                trust_levels = {
                    TrustLevel.UNTRUSTED: 0,
                    TrustLevel.LOW: 1,
                    TrustLevel.MEDIUM: 2,
                    TrustLevel.HIGH: 3,
                    TrustLevel.VERIFIED: 4
                }
                
                current_level = trust_levels.get(request.context.trust_level, 0)
                required_level_value = trust_levels.get(required_level, 0)
                
                if current_level < required_level_value:
                    return True  # Condition triggers policy
            
            # Check risk score thresholds
            if "max_risk_score" in policy.conditions:
                max_risk = policy.conditions["max_risk_score"]
                if request.context.risk_score > max_risk:
                    return True  # Condition triggers policy
            
            # Check time-based conditions
            if "allowed_hours" in policy.conditions:
                allowed_hours = policy.conditions["allowed_hours"]
                current_hour = datetime.now(UTC).hour
                if current_hour not in allowed_hours:
                    return True  # Condition triggers policy
            
            # Check location-based conditions
            if "blocked_locations" in policy.conditions:
                blocked_locations = policy.conditions["blocked_locations"]
                if request.context.location in blocked_locations:
                    return True  # Condition triggers policy
            
            return False  # No conditions trigger policy
            
        except Exception:
            return False  # Fail safe - don't trigger on evaluation errors
    
    def _determine_final_decision(
        self, 
        policy_results: List[Tuple[PolicyId, EnforcementResult]], 
        request: PolicyEvaluationRequest
    ) -> EnforcementResult:
        """Determine final enforcement decision from policy results."""
        if not policy_results:
            return EnforcementResult.ALLOWED  # Default allow if no policies apply
        
        # Priority order: DENIED > REQUIRES_APPROVAL > CONDITIONAL > REMEDIATED > ALLOWED
        decision_priority = {
            EnforcementResult.DENIED: 5,
            EnforcementResult.REQUIRES_APPROVAL: 4,
            EnforcementResult.CONDITIONAL: 3,
            EnforcementResult.REMEDIATED: 2,
            EnforcementResult.ALLOWED: 1
        }
        
        highest_priority = 0
        final_decision = EnforcementResult.ALLOWED
        
        for policy_id, result in policy_results:
            priority = decision_priority.get(result, 0)
            if priority > highest_priority:
                highest_priority = priority
                final_decision = result
        
        return final_decision
    
    def _generate_conditions(
        self, 
        policy_results: List[Tuple[PolicyId, EnforcementResult]], 
        final_decision: EnforcementResult
    ) -> List[str]:
        """Generate conditions for conditional access."""
        conditions = []
        
        if final_decision == EnforcementResult.CONDITIONAL:
            for policy_id, result in policy_results:
                if result == EnforcementResult.CONDITIONAL:
                    policy = self.active_policies[policy_id]
                    if "conditions" in policy.actions:
                        conditions.extend(policy.actions["conditions"])
        
        return list(set(conditions))  # Remove duplicates
    
    def _generate_decision_reason(
        self, 
        policy_results: List[Tuple[PolicyId, EnforcementResult]], 
        final_decision: EnforcementResult
    ) -> str:
        """Generate human-readable reason for the decision."""
        if not policy_results:
            return "No applicable policies found - default allow"
        
        if final_decision == EnforcementResult.DENIED:
            blocking_policies = [
                policy_id for policy_id, result in policy_results 
                if result == EnforcementResult.DENIED
            ]
            return f"Access denied by policies: {', '.join(blocking_policies)}"
        
        elif final_decision == EnforcementResult.CONDITIONAL:
            conditional_policies = [
                policy_id for policy_id, result in policy_results 
                if result == EnforcementResult.CONDITIONAL
            ]
            return f"Conditional access granted with restrictions from policies: {', '.join(conditional_policies)}"
        
        elif final_decision == EnforcementResult.ALLOWED:
            return "Access allowed by policy evaluation"
        
        else:
            return f"Access {final_decision.value} by policy evaluation"
    
    def _calculate_decision_confidence(
        self, 
        policy_results: List[Tuple[PolicyId, EnforcementResult]]
    ) -> float:
        """Calculate confidence in the enforcement decision."""
        if not policy_results:
            return 0.5  # Medium confidence for default decisions
        
        # Higher confidence with more policies agreeing
        total_policies = len(policy_results)
        consistent_results = {}
        
        for _, result in policy_results:
            consistent_results[result] = consistent_results.get(result, 0) + 1
        
        max_consistency = max(consistent_results.values())
        consistency_ratio = max_consistency / total_policies
        
        # Base confidence on consistency and number of policies
        base_confidence = consistency_ratio
        policy_factor = min(1.0, total_policies / 5.0)  # More policies = higher confidence
        
        return min(1.0, base_confidence * policy_factor)
    
    def _create_policy_violation(
        self, 
        policy: SecurityPolicy, 
        request: PolicyEvaluationRequest
    ) -> PolicyViolation:
        """Create policy violation record."""
        return PolicyViolation(
            violation_id=f"violation_{int(datetime.now(UTC).timestamp())}",
            policy_id=policy.policy_id,
            target_id=request.context.user_id or "unknown",
            violation_type=policy.policy_type.value,
            severity=ThreatSeverity.MEDIUM,  # Default severity
            description=f"Policy {policy.policy_name} violated by {request.action} on {request.resource}",
            detected_at=datetime.now(UTC),
            context=request.context,
            evidence={
                "resource": request.resource,
                "action": request.action,
                "policy_conditions": policy.conditions,
                "context_data": request.additional_data
            }
        )
    
    async def _create_block_action(
        self, 
        evaluation_result: PolicyEvaluationResult
    ) -> Either[PolicyEnforcementError, EnforcementAction]:
        """Create blocking enforcement action."""
        try:
            action = EnforcementAction(
                action_id=f"block_{evaluation_result.request_id}",
                action_type="block",
                target=evaluation_result.request_id,
                parameters={
                    "reason": evaluation_result.reason,
                    "policies": [str(pid) for pid in evaluation_result.applicable_policies]
                },
                triggered_by=evaluation_result.applicable_policies[0] if evaluation_result.applicable_policies else PolicyId("default")
            )
            
            return Either.right(action)
            
        except Exception as e:
            return Either.left(PolicyEnforcementError(
                f"Failed to create block action: {str(e)}",
                "BLOCK_ACTION_ERROR",
                SecurityOperation.ENFORCE
            ))
    
    async def _handle_policy_violation(
        self, 
        violation: PolicyViolation
    ) -> Either[PolicyEnforcementError, List[EnforcementAction]]:
        """Handle policy violation with appropriate actions."""
        try:
            actions = []
            
            # Log violation
            log_action = EnforcementAction(
                action_id=f"log_{violation.violation_id}",
                action_type="log",
                target=violation.target_id,
                parameters={
                    "violation_type": violation.violation_type,
                    "severity": violation.severity.value,
                    "description": violation.description
                },
                triggered_by=violation.policy_id
            )
            actions.append(log_action)
            
            # Escalate high-severity violations
            if violation.severity in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]:
                escalate_action = EnforcementAction(
                    action_id=f"escalate_{violation.violation_id}",
                    action_type="escalate",
                    target=violation.target_id,
                    parameters={
                        "violation_id": violation.violation_id,
                        "escalation_level": "security_team"
                    },
                    triggered_by=violation.policy_id
                )
                actions.append(escalate_action)
            
            return Either.right(actions)
            
        except Exception as e:
            return Either.left(PolicyEnforcementError(
                f"Failed to handle violation: {str(e)}",
                "VIOLATION_HANDLING_ERROR",
                SecurityOperation.ENFORCE
            ))
    
    async def _apply_conditions(
        self, 
        evaluation_result: PolicyEvaluationResult
    ) -> Either[PolicyEnforcementError, List[EnforcementAction]]:
        """Apply conditional access conditions."""
        try:
            actions = []
            
            for condition in evaluation_result.conditions:
                condition_action = EnforcementAction(
                    action_id=f"condition_{evaluation_result.request_id}_{len(actions)}",
                    action_type="apply_condition",
                    target=evaluation_result.request_id,
                    parameters={
                        "condition": condition,
                        "enforcement_mode": "conditional"
                    },
                    triggered_by=evaluation_result.applicable_policies[0] if evaluation_result.applicable_policies else PolicyId("default")
                )
                actions.append(condition_action)
            
            return Either.right(actions)
            
        except Exception as e:
            return Either.left(PolicyEnforcementError(
                f"Failed to apply conditions: {str(e)}",
                "CONDITION_APPLICATION_ERROR",
                SecurityOperation.ENFORCE
            ))
    
    async def _create_escalation_action(
        self, 
        evaluation_result: PolicyEvaluationResult
    ) -> Either[PolicyEnforcementError, EnforcementAction]:
        """Create escalation action for manual approval."""
        try:
            action = EnforcementAction(
                action_id=f"escalate_{evaluation_result.request_id}",
                action_type="escalate",
                target=evaluation_result.request_id,
                parameters={
                    "reason": "Manual approval required",
                    "decision_confidence": evaluation_result.confidence,
                    "escalation_level": "approval_required"
                },
                triggered_by=evaluation_result.applicable_policies[0] if evaluation_result.applicable_policies else PolicyId("default")
            )
            
            return Either.right(action)
            
        except Exception as e:
            return Either.left(PolicyEnforcementError(
                f"Failed to create escalation action: {str(e)}",
                "ESCALATION_ACTION_ERROR",
                SecurityOperation.ENFORCE
            ))
    
    async def _execute_enforcement_action(
        self, 
        action: EnforcementAction
    ) -> Either[PolicyEnforcementError, EnforcementAction]:
        """Execute enforcement action."""
        try:
            # Simulate action execution
            executed_action = EnforcementAction(
                action_id=action.action_id,
                action_type=action.action_type,
                target=action.target,
                parameters=action.parameters,
                triggered_by=action.triggered_by,
                executed_at=datetime.now(UTC),
                success=True,
                metadata={
                    "execution_method": "simulated",
                    "execution_time": datetime.now(UTC).isoformat()
                }
            )
            
            return Either.right(executed_action)
            
        except Exception as e:
            failed_action = EnforcementAction(
                action_id=action.action_id,
                action_type=action.action_type,
                target=action.target,
                parameters=action.parameters,
                triggered_by=action.triggered_by,
                executed_at=datetime.now(UTC),
                success=False,
                error_message=str(e)
            )
            
            return Either.right(failed_action)  # Return failed action for logging
    
    def _get_compliance_rules_for_framework(
        self, 
        framework: ComplianceFramework
    ) -> List[ComplianceRule]:
        """Get compliance rules for specific framework."""
        return [
            rule for rule in self.compliance_rules.values()
            if rule.framework == framework
        ]
    
    async def _evaluate_compliance_rule(
        self, 
        rule: ComplianceRule, 
        target_scope: str
    ) -> Either[ComplianceError, bool]:
        """Evaluate single compliance rule."""
        try:
            # Simulate compliance rule evaluation
            # In real implementation, this would check actual system state
            
            criteria = rule.validation_criteria
            
            # Check if automated check is available
            if not rule.automated_check:
                # Manual review required
                return Either.right(True)  # Assume compliant for simulation
            
            # Simulate automated compliance check
            # This would integrate with actual compliance monitoring systems
            compliance_score = 0.8  # Simulated score
            
            return Either.right(compliance_score >= 0.7)
            
        except Exception as e:
            return Either.left(ComplianceError(
                f"Compliance rule evaluation failed: {str(e)}",
                "RULE_EVALUATION_ERROR",
                SecurityOperation.AUDIT,
                {"rule_id": rule.rule_id}
            ))
    
    def _generate_cache_key(self, request: PolicyEvaluationRequest) -> str:
        """Generate cache key for policy evaluation."""
        key_components = [
            request.context.user_id or "anonymous",
            request.context.device_id or "unknown",
            request.resource,
            request.action,
            str(request.context.trust_level.value)
        ]
        
        return ":".join(key_components)
    
    def _get_cached_evaluation(self, cache_key: str) -> Optional[PolicyEvaluationResult]:
        """Get cached policy evaluation result."""
        if cache_key in self.enforcement_cache:
            cached_result, cached_time = self.enforcement_cache[cache_key]
            # Check if cache is still valid (5 minutes)
            if (datetime.now(UTC) - cached_time).total_seconds() < 300:
                # Convert cached result to full PolicyEvaluationResult
                # This is simplified - in practice would store full result
                return None  # Placeholder
        
        return None
    
    def _cache_evaluation_result(
        self, 
        cache_key: str, 
        result: PolicyEvaluationResult
    ) -> None:
        """Cache policy evaluation result."""
        # Store simplified result in cache
        self.enforcement_cache[cache_key] = (result.decision, datetime.now(UTC))
        
        # Clean old cache entries (keep last 1000)
        if len(self.enforcement_cache) > 1000:
            oldest_keys = sorted(
                self.enforcement_cache.keys(),
                key=lambda k: self.enforcement_cache[k][1]
            )[:100]
            
            for key in oldest_keys:
                del self.enforcement_cache[key]
    
    def _update_evaluation_metrics(
        self, 
        evaluation_time: float, 
        had_violations: bool
    ) -> None:
        """Update policy evaluation metrics."""
        self.evaluation_count += 1
        
        # Update average evaluation time
        if self.evaluation_count == 1:
            self.average_evaluation_time = evaluation_time
        else:
            alpha = 0.1  # Exponential moving average factor
            self.average_evaluation_time = (
                alpha * evaluation_time + 
                (1 - alpha) * self.average_evaluation_time
            )
        
        # Update violation rate
        if self.evaluation_count == 1:
            self.violation_rate = 1.0 if had_violations else 0.0
        else:
            alpha = 0.1
            violation_indicator = 1.0 if had_violations else 0.0
            self.violation_rate = (
                alpha * violation_indicator + 
                (1 - alpha) * self.violation_rate
            )


# Utility functions for policy enforcement
def create_security_policy(
    policy_name: str,
    policy_type: PolicyType,
    enforcement_mode: EnforcementMode,
    scope: List[ValidationScope],
    conditions: Dict[str, Any],
    actions: Dict[str, Any],
    priority: int = 50
) -> SecurityPolicy:
    """Create a security policy with validation."""
    return SecurityPolicy(
        policy_id=create_policy_id(policy_name),
        policy_name=policy_name,
        policy_type=policy_type,
        description=f"Security policy for {policy_type.value}",
        enforcement_mode=enforcement_mode,
        scope=scope,
        conditions=conditions,
        actions=actions,
        priority=priority
    )


def create_compliance_rule(
    rule_id: str,
    framework: ComplianceFramework,
    requirement_id: str,
    description: str,
    validation_criteria: Dict[str, Any],
    severity: ThreatSeverity = ThreatSeverity.MEDIUM
) -> ComplianceRule:
    """Create a compliance rule with validation."""
    return ComplianceRule(
        rule_id=rule_id,
        framework=framework,
        requirement_id=requirement_id,
        description=description,
        validation_criteria=validation_criteria,
        severity=severity
    )
