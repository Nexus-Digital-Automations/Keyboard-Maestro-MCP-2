"""
Safety validation and constraint enforcement for autonomous agents.

This module provides comprehensive safety checks, constraint validation, and
risk assessment for autonomous agent operations. Implements defense-in-depth
security with configurable safety policies.

Security: Multi-layer validation with fail-safe defaults
Performance: <50ms validation checks, <200ms comprehensive assessment
Enterprise: Complete audit trail and policy compliance
"""

import asyncio
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
import logging

from ..core.autonomous_systems import (
    AgentId, GoalId, ActionId, AgentGoal, AgentAction,
    RiskScore, ConfidenceScore, AutonomousAgentError,
    AgentType, ActionType
)
from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.errors import ValidationError


class SafetyLevel(Enum):
    """Safety constraint levels."""
    MINIMAL = "minimal"          # Basic safety checks
    STANDARD = "standard"        # Standard safety validation
    ENHANCED = "enhanced"        # Enhanced safety with extra checks
    MAXIMUM = "maximum"          # Maximum safety, most restrictive


class RiskCategory(Enum):
    """Categories of operational risks."""
    RESOURCE = "resource"        # Resource exhaustion or conflict
    SECURITY = "security"        # Security vulnerabilities
    PERFORMANCE = "performance"  # Performance degradation
    DATA = "data"               # Data integrity or loss
    SYSTEM = "system"           # System stability
    COMPLIANCE = "compliance"    # Policy or regulatory compliance


@dataclass
class SafetyPolicy:
    """Safety policy configuration."""
    policy_id: str
    name: str
    description: str
    safety_level: SafetyLevel
    max_risk_tolerance: RiskScore
    forbidden_actions: Set[ActionType]
    resource_limits: Dict[str, float]
    require_human_approval: Set[ActionType]
    enabled: bool = True
    
    def is_action_forbidden(self, action_type: ActionType) -> bool:
        """Check if action type is forbidden by policy."""
        return action_type in self.forbidden_actions
    
    def requires_human_approval(self, action_type: ActionType) -> bool:
        """Check if action requires human approval."""
        return action_type in self.require_human_approval


@dataclass
class SafetyViolation:
    """Safety constraint violation details."""
    violation_id: str
    timestamp: datetime
    agent_id: AgentId
    violation_type: RiskCategory
    severity: str  # low, medium, high, critical
    description: str
    context: Dict[str, Any]
    resolved: bool = False
    resolution: Optional[str] = None


class SafetyValidator:
    """Comprehensive safety validation system for autonomous agents."""
    
    def __init__(self, default_safety_level: SafetyLevel = SafetyLevel.STANDARD):
        self.default_safety_level = default_safety_level
        self.safety_policies: Dict[str, SafetyPolicy] = {}
        self.violations: List[SafetyViolation] = []
        self.safety_metrics = {
            "total_validations": 0,
            "violations_detected": 0,
            "violations_by_category": {},
            "average_risk_score": 0.0
        }
        self._initialize_default_policies()
    
    async def validate_goal_safety(self, goal: AgentGoal) -> Either[AutonomousAgentError, None]:
        """Validate goal safety constraints."""
        self.safety_metrics["total_validations"] += 1
        
        try:
            # Check goal description for forbidden patterns
            forbidden_patterns = [
                "delete all", "rm -rf", "drop database", "shutdown",
                "disable security", "bypass", "hack", "exploit"
            ]
            
            goal_desc_lower = goal.description.lower()
            for pattern in forbidden_patterns:
                if pattern in goal_desc_lower:
                    self._record_violation(
                        agent_id=AgentId("system"),
                        violation_type=RiskCategory.SECURITY,
                        severity="high",
                        description=f"Goal contains forbidden pattern: {pattern}"
                    )
                    return Either.left(AutonomousAgentError.safety_constraint_violated(
                        f"Goal contains forbidden operation: {pattern}"
                    ))
            
            # Validate resource requirements
            total_resources = sum(goal.resource_requirements.values())
            if total_resources > 100:  # Assuming 100% is max
                self._record_violation(
                    agent_id=AgentId("system"),
                    violation_type=RiskCategory.RESOURCE,
                    severity="medium",
                    description=f"Goal requires excessive resources: {total_resources}%"
                )
                return Either.left(AutonomousAgentError.resource_limit_exceeded(
                    "total_resources", total_resources, 100
                ))
            
            # Check deadline feasibility
            if goal.deadline:
                time_until_deadline = goal.deadline - datetime.now(UTC)
                if time_until_deadline < timedelta(minutes=5):
                    return Either.left(AutonomousAgentError.safety_constraint_violated(
                        "Goal deadline is too aggressive"
                    ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(AutonomousAgentError.unexpected_error(f"Goal validation failed: {str(e)}"))
    
    async def validate_action_safety(self, agent, action: AgentAction) -> Either[AutonomousAgentError, None]:
        """Validate action safety constraints."""
        self.safety_metrics["total_validations"] += 1
        
        try:
            # Get applicable policy
            policy = self._get_agent_policy(agent.state.configuration.agent_type)
            
            # Check if action is forbidden
            if policy.is_action_forbidden(action.action_type):
                self._record_violation(
                    agent_id=agent.state.agent_id,
                    violation_type=RiskCategory.COMPLIANCE,
                    severity="critical",
                    description=f"Forbidden action attempted: {action.action_type.value}"
                )
                return Either.left(AutonomousAgentError.safety_constraint_violated(
                    f"Action {action.action_type.value} is forbidden by safety policy"
                ))
            
            # Calculate risk score
            risk_score = action.get_risk_score()
            if risk_score > policy.max_risk_tolerance:
                self._record_violation(
                    agent_id=agent.state.agent_id,
                    violation_type=RiskCategory.SECURITY,
                    severity="high",
                    description=f"Action risk {risk_score} exceeds tolerance {policy.max_risk_tolerance}"
                )
                return Either.left(AutonomousAgentError.safety_constraint_violated(
                    f"Action risk score {risk_score} exceeds safety threshold"
                ))
            
            # Check resource limits
            for resource, cost in action.resource_cost.items():
                limit = policy.resource_limits.get(resource, float('inf'))
                if cost > limit:
                    self._record_violation(
                        agent_id=agent.state.agent_id,
                        violation_type=RiskCategory.RESOURCE,
                        severity="medium",
                        description=f"Resource {resource} usage {cost} exceeds limit {limit}"
                    )
                    return Either.left(AutonomousAgentError.resource_limit_exceeded(
                        resource, cost, limit
                    ))
            
            # Mark if human approval required
            if policy.requires_human_approval(action.action_type):
                action.human_approval_required = True
            
            action.safety_validated = True
            return Either.right(None)
            
        except Exception as e:
            return Either.left(AutonomousAgentError.unexpected_error(f"Action validation failed: {str(e)}"))
    
    def assess_system_risk(self) -> Dict[str, Any]:
        """Assess overall system risk level."""
        recent_violations = [v for v in self.violations[-100:] if not v.resolved]
        
        risk_assessment = {
            "overall_risk": self._calculate_overall_risk(recent_violations),
            "risk_by_category": {},
            "critical_violations": [],
            "recommendations": []
        }
        
        # Risk by category
        for category in RiskCategory:
            category_violations = [v for v in recent_violations if v.violation_type == category]
            if category_violations:
                risk_assessment["risk_by_category"][category.value] = len(category_violations)
        
        # Critical violations
        critical = [v for v in recent_violations if v.severity == "critical"]
        risk_assessment["critical_violations"] = [
            {
                "id": v.violation_id,
                "type": v.violation_type.value,
                "description": v.description,
                "timestamp": v.timestamp.isoformat()
            }
            for v in critical
        ]
        
        # Recommendations
        if len(recent_violations) > 10:
            risk_assessment["recommendations"].append(
                "High violation rate detected. Consider increasing safety constraints."
            )
        
        if risk_assessment["overall_risk"] > 0.7:
            risk_assessment["recommendations"].append(
                "System risk level is high. Manual review recommended."
            )
        
        return risk_assessment
    
    def _initialize_default_policies(self) -> None:
        """Initialize default safety policies."""
        # Standard policy
        standard_policy = SafetyPolicy(
            policy_id="standard",
            name="Standard Safety Policy",
            description="Default safety policy for most agents",
            safety_level=SafetyLevel.STANDARD,
            max_risk_tolerance=RiskScore(0.7),
            forbidden_actions={
                ActionType.DELETE_ALL_DATA,
                ActionType.DISABLE_SECURITY,
                ActionType.MODIFY_SYSTEM_CONFIG
            },
            resource_limits={
                "cpu": 80.0,
                "memory": 80.0,
                "disk": 90.0,
                "network": 100.0
            },
            require_human_approval={
                ActionType.EXECUTE_SYSTEM_COMMAND,
                ActionType.MODIFY_CRITICAL_CONFIG
            }
        )
        self.safety_policies["standard"] = standard_policy
        
        # Enhanced policy
        enhanced_policy = SafetyPolicy(
            policy_id="enhanced",
            name="Enhanced Safety Policy",
            description="Enhanced safety for critical operations",
            safety_level=SafetyLevel.ENHANCED,
            max_risk_tolerance=RiskScore(0.5),
            forbidden_actions={
                ActionType.DELETE_ALL_DATA,
                ActionType.DISABLE_SECURITY,
                ActionType.MODIFY_SYSTEM_CONFIG,
                ActionType.EXECUTE_SYSTEM_COMMAND
            },
            resource_limits={
                "cpu": 60.0,
                "memory": 60.0,
                "disk": 70.0,
                "network": 80.0
            },
            require_human_approval={
                ActionType.MODIFY_CRITICAL_CONFIG,
                ActionType.ACCESS_SENSITIVE_DATA,
                ActionType.COORDINATE_AGENTS
            }
        )
        self.safety_policies["enhanced"] = enhanced_policy
    
    def _get_agent_policy(self, agent_type: AgentType) -> SafetyPolicy:
        """Get appropriate safety policy for agent type."""
        # Critical agent types get enhanced safety
        if agent_type in [AgentType.HEALER, AgentType.RESOURCE_MANAGER]:
            return self.safety_policies.get("enhanced", self.safety_policies["standard"])
        
        return self.safety_policies["standard"]
    
    def _record_violation(self, agent_id: AgentId, violation_type: RiskCategory,
                         severity: str, description: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record a safety violation."""
        violation = SafetyViolation(
            violation_id=f"viol_{datetime.now(UTC).timestamp()}",
            timestamp=datetime.now(UTC),
            agent_id=agent_id,
            violation_type=violation_type,
            severity=severity,
            description=description,
            context=context or {}
        )
        
        self.violations.append(violation)
        self.safety_metrics["violations_detected"] += 1
        
        # Update category metrics
        category_key = violation_type.value
        if category_key not in self.safety_metrics["violations_by_category"]:
            self.safety_metrics["violations_by_category"][category_key] = 0
        self.safety_metrics["violations_by_category"][category_key] += 1
        
        # Log critical violations
        if severity == "critical":
            logging.error(f"CRITICAL SAFETY VIOLATION: {description}")
    
    def _calculate_overall_risk(self, violations: List[SafetyViolation]) -> float:
        """Calculate overall system risk score."""
        if not violations:
            return 0.0
        
        severity_weights = {
            "low": 0.2,
            "medium": 0.4,
            "high": 0.7,
            "critical": 1.0
        }
        
        total_weight = sum(severity_weights.get(v.severity, 0.5) for v in violations)
        max_possible = len(violations) * 1.0
        
        return min(1.0, total_weight / max_possible) if max_possible > 0 else 0.0