"""
Autonomous systems type definitions and data structures for self-managing automation agents.

This module provides comprehensive type safety and data structures for autonomous agents,
goal management, learning systems, and decision-making with enterprise-grade security
and performance optimization.

Security: Complete validation and safety constraints for autonomous operations
Performance: <5s agent initialization, <10s decision-making, <60s autonomous cycles
Type Safety: Complete branded types and contract validation
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Callable, Type, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
import asyncio
import uuid
import hashlib

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.errors import ValidationError


# Branded types for autonomous systems
class AgentId(str):
    """Unique identifier for autonomous agents."""
    pass

class GoalId(str):
    """Unique identifier for agent goals."""
    pass

class ActionId(str):
    """Unique identifier for agent actions."""
    pass

class ExperienceId(str):
    """Unique identifier for learning experiences."""
    pass

class ConfidenceScore(float):
    """Confidence score between 0.0 and 1.0."""
    pass

class RiskScore(float):
    """Risk score between 0.0 and 1.0."""
    pass

class PerformanceMetric(float):
    """Performance metric value."""
    pass


class AgentType(Enum):
    """Types of autonomous agents with specific capabilities."""
    GENERAL = "general"              # General-purpose automation agent
    OPTIMIZER = "optimizer"          # Performance optimization specialist
    MONITOR = "monitor"              # System monitoring and alerting
    LEARNER = "learner"              # Learning and pattern recognition
    COORDINATOR = "coordinator"      # Multi-agent coordination
    HEALER = "healer"                # Self-healing and recovery
    PLANNER = "planner"              # Predictive planning and scheduling
    RESOURCE_MANAGER = "resource_manager"  # Resource allocation and optimization


class AutonomyLevel(Enum):
    """Levels of agent autonomy and human oversight."""
    MANUAL = "manual"                # Manual control, no autonomous actions
    SUPERVISED = "supervised"        # Autonomous with human oversight required
    AUTONOMOUS = "autonomous"        # Full autonomy within safety constraints
    FULL = "full"                    # Maximum autonomy with minimal constraints


class GoalPriority(Enum):
    """Priority levels for agent goals."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AgentStatus(Enum):
    """Current operational status of autonomous agents."""
    CREATED = "created"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    LEARNING = "learning"
    OPTIMIZING = "optimizing"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


class ActionType(Enum):
    """Types of actions autonomous agents can perform."""
    ANALYZE_PERFORMANCE = "analyze_performance"
    OPTIMIZE_WORKFLOW = "optimize_workflow"
    MONITOR_SYSTEM = "monitor_system"
    EXECUTE_AUTOMATION = "execute_automation"
    LEARN_PATTERN = "learn_pattern"
    COORDINATE_AGENTS = "coordinate_agents"
    HEAL_SYSTEM = "heal_system"
    PLAN_SCHEDULE = "plan_schedule"
    ALLOCATE_RESOURCES = "allocate_resources"
    UPDATE_CONFIGURATION = "update_configuration"
    DELETE_ALL_DATA = "delete_all_data"
    DISABLE_SECURITY = "disable_security"
    MODIFY_SYSTEM_CONFIG = "modify_system_config"
    EXECUTE_SYSTEM_COMMAND = "execute_system_command"
    MODIFY_CRITICAL_CONFIG = "modify_critical_config"
    ACCESS_SENSITIVE_DATA = "access_sensitive_data"


def create_agent_id() -> AgentId:
    """Create unique agent identifier."""
    return AgentId(f"agent_{datetime.now(UTC).timestamp()}_{uuid.uuid4().hex[:8]}")


def create_goal_id() -> GoalId:
    """Create unique goal identifier."""
    return GoalId(f"goal_{uuid.uuid4().hex}")


def create_action_id() -> ActionId:
    """Create unique action identifier."""
    return ActionId(f"action_{uuid.uuid4().hex}")


def create_experience_id() -> ExperienceId:
    """Create unique experience identifier."""
    return ExperienceId(f"exp_{uuid.uuid4().hex}")


@dataclass(frozen=True)
class AgentGoal:
    """Comprehensive goal specification for autonomous agents."""
    goal_id: GoalId
    description: str
    priority: GoalPriority
    target_metrics: Dict[str, PerformanceMetric]
    success_criteria: List[str]
    constraints: Dict[str, Any]
    deadline: Optional[datetime] = None
    dependencies: List[GoalId] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    estimated_duration: Optional[timedelta] = None
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    @require(lambda self: len(self.goal_id) > 0)
    @require(lambda self: len(self.description) > 0)
    @require(lambda self: len(self.target_metrics) > 0)
    @require(lambda self: len(self.success_criteria) > 0)
    def __post_init__(self):
        pass
    
    def is_overdue(self) -> bool:
        """Check if goal is past its deadline."""
        return self.deadline is not None and datetime.now(UTC) > self.deadline
    
    def get_urgency_score(self) -> ConfidenceScore:
        """Calculate urgency score based on priority and deadline proximity."""
        priority_weights = {
            GoalPriority.LOW: 0.2,
            GoalPriority.MEDIUM: 0.4,
            GoalPriority.HIGH: 0.6,
            GoalPriority.CRITICAL: 0.8,
            GoalPriority.EMERGENCY: 1.0
        }
        
        base_score = priority_weights[self.priority]
        
        # Adjust for deadline urgency
        if self.deadline:
            time_remaining = (self.deadline - datetime.now(UTC)).total_seconds()
            if time_remaining <= 0:
                return ConfidenceScore(1.0)  # Overdue goals get maximum urgency
            elif time_remaining < 3600:  # Less than 1 hour
                return ConfidenceScore(min(1.0, base_score * 1.5))
            elif time_remaining < 86400:  # Less than 1 day
                return ConfidenceScore(min(1.0, base_score * 1.2))
        
        return ConfidenceScore(base_score)
    
    def estimate_completion_time(self) -> Optional[datetime]:
        """Estimate when goal will be completed."""
        if self.estimated_duration:
            return datetime.now(UTC) + self.estimated_duration
        return None
    
    def is_achievable(self, available_resources: Dict[str, float]) -> bool:
        """Check if goal is achievable with available resources."""
        for resource, required in self.resource_requirements.items():
            if available_resources.get(resource, 0.0) < required:
                return False
        return True


@dataclass(frozen=True)
class AgentAction:
    """Comprehensive action specification for autonomous agents."""
    action_id: ActionId
    agent_id: AgentId
    action_type: ActionType
    parameters: Dict[str, Any]
    goal_id: Optional[GoalId] = None
    rationale: Optional[str] = None
    confidence: ConfidenceScore = ConfidenceScore(0.5)
    estimated_impact: PerformanceMetric = PerformanceMetric(0.0)
    estimated_duration: Optional[timedelta] = None
    resource_cost: Dict[str, float] = field(default_factory=dict)
    prerequisites: List[ActionId] = field(default_factory=list)
    safety_validated: bool = False
    human_approval_required: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    executed_at: Optional[datetime] = None
    
    @require(lambda self: len(self.action_id) > 0)
    @require(lambda self: len(self.agent_id) > 0)
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: self.estimated_impact >= 0.0)
    def __post_init__(self):
        pass
    
    def is_high_confidence(self, threshold: ConfidenceScore = ConfidenceScore(0.8)) -> bool:
        """Check if action has high confidence."""
        return self.confidence >= threshold
    
    def get_risk_score(self) -> RiskScore:
        """Calculate comprehensive risk score for the action."""
        base_risk = 1.0 - self.confidence
        
        # Adjust for estimated impact magnitude
        if self.estimated_impact > 0.8:
            base_risk *= 1.5  # High impact increases risk
        elif self.estimated_impact < 0.2:
            base_risk *= 0.5  # Low impact reduces risk
        
        # Adjust for safety validation
        if not self.safety_validated:
            base_risk *= 2.0  # Unvalidated actions are riskier
        
        # Adjust for human approval requirement
        if self.human_approval_required:
            base_risk *= 0.7  # Human oversight reduces risk
        
        return RiskScore(min(1.0, base_risk))
    
    def can_execute_now(self, completed_actions: Set[ActionId]) -> bool:
        """Check if all prerequisites are satisfied."""
        return all(prereq in completed_actions for prereq in self.prerequisites)
    
    def estimate_total_cost(self) -> float:
        """Calculate total estimated cost of action execution."""
        return sum(self.resource_cost.values())


@dataclass(frozen=True)
class LearningExperience:
    """Comprehensive learning experience for agent improvement."""
    experience_id: ExperienceId
    agent_id: AgentId
    context: Dict[str, Any]
    action_taken: AgentAction
    outcome: Dict[str, Any]
    success: bool
    learning_value: ConfidenceScore
    performance_impact: PerformanceMetric
    unexpected_results: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @require(lambda self: len(self.experience_id) > 0)
    @require(lambda self: len(self.agent_id) > 0)
    @require(lambda self: 0.0 <= self.learning_value <= 1.0)
    def __post_init__(self):
        pass
    
    def extract_patterns(self) -> Dict[str, Any]:
        """Extract learnable patterns from experience."""
        patterns = {
            "success_indicators": [],
            "failure_indicators": [],
            "context_factors": [],
            "optimal_parameters": {},
            "performance_correlations": {}
        }
        
        # Extract success/failure indicators
        if self.success:
            patterns["success_indicators"] = [
                key for key, value in self.context.items() 
                if isinstance(value, (int, float, bool, str))
            ]
            patterns["optimal_parameters"] = self.action_taken.parameters.copy()
        else:
            patterns["failure_indicators"] = [
                key for key, value in self.context.items() 
                if isinstance(value, (int, float, bool, str))
            ]
        
        # Extract context factors that influenced outcome
        patterns["context_factors"] = [
            key for key, value in self.context.items() 
            if isinstance(value, (int, float, bool, str))
        ]
        
        # Performance correlations
        patterns["performance_correlations"] = {
            "confidence_vs_success": self.action_taken.confidence if self.success else -self.action_taken.confidence,
            "impact_vs_performance": self.performance_impact,
            "cost_efficiency": self.performance_impact / max(self.action_taken.estimate_total_cost(), 0.1)
        }
        
        return patterns
    
    def get_learning_weight(self) -> float:
        """Calculate learning weight based on experience characteristics."""
        weight = self.learning_value
        
        # Increase weight for unexpected results
        if self.unexpected_results:
            weight *= 1.5
        
        # Increase weight for high-impact outcomes
        if abs(self.performance_impact) > 0.5:
            weight *= 1.3
        
        # Increase weight for failures (they teach more)
        if not self.success:
            weight *= 1.2
        
        return min(1.0, weight)


@dataclass
class AgentConfiguration:
    """Comprehensive configuration for autonomous agents."""
    agent_type: AgentType
    autonomy_level: AutonomyLevel
    max_concurrent_actions: int = 3
    decision_threshold: ConfidenceScore = ConfidenceScore(0.7)
    risk_tolerance: RiskScore = RiskScore(0.3)
    learning_rate: float = 0.1
    optimization_frequency: timedelta = timedelta(hours=1)
    safety_constraints: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, float] = field(default_factory=dict)
    human_approval_required: bool = False
    monitoring_interval: timedelta = timedelta(minutes=5)
    performance_targets: Dict[str, PerformanceMetric] = field(default_factory=dict)
    
    @require(lambda self: 0 < self.max_concurrent_actions <= 10)
    @require(lambda self: 0.0 <= self.decision_threshold <= 1.0)
    @require(lambda self: 0.0 <= self.risk_tolerance <= 1.0)
    @require(lambda self: 0.0 < self.learning_rate <= 1.0)
    def __post_init__(self):
        pass
    
    def is_action_within_limits(self, action: AgentAction) -> bool:
        """Check if action is within configured limits."""
        # Check risk tolerance
        if action.get_risk_score() > self.risk_tolerance:
            return False
        
        # Check confidence threshold
        if action.confidence < self.decision_threshold:
            return False
        
        # Check resource limits
        for resource, usage in action.resource_cost.items():
            limit = self.resource_limits.get(resource, float('inf'))
            if usage > limit:
                return False
        
        return True
    
    def should_request_human_approval(self, action: AgentAction) -> bool:
        """Determine if human approval is required for action."""
        if self.human_approval_required:
            return True
        
        # High-risk actions require approval
        if action.get_risk_score() > 0.8:
            return True
        
        # High-impact actions require approval
        if action.estimated_impact > 0.9:
            return True
        
        # Manual autonomy level requires approval
        if self.autonomy_level == AutonomyLevel.MANUAL:
            return True
        
        return False


class AutonomousAgentError(ValidationError):
    """Errors specific to autonomous agent operations."""
    
    @classmethod
    def agent_not_found(cls, agent_id: AgentId) -> 'AutonomousAgentError':
        return cls("agent_id", agent_id, f"Agent {agent_id} not found")
    
    @classmethod
    def invalid_goal_constraints(cls) -> 'AutonomousAgentError':
        return cls("goal_constraints", None, "Goal constraints are invalid or conflicting")
    
    @classmethod
    def conflicting_goals(cls, conflicts: List[str]) -> 'AutonomousAgentError':
        return cls("goal_conflicts", conflicts, f"Goals conflict: {', '.join(conflicts)}")
    
    @classmethod
    def agent_not_active(cls) -> 'AutonomousAgentError':
        return cls("agent_status", "not_active", "Agent is not in active state")
    
    @classmethod
    def action_too_risky(cls, risk_score: RiskScore, max_risk: RiskScore) -> 'AutonomousAgentError':
        return cls("risk_score", risk_score, f"Action risk {risk_score} exceeds limit {max_risk}")
    
    @classmethod
    def resource_limit_exceeded(cls, resource: str, usage: float, limit: float) -> 'AutonomousAgentError':
        return cls("resource_usage", usage, f"Resource {resource} usage {usage} exceeds limit {limit}")
    
    @classmethod
    def initialization_failed(cls, reason: str) -> 'AutonomousAgentError':
        return cls("agent_initialization", None, f"Agent initialization failed: {reason}")
    
    @classmethod
    def execution_cycle_failed(cls, reason: str) -> 'AutonomousAgentError':
        return cls("execution_cycle", None, f"Execution cycle failed: {reason}")
    
    @classmethod
    def action_execution_failed(cls, reason: str) -> 'AutonomousAgentError':
        return cls("action_execution", None, f"Action execution failed: {reason}")
    
    @classmethod
    def dangerous_goal_detected(cls) -> 'AutonomousAgentError':
        return cls("goal_safety", None, "Goal contains potentially dangerous operations")
    
    @classmethod
    def manual_mode_action_blocked(cls) -> 'AutonomousAgentError':
        return cls("autonomy_level", "manual", "Autonomous actions blocked in manual mode")
    
    @classmethod
    def agent_creation_failed(cls, reason: str) -> 'AutonomousAgentError':
        return cls("agent_creation", None, f"Agent creation failed: {reason}")
    
    @classmethod
    def recovery_in_progress(cls) -> 'AutonomousAgentError':
        return cls("recovery_status", "in_progress", "Recovery operation already in progress")
    
    @classmethod
    def recovery_planning_failed(cls, reason: str) -> 'AutonomousAgentError':
        return cls("recovery_planning", None, f"Recovery planning failed: {reason}")
    
    @classmethod
    def recovery_execution_failed(cls, reason: str) -> 'AutonomousAgentError':
        return cls("recovery_execution", None, f"Recovery execution failed: {reason}")
    
    @classmethod
    def diagnostic_failure(cls, reason: str) -> 'AutonomousAgentError':
        return cls("diagnostic", None, f"Error diagnosis failed: {reason}")
    
    @classmethod
    def execution_start_failed(cls, reason: str) -> 'AutonomousAgentError':
        return cls("execution_start", None, f"Failed to start execution: {reason}")
    
    @classmethod
    def unexpected_error(cls, reason: str) -> 'AutonomousAgentError':
        return cls("unexpected_error", None, f"Unexpected error: {reason}")


# Default agent configurations for different agent types
DEFAULT_AGENT_CONFIGS = {
    AgentType.GENERAL: AgentConfiguration(
        agent_type=AgentType.GENERAL,
        autonomy_level=AutonomyLevel.SUPERVISED,
        decision_threshold=ConfidenceScore(0.7),
        risk_tolerance=RiskScore(0.3),
        learning_rate=0.1,
        resource_limits={"cpu": 50.0, "memory": 1024.0, "actions_per_minute": 10.0}
    ),
    AgentType.OPTIMIZER: AgentConfiguration(
        agent_type=AgentType.OPTIMIZER,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        decision_threshold=ConfidenceScore(0.8),
        risk_tolerance=RiskScore(0.4),
        learning_rate=0.15,
        optimization_frequency=timedelta(minutes=30),
        resource_limits={"cpu": 70.0, "memory": 2048.0, "actions_per_minute": 15.0}
    ),
    AgentType.MONITOR: AgentConfiguration(
        agent_type=AgentType.MONITOR,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        decision_threshold=ConfidenceScore(0.9),
        risk_tolerance=RiskScore(0.2),
        learning_rate=0.05,
        monitoring_interval=timedelta(minutes=1),
        resource_limits={"cpu": 30.0, "memory": 512.0, "actions_per_minute": 20.0}
    ),
    AgentType.LEARNER: AgentConfiguration(
        agent_type=AgentType.LEARNER,
        autonomy_level=AutonomyLevel.SUPERVISED,
        decision_threshold=ConfidenceScore(0.6),
        risk_tolerance=RiskScore(0.5),
        learning_rate=0.3,
        resource_limits={"cpu": 60.0, "memory": 1536.0, "actions_per_minute": 8.0}
    ),
    AgentType.HEALER: AgentConfiguration(
        agent_type=AgentType.HEALER,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        decision_threshold=ConfidenceScore(0.8),
        risk_tolerance=RiskScore(0.6),
        learning_rate=0.2,
        human_approval_required=False,  # Self-healing should be automatic
        resource_limits={"cpu": 80.0, "memory": 2048.0, "actions_per_minute": 12.0}
    )
}


def get_default_config(agent_type: AgentType) -> AgentConfiguration:
    """Get default configuration for agent type."""
    return DEFAULT_AGENT_CONFIGS.get(agent_type, DEFAULT_AGENT_CONFIGS[AgentType.GENERAL])