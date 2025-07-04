"""
Ecosystem orchestration type definitions and architecture.

This module provides comprehensive type definitions for orchestrating and managing
the complete 49-tool enterprise automation ecosystem with intelligent coordination,
performance optimization, and strategic automation planning.

Security: All ecosystem operations include safety validation and authorization checks
Performance: <2s orchestration planning, <5s workflow execution, <10s optimization cycles
Enterprise: Complete audit integration, resource optimization, and scalable coordination
"""

from typing import Dict, List, Optional, Set, Any, Union, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
import uuid
from decimal import Decimal

from .either import Either
from .contracts import require, ensure


# Branded types for ecosystem orchestration
class OrchestrationId(str):
    """Unique identifier for orchestration sessions."""
    pass

class WorkflowId(str):
    """Unique identifier for workflow definitions."""
    pass

class ToolId(str):
    """Unique identifier for registered tools."""
    pass

class OptimizationScore(float):
    """Performance optimization score (0.0 to 1.0)."""
    def __new__(cls, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"OptimizationScore must be between 0.0 and 1.0, got {value}")
        return super().__new__(cls, value)

class EfficiencyMetric(float):
    """System efficiency metric (0.0 to 1.0)."""
    def __new__(cls, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"EfficiencyMetric must be between 0.0 and 1.0, got {value}")
        return super().__new__(cls, value)

class ResourceUtilization(float):
    """Resource utilization percentage (0.0 to 100.0)."""
    def __new__(cls, value: float):
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"ResourceUtilization must be between 0.0 and 100.0, got {value}")
        return super().__new__(cls, value)


class OrchestrationMode(Enum):
    """Orchestration execution modes."""
    SEQUENTIAL = "sequential"      # Execute tools in sequence
    PARALLEL = "parallel"          # Execute compatible tools in parallel
    ADAPTIVE = "adaptive"          # Adapt execution based on performance
    PIPELINE = "pipeline"          # Pipeline execution with data flow
    INTELLIGENT = "intelligent"    # AI-driven intelligent orchestration

class OptimizationTarget(Enum):
    """System optimization targets."""
    PERFORMANCE = "performance"    # Optimize for speed and throughput
    EFFICIENCY = "efficiency"      # Optimize for resource efficiency
    RELIABILITY = "reliability"    # Optimize for stability and error reduction
    COST = "cost"                 # Optimize for cost minimization
    USER_EXPERIENCE = "user_experience"  # Optimize for user satisfaction
    BALANCED = "balanced"         # Balanced optimization across all metrics

class ResourceStrategy(Enum):
    """Resource allocation strategies."""
    CONSERVATIVE = "conservative"  # Conservative resource usage
    BALANCED = "balanced"         # Balanced resource allocation
    AGGRESSIVE = "aggressive"     # Aggressive resource utilization
    UNLIMITED = "unlimited"       # No resource limits (enterprise only)

class MonitoringLevel(Enum):
    """System monitoring levels."""
    MINIMAL = "minimal"           # Basic health monitoring
    STANDARD = "standard"         # Standard performance monitoring
    DETAILED = "detailed"         # Detailed metrics and analytics
    COMPREHENSIVE = "comprehensive"  # Full ecosystem monitoring

class CacheStrategy(Enum):
    """Caching strategies for performance optimization."""
    NONE = "none"                 # No caching
    BASIC = "basic"               # Basic result caching
    INTELLIGENT = "intelligent"   # AI-driven cache optimization
    PREDICTIVE = "predictive"     # Predictive pre-caching

class ErrorHandling(Enum):
    """Error handling strategies."""
    FAIL_FAST = "fail_fast"       # Stop on first error
    RESILIENT = "resilient"       # Continue with error isolation
    RECOVERY = "recovery"         # Attempt automatic recovery
    ADAPTIVE = "adaptive"         # Learn from errors and adapt


@dataclass(frozen=True)
class ToolCapability:
    """Capability definition for ecosystem tools."""
    tool_id: ToolId
    name: str
    category: str
    operations: List[str]
    input_types: List[str]
    output_types: List[str]
    dependencies: Set[ToolId]
    resource_requirements: Dict[str, float]
    performance_metrics: Dict[str, float]
    compatibility_matrix: Dict[ToolId, bool]
    
    @require(lambda self: len(self.tool_id) > 0)
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: len(self.operations) > 0)
    def __post_init__(self):
        pass

    def is_compatible_with(self, other_tool: ToolId) -> bool:
        """Check compatibility with another tool."""
        return self.compatibility_matrix.get(other_tool, True)
    
    def get_resource_score(self) -> float:
        """Calculate resource usage score."""
        return sum(self.resource_requirements.values()) / len(self.resource_requirements)
    
    def can_run_parallel_with(self, other: 'ToolCapability') -> bool:
        """Check if tool can run in parallel with another tool."""
        # Check resource conflicts
        for resource, usage in self.resource_requirements.items():
            other_usage = other.resource_requirements.get(resource, 0)
            if usage + other_usage > 100.0:  # Exceeds 100% utilization
                return False
        
        # Check dependency conflicts
        if other.tool_id in self.dependencies or self.tool_id in other.dependencies:
            return False
        
        return self.is_compatible_with(other.tool_id)


@dataclass(frozen=True)
class WorkflowStep:
    """Individual step in an orchestrated workflow."""
    step_id: str
    tool_id: ToolId
    operation: str
    parameters: Dict[str, Any]
    input_mapping: Dict[str, str]
    output_mapping: Dict[str, str]
    conditions: Dict[str, Any]
    timeout: timedelta
    retry_count: int = 3
    parallel_group: Optional[str] = None
    
    @require(lambda self: len(self.step_id) > 0)
    @require(lambda self: len(self.operation) > 0)
    @require(lambda self: self.retry_count >= 0)
    def __post_init__(self):
        pass

    def can_execute_parallel_with(self, other: 'WorkflowStep') -> bool:
        """Check if step can execute in parallel with another step."""
        return (self.parallel_group is not None and 
                self.parallel_group == other.parallel_group)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition for ecosystem orchestration."""
    workflow_id: WorkflowId
    name: str
    description: str
    steps: List[WorkflowStep]
    global_timeout: timedelta
    optimization_target: OptimizationTarget
    resource_limits: Dict[str, float]
    success_criteria: Dict[str, Any]
    failure_handling: ErrorHandling
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @require(lambda self: len(self.workflow_id) > 0)
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: len(self.steps) > 0)
    def __post_init__(self):
        pass

    def get_parallel_groups(self) -> Dict[str, List[WorkflowStep]]:
        """Get steps grouped by parallel execution groups."""
        groups = {}
        for step in self.steps:
            if step.parallel_group:
                if step.parallel_group not in groups:
                    groups[step.parallel_group] = []
                groups[step.parallel_group].append(step)
        return groups
    
    def estimate_execution_time(self, tool_registry: Dict[ToolId, ToolCapability]) -> timedelta:
        """Estimate workflow execution time."""
        sequential_time = 0.0
        parallel_groups = self.get_parallel_groups()
        
        for step in self.steps:
            if step.parallel_group and step.parallel_group in parallel_groups:
                # For parallel steps, take the maximum time in the group
                group_steps = parallel_groups[step.parallel_group]
                group_time = max(step.timeout.total_seconds() for step in group_steps)
                sequential_time += group_time
                # Remove processed group to avoid double counting
                del parallel_groups[step.parallel_group]
            elif not step.parallel_group:
                # Sequential step
                sequential_time += step.timeout.total_seconds()
        
        return timedelta(seconds=sequential_time)


@dataclass
class OrchestrationContext:
    """Runtime context for ecosystem orchestration."""
    orchestration_id: OrchestrationId
    workflow_id: WorkflowId
    execution_mode: OrchestrationMode
    optimization_target: OptimizationTarget
    resource_strategy: ResourceStrategy
    monitoring_level: MonitoringLevel
    cache_strategy: CacheStrategy
    error_handling: ErrorHandling
    available_resources: Dict[str, float]
    performance_requirements: Dict[str, float]
    security_constraints: Dict[str, Any]
    user_preferences: Dict[str, Any]
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @require(lambda self: len(self.orchestration_id) > 0)
    @require(lambda self: len(self.workflow_id) > 0)
    def __post_init__(self):
        pass

    def has_sufficient_resources(self, required_resources: Dict[str, float]) -> bool:
        """Check if sufficient resources are available."""
        for resource, required in required_resources.items():
            available = self.available_resources.get(resource, 0.0)
            if available < required:
                return False
        return True
    
    def get_execution_priority(self, step: WorkflowStep) -> float:
        """Calculate execution priority for a workflow step."""
        # Base priority from optimization target
        if self.optimization_target == OptimizationTarget.PERFORMANCE:
            return 1.0 - step.timeout.total_seconds() / 3600.0  # Favor faster steps
        elif self.optimization_target == OptimizationTarget.EFFICIENCY:
            # Favor steps with lower resource requirements
            return 1.0 - sum(step.parameters.get('resource_cost', {}).values()) / 100.0
        elif self.optimization_target == OptimizationTarget.RELIABILITY:
            return 1.0 - (step.retry_count / 10.0)  # Favor steps with fewer retries needed
        else:
            return 0.5  # Balanced priority


@dataclass
class OrchestrationResult:
    """Result of ecosystem orchestration execution."""
    orchestration_id: OrchestrationId
    workflow_id: WorkflowId
    success: bool
    steps_executed: int
    steps_failed: int
    execution_time: timedelta
    resource_utilization: Dict[str, ResourceUtilization]
    performance_metrics: Dict[str, EfficiencyMetric]
    optimization_score: OptimizationScore
    error_details: List[Dict[str, Any]]
    step_results: Dict[str, Any]
    recommendations: List[str]
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @require(lambda self: len(self.orchestration_id) > 0)
    @require(lambda self: self.steps_executed >= 0)
    @require(lambda self: self.steps_failed >= 0)
    def __post_init__(self):
        pass

    def get_success_rate(self) -> float:
        """Calculate overall success rate."""
        total_steps = self.steps_executed + self.steps_failed
        if total_steps == 0:
            return 0.0
        return self.steps_executed / total_steps
    
    def get_overall_efficiency(self) -> EfficiencyMetric:
        """Calculate overall system efficiency."""
        if not self.performance_metrics:
            return EfficiencyMetric(0.0)
        
        efficiency_sum = sum(self.performance_metrics.values())
        return EfficiencyMetric(efficiency_sum / len(self.performance_metrics))


@dataclass
class SystemHealth:
    """System-wide health and performance status."""
    timestamp: datetime
    overall_health: EfficiencyMetric
    tool_availability: Dict[ToolId, bool]
    resource_status: Dict[str, ResourceUtilization]
    performance_trends: Dict[str, List[float]]
    active_orchestrations: int
    error_rate: float
    optimization_opportunities: List[str]
    alerts: List[Dict[str, Any]]
    
    def is_healthy(self, threshold: float = 0.8) -> bool:
        """Check if system is in healthy state."""
        return (self.overall_health >= threshold and 
                self.error_rate < 0.05 and
                all(self.tool_availability.values()))
    
    def get_critical_issues(self) -> List[str]:
        """Get list of critical system issues."""
        issues = []
        
        if self.overall_health < 0.5:
            issues.append("Critical: Overall system health below 50%")
        
        if self.error_rate > 0.1:
            issues.append(f"Critical: High error rate ({self.error_rate:.1%})")
        
        unavailable_tools = [tool_id for tool_id, available in self.tool_availability.items() if not available]
        if unavailable_tools:
            issues.append(f"Critical: Tools unavailable: {', '.join(unavailable_tools)}")
        
        overloaded_resources = [
            resource for resource, usage in self.resource_status.items() 
            if usage > 95.0
        ]
        if overloaded_resources:
            issues.append(f"Critical: Resources overloaded: {', '.join(overloaded_resources)}")
        
        return issues


class EcosystemOrchestratorError(Exception):
    """Base exception for ecosystem orchestration errors."""
    
    def __init__(self, message: str, error_code: str = "ORCHESTRATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)
    
    @classmethod
    def workflow_not_found(cls, workflow_id: WorkflowId) -> 'EcosystemOrchestratorError':
        return cls(f"Workflow not found: {workflow_id}", "WORKFLOW_NOT_FOUND")
    
    @classmethod
    def tool_not_available(cls, tool_id: ToolId) -> 'EcosystemOrchestratorError':
        return cls(f"Tool not available: {tool_id}", "TOOL_NOT_AVAILABLE")
    
    @classmethod
    def resource_limit_exceeded(cls, resource: str, required: float, available: float) -> 'EcosystemOrchestratorError':
        return cls(f"Resource limit exceeded for {resource}: required {required}, available {available}", "RESOURCE_LIMIT_EXCEEDED")
    
    @classmethod
    def orchestration_timeout(cls, timeout: timedelta) -> 'EcosystemOrchestratorError':
        return cls(f"Orchestration timed out after {timeout}", "ORCHESTRATION_TIMEOUT")
    
    @classmethod
    def workflow_validation_failed(cls, details: str) -> 'EcosystemOrchestratorError':
        return cls(f"Workflow validation failed: {details}", "WORKFLOW_VALIDATION_FAILED")
    
    @classmethod
    def optimization_failed(cls, details: str) -> 'EcosystemOrchestratorError':
        return cls(f"System optimization failed: {details}", "OPTIMIZATION_FAILED")


def create_orchestration_id() -> OrchestrationId:
    """Create unique orchestration identifier."""
    return OrchestrationId(f"orch_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")

def create_workflow_id() -> WorkflowId:
    """Create unique workflow identifier."""
    return WorkflowId(f"wf_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")

def create_tool_id(tool_name: str) -> ToolId:
    """Create tool identifier from tool name."""
    return ToolId(f"km_{tool_name.lower().replace(' ', '_')}")


# Default configurations
def get_default_orchestration_context(
    workflow_id: WorkflowId,
    execution_mode: OrchestrationMode = OrchestrationMode.INTELLIGENT
) -> OrchestrationContext:
    """Get default orchestration context configuration."""
    return OrchestrationContext(
        orchestration_id=create_orchestration_id(),
        workflow_id=workflow_id,
        execution_mode=execution_mode,
        optimization_target=OptimizationTarget.BALANCED,
        resource_strategy=ResourceStrategy.BALANCED,
        monitoring_level=MonitoringLevel.COMPREHENSIVE,
        cache_strategy=CacheStrategy.INTELLIGENT,
        error_handling=ErrorHandling.RESILIENT,
        available_resources={
            "cpu": 100.0,
            "memory": 100.0,
            "disk": 100.0,
            "network": 100.0,
            "api_calls": 1000.0,
            "concurrent_operations": 50.0
        },
        performance_requirements={
            "response_time": 5.0,
            "throughput": 100.0,
            "reliability": 0.99
        },
        security_constraints={
            "require_authentication": True,
            "audit_all_operations": True,
            "encrypt_sensitive_data": True
        },
        user_preferences={}
    )