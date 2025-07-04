"""
Core ecosystem architecture types and patterns for master orchestration.

This module provides comprehensive type definitions, data structures, and patterns
for orchestrating all 48 tools in the enterprise automation ecosystem.

Security: Enterprise-grade type safety with comprehensive validation.
Performance: Optimized data structures for large-scale orchestration.
Type Safety: Complete branded types and contract validation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
import asyncio
import uuid
import hashlib

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.errors import ValidationError


# Ecosystem orchestration branded types
class WorkflowId(str):
    """Unique identifier for ecosystem workflows."""
    pass

class StepId(str):
    """Unique identifier for workflow steps."""
    pass

class OrchestrationId(str):
    """Unique identifier for orchestration sessions."""
    pass

class CapabilityId(str):
    """Unique identifier for tool capabilities."""
    pass


class ToolCategory(Enum):
    """Categories of tools in the ecosystem."""
    FOUNDATION = "foundation"                    # Core platform tools (TASK_1-20)
    INTELLIGENCE = "intelligence"                # AI and smart automation (TASK_21-23, 40-41)
    CREATION = "creation"                        # Macro creation and editing (TASK_28-31)
    COMMUNICATION = "communication"              # Communication and integration (TASK_32-34)
    VISUAL_MEDIA = "visual_media"                # Visual and media automation (TASK_35-37)
    DATA_MANAGEMENT = "data_management"          # Data and plugin systems (TASK_38-39)
    ENTERPRISE = "enterprise"                    # Enterprise and cloud integration (TASK_43, 46-47)
    AUTONOMOUS = "autonomous"                    # Autonomous and orchestration (TASK_48-49)


class ExecutionMode(Enum):
    """Workflow execution modes."""
    SEQUENTIAL = "sequential"                    # Execute tools one after another
    PARALLEL = "parallel"                        # Execute compatible tools in parallel
    ADAPTIVE = "adaptive"                        # Adapt execution based on performance
    PIPELINE = "pipeline"                        # Pipeline execution with streaming


class OptimizationTarget(Enum):
    """System optimization targets."""
    PERFORMANCE = "performance"                  # Maximize speed and throughput
    EFFICIENCY = "efficiency"                    # Optimize resource utilization
    RELIABILITY = "reliability"                  # Maximize success rates and stability
    COST = "cost"                               # Minimize resource and operational costs
    USER_EXPERIENCE = "user_experience"          # Optimize for user satisfaction


class ResourceType(Enum):
    """System resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    API_CALLS = "api_calls"
    ACTIONS = "actions"
    TIME = "time"


class SecurityLevel(Enum):
    """Security levels for tools and operations."""
    STANDARD = "standard"
    HIGH = "high"
    ENTERPRISE = "enterprise"
    CRITICAL = "critical"


def create_workflow_id() -> WorkflowId:
    """Create unique workflow identifier."""
    return WorkflowId(f"workflow_{datetime.now(UTC).timestamp()}_{uuid.uuid4().hex[:8]}")


def create_step_id() -> StepId:
    """Create unique step identifier."""
    return StepId(f"step_{uuid.uuid4().hex[:8]}")


def create_orchestration_id() -> OrchestrationId:
    """Create unique orchestration identifier."""
    return OrchestrationId(f"orch_{datetime.now(UTC).timestamp()}_{uuid.uuid4().hex[:8]}")


@dataclass(frozen=True)
class ToolDescriptor:
    """Complete descriptor for ecosystem tools."""
    tool_id: str
    tool_name: str
    category: ToolCategory
    capabilities: Set[str]
    dependencies: List[str]
    resource_requirements: Dict[str, float]
    performance_characteristics: Dict[str, float]
    integration_points: List[str]
    security_level: SecurityLevel
    enterprise_ready: bool = False
    ai_enhanced: bool = False
    
    @require(lambda self: len(self.tool_id) > 0)
    @require(lambda self: len(self.tool_name) > 0)
    @require(lambda self: len(self.capabilities) > 0)
    def __post_init__(self):
        pass
    
    def is_compatible_with(self, other: 'ToolDescriptor') -> bool:
        """Check if this tool is compatible with another for parallel execution."""
        # Check for resource conflicts
        for resource in self.resource_requirements:
            if resource in other.resource_requirements:
                combined_usage = (self.resource_requirements[resource] + 
                                other.resource_requirements[resource])
                if combined_usage > 1.0:  # Assuming 1.0 is maximum capacity
                    return False
        
        # Check for dependency conflicts
        if self.tool_id in other.dependencies or other.tool_id in self.dependencies:
            return False
        
        return True
    
    def get_synergy_score(self, other: 'ToolDescriptor') -> float:
        """Calculate synergy score with another tool."""
        synergy_score = 0.0
        
        # Capability complementarity
        shared_capabilities = self.capabilities & other.capabilities
        total_capabilities = self.capabilities | other.capabilities
        if len(total_capabilities) > 0:
            synergy_score += (len(shared_capabilities) / len(total_capabilities)) * 0.3
        
        # Integration point synergy
        shared_integrations = set(self.integration_points) & set(other.integration_points)
        synergy_score += len(shared_integrations) * 0.2
        
        # Category synergy
        category_synergies = {
            (ToolCategory.INTELLIGENCE, ToolCategory.CREATION): 0.8,
            (ToolCategory.COMMUNICATION, ToolCategory.ENTERPRISE): 0.9,
            (ToolCategory.VISUAL_MEDIA, ToolCategory.DATA_MANAGEMENT): 0.7,
            (ToolCategory.AUTONOMOUS, ToolCategory.INTELLIGENCE): 0.9
        }
        
        category_pair = (self.category, other.category)
        reverse_pair = (other.category, self.category)
        synergy_score += category_synergies.get(category_pair, 
                                              category_synergies.get(reverse_pair, 0.1))
        
        return min(1.0, synergy_score)


@dataclass(frozen=True)
class WorkflowStep:
    """Single step in ecosystem workflow."""
    step_id: StepId
    tool_id: str
    parameters: Dict[str, Any]
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 3
    parallel_group: Optional[str] = None
    
    @require(lambda self: len(self.step_id) > 0)
    @require(lambda self: len(self.tool_id) > 0)
    @require(lambda self: self.timeout > 0)
    @require(lambda self: self.retry_count >= 0)
    def __post_init__(self):
        pass


@dataclass(frozen=True)
class EcosystemWorkflow:
    """Complete workflow definition using multiple ecosystem tools."""
    workflow_id: WorkflowId
    name: str
    description: str
    steps: List[WorkflowStep]
    execution_mode: ExecutionMode
    optimization_target: OptimizationTarget
    expected_duration: float
    resource_requirements: Dict[str, float]
    success_criteria: List[str] = field(default_factory=list)
    rollback_strategy: Optional[str] = None
    
    @require(lambda self: len(self.workflow_id) > 0)
    @require(lambda self: len(self.steps) > 0)
    @require(lambda self: self.expected_duration > 0)
    def __post_init__(self):
        pass
    
    def get_tool_dependencies(self) -> Dict[str, List[str]]:
        """Get dependency graph for workflow tools."""
        dependencies = {}
        for step in self.steps:
            tool_deps = []
            for precond in step.preconditions:
                # Find steps that satisfy this precondition
                for other_step in self.steps:
                    if precond in other_step.postconditions:
                        tool_deps.append(other_step.tool_id)
            dependencies[step.tool_id] = tool_deps
        return dependencies
    
    def get_parallel_groups(self) -> Dict[str, List[str]]:
        """Get parallel execution groups."""
        groups = {}
        for step in self.steps:
            if step.parallel_group:
                if step.parallel_group not in groups:
                    groups[step.parallel_group] = []
                groups[step.parallel_group].append(step.step_id)
        return groups


@dataclass(frozen=True)
class SystemPerformanceMetrics:
    """System-wide performance metrics."""
    timestamp: datetime
    total_tools_active: int
    resource_utilization: Dict[str, float]
    average_response_time: float
    success_rate: float
    error_rate: float
    throughput: float
    bottlenecks: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    
    @require(lambda self: self.total_tools_active >= 0)
    @require(lambda self: 0.0 <= self.success_rate <= 1.0)
    @require(lambda self: 0.0 <= self.error_rate <= 1.0)
    @require(lambda self: self.average_response_time >= 0.0)
    @require(lambda self: self.throughput >= 0.0)
    def __post_init__(self):
        pass
    
    def get_health_score(self) -> float:
        """Calculate overall system health score."""
        # Weighted combination of metrics
        health_components = {
            'success_rate': self.success_rate * 0.3,
            'response_time': max(0, 1.0 - (self.average_response_time / 10.0)) * 0.2,
            'resource_efficiency': (1.0 - max(self.resource_utilization.values() or [0])) * 0.2,
            'error_impact': (1.0 - self.error_rate) * 0.2,
            'bottleneck_impact': max(0, 1.0 - (len(self.bottlenecks) / 10.0)) * 0.1
        }
        
        return sum(health_components.values())


@dataclass(frozen=True)
class OrchestrationResult:
    """Result of ecosystem orchestration operation."""
    orchestration_id: OrchestrationId
    operation_type: str
    success: bool
    execution_time: float
    tools_involved: List[str]
    performance_metrics: SystemPerformanceMetrics
    optimization_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    next_recommendations: List[str] = field(default_factory=list)
    
    @require(lambda self: len(self.orchestration_id) > 0)
    @require(lambda self: len(self.operation_type) > 0)
    @require(lambda self: self.execution_time >= 0.0)
    def __post_init__(self):
        pass


@dataclass
class ToolRegistry:
    """Registry of all ecosystem tools with capabilities mapping."""
    tools: Dict[str, ToolDescriptor] = field(default_factory=dict)
    capability_index: Dict[str, Set[str]] = field(default_factory=dict)
    category_index: Dict[ToolCategory, Set[str]] = field(default_factory=dict)
    
    def register_tool(self, tool: ToolDescriptor) -> None:
        """Register a tool in the ecosystem."""
        self.tools[tool.tool_id] = tool
        
        # Update capability index
        for capability in tool.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(tool.tool_id)
        
        # Update category index
        if tool.category not in self.category_index:
            self.category_index[tool.category] = set()
        self.category_index[tool.category].add(tool.tool_id)
    
    def find_tools_by_capability(self, capability: str) -> List[ToolDescriptor]:
        """Find tools that provide a specific capability."""
        tool_ids = self.capability_index.get(capability, set())
        return [self.tools[tool_id] for tool_id in tool_ids]
    
    def find_tools_by_category(self, category: ToolCategory) -> List[ToolDescriptor]:
        """Find tools in a specific category."""
        tool_ids = self.category_index.get(category, set())
        return [self.tools[tool_id] for tool_id in tool_ids]
    
    def get_tool_synergies(self, tool_id: str) -> List[Tuple[str, float]]:
        """Get tools with high synergy scores."""
        if tool_id not in self.tools:
            return []
        
        base_tool = self.tools[tool_id]
        synergies = []
        
        for other_id, other_tool in self.tools.items():
            if other_id != tool_id:
                synergy_score = base_tool.get_synergy_score(other_tool)
                if synergy_score > 0.5:  # Only include significant synergies
                    synergies.append((other_id, synergy_score))
        
        return sorted(synergies, key=lambda x: x[1], reverse=True)


class OrchestrationError(ValidationError):
    """Errors specific to ecosystem orchestration."""
    
    @classmethod
    def workflow_execution_failed(cls, reason: str) -> 'OrchestrationError':
        return cls("workflow_execution", None, f"Workflow execution failed: {reason}")
    
    @classmethod
    def tool_not_found(cls, tool_id: str) -> 'OrchestrationError':
        return cls("tool_id", tool_id, f"Tool {tool_id} not found in registry")
    
    @classmethod
    def precondition_failed(cls, step_id: str, precondition: str) -> 'OrchestrationError':
        return cls("precondition", precondition, f"Precondition '{precondition}' failed for step {step_id}")
    
    @classmethod
    def step_execution_failed(cls, step_id: str, reason: str) -> 'OrchestrationError':
        return cls("step_execution", step_id, f"Step {step_id} execution failed: {reason}")
    
    @classmethod
    def unsupported_execution_mode(cls, mode: ExecutionMode) -> 'OrchestrationError':
        return cls("execution_mode", mode.value, f"Execution mode {mode.value} not supported")
    
    @classmethod
    def parallel_execution_failed(cls, reason: str) -> 'OrchestrationError':
        return cls("parallel_execution", None, f"Parallel execution failed: {reason}")
    
    @classmethod
    def incomplete_tool_registry(cls) -> 'OrchestrationError':
        return cls("tool_registry", None, "Tool registry is incomplete - not all tools registered")
    
    @classmethod
    def ecosystem_initialization_failed(cls, reason: str) -> 'OrchestrationError':
        return cls("ecosystem_init", None, f"Ecosystem initialization failed: {reason}")
    
    @classmethod
    def intelligent_orchestration_failed(cls, reason: str) -> 'OrchestrationError':
        return cls("intelligent_orchestration", None, f"Intelligent orchestration failed: {reason}")
    
    @classmethod
    def optimization_failed(cls, reason: str) -> 'OrchestrationError':
        return cls("optimization", None, f"Ecosystem optimization failed: {reason}")
    
    @classmethod
    def strategic_planning_failed(cls, reason: str) -> 'OrchestrationError':
        return cls("strategic_planning", None, f"Strategic planning failed: {reason}")
    
    @classmethod
    def security_escalation_detected(cls) -> 'OrchestrationError':
        return cls("security_escalation", None, "Security level escalation detected in workflow")
    
    @classmethod
    def sensitive_data_exposure(cls) -> 'OrchestrationError':
        return cls("data_exposure", None, "Sensitive data exposure risk detected")


# Protocol definitions for orchestration components
class PerformanceMonitorProtocol(Protocol):
    """Protocol for performance monitoring systems."""
    
    async def get_current_metrics(self) -> SystemPerformanceMetrics:
        """Get current system performance metrics."""
        ...
    
    async def detect_bottlenecks(self) -> List[str]:
        """Detect system bottlenecks."""
        ...


class WorkflowEngineProtocol(Protocol):
    """Protocol for workflow execution engines."""
    
    async def execute_workflow(self, workflow: EcosystemWorkflow) -> Either[OrchestrationError, Dict[str, Any]]:
        """Execute ecosystem workflow."""
        ...


class ResourceManagerProtocol(Protocol):
    """Protocol for resource management systems."""
    
    async def allocate_resources(self, requirements: Dict[str, float]) -> Either[OrchestrationError, Dict[str, float]]:
        """Allocate system resources."""
        ...
    
    async def optimize_allocation(self) -> Either[OrchestrationError, Dict[str, Any]]:
        """Optimize resource allocation."""
        ...


# Utility functions for ecosystem orchestration
def calculate_workflow_complexity(workflow: EcosystemWorkflow) -> float:
    """Calculate complexity score for workflow."""
    complexity = 0.0
    
    # Base complexity from number of steps
    complexity += len(workflow.steps) * 0.1
    
    # Add complexity for dependencies
    dependencies = workflow.get_tool_dependencies()
    total_deps = sum(len(deps) for deps in dependencies.values())
    complexity += total_deps * 0.05
    
    # Add complexity for parallel execution
    parallel_groups = workflow.get_parallel_groups()
    complexity += len(parallel_groups) * 0.2
    
    # Add complexity for resource requirements
    resource_diversity = len(workflow.resource_requirements)
    complexity += resource_diversity * 0.1
    
    return min(10.0, complexity)  # Cap at 10.0


def estimate_workflow_duration(workflow: EcosystemWorkflow, tool_registry: ToolRegistry) -> float:
    """Estimate workflow execution duration based on tools and complexity."""
    estimated_duration = 0.0
    
    # Calculate based on execution mode
    if workflow.execution_mode == ExecutionMode.SEQUENTIAL:
        # Sum all tool execution times
        for step in workflow.steps:
            tool = tool_registry.tools.get(step.tool_id)
            if tool:
                tool_time = tool.performance_characteristics.get("response_time", 1.0)
                estimated_duration += tool_time + (step.timeout * 0.1)  # Add timeout buffer
    
    elif workflow.execution_mode == ExecutionMode.PARALLEL:
        # Calculate based on parallel groups
        parallel_groups = workflow.get_parallel_groups()
        max_group_time = 0.0
        
        for group_steps in parallel_groups.values():
            group_time = 0.0
            for step_id in group_steps:
                step = next((s for s in workflow.steps if s.step_id == step_id), None)
                if step:
                    tool = tool_registry.tools.get(step.tool_id)
                    if tool:
                        tool_time = tool.performance_characteristics.get("response_time", 1.0)
                        group_time = max(group_time, tool_time)
            max_group_time += group_time
        
        estimated_duration = max_group_time
    
    else:
        # Adaptive/pipeline mode - use workflow's expected duration as base
        estimated_duration = workflow.expected_duration
    
    # Apply complexity multiplier
    complexity = calculate_workflow_complexity(workflow)
    complexity_multiplier = 1.0 + (complexity * 0.1)
    
    return estimated_duration * complexity_multiplier


def validate_workflow_security(workflow: EcosystemWorkflow, tool_registry: ToolRegistry) -> Either[OrchestrationError, None]:
    """Validate workflow security across all tools."""
    # Check for security escalation paths
    security_levels = set()
    for step in workflow.steps:
        tool = tool_registry.tools.get(step.tool_id)
        if tool:
            security_levels.add(tool.security_level)
    
    # Ensure no security level escalation without proper validation
    if SecurityLevel.ENTERPRISE in security_levels and len(security_levels) > 1:
        return Either.left(OrchestrationError.security_escalation_detected())
    
    # Check for sensitive data exposure
    for step in workflow.steps:
        if _contains_sensitive_parameters(step.parameters):
            tool = tool_registry.tools.get(step.tool_id)
            if tool and tool.security_level not in [SecurityLevel.HIGH, SecurityLevel.ENTERPRISE]:
                return Either.left(OrchestrationError.sensitive_data_exposure())
    
    return Either.right(None)


def _contains_sensitive_parameters(parameters: Dict[str, Any]) -> bool:
    """Check if parameters contain sensitive data."""
    sensitive_patterns = ["password", "secret", "token", "key", "credential"]
    
    for key, value in parameters.items():
        key_lower = key.lower()
        if any(pattern in key_lower for pattern in sensitive_patterns):
            return True
        
        if isinstance(value, str) and any(pattern in value.lower() for pattern in sensitive_patterns):
            return True
    
    return False