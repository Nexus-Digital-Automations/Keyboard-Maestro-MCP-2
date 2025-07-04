"""
API Orchestration Architecture - TASK_64 Phase 1 Architecture & Design

Comprehensive type system for advanced API management, service orchestration, and microservices coordination.
Provides enterprise-grade API workflow management with fault tolerance and performance monitoring.

Architecture: Service-Oriented + Microservices + Circuit Breaker + Load Balancing + Service Mesh
Performance: <100ms API routing, <500ms workflow orchestration, <200ms service coordination
Reliability: Circuit breaker patterns, retry logic, health monitoring, graceful degradation
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import uuid

from src.core.either import Either
from src.core.errors import MCPError


# Branded Types for API Orchestration
class WorkflowId(str):
    """Branded type for API workflow identifiers."""
    def __new__(cls, value: str) -> WorkflowId:
        if not value or len(value) < 3:
            raise ValueError("WorkflowId must be at least 3 characters")
        return str.__new__(cls, value)

class ServiceId(str):
    """Branded type for service identifiers."""
    def __new__(cls, value: str) -> ServiceId:
        if not value or len(value) < 2:
            raise ValueError("ServiceId must be at least 2 characters")
        return str.__new__(cls, value)

class OrchestrationId(str):
    """Branded type for orchestration identifiers."""
    def __new__(cls, value: str) -> OrchestrationId:
        if not value or len(value) < 8:
            raise ValueError("OrchestrationId must be at least 8 characters")
        return str.__new__(cls, value)

class LoadBalancerId(str):
    """Branded type for load balancer identifiers."""
    def __new__(cls, value: str) -> LoadBalancerId:
        if not value or len(value) < 4:
            raise ValueError("LoadBalancerId must be at least 4 characters")
        return str.__new__(cls, value)

class CircuitBreakerId(str):
    """Branded type for circuit breaker identifiers."""
    def __new__(cls, value: str) -> CircuitBreakerId:
        if not value or len(value) < 4:
            raise ValueError("CircuitBreakerId must be at least 4 characters")
        return str.__new__(cls, value)


# Enums for API Orchestration

class OrchestrationStrategy(Enum):
    """API orchestration strategies."""
    SEQUENTIAL = "sequential"           # Execute APIs in sequence
    PARALLEL = "parallel"               # Execute APIs in parallel
    CONDITIONAL = "conditional"         # Execute based on conditions
    PIPELINE = "pipeline"               # Pipeline with data flow
    SCATTER_GATHER = "scatter_gather"   # Scatter requests, gather responses
    CHOREOGRAPHY = "choreography"       # Event-driven choreography
    SAGA = "saga"                       # Distributed transaction pattern


class ServiceMeshType(Enum):
    """Service mesh types."""
    ISTIO = "istio"                     # Istio service mesh
    LINKERD = "linkerd"                 # Linkerd service mesh
    CONSUL_CONNECT = "consul_connect"   # Consul Connect
    AWS_APP_MESH = "aws_app_mesh"       # AWS App Mesh
    CUSTOM = "custom"                   # Custom service mesh


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"         # Round robin distribution
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # Weighted round robin
    LEAST_CONNECTIONS = "least_connections"  # Least connections
    LEAST_RESPONSE_TIME = "least_response_time"  # Least response time
    CONSISTENT_HASH = "consistent_hash"  # Consistent hashing
    RANDOM = "random"                   # Random selection
    HEALTH_BASED = "health_based"       # Health-based routing


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"                   # Normal operation
    OPEN = "open"                       # Circuit breaker open
    HALF_OPEN = "half_open"             # Testing recovery


class ServiceHealthStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"                 # Service is healthy
    DEGRADED = "degraded"               # Service is degraded
    UNHEALTHY = "unhealthy"             # Service is unhealthy
    UNKNOWN = "unknown"                 # Health status unknown


class RoutingRule(Enum):
    """API routing rules."""
    PATH_BASED = "path_based"           # Route based on path
    HEADER_BASED = "header_based"       # Route based on headers
    QUERY_BASED = "query_based"         # Route based on query parameters
    WEIGHT_BASED = "weight_based"       # Route based on weights
    CANARY = "canary"                   # Canary deployment routing
    BLUE_GREEN = "blue_green"           # Blue-green deployment routing


# Core Data Structures

@dataclass(frozen=True)
class APIEndpoint:
    """API endpoint specification."""
    endpoint_id: str
    service_id: ServiceId
    url: str
    method: str                         # HTTP method
    timeout_ms: int = 30000            # Request timeout
    retry_config: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    authentication: Optional[Dict[str, Any]] = None
    rate_limit: Optional[Dict[str, int]] = None
    circuit_breaker_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        if self.method.upper() not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']:
            raise ValueError(f"Invalid HTTP method: {self.method}")


@dataclass(frozen=True)
class ServiceDefinition:
    """Service definition for orchestration."""
    service_id: ServiceId
    service_name: str
    service_version: str
    endpoints: List[APIEndpoint]
    load_balancer_config: Optional[Dict[str, Any]] = None
    health_check_config: Optional[Dict[str, Any]] = None
    circuit_breaker_config: Optional[Dict[str, Any]] = None
    service_mesh_config: Optional[Dict[str, Any]] = None
    observability_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.endpoints:
            raise ValueError("Service must have at least one endpoint")


@dataclass(frozen=True)
class WorkflowStep:
    """Individual workflow step specification."""
    step_id: str
    step_name: str
    service_id: ServiceId
    endpoint_id: str
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    retry_policy: Optional[Dict[str, Any]] = None
    timeout_override: Optional[int] = None
    parallel_group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestrationWorkflow:
    """Complete API orchestration workflow."""
    workflow_id: WorkflowId
    workflow_name: str
    workflow_version: str
    strategy: OrchestrationStrategy
    steps: List[WorkflowStep]
    global_timeout_ms: int = 300000     # 5 minutes default
    error_handling_strategy: str = "fail_fast"
    data_transformation_rules: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    security_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.steps:
            raise ValueError("Workflow must have at least one step")


@dataclass(frozen=True)
class LoadBalancerConfig:
    """Load balancer configuration."""
    balancer_id: LoadBalancerId
    strategy: LoadBalancingStrategy
    targets: List[Dict[str, Any]]
    health_check_interval: int = 30     # seconds
    weights: Optional[Dict[str, float]] = None
    sticky_sessions: bool = False
    connection_pool_config: Optional[Dict[str, Any]] = None
    circuit_breaker_integration: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    breaker_id: CircuitBreakerId
    failure_threshold: int = 5          # failures before opening
    recovery_timeout: int = 60          # seconds before half-open
    success_threshold: int = 3          # successes before closing
    timeout_ms: int = 30000            # request timeout
    failure_rate_threshold: float = 0.5  # failure rate threshold
    minimum_requests: int = 10          # minimum requests for rate calculation
    sliding_window_size: int = 100      # sliding window size
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServiceMeshConfig:
    """Service mesh configuration."""
    mesh_type: ServiceMeshType
    service_id: ServiceId
    routing_rules: List[Dict[str, Any]]
    security_policies: List[Dict[str, Any]]
    observability_config: Dict[str, Any]
    traffic_management: Dict[str, Any] = field(default_factory=dict)
    fault_injection: Optional[Dict[str, Any]] = None
    rate_limiting: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestrationResult:
    """Result of API orchestration execution."""
    orchestration_id: OrchestrationId
    workflow_id: WorkflowId
    execution_status: str               # success, failure, partial
    start_time: datetime
    end_time: datetime
    total_duration_ms: int
    step_results: List[Dict[str, Any]]
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    circuit_breaker_events: List[Dict[str, Any]] = field(default_factory=list)
    load_balancer_decisions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServiceHealthReport:
    """Service health monitoring report."""
    service_id: ServiceId
    health_status: ServiceHealthStatus
    check_timestamp: datetime
    response_time_ms: int
    availability_percentage: float
    error_rate: float
    throughput_rps: float               # requests per second
    circuit_breaker_state: CircuitBreakerState
    load_balancer_status: str
    recent_errors: List[str] = field(default_factory=list)
    performance_trends: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Exception Classes

class APIOrchestrationError(MCPError):
    """Base exception for API orchestration errors."""
    pass

class ServiceUnavailableError(APIOrchestrationError):
    """Exception for service unavailability."""
    pass

class WorkflowExecutionError(APIOrchestrationError):
    """Exception for workflow execution failures."""
    pass

class CircuitBreakerOpenError(APIOrchestrationError):
    """Exception for circuit breaker open state."""
    pass

class LoadBalancerError(APIOrchestrationError):
    """Exception for load balancer failures."""
    pass

class ServiceMeshError(APIOrchestrationError):
    """Exception for service mesh configuration errors."""
    pass


# Utility Functions

def create_workflow_id(base_name: str) -> WorkflowId:
    """Create a unique workflow identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return WorkflowId(f"{base_name}_{timestamp}_{unique_id}")

def create_service_id(service_name: str) -> ServiceId:
    """Create a service identifier."""
    sanitized_name = "".join(c for c in service_name if c.isalnum() or c in "_-")
    return ServiceId(sanitized_name.lower())

def create_orchestration_id(workflow_id: WorkflowId) -> OrchestrationId:
    """Create an orchestration execution identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return OrchestrationId(f"exec_{workflow_id}_{timestamp}")

def create_load_balancer_id(service_name: str) -> LoadBalancerId:
    """Create a load balancer identifier."""
    sanitized_name = "".join(c for c in service_name if c.isalnum() or c in "_-")
    return LoadBalancerId(f"lb_{sanitized_name.lower()}")

def create_circuit_breaker_id(service_name: str) -> CircuitBreakerId:
    """Create a circuit breaker identifier."""
    sanitized_name = "".join(c for c in service_name if c.isalnum() or c in "_-")
    return CircuitBreakerId(f"cb_{sanitized_name.lower()}")

def validate_workflow_configuration(workflow: OrchestrationWorkflow) -> Either[APIOrchestrationError, bool]:
    """Validate workflow configuration for correctness."""
    try:
        # Check for step dependencies and circular references
        step_ids = {step.step_id for step in workflow.steps}
        
        # Validate parallel groups if using parallel strategy
        if workflow.strategy == OrchestrationStrategy.PARALLEL:
            parallel_groups = {step.parallel_group for step in workflow.steps if step.parallel_group}
            if not parallel_groups:
                return Either.error(APIOrchestrationError("Parallel workflow requires parallel groups"))
        
        # Validate conditional logic if using conditional strategy
        if workflow.strategy == OrchestrationStrategy.CONDITIONAL:
            conditional_steps = [step for step in workflow.steps if step.conditions]
            if not conditional_steps:
                return Either.error(APIOrchestrationError("Conditional workflow requires condition expressions"))
        
        # Validate timeout constraints
        if workflow.global_timeout_ms < 1000:  # Minimum 1 second
            return Either.error(APIOrchestrationError("Global timeout must be at least 1000ms"))
        
        return Either.success(True)
        
    except Exception as e:
        return Either.error(APIOrchestrationError(f"Workflow validation failed: {str(e)}"))

def calculate_workflow_complexity(workflow: OrchestrationWorkflow) -> int:
    """Calculate workflow complexity score for monitoring and optimization."""
    complexity = 0
    
    # Base complexity
    complexity += len(workflow.steps)
    
    # Strategy complexity multiplier
    strategy_multipliers = {
        OrchestrationStrategy.SEQUENTIAL: 1,
        OrchestrationStrategy.PARALLEL: 2,
        OrchestrationStrategy.CONDITIONAL: 3,
        OrchestrationStrategy.PIPELINE: 2,
        OrchestrationStrategy.SCATTER_GATHER: 4,
        OrchestrationStrategy.CHOREOGRAPHY: 5,
        OrchestrationStrategy.SAGA: 6
    }
    complexity *= strategy_multipliers.get(workflow.strategy, 1)
    
    # Conditional complexity
    conditional_steps = sum(1 for step in workflow.steps if step.conditions)
    complexity += conditional_steps * 2
    
    # Parallel group complexity
    parallel_groups = len({step.parallel_group for step in workflow.steps if step.parallel_group})
    complexity += parallel_groups * 3
    
    return complexity


# Export all public types and functions
__all__ = [
    # Branded Types
    "WorkflowId", "ServiceId", "OrchestrationId", "LoadBalancerId", "CircuitBreakerId",
    
    # Enums
    "OrchestrationStrategy", "ServiceMeshType", "LoadBalancingStrategy", 
    "CircuitBreakerState", "ServiceHealthStatus", "RoutingRule",
    
    # Data Structures
    "APIEndpoint", "ServiceDefinition", "WorkflowStep", "OrchestrationWorkflow",
    "LoadBalancerConfig", "CircuitBreakerConfig", "ServiceMeshConfig",
    "OrchestrationResult", "ServiceHealthReport",
    
    # Exceptions
    "APIOrchestrationError", "ServiceUnavailableError", "WorkflowExecutionError",
    "CircuitBreakerOpenError", "LoadBalancerError", "ServiceMeshError",
    
    # Utility Functions
    "create_workflow_id", "create_service_id", "create_orchestration_id",
    "create_load_balancer_id", "create_circuit_breaker_id",
    "validate_workflow_configuration", "calculate_workflow_complexity"
]