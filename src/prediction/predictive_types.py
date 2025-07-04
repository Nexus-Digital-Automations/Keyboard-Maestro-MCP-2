"""
Branded types for predictive automation system.

This module defines type-safe branded types for predictive operations, model management,
and optimization recommendations. Implements complete type safety with validation.

Security: Type validation prevents injection and ensures data integrity.
Performance: Efficient branded type operations with minimal overhead.
Type Safety: Complete branded type system with contract validation.
"""

from typing import NewType, List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
import uuid

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.errors import ValidationError


# Core Branded Types
PredictiveModelId = NewType('PredictiveModelId', str)
PredictionRequestId = NewType('PredictionRequestId', str)
OptimizationId = NewType('OptimizationId', str)
ForecastId = NewType('ForecastId', str)
PatternId = NewType('PatternId', str)
AnomalyId = NewType('AnomalyId', str)
CapacityPlanId = NewType('CapacityPlanId', str)
WorkflowOptimizationId = NewType('WorkflowOptimizationId', str)
AlertId = NewType('AlertId', str)

# Confidence and probability types
ConfidenceLevel = NewType('ConfidenceLevel', float)  # 0.0 to 1.0
ProbabilityScore = NewType('ProbabilityScore', float)  # 0.0 to 1.0
AccuracyScore = NewType('AccuracyScore', float)  # 0.0 to 1.0

# Performance and resource metrics
PerformanceScore = NewType('PerformanceScore', float)  # 0.0 to 100.0
ResourceUtilization = NewType('ResourceUtilization', float)  # 0.0 to 1.0
OptimizationImpact = NewType('OptimizationImpact', float)  # -100.0 to 100.0


class PredictionType(Enum):
    """Types of predictions supported by the system."""
    PERFORMANCE = "performance"
    RESOURCE_USAGE = "resource_usage"
    CAPACITY_NEEDS = "capacity_needs"
    ANOMALY_DETECTION = "anomaly_detection"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    SYSTEM_HEALTH = "system_health"
    COST_FORECASTING = "cost_forecasting"
    USAGE_PATTERNS = "usage_patterns"


class ModelType(Enum):
    """Types of predictive models."""
    PATTERN_RECOGNITION = "pattern_recognition"
    PERFORMANCE_FORECASTING = "performance_forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    RESOURCE_PREDICTION = "resource_prediction"
    CAPACITY_PLANNING = "capacity_planning"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    COST_OPTIMIZATION = "cost_optimization"
    TREND_ANALYSIS = "trend_analysis"


class OptimizationType(Enum):
    """Types of optimization operations."""
    PERFORMANCE = "performance"
    RESOURCE_ALLOCATION = "resource_allocation"
    COST_REDUCTION = "cost_reduction"
    CAPACITY_SCALING = "capacity_scaling"
    WORKFLOW_EFFICIENCY = "workflow_efficiency"
    SYSTEM_RELIABILITY = "system_reliability"
    ENERGY_EFFICIENCY = "energy_efficiency"
    RESPONSE_TIME = "response_time"


class PredictionPriority(Enum):
    """Priority levels for predictions and optimizations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class AlertSeverity(Enum):
    """Severity levels for predictive alerts."""
    EMERGENCY = "emergency"
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


@dataclass(frozen=True)
class PredictiveModel:
    """Branded type for predictive model configuration."""
    model_id: PredictiveModelId
    model_type: ModelType
    name: str
    description: str
    version: str
    accuracy_score: AccuracyScore
    confidence_threshold: ConfidenceLevel
    last_trained: datetime
    training_data_size: int
    supported_prediction_types: List[PredictionType]
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate model configuration."""
        if not (0.0 <= self.accuracy_score <= 1.0):
            raise ValidationError("accuracy_score must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValidationError("confidence_threshold must be between 0.0 and 1.0")
        if self.training_data_size < 0:
            raise ValidationError("training_data_size cannot be negative")
        if not self.supported_prediction_types:
            raise ValidationError("model must support at least one prediction type")


@dataclass(frozen=True)
class PredictionRequest:
    """Branded type for prediction requests."""
    request_id: PredictionRequestId
    prediction_type: PredictionType
    model_id: PredictiveModelId
    input_data: Dict[str, Any]
    forecast_horizon: timedelta
    confidence_level: ConfidenceLevel
    priority: PredictionPriority
    requesting_component: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    timeout: float = 300.0
    
    def __post_init__(self):
        """Validate prediction request."""
        if not (0.0 <= self.confidence_level <= 1.0):
            raise ValidationError("confidence_level must be between 0.0 and 1.0")
        if self.timeout <= 0:
            raise ValidationError("timeout must be positive")
        if self.forecast_horizon.total_seconds() <= 0:
            raise ValidationError("forecast_horizon must be positive")


@dataclass(frozen=True)
class OptimizationSuggestion:
    """Branded type for optimization recommendations."""
    optimization_id: OptimizationId
    optimization_type: OptimizationType
    title: str
    description: str
    confidence: ConfidenceLevel
    expected_impact: OptimizationImpact
    implementation_effort: str  # "low", "medium", "high"
    priority: PredictionPriority
    affected_components: List[str]
    implementation_steps: List[str]
    estimated_duration: timedelta
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    metrics_to_monitor: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Validate optimization suggestion."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError("confidence must be between 0.0 and 1.0")
        if not (-100.0 <= self.expected_impact <= 100.0):
            raise ValidationError("expected_impact must be between -100.0 and 100.0")
        if self.implementation_effort not in ["low", "medium", "high"]:
            raise ValidationError("implementation_effort must be low, medium, or high")


@dataclass(frozen=True)
class PerformanceForecast:
    """Branded type for performance forecasting results."""
    forecast_id: ForecastId
    metric_name: str
    current_value: float
    predicted_values: List[Tuple[datetime, float, ConfidenceLevel]]
    trend: str  # "increasing", "decreasing", "stable", "volatile"
    forecast_accuracy: AccuracyScore
    confidence_interval: Tuple[float, float]
    anomaly_probability: ProbabilityScore
    recommendation: str
    model_used: PredictiveModelId
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Validate performance forecast."""
        if not (0.0 <= self.forecast_accuracy <= 1.0):
            raise ValidationError("forecast_accuracy must be between 0.0 and 1.0")
        if not (0.0 <= self.anomaly_probability <= 1.0):
            raise ValidationError("anomaly_probability must be between 0.0 and 1.0")
        if self.trend not in ["increasing", "decreasing", "stable", "volatile"]:
            raise ValidationError("trend must be increasing, decreasing, stable, or volatile")


@dataclass(frozen=True)
class ResourcePrediction:
    """Branded type for resource usage predictions."""
    prediction_id: str
    resource_type: str
    current_usage: ResourceUtilization
    predicted_usage: List[Tuple[datetime, ResourceUtilization, ConfidenceLevel]]
    capacity_threshold: ResourceUtilization
    expected_shortage: Optional[datetime]
    optimization_opportunities: List[str]
    scaling_recommendation: str
    model_used: PredictiveModelId
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Validate resource prediction."""
        if not (0.0 <= self.current_usage <= 1.0):
            raise ValidationError("current_usage must be between 0.0 and 1.0")
        if not (0.0 <= self.capacity_threshold <= 1.0):
            raise ValidationError("capacity_threshold must be between 0.0 and 1.0")


@dataclass(frozen=True)
class PatternAnalysis:
    """Branded type for pattern recognition results."""
    pattern_id: PatternId
    pattern_type: str
    description: str
    confidence: ConfidenceLevel
    frequency: str  # "daily", "weekly", "monthly", "irregular"
    detected_at: datetime
    historical_occurrences: List[datetime]
    prediction_accuracy: AccuracyScore
    business_impact: str  # "high", "medium", "low"
    recommendations: List[str]
    model_used: PredictiveModelId
    
    def __post_init__(self):
        """Validate pattern analysis."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError("confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.prediction_accuracy <= 1.0):
            raise ValidationError("prediction_accuracy must be between 0.0 and 1.0")
        if self.frequency not in ["daily", "weekly", "monthly", "irregular"]:
            raise ValidationError("frequency must be daily, weekly, monthly, or irregular")
        if self.business_impact not in ["high", "medium", "low"]:
            raise ValidationError("business_impact must be high, medium, or low")


@dataclass(frozen=True)
class AnomalyPrediction:
    """Branded type for anomaly detection results."""
    anomaly_id: AnomalyId
    anomaly_type: str
    severity: AlertSeverity
    probability: ProbabilityScore
    affected_metric: str
    current_value: float
    expected_range: Tuple[float, float]
    deviation_score: float
    predicted_impact: str
    time_to_resolution: Optional[timedelta]
    mitigation_suggestions: List[str]
    model_used: PredictiveModelId
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Validate anomaly prediction."""
        if not (0.0 <= self.probability <= 1.0):
            raise ValidationError("probability must be between 0.0 and 1.0")


@dataclass(frozen=True)
class CapacityPlan:
    """Branded type for capacity planning results."""
    plan_id: CapacityPlanId
    resource_type: str
    current_capacity: float
    projected_demand: List[Tuple[datetime, float, ConfidenceLevel]]
    scaling_recommendations: List[str]
    optimal_scaling_time: datetime
    cost_implications: Dict[str, float]
    risk_assessment: str
    confidence: ConfidenceLevel
    model_used: PredictiveModelId
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Validate capacity plan."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError("confidence must be between 0.0 and 1.0")
        if self.current_capacity < 0:
            raise ValidationError("current_capacity cannot be negative")


@dataclass(frozen=True)
class WorkflowOptimization:
    """Branded type for workflow optimization results."""
    optimization_id: WorkflowOptimizationId
    workflow_name: str
    current_performance: PerformanceScore
    optimized_performance: PerformanceScore
    optimization_steps: List[str]
    performance_gain: float
    implementation_complexity: str  # "low", "medium", "high"
    estimated_savings: Dict[str, float]
    success_probability: ProbabilityScore
    model_used: PredictiveModelId
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Validate workflow optimization."""
        if not (0.0 <= self.current_performance <= 100.0):
            raise ValidationError("current_performance must be between 0.0 and 100.0")
        if not (0.0 <= self.optimized_performance <= 100.0):
            raise ValidationError("optimized_performance must be between 0.0 and 100.0")
        if not (0.0 <= self.success_probability <= 1.0):
            raise ValidationError("success_probability must be between 0.0 and 1.0")
        if self.implementation_complexity not in ["low", "medium", "high"]:
            raise ValidationError("implementation_complexity must be low, medium, or high")


@dataclass(frozen=True)
class PredictiveAlert:
    """Branded type for predictive alerts."""
    alert_id: AlertId
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    predicted_occurrence: datetime
    confidence: ConfidenceLevel
    affected_systems: List[str]
    recommended_actions: List[str]
    escalation_threshold: timedelta
    auto_resolution: bool
    model_used: PredictiveModelId
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Validate predictive alert."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError("confidence must be between 0.0 and 1.0")


# Factory functions for creating branded types
def create_predictive_model_id() -> PredictiveModelId:
    """Create a new predictive model ID."""
    return PredictiveModelId(f"model_{uuid.uuid4().hex[:12]}")


def create_prediction_request_id() -> PredictionRequestId:
    """Create a new prediction request ID."""
    return PredictionRequestId(f"pred_{uuid.uuid4().hex[:12]}")


def create_optimization_id() -> OptimizationId:
    """Create a new optimization ID."""
    return OptimizationId(f"opt_{uuid.uuid4().hex[:12]}")


def create_forecast_id() -> ForecastId:
    """Create a new forecast ID."""
    return ForecastId(f"forecast_{uuid.uuid4().hex[:12]}")


def create_pattern_id() -> PatternId:
    """Create a new pattern ID."""
    return PatternId(f"pattern_{uuid.uuid4().hex[:12]}")


def create_anomaly_id() -> AnomalyId:
    """Create a new anomaly ID."""
    return AnomalyId(f"anomaly_{uuid.uuid4().hex[:12]}")


def create_capacity_plan_id() -> CapacityPlanId:
    """Create a new capacity plan ID."""
    return CapacityPlanId(f"capacity_{uuid.uuid4().hex[:12]}")


def create_workflow_optimization_id() -> WorkflowOptimizationId:
    """Create a new workflow optimization ID."""
    return WorkflowOptimizationId(f"workflow_{uuid.uuid4().hex[:12]}")


def create_alert_id() -> AlertId:
    """Create a new alert ID."""
    return AlertId(f"alert_{uuid.uuid4().hex[:12]}")


# Validation functions
@require(lambda score: 0.0 <= score <= 1.0)
def create_confidence_level(score: float) -> ConfidenceLevel:
    """Create a validated confidence level."""
    return ConfidenceLevel(score)


@require(lambda score: 0.0 <= score <= 1.0)
def create_probability_score(score: float) -> ProbabilityScore:
    """Create a validated probability score."""
    return ProbabilityScore(score)


@require(lambda score: 0.0 <= score <= 1.0)
def create_accuracy_score(score: float) -> AccuracyScore:
    """Create a validated accuracy score."""
    return AccuracyScore(score)


@require(lambda score: 0.0 <= score <= 100.0)
def create_performance_score(score: float) -> PerformanceScore:
    """Create a validated performance score."""
    return PerformanceScore(score)


@require(lambda utilization: 0.0 <= utilization <= 1.0)
def create_resource_utilization(utilization: float) -> ResourceUtilization:
    """Create a validated resource utilization score."""
    return ResourceUtilization(utilization)


@require(lambda impact: -100.0 <= impact <= 100.0)
def create_optimization_impact(impact: float) -> OptimizationImpact:
    """Create a validated optimization impact score."""
    return OptimizationImpact(impact)