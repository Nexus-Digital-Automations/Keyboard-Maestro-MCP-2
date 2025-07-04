"""
Predictive Modeling Architecture - TASK_59 Phase 1 Implementation

Comprehensive predictive analytics type definitions, modeling frameworks, and forecasting structures.
Extends the existing analytics and AI processing systems with advanced predictive capabilities.

Architecture: Predictive Types + ML Integration + Forecasting Models + Statistical Analysis
Performance: <100ms prediction setup, <5s model training, <1s inference
Security: Safe model execution, validated predictions, comprehensive input sanitization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from abc import ABC, abstractmethod
from enum import Enum
import uuid
import asyncio

from src.core.types import MacroId, create_macro_id
from src.core.either import Either
from src.core.contracts import require, ensure
from src.intelligence.intelligence_types import IntelligenceOperation, AnalysisScope, LearningMode


# Branded Types for Predictive Modeling
PredictionId = str
ModelId = str
ForecastId = str
InsightId = str
ScenarioId = str

def create_prediction_id() -> PredictionId:
    """Create a unique prediction identifier."""
    return f"pred_{uuid.uuid4().hex[:12]}"

def create_model_id() -> ModelId:
    """Create a unique model identifier."""
    return f"model_{uuid.uuid4().hex[:12]}"

def create_forecast_id() -> ForecastId:
    """Create a unique forecast identifier."""
    return f"forecast_{uuid.uuid4().hex[:12]}"

def create_insight_id() -> InsightId:
    """Create a unique insight identifier."""
    return f"insight_{uuid.uuid4().hex[:12]}"

def create_scenario_id() -> ScenarioId:
    """Create a unique scenario identifier."""
    return f"scenario_{uuid.uuid4().hex[:12]}"


class PredictionType(Enum):
    """Types of predictions supported by the system."""
    # Pattern Predictions
    USAGE_PATTERNS = "usage_patterns"
    PERFORMANCE_PATTERNS = "performance_patterns"
    ERROR_PATTERNS = "error_patterns"
    WORKFLOW_PATTERNS = "workflow_patterns"
    
    # Resource Forecasting
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    STORAGE_USAGE = "storage_usage"
    NETWORK_USAGE = "network_usage"
    
    # Business Analytics
    ROI_FORECAST = "roi_forecast"
    COST_PREDICTION = "cost_prediction"
    EFFICIENCY_TRENDS = "efficiency_trends"
    CAPACITY_PLANNING = "capacity_planning"
    
    # Failure Prediction
    EXECUTION_FAILURES = "execution_failures"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SYSTEM_ANOMALIES = "system_anomalies"


class ModelType(Enum):
    """Machine learning model types for predictive analytics."""
    # Time Series Models
    LINEAR_REGRESSION = "linear_regression"
    ARIMA = "arima"
    SEASONAL_ARIMA = "seasonal_arima"
    LSTM = "lstm"
    
    # Classification Models
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    
    # Ensemble Models
    ENSEMBLE = "ensemble"
    VOTING_CLASSIFIER = "voting_classifier"
    STACKING = "stacking"
    
    # Specialized Models
    ANOMALY_DETECTION = "anomaly_detection"
    CHANGE_POINT = "change_point"
    PROPHET = "prophet"


class ForecastGranularity(Enum):
    """Granularity levels for forecasting."""
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class ConfidenceLevel(Enum):
    """Confidence levels for predictions and forecasts."""
    LOW = "low"              # 70-80%
    MEDIUM = "medium"        # 80-90%
    HIGH = "high"           # 90-95%
    VERY_HIGH = "very_high" # 95%+


class InsightType(Enum):
    """Types of insights generated from predictive analytics."""
    OPTIMIZATION = "optimization"
    EFFICIENCY = "efficiency"
    COST_SAVINGS = "cost_savings"
    RISK_MITIGATION = "risk_mitigation"
    CAPACITY_PLANNING = "capacity_planning"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    WORKFLOW_ENHANCEMENT = "workflow_enhancement"
    ANOMALY_ALERT = "anomaly_alert"


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for predictive modeling operations."""
    model_type: ModelType
    prediction_type: PredictionType
    horizon_days: int = 30
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    include_uncertainty: bool = True
    seasonality_detection: bool = True
    trend_analysis: bool = True
    external_factors: bool = False
    
    def __post_init__(self):
        if not (1 <= self.horizon_days <= 365):
            raise ValueError("Prediction horizon must be between 1 and 365 days")


@dataclass(frozen=True)
class TimeSeriesData:
    """Time series data structure for predictive modeling."""
    timestamps: List[datetime]
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have the same length")
        if len(self.timestamps) < 2:
            raise ValueError("Time series must have at least 2 data points")


@dataclass(frozen=True)
class PredictionResult:
    """Result of a predictive modeling operation."""
    prediction_id: PredictionId
    model_id: ModelId
    prediction_type: PredictionType
    forecast_values: List[float]
    forecast_timestamps: List[datetime]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    model_accuracy: Optional[float] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if len(self.forecast_values) != len(self.forecast_timestamps):
            raise ValueError("Forecast values and timestamps must have the same length")
        if self.model_accuracy is not None and not (0.0 <= self.model_accuracy <= 1.0):
            raise ValueError("Model accuracy must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ResourceForecast:
    """Resource usage forecast with capacity planning."""
    forecast_id: ForecastId
    resource_type: str
    granularity: ForecastGranularity
    forecast_period_days: int
    current_usage: float
    predicted_usage: List[float]
    forecast_timestamps: List[datetime]
    capacity_thresholds: Dict[str, float] = field(default_factory=dict)
    growth_rate: Optional[float] = None
    seasonality_patterns: Dict[str, Any] = field(default_factory=dict)
    capacity_recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.predicted_usage) != len(self.forecast_timestamps):
            raise ValueError("Predicted usage and timestamps must have the same length")
        if not (1 <= self.forecast_period_days <= 365):
            raise ValueError("Forecast period must be between 1 and 365 days")


@dataclass(frozen=True)
class PredictiveInsight:
    """Intelligent insight generated from predictive analytics."""
    insight_id: InsightId
    insight_type: InsightType
    title: str
    description: str
    confidence_score: float
    impact_score: float
    priority_level: str  # low, medium, high, critical
    actionable_recommendations: List[str]
    data_sources: List[str]
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    roi_estimate: Optional[float] = None
    implementation_effort: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        if not (0.0 <= self.impact_score <= 1.0):
            raise ValueError("Impact score must be between 0.0 and 1.0")
        if self.priority_level not in ["low", "medium", "high", "critical"]:
            raise ValueError("Priority level must be low, medium, high, or critical")


@dataclass(frozen=True)
class FailurePrediction:
    """Prediction of potential automation failures."""
    prediction_id: PredictionId
    target_id: str
    target_type: str  # macro, workflow, system
    failure_type: str
    probability: float
    time_to_failure: Optional[timedelta] = None
    contributing_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    early_warning_indicators: List[str] = field(default_factory=list)
    severity_level: str = "medium"  # low, medium, high, critical
    
    def __post_init__(self):
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError("Failure probability must be between 0.0 and 1.0")
        if self.severity_level not in ["low", "medium", "high", "critical"]:
            raise ValueError("Severity level must be low, medium, high, or critical")


@dataclass(frozen=True)
class ScenarioModel:
    """Scenario modeling configuration and results."""
    scenario_id: ScenarioId
    scenario_name: str
    scenario_type: str  # what_if, stress_test, capacity, growth
    base_parameters: Dict[str, Any]
    scenario_parameters: Dict[str, Any]
    time_horizon: int  # days
    simulation_results: Dict[str, Any] = field(default_factory=dict)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    uncertainty_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not (1 <= self.time_horizon <= 365):
            raise ValueError("Time horizon must be between 1 and 365 days")


@dataclass(frozen=True)
class ModelPerformance:
    """Performance metrics for predictive models."""
    model_id: ModelId
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    rmse: Optional[float] = None
    mae: Optional[float] = None
    training_time_seconds: float = 0.0
    inference_time_ms: float = 0.0
    model_size_mb: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        for metric in [self.accuracy, self.precision, self.recall, self.f1_score]:
            if not (0.0 <= metric <= 1.0):
                raise ValueError("Performance metrics must be between 0.0 and 1.0")


class PredictiveModelingError(Exception):
    """Base exception for predictive modeling errors."""
    
    def __init__(self, message: str, error_code: str = "PREDICTION_ERROR"):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now(UTC)


class ModelTrainingError(PredictiveModelingError):
    """Model training specific errors."""
    
    @classmethod
    def insufficient_data(cls, required_points: int, available_points: int) -> ModelTrainingError:
        return cls(f"Insufficient data: requires {required_points}, got {available_points}", "INSUFFICIENT_DATA")
    
    @classmethod
    def training_failed(cls, model_type: str, error: str) -> ModelTrainingError:
        return cls(f"Training failed for {model_type}: {error}", "TRAINING_FAILED")


class PredictionError(PredictiveModelingError):
    """Prediction generation specific errors."""
    
    @classmethod
    def model_not_found(cls, model_id: str) -> PredictionError:
        return cls(f"Model not found: {model_id}", "MODEL_NOT_FOUND")
    
    @classmethod
    def prediction_failed(cls, model_id: str, error: str) -> PredictionError:
        return cls(f"Prediction failed for model {model_id}: {error}", "PREDICTION_FAILED")


class ForecastingError(PredictiveModelingError):
    """Forecasting specific errors."""
    
    @classmethod
    def invalid_horizon(cls, horizon: int) -> ForecastingError:
        return cls(f"Invalid forecast horizon: {horizon} days", "INVALID_HORIZON")


# Utility Functions for Predictive Modeling

@require(lambda data: len(data.timestamps) >= 10)
@require(lambda horizon_days: 1 <= horizon_days <= 365)
def validate_time_series_data(data: TimeSeriesData, horizon_days: int) -> Either[ModelTrainingError, None]:
    """Validate time series data for predictive modeling."""
    try:
        # Check for sufficient data points
        min_required = max(10, horizon_days // 2)
        if len(data.timestamps) < min_required:
            return Either.left(ModelTrainingError.insufficient_data(min_required, len(data.timestamps)))
        
        # Check for data quality
        if any(v is None or not isinstance(v, (int, float)) for v in data.values):
            return Either.left(ModelTrainingError("Invalid data values detected", "INVALID_DATA"))
        
        # Check for temporal ordering
        if data.timestamps != sorted(data.timestamps):
            return Either.left(ModelTrainingError("Timestamps must be in chronological order", "UNORDERED_DATA"))
        
        return Either.right(None)
        
    except Exception as e:
        return Either.left(ModelTrainingError(f"Data validation failed: {str(e)}", "VALIDATION_ERROR"))


@ensure(lambda result: 0.0 <= result <= 1.0)
def calculate_prediction_confidence(
    model_performance: ModelPerformance,
    data_quality_score: float,
    horizon_days: int
) -> float:
    """Calculate confidence score for predictions based on model performance and data quality."""
    base_confidence = (model_performance.accuracy + model_performance.f1_score) / 2
    
    # Adjust for data quality
    quality_factor = min(1.0, data_quality_score)
    
    # Adjust for prediction horizon (confidence decreases with longer horizons)
    horizon_factor = max(0.5, 1.0 - (horizon_days / 365) * 0.3)
    
    # Adjust for model freshness
    days_since_training = (datetime.now(UTC) - model_performance.last_updated).days
    freshness_factor = max(0.7, 1.0 - (days_since_training / 30) * 0.1)
    
    confidence = base_confidence * quality_factor * horizon_factor * freshness_factor
    return min(1.0, max(0.0, confidence))


def categorize_confidence_level(confidence_score: float) -> ConfidenceLevel:
    """Categorize numeric confidence score into confidence level enum."""
    if confidence_score >= 0.95:
        return ConfidenceLevel.VERY_HIGH
    elif confidence_score >= 0.90:
        return ConfidenceLevel.HIGH
    elif confidence_score >= 0.80:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


@require(lambda insight_data: len(insight_data) > 0)
def prioritize_insights(insights: List[PredictiveInsight]) -> List[PredictiveInsight]:
    """Prioritize insights based on impact, confidence, and urgency."""
    def insight_score(insight: PredictiveInsight) -> float:
        # Combine impact and confidence with priority weighting
        priority_weights = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
        priority_weight = priority_weights.get(insight.priority_level, 0.5)
        
        return (insight.impact_score * 0.4 + 
                insight.confidence_score * 0.3 + 
                priority_weight * 0.3)
    
    return sorted(insights, key=insight_score, reverse=True)


def generate_capacity_recommendations(
    forecast: ResourceForecast,
    current_capacity: float,
    utilization_threshold: float = 0.8
) -> List[str]:
    """Generate capacity planning recommendations based on resource forecasts."""
    recommendations = []
    
    max_predicted = max(forecast.predicted_usage)
    projected_utilization = max_predicted / current_capacity
    
    if projected_utilization > utilization_threshold:
        excess_percentage = (projected_utilization - utilization_threshold) * 100
        recommendations.append(
            f"Consider increasing {forecast.resource_type} capacity by {excess_percentage:.1f}% "
            f"to maintain utilization below {utilization_threshold*100:.0f}%"
        )
    
    # Check for growth trend
    if forecast.growth_rate and forecast.growth_rate > 0.1:  # >10% growth
        recommendations.append(
            f"High growth rate detected ({forecast.growth_rate*100:.1f}%/month). "
            f"Plan for additional {forecast.resource_type} capacity within 3 months"
        )
    
    # Check for seasonal patterns
    if forecast.seasonality_patterns:
        recommendations.append(
            f"Seasonal patterns detected. Consider auto-scaling for {forecast.resource_type} "
            f"to handle periodic demand spikes"
        )
    
    return recommendations


def create_failure_mitigation_strategies(failure_prediction: FailurePrediction) -> List[str]:
    """Create mitigation strategies for predicted failures."""
    strategies = []
    
    failure_type = failure_prediction.failure_type.lower()
    
    if "execution" in failure_type:
        strategies.extend([
            "Implement pre-execution validation checks",
            "Add retry logic with exponential backoff",
            "Set up alternative execution paths"
        ])
    
    if "performance" in failure_type:
        strategies.extend([
            "Optimize resource allocation",
            "Implement performance monitoring",
            "Set up automatic scaling triggers"
        ])
    
    if "resource" in failure_type:
        strategies.extend([
            "Increase resource limits",
            "Implement resource cleanup procedures",
            "Add resource usage monitoring"
        ])
    
    # Add severity-specific strategies
    if failure_prediction.severity_level == "critical":
        strategies.insert(0, "Implement immediate failover mechanisms")
        strategies.append("Set up 24/7 monitoring alerts")
    
    return strategies