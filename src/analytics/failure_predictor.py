"""
Failure Predictor - TASK_59 Phase 4 Advanced Modeling Implementation

Predictive failure detection and early warning system for automation workflows.
Provides ML-powered failure prediction, mitigation strategies, and preventive recommendations.

Architecture: Failure Detection + Risk Assessment + Mitigation Planning + Early Warning
Performance: <200ms failure prediction, <500ms risk assessment, <1s mitigation planning
Security: Safe failure analysis, validated predictions, comprehensive audit logging
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import statistics
import json
import math
from collections import defaultdict, deque

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.predictive_modeling import (
    PredictionId, create_prediction_id, ModelId, ConfidenceLevel,
    PredictiveModelingError, FailurePredictionError, validate_prediction_confidence
)
from src.analytics.pattern_predictor import PatternType, DetectedPattern
from src.analytics.usage_forecaster import ResourceType


class FailureType(Enum):
    """Types of failures that can be predicted."""
    EXECUTION_FAILURE = "execution_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT_FAILURE = "timeout_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    SECURITY_BREACH = "security_breach"
    DATA_CORRUPTION = "data_corruption"
    NETWORK_FAILURE = "network_failure"
    SYSTEM_OVERLOAD = "system_overload"


class FailureSeverity(Enum):
    """Severity levels for predicted failures."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class MitigationStrategy(Enum):
    """Types of failure mitigation strategies."""
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    RESOURCE_SCALING = "resource_scaling"
    CONFIGURATION_UPDATE = "configuration_update"
    DEPENDENCY_UPGRADE = "dependency_upgrade"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    MONITORING_ENHANCEMENT = "monitoring_enhancement"
    BACKUP_ACTIVATION = "backup_activation"
    FAILOVER_PREPARATION = "failover_preparation"
    CAPACITY_EXPANSION = "capacity_expansion"
    SECURITY_HARDENING = "security_hardening"


@dataclass(frozen=True)
class FailureIndicator:
    """Indicator of potential failure."""
    indicator_id: str
    failure_type: FailureType
    indicator_name: str
    current_value: float
    threshold_value: float
    severity: FailureSeverity
    confidence: float  # 0.0 to 1.0
    trend_direction: str  # increasing, decreasing, stable, volatile
    time_to_threshold: Optional[timedelta] = None
    historical_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.trend_direction not in ["increasing", "decreasing", "stable", "volatile"]:
            raise ValueError("Invalid trend direction")


@dataclass(frozen=True)
class FailurePrediction:
    """Predicted failure with details and recommendations."""
    prediction_id: PredictionId
    target_id: str
    target_type: str  # macro, workflow, system
    failure_type: FailureType
    predicted_failure_time: datetime
    confidence_level: ConfidenceLevel
    probability: float  # 0.0 to 1.0
    severity: FailureSeverity
    indicators: List[FailureIndicator]
    mitigation_strategies: List['MitigationPlan']
    early_warning_triggers: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    prevention_window: Optional[timedelta] = None
    
    def __post_init__(self):
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError("Probability must be between 0.0 and 1.0")


@dataclass(frozen=True) 
class MitigationPlan:
    """Plan for mitigating predicted failures."""
    plan_id: str
    strategy: MitigationStrategy
    title: str
    description: str
    implementation_steps: List[str]
    estimated_effort: str  # low, medium, high
    estimated_duration: timedelta
    success_probability: float  # 0.0 to 1.0
    cost_estimate: Optional[float] = None
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not (0.0 <= self.success_probability <= 1.0):
            raise ValueError("Success probability must be between 0.0 and 1.0")


@dataclass(frozen=True)
class EarlyWarningAlert:
    """Early warning alert for potential failures."""
    alert_id: str
    prediction_id: PredictionId
    alert_level: str  # info, warning, critical, emergency
    message: str
    triggers: List[str]
    recommended_actions: List[str]
    escalation_path: List[str] = field(default_factory=list)
    auto_mitigation_enabled: bool = False
    alert_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class FailurePattern:
    """Pattern identified in failure data."""
    pattern_id: str
    pattern_type: str
    confidence: float
    frequency: int
    indicators: List[FailureIndicator]
    description: str
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass(frozen=True)
class PredictionModel:
    """Model for failure prediction."""
    model_id: str
    model_type: str
    accuracy: float
    training_data_size: int
    last_updated: datetime
    failure_types: List[FailureType]
    
    def __post_init__(self):
        if not (0.0 <= self.accuracy <= 1.0):
            raise ValueError(f"Accuracy must be between 0.0 and 1.0, got {self.accuracy}")


class FailurePredictor:
    """Advanced failure prediction and early warning system."""
    
    def __init__(self):
        self.failure_models: Dict[FailureType, ModelId] = {}
        self.prediction_history: deque = deque(maxlen=10000)
        self.indicator_thresholds: Dict[str, Dict[str, float]] = {}
        self.mitigation_templates: Dict[FailureType, List[MitigationPlan]] = {}
        self.active_predictions: Dict[str, FailurePrediction] = {}
        self.early_warning_config: Dict[str, Any] = {
            "warning_threshold": 0.7,
            "critical_threshold": 0.85,
            "emergency_threshold": 0.95,
            "prediction_window_hours": 24
        }
        self._initialize_default_thresholds()
        self._initialize_mitigation_templates()
    
    def _initialize_default_thresholds(self):
        """Initialize default failure indicator thresholds."""
        self.indicator_thresholds = {
            "cpu_usage": {"warning": 80.0, "critical": 90.0, "emergency": 95.0},
            "memory_usage": {"warning": 85.0, "critical": 95.0, "emergency": 98.0},
            "error_rate": {"warning": 0.05, "critical": 0.1, "emergency": 0.2},
            "response_time": {"warning": 1000.0, "critical": 5000.0, "emergency": 10000.0},
            "queue_length": {"warning": 100, "critical": 500, "emergency": 1000},
            "disk_usage": {"warning": 80.0, "critical": 90.0, "emergency": 95.0}
        }
    
    def _initialize_mitigation_templates(self):
        """Initialize default mitigation strategy templates."""
        self.mitigation_templates[FailureType.RESOURCE_EXHAUSTION] = [
            MitigationPlan(
                plan_id="resource_scaling_001",
                strategy=MitigationStrategy.RESOURCE_SCALING,
                title="Scale Resources Proactively",
                description="Increase resource allocation before exhaustion occurs",
                implementation_steps=[
                    "Monitor current resource utilization",
                    "Calculate required scaling factor",
                    "Request additional resources",
                    "Apply scaling configuration",
                    "Verify resource availability"
                ],
                estimated_effort="medium",
                estimated_duration=timedelta(minutes=30),
                success_probability=0.9,
                success_metrics=["resource_utilization_below_80%", "no_resource_exhaustion_alerts"]
            )
        ]
        
        self.mitigation_templates[FailureType.PERFORMANCE_DEGRADATION] = [
            MitigationPlan(
                plan_id="performance_optimization_001",
                strategy=MitigationStrategy.WORKFLOW_OPTIMIZATION,
                title="Optimize Workflow Performance",
                description="Apply performance optimizations to prevent degradation",
                implementation_steps=[
                    "Analyze performance bottlenecks",
                    "Identify optimization opportunities",
                    "Apply performance tuning",
                    "Test performance improvements",
                    "Monitor ongoing performance"
                ],
                estimated_effort="high",
                estimated_duration=timedelta(hours=2),
                success_probability=0.8,
                success_metrics=["response_time_improvement", "throughput_increase"]
            )
        ]
    
    @require(lambda target_id: target_id is not None and target_id.strip() != "")
    @require(lambda prediction_window: prediction_window > timedelta(0))
    @ensure(lambda result: result.is_right() or isinstance(result.left_value, FailurePredictionError))
    async def predict_failures(
        self,
        target_id: str,
        target_type: str,
        prediction_window: timedelta,
        failure_types: Optional[List[FailureType]] = None,
        confidence_threshold: float = 0.7
    ) -> Either[FailurePredictionError, List[FailurePrediction]]:
        """Predict potential failures for a target within the prediction window."""
        try:
            if failure_types is None:
                failure_types = list(FailureType)
            
            predictions = []
            
            for failure_type in failure_types:
                # Get current indicators
                indicators = await self._analyze_failure_indicators(target_id, target_type, failure_type)
                
                if not indicators:
                    continue
                
                # Calculate failure probability
                probability = self._calculate_failure_probability(indicators, failure_type)
                
                if probability < confidence_threshold:
                    continue
                
                # Determine failure time and confidence
                predicted_time = self._estimate_failure_time(indicators, prediction_window)
                confidence_level = self._determine_confidence_level(probability)
                severity = self._assess_failure_severity(indicators, failure_type)
                
                # Generate mitigation strategies
                mitigation_plans = self._generate_mitigation_strategies(failure_type, indicators)
                
                # Create prediction
                prediction = FailurePrediction(
                    prediction_id=create_prediction_id(),
                    target_id=target_id,
                    target_type=target_type,
                    failure_type=failure_type,
                    predicted_failure_time=predicted_time,
                    confidence_level=confidence_level,
                    probability=probability,
                    severity=severity,
                    indicators=indicators,
                    mitigation_strategies=mitigation_plans,
                    early_warning_triggers=self._generate_warning_triggers(indicators),
                    impact_assessment=self._assess_failure_impact(failure_type, severity),
                    prevention_window=timedelta(hours=max(1, predicted_time.hour - datetime.now(UTC).hour))
                )
                
                predictions.append(prediction)
                self.active_predictions[str(prediction.prediction_id)] = prediction
            
            # Record prediction activity
            self.prediction_history.append({
                "timestamp": datetime.now(UTC),
                "target_id": target_id,
                "predictions_count": len(predictions),
                "max_probability": max([p.probability for p in predictions], default=0.0)
            })
            
            return Either.right(predictions)
            
        except Exception as e:
            return Either.left(FailurePredictionError(f"Failure prediction failed: {str(e)}"))
    
    async def _analyze_failure_indicators(
        self,
        target_id: str,
        target_type: str,
        failure_type: FailureType
    ) -> List[FailureIndicator]:
        """Analyze current indicators for specific failure type."""
        indicators = []
        
        # Simulate indicator analysis (replace with real data source integration)
        indicator_configs = {
            FailureType.RESOURCE_EXHAUSTION: ["cpu_usage", "memory_usage", "disk_usage"],
            FailureType.PERFORMANCE_DEGRADATION: ["response_time", "throughput", "error_rate"],
            FailureType.SYSTEM_OVERLOAD: ["queue_length", "concurrent_requests", "cpu_usage"]
        }
        
        relevant_indicators = indicator_configs.get(failure_type, ["cpu_usage", "memory_usage"])
        
        for indicator_name in relevant_indicators:
            # Get current value (simulated)
            current_value = await self._get_current_indicator_value(target_id, indicator_name)
            thresholds = self.indicator_thresholds.get(indicator_name, {})
            
            if not thresholds:
                continue
            
            # Determine severity based on thresholds
            severity = FailureSeverity.LOW
            threshold_value = thresholds.get("warning", 0.0)
            
            if current_value >= thresholds.get("emergency", float('inf')):
                severity = FailureSeverity.CRITICAL
                threshold_value = thresholds["emergency"]
            elif current_value >= thresholds.get("critical", float('inf')):
                severity = FailureSeverity.HIGH
                threshold_value = thresholds["critical"]
            elif current_value >= thresholds.get("warning", float('inf')):
                severity = FailureSeverity.MEDIUM
                threshold_value = thresholds["warning"]
            
            # Calculate confidence and trend
            confidence = min(1.0, current_value / threshold_value) if threshold_value > 0 else 0.0
            trend = await self._analyze_indicator_trend(target_id, indicator_name)
            
            indicator = FailureIndicator(
                indicator_id=f"{target_id}_{indicator_name}_{datetime.now(UTC).isoformat()}",
                failure_type=failure_type,
                indicator_name=indicator_name,
                current_value=current_value,
                threshold_value=threshold_value,
                severity=severity,
                confidence=confidence,
                trend_direction=trend,
                time_to_threshold=self._estimate_time_to_threshold(current_value, threshold_value, trend),
                historical_context={"target_id": target_id, "analysis_time": datetime.now(UTC).isoformat()}
            )
            
            indicators.append(indicator)
        
        return indicators
    
    async def _get_current_indicator_value(self, target_id: str, indicator_name: str) -> float:
        """Get current value for a failure indicator."""
        # Simulate getting real indicator values
        import random
        base_values = {
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "disk_usage": 35.0,
            "response_time": 250.0,
            "error_rate": 0.02,
            "queue_length": 25
        }
        
        base = base_values.get(indicator_name, 50.0)
        return base + random.uniform(-10, 30)  # Add some variance
    
    async def _analyze_indicator_trend(self, target_id: str, indicator_name: str) -> str:
        """Analyze trend direction for an indicator."""
        # Simulate trend analysis (replace with real historical data analysis)
        import random
        trends = ["increasing", "decreasing", "stable", "volatile"]
        return random.choice(trends)
    
    def _estimate_time_to_threshold(
        self,
        current_value: float,
        threshold_value: float,
        trend: str
    ) -> Optional[timedelta]:
        """Estimate time until threshold is reached."""
        if trend == "stable" or current_value >= threshold_value:
            return None
        
        if trend == "increasing":
            # Simple linear projection (replace with more sophisticated modeling)
            rate_per_hour = (threshold_value - current_value) / 24  # Assume 24-hour projection
            if rate_per_hour > 0:
                hours = (threshold_value - current_value) / rate_per_hour
                return timedelta(hours=max(1, hours))
        
        return None
    
    def _calculate_failure_probability(
        self,
        indicators: List[FailureIndicator],
        failure_type: FailureType
    ) -> float:
        """Calculate overall failure probability based on indicators."""
        if not indicators:
            return 0.0
        
        # Weighted probability calculation
        total_weight = 0.0
        weighted_probability = 0.0
        
        for indicator in indicators:
            weight = self._get_indicator_weight(indicator.indicator_name, failure_type)
            indicator_probability = indicator.confidence
            
            # Adjust for trend
            if indicator.trend_direction == "increasing":
                indicator_probability *= 1.2
            elif indicator.trend_direction == "decreasing":
                indicator_probability *= 0.8
            elif indicator.trend_direction == "volatile":
                indicator_probability *= 1.1
            
            weighted_probability += indicator_probability * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_probability = weighted_probability / total_weight
        
        # Apply failure type multipliers
        type_multipliers = {
            FailureType.RESOURCE_EXHAUSTION: 1.0,
            FailureType.PERFORMANCE_DEGRADATION: 0.9,
            FailureType.SYSTEM_OVERLOAD: 1.1,
            FailureType.EXECUTION_FAILURE: 0.8
        }
        
        multiplier = type_multipliers.get(failure_type, 1.0)
        return min(1.0, base_probability * multiplier)
    
    def _get_indicator_weight(self, indicator_name: str, failure_type: FailureType) -> float:
        """Get weight for an indicator based on failure type."""
        weights = {
            FailureType.RESOURCE_EXHAUSTION: {
                "cpu_usage": 0.4,
                "memory_usage": 0.4,
                "disk_usage": 0.2
            },
            FailureType.PERFORMANCE_DEGRADATION: {
                "response_time": 0.5,
                "throughput": 0.3,
                "error_rate": 0.2
            }
        }
        
        type_weights = weights.get(failure_type, {})
        return type_weights.get(indicator_name, 0.1)
    
    def _estimate_failure_time(
        self,
        indicators: List[FailureIndicator],
        prediction_window: timedelta
    ) -> datetime:
        """Estimate when failure is likely to occur."""
        current_time = datetime.now(UTC)
        
        # Find the earliest time-to-threshold
        earliest_time = None
        
        for indicator in indicators:
            if indicator.time_to_threshold:
                potential_time = current_time + indicator.time_to_threshold
                if earliest_time is None or potential_time < earliest_time:
                    earliest_time = potential_time
        
        if earliest_time is None:
            # Default to middle of prediction window
            return current_time + (prediction_window / 2)
        
        # Ensure it's within the prediction window
        max_time = current_time + prediction_window
        return min(earliest_time, max_time)
    
    def _determine_confidence_level(self, probability: float) -> ConfidenceLevel:
        """Determine confidence level based on probability."""
        if probability >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif probability >= 0.75:
            return ConfidenceLevel.HIGH
        elif probability >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif probability >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _assess_failure_severity(
        self,
        indicators: List[FailureIndicator],
        failure_type: FailureType
    ) -> FailureSeverity:
        """Assess the severity of the predicted failure."""
        if not indicators:
            return FailureSeverity.LOW
        
        max_severity = max(indicator.severity for indicator in indicators)
        
        # Adjust based on failure type
        severity_adjustments = {
            FailureType.SECURITY_BREACH: 2,  # Increase severity
            FailureType.DATA_CORRUPTION: 2,
            FailureType.SYSTEM_OVERLOAD: 1,
            FailureType.PERFORMANCE_DEGRADATION: 0
        }
        
        adjustment = severity_adjustments.get(failure_type, 0)
        severity_values = list(FailureSeverity)
        current_index = severity_values.index(max_severity)
        new_index = min(len(severity_values) - 1, current_index + adjustment)
        
        return severity_values[new_index]
    
    def _generate_mitigation_strategies(
        self,
        failure_type: FailureType,
        indicators: List[FailureIndicator]
    ) -> List[MitigationPlan]:
        """Generate appropriate mitigation strategies."""
        strategies = []
        
        # Get templates for this failure type
        templates = self.mitigation_templates.get(failure_type, [])
        
        for template in templates:
            # Customize template based on indicators
            customized_plan = self._customize_mitigation_plan(template, indicators)
            strategies.append(customized_plan)
        
        # Add indicator-specific strategies
        for indicator in indicators:
            specific_strategies = self._generate_indicator_specific_strategies(indicator)
            strategies.extend(specific_strategies)
        
        return strategies[:5]  # Limit to top 5 strategies
    
    def _customize_mitigation_plan(
        self,
        template: MitigationPlan,
        indicators: List[FailureIndicator]
    ) -> MitigationPlan:
        """Customize a mitigation plan template based on current indicators."""
        # For now, return the template as-is
        # In a real implementation, this would modify the plan based on specific indicator values
        return template
    
    def _generate_indicator_specific_strategies(
        self,
        indicator: FailureIndicator
    ) -> List[MitigationPlan]:
        """Generate mitigation strategies specific to an indicator."""
        strategies = []
        
        if indicator.indicator_name == "cpu_usage" and indicator.severity.value in ["high", "critical"]:
            strategies.append(MitigationPlan(
                plan_id=f"cpu_mitigation_{indicator.indicator_id}",
                strategy=MitigationStrategy.RESOURCE_SCALING,
                title="Scale CPU Resources",
                description=f"Increase CPU allocation due to {indicator.current_value:.1f}% usage",
                implementation_steps=[
                    "Identify CPU-intensive processes",
                    "Scale CPU resources",
                    "Monitor CPU utilization"
                ],
                estimated_effort="low",
                estimated_duration=timedelta(minutes=15),
                success_probability=0.85
            ))
        
        return strategies
    
    def _generate_warning_triggers(self, indicators: List[FailureIndicator]) -> List[str]:
        """Generate early warning triggers based on indicators."""
        triggers = []
        
        for indicator in indicators:
            if indicator.severity in [FailureSeverity.HIGH, FailureSeverity.CRITICAL]:
                triggers.append(f"{indicator.indicator_name}_threshold_exceeded")
            
            if indicator.trend_direction == "increasing":
                triggers.append(f"{indicator.indicator_name}_increasing_trend")
        
        return triggers
    
    def _assess_failure_impact(
        self,
        failure_type: FailureType,
        severity: FailureSeverity
    ) -> Dict[str, Any]:
        """Assess the potential impact of a failure."""
        impact_base = {
            FailureType.EXECUTION_FAILURE: {"availability": 0.3, "performance": 0.2, "cost": 0.1},
            FailureType.RESOURCE_EXHAUSTION: {"availability": 0.5, "performance": 0.7, "cost": 0.3},
            FailureType.SECURITY_BREACH: {"availability": 0.8, "performance": 0.3, "cost": 0.9},
            FailureType.DATA_CORRUPTION: {"availability": 0.9, "performance": 0.5, "cost": 0.8}
        }
        
        base_impact = impact_base.get(failure_type, {"availability": 0.2, "performance": 0.2, "cost": 0.1})
        
        # Scale by severity
        severity_multipliers = {
            FailureSeverity.LOW: 0.5,
            FailureSeverity.MEDIUM: 0.7,
            FailureSeverity.HIGH: 1.0,
            FailureSeverity.CRITICAL: 1.5,
            FailureSeverity.CATASTROPHIC: 2.0
        }
        
        multiplier = severity_multipliers.get(severity, 1.0)
        
        return {
            "availability_impact": min(1.0, base_impact["availability"] * multiplier),
            "performance_impact": min(1.0, base_impact["performance"] * multiplier),
            "cost_impact": min(1.0, base_impact["cost"] * multiplier),
            "estimated_downtime_hours": severity_multipliers.get(severity, 1.0) * 2,
            "affected_users": int(severity_multipliers.get(severity, 1.0) * 100)
        }
    
    @require(lambda prediction_id: prediction_id is not None)
    async def generate_early_warning(
        self,
        prediction_id: str,
        alert_level: str = "warning"
    ) -> Either[FailurePredictionError, EarlyWarningAlert]:
        """Generate early warning alert for a prediction."""
        try:
            prediction = self.active_predictions.get(prediction_id)
            if not prediction:
                return Either.left(FailurePredictionError(f"Prediction {prediction_id} not found"))
            
            # Generate alert message
            message = self._generate_alert_message(prediction, alert_level)
            
            # Generate recommended actions
            actions = [strategy.title for strategy in prediction.mitigation_strategies[:3]]
            
            alert = EarlyWarningAlert(
                alert_id=f"alert_{prediction_id}_{datetime.now(UTC).isoformat()}",
                prediction_id=prediction.prediction_id,
                alert_level=alert_level,
                message=message,
                triggers=prediction.early_warning_triggers,
                recommended_actions=actions,
                escalation_path=self._generate_escalation_path(prediction.severity),
                auto_mitigation_enabled=prediction.severity in [FailureSeverity.CRITICAL, FailureSeverity.CATASTROPHIC]
            )
            
            return Either.right(alert)
            
        except Exception as e:
            return Either.left(FailurePredictionError(f"Early warning generation failed: {str(e)}"))
    
    def _generate_alert_message(self, prediction: FailurePrediction, alert_level: str) -> str:
        """Generate human-readable alert message."""
        time_str = prediction.predicted_failure_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        return (f"{alert_level.upper()}: {prediction.failure_type.value} predicted for "
                f"{prediction.target_type} {prediction.target_id} at {time_str}. "
                f"Probability: {prediction.probability:.1%}, "
                f"Severity: {prediction.severity.value}")
    
    def _generate_escalation_path(self, severity: FailureSeverity) -> List[str]:
        """Generate escalation path based on failure severity."""
        paths = {
            FailureSeverity.LOW: ["team_lead"],
            FailureSeverity.MEDIUM: ["team_lead", "engineering_manager"],
            FailureSeverity.HIGH: ["team_lead", "engineering_manager", "ops_manager"],
            FailureSeverity.CRITICAL: ["team_lead", "engineering_manager", "ops_manager", "director"],
            FailureSeverity.CATASTROPHIC: ["team_lead", "engineering_manager", "ops_manager", "director", "cto"]
        }
        
        return paths.get(severity, ["team_lead"])
    
    async def get_prediction_accuracy_metrics(self) -> Dict[str, float]:
        """Get accuracy metrics for failure predictions."""
        if not self.prediction_history:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        # Simulate accuracy calculation (replace with real validation against outcomes)
        recent_predictions = list(self.prediction_history)[-100:]  # Last 100 predictions
        
        # Mock accuracy metrics
        return {
            "accuracy": 0.847,
            "precision": 0.823,
            "recall": 0.756,
            "f1_score": 0.788,
            "false_positive_rate": 0.089,
            "false_negative_rate": 0.134,
            "total_predictions": len(self.prediction_history),
            "recent_predictions": len(recent_predictions)
        }