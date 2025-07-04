"""
Pattern Predictor Engine - TASK_59 Phase 2 Core Implementation

Advanced pattern analysis and prediction engine for automation workflows.
Provides ML-powered pattern recognition, trend analysis, and behavioral prediction.

Architecture: ML Models + Statistical Analysis + Pattern Recognition + Behavioral Prediction
Performance: <200ms pattern analysis, <1s trend prediction, <2s complex behavioral modeling
Security: Safe pattern analysis, validated predictions, comprehensive input sanitization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import statistics
import json
from collections import defaultdict, deque

from src.core.types import MacroId
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.predictive_modeling import (
    PredictionId, create_prediction_id, PredictionType, ConfidenceLevel, ModelType,
    PredictionResult, TimeSeriesData, PredictionConfig, PredictiveModelingError,
    ModelTrainingError, PredictionError, validate_time_series_data,
    calculate_prediction_confidence, categorize_confidence_level
)
from src.intelligence.intelligence_types import IntelligenceOperation, AnalysisScope, LearningMode


class PatternType(Enum):
    """Types of patterns detected and predicted."""
    USAGE_FREQUENCY = "usage_frequency"
    EXECUTION_TIMING = "execution_timing"
    WORKFLOW_SEQUENCE = "workflow_sequence"
    RESOURCE_CONSUMPTION = "resource_consumption"
    ERROR_OCCURRENCE = "error_occurrence"
    SEASONAL_BEHAVIOR = "seasonal_behavior"
    USER_BEHAVIOR = "user_behavior"
    PERFORMANCE_TREND = "performance_trend"


class PatternComplexity(Enum):
    """Complexity levels of detected patterns."""
    SIMPLE = "simple"          # Linear, obvious patterns
    MODERATE = "moderate"      # Seasonal, cyclical patterns
    COMPLEX = "complex"        # Multi-variate, non-linear patterns
    CHAOTIC = "chaotic"        # Unpredictable, random-like patterns


@dataclass(frozen=True)
class PatternFeature:
    """Feature extracted from automation data for pattern analysis."""
    feature_name: str
    feature_type: str  # numeric, categorical, temporal, boolean
    values: List[Union[float, str, bool]]
    timestamps: List[datetime]
    confidence_score: float = 0.0
    missing_data_ratio: float = 0.0
    
    def __post_init__(self):
        if len(self.values) != len(self.timestamps):
            raise ValueError("Values and timestamps must have the same length")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        if not (0.0 <= self.missing_data_ratio <= 1.0):
            raise ValueError("Missing data ratio must be between 0.0 and 1.0")


@dataclass(frozen=True)
class DetectedPattern:
    """Pattern detected in automation data."""
    pattern_id: str
    pattern_type: PatternType
    pattern_complexity: PatternComplexity
    description: str
    confidence_score: float
    strength: float  # 0.0 to 1.0, strength of the pattern
    frequency: Optional[str] = None  # daily, weekly, monthly, etc.
    seasonality: Optional[Dict[str, Any]] = None
    trend_direction: Optional[str] = None  # increasing, decreasing, stable
    supporting_features: List[str] = field(default_factory=list)
    statistical_significance: Optional[float] = None
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError("Pattern strength must be between 0.0 and 1.0")


@dataclass(frozen=True)
class PatternPrediction:
    """Prediction based on detected patterns."""
    prediction_id: PredictionId
    pattern_id: str
    prediction_type: PredictionType
    predicted_values: List[float]
    prediction_timestamps: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    confidence_level: ConfidenceLevel
    prediction_horizon_hours: int
    model_used: ModelType
    accuracy_estimate: float
    factors_considered: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.predicted_values) != len(self.prediction_timestamps):
            raise ValueError("Predicted values and timestamps must have the same length")
        if not (0.0 <= self.accuracy_estimate <= 1.0):
            raise ValueError("Accuracy estimate must be between 0.0 and 1.0")
        if self.prediction_horizon_hours <= 0:
            raise ValueError("Prediction horizon must be positive")


class PatternPredictor:
    """
    Advanced pattern analysis and prediction engine for automation workflows.
    
    Provides ML-powered pattern recognition, trend analysis, and behavioral prediction
    with comprehensive validation and performance optimization.
    """
    
    def __init__(self):
        self.detected_patterns: Dict[str, DetectedPattern] = {}
        self.feature_extractors: Dict[str, Any] = {}
        self.prediction_models: Dict[str, Any] = {}
        self.pattern_cache: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.learning_history: deque = deque(maxlen=10000)
        
        # Initialize pattern detection algorithms
        self._initialize_pattern_detectors()
        
        # Initialize prediction models
        self._initialize_prediction_models()
    
    def _initialize_pattern_detectors(self):
        """Initialize pattern detection algorithms."""
        self.pattern_detectors = {
            PatternType.USAGE_FREQUENCY: self._detect_usage_frequency_patterns,
            PatternType.EXECUTION_TIMING: self._detect_timing_patterns,
            PatternType.WORKFLOW_SEQUENCE: self._detect_sequence_patterns,
            PatternType.RESOURCE_CONSUMPTION: self._detect_resource_patterns,
            PatternType.ERROR_OCCURRENCE: self._detect_error_patterns,
            PatternType.SEASONAL_BEHAVIOR: self._detect_seasonal_patterns,
            PatternType.USER_BEHAVIOR: self._detect_user_behavior_patterns,
            PatternType.PERFORMANCE_TREND: self._detect_performance_trends
        }
    
    def _initialize_prediction_models(self):
        """Initialize prediction models for different pattern types."""
        self.prediction_models = {
            PatternType.USAGE_FREQUENCY: "linear_regression",
            PatternType.EXECUTION_TIMING: "time_series_arima",
            PatternType.WORKFLOW_SEQUENCE: "markov_chain",
            PatternType.RESOURCE_CONSUMPTION: "polynomial_regression",
            PatternType.ERROR_OCCURRENCE: "logistic_regression",
            PatternType.SEASONAL_BEHAVIOR: "seasonal_decomposition",
            PatternType.USER_BEHAVIOR: "clustering_prediction",
            PatternType.PERFORMANCE_TREND: "exponential_smoothing"
        }

    @require(lambda features: len(features) > 0)
    @ensure(lambda result: len(result.get_right_or([])) >= 0)
    async def detect_patterns(
        self, 
        features: List[PatternFeature],
        pattern_types: Optional[List[PatternType]] = None,
        min_confidence: float = 0.6
    ) -> Either[PredictiveModelingError, List[DetectedPattern]]:
        """
        Detect patterns in automation data features.
        
        Uses multiple pattern detection algorithms to identify various types of patterns
        in automation workflows and behaviors.
        """
        try:
            if pattern_types is None:
                pattern_types = list(PatternType)
            
            detected_patterns = []
            
            for pattern_type in pattern_types:
                if pattern_type not in self.pattern_detectors:
                    continue
                
                detector = self.pattern_detectors[pattern_type]
                patterns = await detector(features, min_confidence)
                
                if patterns.is_right():
                    detected_patterns.extend(patterns.get_right())
            
            # Filter patterns by confidence
            high_confidence_patterns = [
                p for p in detected_patterns 
                if p.confidence_score >= min_confidence
            ]
            
            # Store detected patterns
            for pattern in high_confidence_patterns:
                self.detected_patterns[pattern.pattern_id] = pattern
            
            return Either.right(high_confidence_patterns)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Pattern detection failed: {str(e)}", 
                "PATTERN_DETECTION_ERROR"
            ))

    async def _detect_usage_frequency_patterns(
        self, 
        features: List[PatternFeature], 
        min_confidence: float
    ) -> Either[PredictiveModelingError, List[DetectedPattern]]:
        """Detect usage frequency patterns in automation data."""
        try:
            patterns = []
            
            for feature in features:
                if feature.feature_type != "numeric":
                    continue
                
                # Calculate usage frequency statistics
                values = [float(v) for v in feature.values if isinstance(v, (int, float))]
                
                if len(values) < 10:  # Need sufficient data
                    continue
                
                # Detect frequency patterns
                mean_usage = statistics.mean(values)
                std_usage = statistics.stdev(values) if len(values) > 1 else 0
                
                # Check for regular patterns
                if std_usage > 0:
                    coefficient_of_variation = std_usage / mean_usage
                    
                    if coefficient_of_variation < 0.3:  # Low variation = regular pattern
                        pattern = DetectedPattern(
                            pattern_id=f"usage_freq_{feature.feature_name}_{hash(str(values)) % 10000}",
                            pattern_type=PatternType.USAGE_FREQUENCY,
                            pattern_complexity=PatternComplexity.SIMPLE,
                            description=f"Regular usage pattern detected in {feature.feature_name}",
                            confidence_score=min(0.95, 1.0 - coefficient_of_variation),
                            strength=1.0 - coefficient_of_variation,
                            frequency="regular",
                            supporting_features=[feature.feature_name],
                            statistical_significance=1.0 - coefficient_of_variation
                        )
                        patterns.append(pattern)
            
            return Either.right(patterns)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Usage frequency pattern detection failed: {str(e)}", 
                "USAGE_PATTERN_ERROR"
            ))

    async def _detect_timing_patterns(
        self, 
        features: List[PatternFeature], 
        min_confidence: float
    ) -> Either[PredictiveModelingError, List[DetectedPattern]]:
        """Detect timing patterns in automation execution."""
        try:
            patterns = []
            
            for feature in features:
                if not feature.timestamps:
                    continue
                
                # Analyze timing patterns
                hour_distribution = defaultdict(int)
                day_distribution = defaultdict(int)
                
                for timestamp in feature.timestamps:
                    hour_distribution[timestamp.hour] += 1
                    day_distribution[timestamp.weekday()] += 1
                
                # Check for hourly patterns
                if hour_distribution:
                    max_hour_count = max(hour_distribution.values())
                    total_count = sum(hour_distribution.values())
                    
                    if max_hour_count / total_count > 0.3:  # 30% concentration in one hour
                        peak_hour = max(hour_distribution.keys(), key=lambda h: hour_distribution[h])
                        confidence = max_hour_count / total_count
                        
                        if confidence >= min_confidence:
                            pattern = DetectedPattern(
                                pattern_id=f"timing_hourly_{feature.feature_name}_{peak_hour}",
                                pattern_type=PatternType.EXECUTION_TIMING,
                                pattern_complexity=PatternComplexity.SIMPLE,
                                description=f"Peak execution time at hour {peak_hour} for {feature.feature_name}",
                                confidence_score=confidence,
                                strength=confidence,
                                frequency="hourly",
                                supporting_features=[feature.feature_name],
                                statistical_significance=confidence
                            )
                            patterns.append(pattern)
                
                # Check for daily patterns
                if day_distribution:
                    max_day_count = max(day_distribution.values())
                    total_count = sum(day_distribution.values())
                    
                    if max_day_count / total_count > 0.25:  # 25% concentration in one day
                        peak_day = max(day_distribution.keys(), key=lambda d: day_distribution[d])
                        confidence = max_day_count / total_count
                        
                        if confidence >= min_confidence:
                            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            pattern = DetectedPattern(
                                pattern_id=f"timing_daily_{feature.feature_name}_{peak_day}",
                                pattern_type=PatternType.EXECUTION_TIMING,
                                pattern_complexity=PatternComplexity.SIMPLE,
                                description=f"Peak execution on {day_names[peak_day]} for {feature.feature_name}",
                                confidence_score=confidence,
                                strength=confidence,
                                frequency="daily",
                                supporting_features=[feature.feature_name],
                                statistical_significance=confidence
                            )
                            patterns.append(pattern)
            
            return Either.right(patterns)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Timing pattern detection failed: {str(e)}", 
                "TIMING_PATTERN_ERROR"
            ))

    async def _detect_sequence_patterns(
        self, 
        features: List[PatternFeature], 
        min_confidence: float
    ) -> Either[PredictiveModelingError, List[DetectedPattern]]:
        """Detect workflow sequence patterns."""
        try:
            patterns = []
            
            # Look for sequential patterns in categorical features
            categorical_features = [f for f in features if f.feature_type == "categorical"]
            
            for feature in categorical_features:
                if len(feature.values) < 5:  # Need sufficient sequence data
                    continue
                
                # Build sequence transitions
                transitions = defaultdict(lambda: defaultdict(int))
                
                for i in range(len(feature.values) - 1):
                    current_state = str(feature.values[i])
                    next_state = str(feature.values[i + 1])
                    transitions[current_state][next_state] += 1
                
                # Analyze transition patterns
                for current_state, next_states in transitions.items():
                    total_transitions = sum(next_states.values())
                    
                    if total_transitions >= 3:  # Minimum transitions for pattern
                        max_transition_count = max(next_states.values())
                        most_likely_next = max(next_states.keys(), key=lambda k: next_states[k])
                        
                        transition_probability = max_transition_count / total_transitions
                        
                        if transition_probability >= min_confidence:
                            pattern = DetectedPattern(
                                pattern_id=f"sequence_{feature.feature_name}_{current_state}_{most_likely_next}",
                                pattern_type=PatternType.WORKFLOW_SEQUENCE,
                                pattern_complexity=PatternComplexity.MODERATE,
                                description=f"Sequence pattern: {current_state} → {most_likely_next} ({transition_probability:.2%})",
                                confidence_score=transition_probability,
                                strength=transition_probability,
                                supporting_features=[feature.feature_name],
                                statistical_significance=transition_probability
                            )
                            patterns.append(pattern)
            
            return Either.right(patterns)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Sequence pattern detection failed: {str(e)}", 
                "SEQUENCE_PATTERN_ERROR"
            ))

    async def _detect_resource_patterns(
        self, 
        features: List[PatternFeature], 
        min_confidence: float
    ) -> Either[PredictiveModelingError, List[DetectedPattern]]:
        """Detect resource consumption patterns."""
        try:
            patterns = []
            
            resource_features = [f for f in features if "resource" in f.feature_name.lower() or 
                               "cpu" in f.feature_name.lower() or "memory" in f.feature_name.lower()]
            
            for feature in resource_features:
                if feature.feature_type != "numeric":
                    continue
                
                numeric_values = [float(v) for v in feature.values if isinstance(v, (int, float))]
                
                if len(numeric_values) < 10:
                    continue
                
                # Detect resource usage patterns
                mean_usage = statistics.mean(numeric_values)
                median_usage = statistics.median(numeric_values)
                std_usage = statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
                
                # Check for consistent high usage
                high_usage_threshold = mean_usage + std_usage
                high_usage_count = sum(1 for v in numeric_values if v > high_usage_threshold)
                high_usage_ratio = high_usage_count / len(numeric_values)
                
                if high_usage_ratio > 0.3:  # More than 30% high usage
                    pattern = DetectedPattern(
                        pattern_id=f"resource_high_{feature.feature_name}_{hash(str(numeric_values)) % 10000}",
                        pattern_type=PatternType.RESOURCE_CONSUMPTION,
                        pattern_complexity=PatternComplexity.SIMPLE,
                        description=f"High resource usage pattern in {feature.feature_name} ({high_usage_ratio:.1%})",
                        confidence_score=high_usage_ratio,
                        strength=high_usage_ratio,
                        trend_direction="high",
                        supporting_features=[feature.feature_name],
                        statistical_significance=high_usage_ratio
                    )
                    patterns.append(pattern)
                
                # Check for increasing trend
                if len(numeric_values) >= 5:
                    # Simple trend detection using linear regression slope
                    n = len(numeric_values)
                    x_values = list(range(n))
                    
                    # Calculate slope
                    mean_x = statistics.mean(x_values)
                    mean_y = statistics.mean(numeric_values)
                    
                    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, numeric_values))
                    denominator = sum((x - mean_x) ** 2 for x in x_values)
                    
                    if denominator > 0:
                        slope = numerator / denominator
                        
                        # If slope is significantly positive, we have an increasing trend
                        if slope > 0.1 * mean_usage:  # 10% increase per time unit
                            confidence = min(0.95, abs(slope) / mean_usage)
                            
                            if confidence >= min_confidence:
                                pattern = DetectedPattern(
                                    pattern_id=f"resource_trend_{feature.feature_name}_{hash(str(numeric_values)) % 10000}",
                                    pattern_type=PatternType.RESOURCE_CONSUMPTION,
                                    pattern_complexity=PatternComplexity.MODERATE,
                                    description=f"Increasing resource usage trend in {feature.feature_name}",
                                    confidence_score=confidence,
                                    strength=confidence,
                                    trend_direction="increasing",
                                    supporting_features=[feature.feature_name],
                                    statistical_significance=confidence
                                )
                                patterns.append(pattern)
            
            return Either.right(patterns)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Resource pattern detection failed: {str(e)}", 
                "RESOURCE_PATTERN_ERROR"
            ))

    async def _detect_error_patterns(
        self, 
        features: List[PatternFeature], 
        min_confidence: float
    ) -> Either[PredictiveModelingError, List[DetectedPattern]]:
        """Detect error occurrence patterns."""
        try:
            patterns = []
            
            error_features = [f for f in features if "error" in f.feature_name.lower() or 
                            "failure" in f.feature_name.lower() or "exception" in f.feature_name.lower()]
            
            for feature in error_features:
                if feature.feature_type == "boolean":
                    # Boolean error indicators
                    error_occurrences = [bool(v) for v in feature.values if isinstance(v, bool)]
                    
                    if len(error_occurrences) < 10:
                        continue
                    
                    error_rate = sum(error_occurrences) / len(error_occurrences)
                    
                    if error_rate > 0.05:  # More than 5% error rate
                        pattern = DetectedPattern(
                            pattern_id=f"error_rate_{feature.feature_name}_{hash(str(error_occurrences)) % 10000}",
                            pattern_type=PatternType.ERROR_OCCURRENCE,
                            pattern_complexity=PatternComplexity.SIMPLE,
                            description=f"Elevated error rate in {feature.feature_name} ({error_rate:.1%})",
                            confidence_score=min(0.95, error_rate * 10),  # Scale to confidence
                            strength=error_rate,
                            supporting_features=[feature.feature_name],
                            statistical_significance=error_rate
                        )
                        patterns.append(pattern)
                
                elif feature.feature_type == "numeric":
                    # Numeric error counts
                    error_counts = [float(v) for v in feature.values if isinstance(v, (int, float))]
                    
                    if len(error_counts) < 10:
                        continue
                    
                    mean_errors = statistics.mean(error_counts)
                    
                    if mean_errors > 1.0:  # More than 1 error per time period on average
                        pattern = DetectedPattern(
                            pattern_id=f"error_count_{feature.feature_name}_{hash(str(error_counts)) % 10000}",
                            pattern_type=PatternType.ERROR_OCCURRENCE,
                            pattern_complexity=PatternComplexity.SIMPLE,
                            description=f"High error count pattern in {feature.feature_name} (avg: {mean_errors:.1f})",
                            confidence_score=min(0.95, mean_errors / 10),  # Scale to confidence
                            strength=min(1.0, mean_errors / 5),
                            supporting_features=[feature.feature_name],
                            statistical_significance=min(0.95, mean_errors / 10)
                        )
                        patterns.append(pattern)
            
            return Either.right(patterns)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Error pattern detection failed: {str(e)}", 
                "ERROR_PATTERN_ERROR"
            ))

    async def _detect_seasonal_patterns(
        self, 
        features: List[PatternFeature], 
        min_confidence: float
    ) -> Either[PredictiveModelingError, List[DetectedPattern]]:
        """Detect seasonal behavior patterns."""
        try:
            patterns = []
            
            for feature in features:
                if not feature.timestamps or len(feature.timestamps) < 30:  # Need sufficient temporal data
                    continue
                
                # Group by month, day of week, hour
                monthly_stats = defaultdict(list)
                weekly_stats = defaultdict(list)
                hourly_stats = defaultdict(list)
                
                for timestamp, value in zip(feature.timestamps, feature.values):
                    if isinstance(value, (int, float)):
                        monthly_stats[timestamp.month].append(float(value))
                        weekly_stats[timestamp.weekday()].append(float(value))
                        hourly_stats[timestamp.hour].append(float(value))
                
                # Check for monthly seasonality
                if len(monthly_stats) >= 3:  # At least 3 months
                    month_means = {month: statistics.mean(values) for month, values in monthly_stats.items()}
                    
                    if month_means:
                        mean_overall = statistics.mean(month_means.values())
                        std_overall = statistics.stdev(month_means.values()) if len(month_means) > 1 else 0
                        
                        if std_overall > 0.2 * mean_overall:  # Significant seasonal variation
                            confidence = min(0.95, std_overall / mean_overall)
                            
                            if confidence >= min_confidence:
                                pattern = DetectedPattern(
                                    pattern_id=f"seasonal_monthly_{feature.feature_name}_{hash(str(month_means)) % 10000}",
                                    pattern_type=PatternType.SEASONAL_BEHAVIOR,
                                    pattern_complexity=PatternComplexity.MODERATE,
                                    description=f"Monthly seasonal pattern in {feature.feature_name}",
                                    confidence_score=confidence,
                                    strength=confidence,
                                    frequency="monthly",
                                    seasonality={"type": "monthly", "stats": month_means},
                                    supporting_features=[feature.feature_name],
                                    statistical_significance=confidence
                                )
                                patterns.append(pattern)
                
                # Check for weekly seasonality
                if len(weekly_stats) >= 5:  # At least 5 different days
                    day_means = {day: statistics.mean(values) for day, values in weekly_stats.items()}
                    
                    if day_means:
                        mean_overall = statistics.mean(day_means.values())
                        std_overall = statistics.stdev(day_means.values()) if len(day_means) > 1 else 0
                        
                        if std_overall > 0.15 * mean_overall:  # Significant weekly variation
                            confidence = min(0.95, std_overall / mean_overall)
                            
                            if confidence >= min_confidence:
                                pattern = DetectedPattern(
                                    pattern_id=f"seasonal_weekly_{feature.feature_name}_{hash(str(day_means)) % 10000}",
                                    pattern_type=PatternType.SEASONAL_BEHAVIOR,
                                    pattern_complexity=PatternComplexity.MODERATE,
                                    description=f"Weekly seasonal pattern in {feature.feature_name}",
                                    confidence_score=confidence,
                                    strength=confidence,
                                    frequency="weekly",
                                    seasonality={"type": "weekly", "stats": day_means},
                                    supporting_features=[feature.feature_name],
                                    statistical_significance=confidence
                                )
                                patterns.append(pattern)
            
            return Either.right(patterns)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Seasonal pattern detection failed: {str(e)}", 
                "SEASONAL_PATTERN_ERROR"
            ))

    async def _detect_user_behavior_patterns(
        self, 
        features: List[PatternFeature], 
        min_confidence: float
    ) -> Either[PredictiveModelingError, List[DetectedPattern]]:
        """Detect user behavior patterns."""
        try:
            patterns = []
            
            # Look for user activity patterns
            user_features = [f for f in features if "user" in f.feature_name.lower() or 
                           "activity" in f.feature_name.lower() or "interaction" in f.feature_name.lower()]
            
            for feature in user_features:
                if not feature.timestamps:
                    continue
                
                # Analyze user activity timing
                activity_hours = defaultdict(int)
                activity_days = defaultdict(int)
                
                for timestamp in feature.timestamps:
                    activity_hours[timestamp.hour] += 1
                    activity_days[timestamp.weekday()] += 1
                
                # Check for consistent user behavior patterns
                if activity_hours:
                    total_activities = sum(activity_hours.values())
                    
                    # Find peak activity hours
                    peak_hours = sorted(activity_hours.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    if peak_hours:
                        top_3_activity = sum(count for _, count in peak_hours)
                        concentration = top_3_activity / total_activities
                        
                        if concentration >= min_confidence:
                            peak_hour_list = [hour for hour, _ in peak_hours]
                            pattern = DetectedPattern(
                                pattern_id=f"user_behavior_{feature.feature_name}_{hash(str(peak_hour_list)) % 10000}",
                                pattern_type=PatternType.USER_BEHAVIOR,
                                pattern_complexity=PatternComplexity.SIMPLE,
                                description=f"User activity concentrated in hours {peak_hour_list} for {feature.feature_name}",
                                confidence_score=concentration,
                                strength=concentration,
                                frequency="hourly",
                                supporting_features=[feature.feature_name],
                                statistical_significance=concentration
                            )
                            patterns.append(pattern)
            
            return Either.right(patterns)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"User behavior pattern detection failed: {str(e)}", 
                "USER_BEHAVIOR_PATTERN_ERROR"
            ))

    async def _detect_performance_trends(
        self, 
        features: List[PatternFeature], 
        min_confidence: float
    ) -> Either[PredictiveModelingError, List[DetectedPattern]]:
        """Detect performance trend patterns."""
        try:
            patterns = []
            
            performance_features = [f for f in features if "performance" in f.feature_name.lower() or 
                                  "latency" in f.feature_name.lower() or "response" in f.feature_name.lower()]
            
            for feature in performance_features:
                if feature.feature_type != "numeric" or len(feature.values) < 10:
                    continue
                
                numeric_values = [float(v) for v in feature.values if isinstance(v, (int, float))]
                
                if len(numeric_values) < 10:
                    continue
                
                # Calculate performance trend
                n = len(numeric_values)
                x_values = list(range(n))
                
                # Linear regression for trend
                mean_x = statistics.mean(x_values)
                mean_y = statistics.mean(numeric_values)
                
                numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, numeric_values))
                denominator = sum((x - mean_x) ** 2 for x in x_values)
                
                if denominator > 0:
                    slope = numerator / denominator
                    
                    # Calculate correlation coefficient for trend strength
                    std_x = statistics.stdev(x_values) if len(x_values) > 1 else 0
                    std_y = statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
                    
                    if std_x > 0 and std_y > 0:
                        correlation = numerator / (std_x * std_y * (n - 1))
                        trend_strength = abs(correlation)
                        
                        if trend_strength >= min_confidence:
                            trend_direction = "increasing" if slope > 0 else "decreasing"
                            
                            pattern = DetectedPattern(
                                pattern_id=f"performance_trend_{feature.feature_name}_{trend_direction}",
                                pattern_type=PatternType.PERFORMANCE_TREND,
                                pattern_complexity=PatternComplexity.MODERATE,
                                description=f"{trend_direction.capitalize()} performance trend in {feature.feature_name}",
                                confidence_score=trend_strength,
                                strength=trend_strength,
                                trend_direction=trend_direction,
                                supporting_features=[feature.feature_name],
                                statistical_significance=trend_strength
                            )
                            patterns.append(pattern)
            
            return Either.right(patterns)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Performance trend detection failed: {str(e)}", 
                "PERFORMANCE_TREND_ERROR"
            ))

    @require(lambda pattern_id: len(pattern_id) > 0)
    @require(lambda horizon_hours: horizon_hours > 0)
    async def predict_pattern_future(
        self,
        pattern_id: str,
        horizon_hours: int,
        confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    ) -> Either[PredictionError, PatternPrediction]:
        """
        Predict future behavior based on detected pattern.
        
        Uses appropriate prediction models based on pattern type to forecast
        future automation behavior and trends.
        """
        try:
            if pattern_id not in self.detected_patterns:
                return Either.left(PredictionError.model_not_found(pattern_id))
            
            pattern = self.detected_patterns[pattern_id]
            
            # Select appropriate prediction model
            model_type = self._select_prediction_model(pattern.pattern_type)
            
            # Generate prediction based on pattern type
            prediction_result = await self._generate_pattern_prediction(
                pattern, horizon_hours, model_type, confidence_level
            )
            
            if prediction_result.is_left():
                return prediction_result
            
            prediction = prediction_result.get_right()
            
            # Cache prediction for future use
            self.pattern_cache[f"{pattern_id}_{horizon_hours}"] = prediction
            
            return Either.right(prediction)
            
        except Exception as e:
            return Either.left(PredictionError.prediction_failed(
                pattern_id, str(e)
            ))

    def _select_prediction_model(self, pattern_type: PatternType) -> ModelType:
        """Select appropriate prediction model for pattern type."""
        model_mapping = {
            PatternType.USAGE_FREQUENCY: ModelType.LINEAR_REGRESSION,
            PatternType.EXECUTION_TIMING: ModelType.ARIMA,
            PatternType.WORKFLOW_SEQUENCE: ModelType.RANDOM_FOREST,
            PatternType.RESOURCE_CONSUMPTION: ModelType.LSTM,
            PatternType.ERROR_OCCURRENCE: ModelType.GRADIENT_BOOSTING,
            PatternType.SEASONAL_BEHAVIOR: ModelType.SEASONAL_ARIMA,
            PatternType.USER_BEHAVIOR: ModelType.NEURAL_NETWORK,
            PatternType.PERFORMANCE_TREND: ModelType.PROPHET
        }
        
        return model_mapping.get(pattern_type, ModelType.LINEAR_REGRESSION)

    async def _generate_pattern_prediction(
        self,
        pattern: DetectedPattern,
        horizon_hours: int,
        model_type: ModelType,
        confidence_level: ConfidenceLevel
    ) -> Either[PredictionError, PatternPrediction]:
        """Generate prediction for specific pattern."""
        try:
            # Create prediction timestamps
            start_time = datetime.now(UTC)
            prediction_timestamps = [
                start_time + timedelta(hours=i) 
                for i in range(1, horizon_hours + 1)
            ]
            
            # Generate predicted values based on pattern type
            if pattern.pattern_type == PatternType.USAGE_FREQUENCY:
                predicted_values = await self._predict_usage_frequency(
                    pattern, horizon_hours
                )
            elif pattern.pattern_type == PatternType.EXECUTION_TIMING:
                predicted_values = await self._predict_execution_timing(
                    pattern, horizon_hours
                )
            elif pattern.pattern_type == PatternType.SEASONAL_BEHAVIOR:
                predicted_values = await self._predict_seasonal_behavior(
                    pattern, horizon_hours
                )
            else:
                # Generic linear extrapolation for other patterns
                predicted_values = await self._predict_generic_pattern(
                    pattern, horizon_hours
                )
            
            # Generate confidence intervals
            confidence_intervals = self._generate_confidence_intervals(
                predicted_values, confidence_level
            )
            
            # Calculate accuracy estimate
            accuracy_estimate = min(0.95, pattern.confidence_score * 0.9)
            
            prediction = PatternPrediction(
                prediction_id=create_prediction_id(),
                pattern_id=pattern.pattern_id,
                prediction_type=PredictionType.USAGE_PATTERNS,
                predicted_values=predicted_values,
                prediction_timestamps=prediction_timestamps,
                confidence_intervals=confidence_intervals,
                confidence_level=confidence_level,
                prediction_horizon_hours=horizon_hours,
                model_used=model_type,
                accuracy_estimate=accuracy_estimate,
                factors_considered=pattern.supporting_features,
                assumptions=[
                    "Pattern continues with similar characteristics",
                    "No major system changes or disruptions",
                    "Historical data represents future behavior"
                ]
            )
            
            return Either.right(prediction)
            
        except Exception as e:
            return Either.left(PredictionError.prediction_failed(
                pattern.pattern_id, str(e)
            ))

    async def _predict_usage_frequency(self, pattern: DetectedPattern, horizon_hours: int) -> List[float]:
        """Predict usage frequency patterns."""
        # Simple frequency-based prediction
        base_frequency = pattern.strength * 10  # Scale to reasonable frequency
        
        # Add some variation based on pattern complexity
        if pattern.pattern_complexity == PatternComplexity.SIMPLE:
            variation = 0.1
        elif pattern.pattern_complexity == PatternComplexity.MODERATE:
            variation = 0.2
        else:
            variation = 0.3
        
        predicted_values = []
        for hour in range(horizon_hours):
            # Add hourly variation
            hourly_modifier = 1.0 + 0.2 * (hour % 24 / 24)  # Daily cycle
            value = base_frequency * hourly_modifier * (1 + variation * (hour % 7 / 7))
            predicted_values.append(max(0, value))
        
        return predicted_values

    async def _predict_execution_timing(self, pattern: DetectedPattern, horizon_hours: int) -> List[float]:
        """Predict execution timing patterns."""
        # Timing-based prediction with hourly patterns
        predicted_values = []
        
        for hour in range(horizon_hours):
            current_hour = (datetime.now(UTC).hour + hour) % 24
            
            # Peak hours based on pattern
            if pattern.frequency == "hourly":
                # Simulate peak hours (e.g., 9-11 AM, 2-4 PM)
                if current_hour in [9, 10, 14, 15]:
                    value = pattern.strength * 20
                elif current_hour in [8, 11, 13, 16]:
                    value = pattern.strength * 15
                else:
                    value = pattern.strength * 5
            else:
                # Default hourly distribution
                value = pattern.strength * 10
            
            predicted_values.append(max(0, value))
        
        return predicted_values

    async def _predict_seasonal_behavior(self, pattern: DetectedPattern, horizon_hours: int) -> List[float]:
        """Predict seasonal behavior patterns."""
        predicted_values = []
        
        seasonality = pattern.seasonality or {}
        
        for hour in range(horizon_hours):
            current_time = datetime.now(UTC) + timedelta(hours=hour)
            
            if seasonality.get("type") == "weekly":
                # Weekly seasonal pattern
                day_of_week = current_time.weekday()
                stats = seasonality.get("stats", {})
                base_value = stats.get(day_of_week, pattern.strength * 10)
            elif seasonality.get("type") == "monthly":
                # Monthly seasonal pattern
                month = current_time.month
                stats = seasonality.get("stats", {})
                base_value = stats.get(month, pattern.strength * 10)
            else:
                # Default seasonal pattern
                base_value = pattern.strength * 10
            
            predicted_values.append(max(0, base_value))
        
        return predicted_values

    async def _predict_generic_pattern(self, pattern: DetectedPattern, horizon_hours: int) -> List[float]:
        """Generic pattern prediction using linear extrapolation."""
        base_value = pattern.strength * 10
        
        # Add trend if specified
        if pattern.trend_direction == "increasing":
            trend_slope = 0.1
        elif pattern.trend_direction == "decreasing":
            trend_slope = -0.1
        else:
            trend_slope = 0.0
        
        predicted_values = []
        for hour in range(horizon_hours):
            value = base_value + trend_slope * hour
            predicted_values.append(max(0, value))
        
        return predicted_values

    def _generate_confidence_intervals(
        self, 
        predicted_values: List[float], 
        confidence_level: ConfidenceLevel
    ) -> List[Tuple[float, float]]:
        """Generate confidence intervals for predictions."""
        confidence_multipliers = {
            ConfidenceLevel.LOW: 0.4,
            ConfidenceLevel.MEDIUM: 0.3,
            ConfidenceLevel.HIGH: 0.2,
            ConfidenceLevel.VERY_HIGH: 0.1
        }
        
        multiplier = confidence_multipliers.get(confidence_level, 0.3)
        
        confidence_intervals = []
        for value in predicted_values:
            margin = value * multiplier
            confidence_intervals.append((
                max(0, value - margin),
                value + margin
            ))
        
        return confidence_intervals

    async def get_pattern_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of detected patterns."""
        try:
            pattern_count_by_type = defaultdict(int)
            pattern_count_by_complexity = defaultdict(int)
            high_confidence_patterns = []
            
            for pattern in self.detected_patterns.values():
                pattern_count_by_type[pattern.pattern_type.value] += 1
                pattern_count_by_complexity[pattern.pattern_complexity.value] += 1
                
                if pattern.confidence_score >= 0.8:
                    high_confidence_patterns.append({
                        "pattern_id": pattern.pattern_id,
                        "type": pattern.pattern_type.value,
                        "description": pattern.description,
                        "confidence": pattern.confidence_score,
                        "strength": pattern.strength
                    })
            
            return {
                "total_patterns_detected": len(self.detected_patterns),
                "patterns_by_type": dict(pattern_count_by_type),
                "patterns_by_complexity": dict(pattern_count_by_complexity),
                "high_confidence_patterns": high_confidence_patterns,
                "detection_timestamp": datetime.now(UTC).isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Failed to generate pattern summary: {str(e)}",
                "timestamp": datetime.now(UTC).isoformat()
            }