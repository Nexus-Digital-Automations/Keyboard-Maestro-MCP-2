"""
Usage Forecaster - TASK_59 Phase 2 Core Implementation

Resource usage and capacity forecasting system for automation workflows.
Provides advanced forecasting models, capacity planning, and resource optimization.

Architecture: Time Series Models + Capacity Planning + Growth Prediction + Optimization
Performance: <500ms forecast generation, <1s capacity analysis, <2s optimization recommendations
Security: Safe resource modeling, validated forecasts, comprehensive input sanitization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import statistics
import math
from collections import defaultdict, deque

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.predictive_modeling import (
    ForecastId, create_forecast_id, ForecastGranularity, ConfidenceLevel, ModelType,
    ResourceForecast, TimeSeriesData, PredictiveModelingError, ForecastingError,
    validate_time_series_data, calculate_prediction_confidence,
    generate_capacity_recommendations
)


class ResourceType(Enum):
    """Types of resources that can be forecasted."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage" 
    STORAGE_USAGE = "storage_usage"
    NETWORK_BANDWIDTH = "network_bandwidth"
    DISK_IO = "disk_io"
    AUTOMATION_EXECUTIONS = "automation_executions"
    API_CALLS = "api_calls"
    CONCURRENT_WORKFLOWS = "concurrent_workflows"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"


class GrowthPattern(Enum):
    """Growth patterns for resource usage."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    VOLATILE = "volatile"
    STABLE = "stable"


class CapacityStatus(Enum):
    """Capacity status indicators."""
    OPTIMAL = "optimal"
    APPROACHING_LIMIT = "approaching_limit"
    AT_CAPACITY = "at_capacity"
    OVER_CAPACITY = "over_capacity"
    SCALING_NEEDED = "scaling_needed"


@dataclass(frozen=True)
class UsageTrend:
    """Usage trend analysis for a specific resource."""
    resource_type: ResourceType
    trend_direction: str  # increasing, decreasing, stable
    growth_rate: float  # percentage per time unit
    growth_pattern: GrowthPattern
    seasonality_detected: bool
    seasonal_periods: List[str] = field(default_factory=list)
    trend_confidence: float = 0.0
    data_quality_score: float = 0.0
    
    def __post_init__(self):
        if not (0.0 <= self.trend_confidence <= 1.0):
            raise ValueError("Trend confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.data_quality_score <= 1.0):
            raise ValueError("Data quality score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class CapacityAnalysis:
    """Capacity analysis for resource planning."""
    resource_type: ResourceType
    current_capacity: float
    current_utilization: float
    utilization_percentage: float
    capacity_status: CapacityStatus
    time_to_capacity: Optional[timedelta] = None
    recommended_actions: List[str] = field(default_factory=list)
    scaling_recommendations: Dict[str, Any] = field(default_factory=dict)
    cost_impact: Optional[float] = None
    
    def __post_init__(self):
        if self.current_capacity <= 0:
            raise ValueError("Current capacity must be positive")
        if self.current_utilization < 0:
            raise ValueError("Current utilization must be non-negative")
        if not (0.0 <= self.utilization_percentage <= 100.0):
            raise ValueError("Utilization percentage must be between 0.0 and 100.0")


@dataclass(frozen=True)
class ForecastScenario:
    """Scenario-based forecasting configuration."""
    scenario_name: str
    growth_multiplier: float = 1.0
    seasonal_adjustment: float = 1.0
    external_factors: Dict[str, float] = field(default_factory=dict)
    confidence_adjustment: float = 1.0
    description: str = ""
    
    def __post_init__(self):
        if self.growth_multiplier <= 0:
            raise ValueError("Growth multiplier must be positive")
        if not (0.1 <= self.confidence_adjustment <= 2.0):
            raise ValueError("Confidence adjustment must be between 0.1 and 2.0")


class UsageForecaster:
    """
    Advanced resource usage and capacity forecasting system.
    
    Provides comprehensive forecasting capabilities including trend analysis,
    capacity planning, and optimization recommendations with multiple modeling approaches.
    """
    
    def __init__(self):
        self.resource_data: Dict[ResourceType, List[TimeSeriesData]] = defaultdict(list)
        self.forecasting_models: Dict[ResourceType, Any] = {}
        self.capacity_thresholds: Dict[ResourceType, Dict[str, float]] = {}
        self.forecast_cache: Dict[str, ResourceForecast] = {}
        self.historical_accuracy: Dict[str, float] = {}
        
        # Initialize default capacity thresholds
        self._initialize_capacity_thresholds()
        
        # Initialize forecasting models
        self._initialize_forecasting_models()
    
    def _initialize_capacity_thresholds(self):
        """Initialize default capacity thresholds for different resource types."""
        self.capacity_thresholds = {
            ResourceType.CPU_USAGE: {
                "warning": 70.0,
                "critical": 85.0,
                "maximum": 95.0
            },
            ResourceType.MEMORY_USAGE: {
                "warning": 75.0,
                "critical": 90.0,
                "maximum": 98.0
            },
            ResourceType.STORAGE_USAGE: {
                "warning": 80.0,
                "critical": 90.0,
                "maximum": 95.0
            },
            ResourceType.NETWORK_BANDWIDTH: {
                "warning": 60.0,
                "critical": 80.0,
                "maximum": 90.0
            },
            ResourceType.AUTOMATION_EXECUTIONS: {
                "warning": 1000.0,
                "critical": 5000.0,
                "maximum": 10000.0
            }
        }
    
    def _initialize_forecasting_models(self):
        """Initialize forecasting models for different resource types."""
        self.forecasting_models = {
            ResourceType.CPU_USAGE: {
                "primary": ModelType.ARIMA,
                "secondary": ModelType.LINEAR_REGRESSION,
                "seasonal": ModelType.SEASONAL_ARIMA
            },
            ResourceType.MEMORY_USAGE: {
                "primary": ModelType.LSTM,
                "secondary": ModelType.ARIMA,
                "seasonal": ModelType.PROPHET
            },
            ResourceType.STORAGE_USAGE: {
                "primary": ModelType.LINEAR_REGRESSION,
                "secondary": ModelType.POLYNOMIAL_REGRESSION,
                "seasonal": ModelType.SEASONAL_ARIMA
            },
            ResourceType.AUTOMATION_EXECUTIONS: {
                "primary": ModelType.PROPHET,
                "secondary": ModelType.ARIMA,
                "seasonal": ModelType.SEASONAL_ARIMA
            }
        }

    @require(lambda time_series_data: len(time_series_data.timestamps) >= 10)
    async def add_usage_data(
        self, 
        resource_type: ResourceType, 
        time_series_data: TimeSeriesData
    ) -> Either[PredictiveModelingError, None]:
        """
        Add historical usage data for forecasting.
        
        Validates and stores time series data for specific resource types
        to enable accurate forecasting and trend analysis.
        """
        try:
            # Validate time series data
            validation_result = validate_time_series_data(time_series_data, 30)
            if validation_result.is_left():
                return validation_result
            
            # Add to resource data
            self.resource_data[resource_type].append(time_series_data)
            
            # Keep only recent data (last 90 days worth)
            cutoff_date = datetime.now(UTC) - timedelta(days=90)
            self.resource_data[resource_type] = [
                data for data in self.resource_data[resource_type]
                if any(ts >= cutoff_date for ts in data.timestamps)
            ]
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Failed to add usage data: {str(e)}", 
                "DATA_ADDITION_ERROR"
            ))

    @require(lambda resource_type: isinstance(resource_type, ResourceType))
    @require(lambda forecast_period_days: 1 <= forecast_period_days <= 365)
    async def generate_forecast(
        self,
        resource_type: ResourceType,
        forecast_period_days: int,
        granularity: ForecastGranularity = ForecastGranularity.DAILY,
        confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        scenario: Optional[ForecastScenario] = None
    ) -> Either[ForecastingError, ResourceForecast]:
        """
        Generate resource usage forecast for specified period.
        
        Uses appropriate forecasting models based on resource type and historical data
        to predict future usage patterns with confidence intervals.
        """
        try:
            # Check if we have sufficient data
            if resource_type not in self.resource_data or not self.resource_data[resource_type]:
                return Either.left(ForecastingError(
                    f"No historical data available for {resource_type.value}",
                    "INSUFFICIENT_DATA"
                ))
            
            # Get the most recent and comprehensive data
            latest_data = self._get_latest_consolidated_data(resource_type)
            
            if len(latest_data.values) < 10:
                return Either.left(ForecastingError(
                    f"Insufficient data points for {resource_type.value}: {len(latest_data.values)}",
                    "INSUFFICIENT_DATA_POINTS"
                ))
            
            # Analyze usage trends
            trend_analysis = await self._analyze_usage_trends(latest_data)
            
            # Select appropriate forecasting model
            model_type = self._select_forecasting_model(resource_type, trend_analysis)
            
            # Generate base forecast
            forecast_result = await self._generate_base_forecast(
                latest_data, forecast_period_days, granularity, model_type
            )
            
            if forecast_result.is_left():
                return forecast_result
            
            base_forecast = forecast_result.get_right()
            
            # Apply scenario adjustments if provided
            if scenario:
                base_forecast = self._apply_scenario_adjustments(base_forecast, scenario)
            
            # Generate capacity analysis
            capacity_analysis = await self._analyze_capacity_requirements(
                resource_type, base_forecast, latest_data
            )
            
            # Create final forecast
            forecast = ResourceForecast(
                forecast_id=create_forecast_id(),
                resource_type=resource_type.value,
                granularity=granularity,
                forecast_period_days=forecast_period_days,
                current_usage=latest_data.values[-1] if latest_data.values else 0.0,
                predicted_usage=base_forecast,
                forecast_timestamps=self._generate_forecast_timestamps(
                    forecast_period_days, granularity
                ),
                capacity_thresholds=self.capacity_thresholds.get(resource_type, {}),
                growth_rate=trend_analysis.growth_rate if trend_analysis else 0.0,
                seasonality_patterns=self._extract_seasonality_patterns(latest_data),
                capacity_recommendations=capacity_analysis.recommended_actions if capacity_analysis else []
            )
            
            # Cache forecast
            cache_key = f"{resource_type.value}_{forecast_period_days}_{granularity.value}"
            self.forecast_cache[cache_key] = forecast
            
            return Either.right(forecast)
            
        except Exception as e:
            return Either.left(ForecastingError(
                f"Forecast generation failed for {resource_type.value}: {str(e)}",
                "FORECAST_GENERATION_ERROR"
            ))

    def _get_latest_consolidated_data(self, resource_type: ResourceType) -> TimeSeriesData:
        """Consolidate and return the latest data for a resource type."""
        all_data = self.resource_data[resource_type]
        
        if not all_data:
            return TimeSeriesData(timestamps=[], values=[])
        
        # Combine all time series data
        all_timestamps = []
        all_values = []
        
        for data in all_data:
            all_timestamps.extend(data.timestamps)
            all_values.extend(data.values)
        
        # Sort by timestamp
        combined_data = list(zip(all_timestamps, all_values))
        combined_data.sort(key=lambda x: x[0])
        
        timestamps, values = zip(*combined_data) if combined_data else ([], [])
        
        return TimeSeriesData(
            timestamps=list(timestamps),
            values=list(values)
        )

    async def _analyze_usage_trends(self, data: TimeSeriesData) -> Optional[UsageTrend]:
        """Analyze usage trends in time series data."""
        try:
            if len(data.values) < 5:
                return None
            
            numeric_values = [float(v) for v in data.values if isinstance(v, (int, float))]
            
            if len(numeric_values) < 5:
                return None
            
            # Calculate linear trend
            n = len(numeric_values)
            x_values = list(range(n))
            
            mean_x = statistics.mean(x_values)
            mean_y = statistics.mean(numeric_values)
            
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, numeric_values))
            denominator = sum((x - mean_x) ** 2 for x in x_values)
            
            if denominator == 0:
                return None
            
            slope = numerator / denominator
            
            # Determine trend direction and growth rate
            if abs(slope) < 0.01 * mean_y:
                trend_direction = "stable"
                growth_rate = 0.0
            elif slope > 0:
                trend_direction = "increasing"
                growth_rate = (slope / mean_y) * 100  # Percentage growth per period
            else:
                trend_direction = "decreasing"
                growth_rate = (slope / mean_y) * 100  # Negative growth
            
            # Determine growth pattern
            growth_pattern = self._determine_growth_pattern(numeric_values)
            
            # Check for seasonality
            seasonality_detected, seasonal_periods = self._detect_seasonality(data)
            
            # Calculate trend confidence
            std_x = statistics.stdev(x_values) if len(x_values) > 1 else 0
            std_y = statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
            
            if std_x > 0 and std_y > 0:
                correlation = numerator / (std_x * std_y * (n - 1))
                trend_confidence = abs(correlation)
            else:
                trend_confidence = 0.0
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality(data)
            
            return UsageTrend(
                resource_type=ResourceType.CPU_USAGE,  # This will be set by caller
                trend_direction=trend_direction,
                growth_rate=growth_rate,
                growth_pattern=growth_pattern,
                seasonality_detected=seasonality_detected,
                seasonal_periods=seasonal_periods,
                trend_confidence=trend_confidence,
                data_quality_score=data_quality_score
            )
            
        except Exception:
            return None

    def _determine_growth_pattern(self, values: List[float]) -> GrowthPattern:
        """Determine the growth pattern from values."""
        if len(values) < 5:
            return GrowthPattern.STABLE
        
        # Calculate coefficient of variation
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        if mean_val == 0:
            return GrowthPattern.STABLE
        
        cv = std_val / mean_val
        
        # High volatility
        if cv > 0.5:
            return GrowthPattern.VOLATILE
        
        # Check for exponential growth
        ratios = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                ratios.append(values[i] / values[i-1])
        
        if ratios:
            ratio_std = statistics.stdev(ratios) if len(ratios) > 1 else 0
            ratio_mean = statistics.mean(ratios)
            
            if ratio_mean > 1.1 and ratio_std < 0.2:  # Consistent growth > 10%
                return GrowthPattern.EXPONENTIAL
            elif ratio_mean < 0.9 and ratio_std < 0.2:  # Consistent decline
                return GrowthPattern.LOGARITHMIC
        
        # Check for seasonal patterns
        if len(values) >= 12:  # Need at least a year of monthly data
            # Simple seasonality check - compare first half to second half
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            if len(first_half) == len(second_half):
                correlation = self._calculate_correlation(first_half, second_half)
                if correlation > 0.7:
                    return GrowthPattern.SEASONAL
        
        # Linear growth
        return GrowthPattern.LINEAR

    def _detect_seasonality(self, data: TimeSeriesData) -> Tuple[bool, List[str]]:
        """Detect seasonality patterns in the data."""
        if len(data.timestamps) < 30:  # Need sufficient data
            return False, []
        
        seasonal_periods = []
        
        # Group by different time periods
        monthly_groups = defaultdict(list)
        weekly_groups = defaultdict(list)
        daily_groups = defaultdict(list)
        
        for timestamp, value in zip(data.timestamps, data.values):
            if isinstance(value, (int, float)):
                monthly_groups[timestamp.month].append(float(value))
                weekly_groups[timestamp.weekday()].append(float(value))
                daily_groups[timestamp.hour].append(float(value))
        
        # Check monthly seasonality
        if len(monthly_groups) >= 6:  # At least 6 months
            month_means = [statistics.mean(values) for values in monthly_groups.values()]
            if len(month_means) > 1 and statistics.stdev(month_means) > 0.2 * statistics.mean(month_means):
                seasonal_periods.append("monthly")
        
        # Check weekly seasonality
        if len(weekly_groups) >= 5:  # At least 5 different days
            day_means = [statistics.mean(values) for values in weekly_groups.values()]
            if len(day_means) > 1 and statistics.stdev(day_means) > 0.15 * statistics.mean(day_means):
                seasonal_periods.append("weekly")
        
        # Check daily seasonality
        if len(daily_groups) >= 12:  # At least 12 different hours
            hour_means = [statistics.mean(values) for values in daily_groups.values()]
            if len(hour_means) > 1 and statistics.stdev(hour_means) > 0.1 * statistics.mean(hour_means):
                seasonal_periods.append("daily")
        
        return len(seasonal_periods) > 0, seasonal_periods

    def _calculate_correlation(self, list1: List[float], list2: List[float]) -> float:
        """Calculate correlation between two lists."""
        if len(list1) != len(list2) or len(list1) < 2:
            return 0.0
        
        mean1 = statistics.mean(list1)
        mean2 = statistics.mean(list2)
        
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(list1, list2))
        std1 = statistics.stdev(list1)
        std2 = statistics.stdev(list2)
        
        if std1 == 0 or std2 == 0:
            return 0.0
        
        denominator = std1 * std2 * len(list1)
        
        return numerator / denominator if denominator != 0 else 0.0

    def _calculate_data_quality(self, data: TimeSeriesData) -> float:
        """Calculate data quality score."""
        if not data.values:
            return 0.0
        
        # Check for missing values
        total_points = len(data.values)
        valid_points = sum(1 for v in data.values if v is not None and isinstance(v, (int, float)))
        
        completeness_score = valid_points / total_points if total_points > 0 else 0.0
        
        # Check for temporal consistency
        if len(data.timestamps) > 1:
            time_diffs = [
                (data.timestamps[i] - data.timestamps[i-1]).total_seconds() 
                for i in range(1, len(data.timestamps))
            ]
            
            if time_diffs:
                mean_diff = statistics.mean(time_diffs)
                std_diff = statistics.stdev(time_diffs) if len(time_diffs) > 1 else 0
                
                # Consistency score - lower std relative to mean is better
                consistency_score = 1.0 - min(1.0, std_diff / mean_diff) if mean_diff > 0 else 0.0
            else:
                consistency_score = 1.0
        else:
            consistency_score = 1.0
        
        # Overall quality score
        return (completeness_score + consistency_score) / 2

    def _select_forecasting_model(
        self, 
        resource_type: ResourceType, 
        trend_analysis: Optional[UsageTrend]
    ) -> ModelType:
        """Select the best forecasting model for the resource type and trend."""
        default_model = self.forecasting_models.get(resource_type, {}).get("primary", ModelType.LINEAR_REGRESSION)
        
        if not trend_analysis:
            return default_model
        
        # Select model based on trend characteristics
        if trend_analysis.seasonality_detected:
            return self.forecasting_models.get(resource_type, {}).get("seasonal", ModelType.SEASONAL_ARIMA)
        elif trend_analysis.growth_pattern == GrowthPattern.EXPONENTIAL:
            return ModelType.LSTM
        elif trend_analysis.growth_pattern == GrowthPattern.VOLATILE:
            return ModelType.ENSEMBLE
        elif trend_analysis.growth_pattern == GrowthPattern.LINEAR:
            return ModelType.LINEAR_REGRESSION
        else:
            return default_model

    async def _generate_base_forecast(
        self,
        data: TimeSeriesData,
        forecast_period_days: int,
        granularity: ForecastGranularity,
        model_type: ModelType
    ) -> Either[ForecastingError, List[float]]:
        """Generate base forecast using selected model."""
        try:
            numeric_values = [float(v) for v in data.values if isinstance(v, (int, float))]
            
            if len(numeric_values) < 5:
                return Either.left(ForecastingError(
                    "Insufficient numeric values for forecasting",
                    "INSUFFICIENT_NUMERIC_DATA"
                ))
            
            # Calculate number of forecast points based on granularity
            points_per_day = {
                ForecastGranularity.HOURLY: 24,
                ForecastGranularity.DAILY: 1,
                ForecastGranularity.WEEKLY: 1/7,
                ForecastGranularity.MONTHLY: 1/30
            }
            
            forecast_points = int(forecast_period_days * points_per_day.get(granularity, 1))
            
            # Generate forecast based on model type
            if model_type == ModelType.LINEAR_REGRESSION:
                forecast = self._linear_regression_forecast(numeric_values, forecast_points)
            elif model_type == ModelType.ARIMA:
                forecast = self._arima_forecast(numeric_values, forecast_points)
            elif model_type == ModelType.SEASONAL_ARIMA:
                forecast = self._seasonal_arima_forecast(numeric_values, forecast_points)
            elif model_type == ModelType.LSTM:
                forecast = self._lstm_forecast(numeric_values, forecast_points)
            elif model_type == ModelType.PROPHET:
                forecast = self._prophet_forecast(data, forecast_points)
            else:
                forecast = self._linear_regression_forecast(numeric_values, forecast_points)
            
            return Either.right(forecast)
            
        except Exception as e:
            return Either.left(ForecastingError(
                f"Base forecast generation failed: {str(e)}",
                "BASE_FORECAST_ERROR"
            ))

    def _linear_regression_forecast(self, values: List[float], forecast_points: int) -> List[float]:
        """Generate forecast using linear regression."""
        n = len(values)
        x_values = list(range(n))
        
        # Calculate linear regression
        mean_x = statistics.mean(x_values)
        mean_y = statistics.mean(values)
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, values))
        denominator = sum((x - mean_x) ** 2 for x in x_values)
        
        if denominator == 0:
            # No trend, use last value
            return [values[-1]] * forecast_points
        
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        
        # Generate forecast
        forecast = []
        for i in range(forecast_points):
            x = n + i
            predicted_value = slope * x + intercept
            # Ensure non-negative values for most resource types
            forecast.append(max(0, predicted_value))
        
        return forecast

    def _arima_forecast(self, values: List[float], forecast_points: int) -> List[float]:
        """Generate forecast using ARIMA model (simplified implementation)."""
        # Simplified ARIMA - using moving average with trend
        if len(values) < 3:
            return [values[-1]] * forecast_points
        
        # Calculate trend using last few points
        window_size = min(5, len(values))
        recent_values = values[-window_size:]
        
        # Simple trend calculation
        if len(recent_values) > 1:
            trend = (recent_values[-1] - recent_values[0]) / (len(recent_values) - 1)
        else:
            trend = 0
        
        # Generate forecast
        forecast = []
        last_value = values[-1]
        
        for i in range(forecast_points):
            predicted_value = last_value + trend * (i + 1)
            # Add some noise reduction for stability
            if i > 0:
                predicted_value = 0.8 * predicted_value + 0.2 * forecast[i-1]
            
            forecast.append(max(0, predicted_value))
        
        return forecast

    def _seasonal_arima_forecast(self, values: List[float], forecast_points: int) -> List[float]:
        """Generate forecast using seasonal ARIMA (simplified implementation)."""
        if len(values) < 12:  # Need at least a seasonal cycle
            return self._arima_forecast(values, forecast_points)
        
        # Detect seasonal period (assume monthly if we have enough data)
        seasonal_period = min(12, len(values) // 2)
        
        # Decompose into trend and seasonal components
        seasonal_components = []
        for i in range(seasonal_period):
            seasonal_values = [values[j] for j in range(i, len(values), seasonal_period)]
            if seasonal_values:
                seasonal_components.append(statistics.mean(seasonal_values))
            else:
                seasonal_components.append(0)
        
        # Calculate overall trend
        trend = self._calculate_overall_trend(values)
        
        # Generate forecast
        forecast = []
        base_value = values[-1]
        
        for i in range(forecast_points):
            seasonal_index = i % len(seasonal_components)
            seasonal_factor = seasonal_components[seasonal_index] / statistics.mean(seasonal_components) if seasonal_components else 1.0
            
            predicted_value = (base_value + trend * (i + 1)) * seasonal_factor
            forecast.append(max(0, predicted_value))
        
        return forecast

    def _lstm_forecast(self, values: List[float], forecast_points: int) -> List[float]:
        """Generate forecast using LSTM-like approach (simplified implementation)."""
        # Simplified LSTM - using exponential smoothing with memory
        if len(values) < 5:
            return self._linear_regression_forecast(values, forecast_points)
        
        # Calculate exponential smoothing parameters
        alpha = 0.3  # Smoothing factor
        
        # Apply exponential smoothing
        smoothed_values = [values[0]]
        for i in range(1, len(values)):
            smoothed_value = alpha * values[i] + (1 - alpha) * smoothed_values[i-1]
            smoothed_values.append(smoothed_value)
        
        # Calculate trend from smoothed values
        trend = self._calculate_overall_trend(smoothed_values)
        
        # Generate forecast with dampening
        forecast = []
        last_value = smoothed_values[-1]
        dampening_factor = 0.98  # Dampen trend over time
        
        for i in range(forecast_points):
            trend_component = trend * (dampening_factor ** i)
            predicted_value = last_value + trend_component * (i + 1)
            forecast.append(max(0, predicted_value))
        
        return forecast

    def _prophet_forecast(self, data: TimeSeriesData, forecast_points: int) -> List[float]:
        """Generate forecast using Prophet-like approach (simplified implementation)."""
        numeric_values = [float(v) for v in data.values if isinstance(v, (int, float))]
        
        if len(numeric_values) < 10:
            return self._linear_regression_forecast(numeric_values, forecast_points)
        
        # Decompose into trend, seasonal, and residual components
        trend = self._extract_trend(numeric_values)
        seasonal = self._extract_seasonal_component(data)
        
        # Generate forecast
        forecast = []
        
        for i in range(forecast_points):
            # Project trend
            trend_value = trend[-1] + (trend[-1] - trend[0]) / len(trend) * (i + 1)
            
            # Add seasonal component if detected
            if seasonal:
                seasonal_index = i % len(seasonal)
                seasonal_value = seasonal[seasonal_index]
            else:
                seasonal_value = 0
            
            predicted_value = trend_value + seasonal_value
            forecast.append(max(0, predicted_value))
        
        return forecast

    def _calculate_overall_trend(self, values: List[float]) -> float:
        """Calculate overall trend in values."""
        if len(values) < 2:
            return 0
        
        # Simple linear trend
        n = len(values)
        x_values = list(range(n))
        
        mean_x = statistics.mean(x_values)
        mean_y = statistics.mean(values)
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, values))
        denominator = sum((x - mean_x) ** 2 for x in x_values)
        
        return numerator / denominator if denominator != 0 else 0

    def _extract_trend(self, values: List[float]) -> List[float]:
        """Extract trend component from values."""
        # Simple moving average for trend extraction
        window_size = min(5, len(values) // 3)
        trend = []
        
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            trend.append(statistics.mean(values[start_idx:end_idx]))
        
        return trend

    def _extract_seasonal_component(self, data: TimeSeriesData) -> List[float]:
        """Extract seasonal component from data."""
        if len(data.timestamps) < 24:  # Need sufficient data
            return []
        
        # Group by hour of day for daily seasonality
        hourly_groups = defaultdict(list)
        
        for timestamp, value in zip(data.timestamps, data.values):
            if isinstance(value, (int, float)):
                hourly_groups[timestamp.hour].append(float(value))
        
        if len(hourly_groups) < 12:  # Need enough hours
            return []
        
        # Calculate hourly averages
        seasonal_component = []
        for hour in range(24):
            if hour in hourly_groups:
                seasonal_component.append(statistics.mean(hourly_groups[hour]))
            else:
                seasonal_component.append(0)
        
        # Normalize around zero
        mean_seasonal = statistics.mean(seasonal_component)
        return [v - mean_seasonal for v in seasonal_component]

    def _apply_scenario_adjustments(
        self, 
        base_forecast: List[float], 
        scenario: ForecastScenario
    ) -> List[float]:
        """Apply scenario adjustments to base forecast."""
        adjusted_forecast = []
        
        for i, value in enumerate(base_forecast):
            # Apply growth multiplier
            adjusted_value = value * scenario.growth_multiplier
            
            # Apply seasonal adjustment
            adjusted_value *= scenario.seasonal_adjustment
            
            # Apply external factors
            for factor, multiplier in scenario.external_factors.items():
                adjusted_value *= multiplier
            
            adjusted_forecast.append(max(0, adjusted_value))
        
        return adjusted_forecast

    async def _analyze_capacity_requirements(
        self,
        resource_type: ResourceType,
        forecast: List[float],
        historical_data: TimeSeriesData
    ) -> Optional[CapacityAnalysis]:
        """Analyze capacity requirements based on forecast."""
        try:
            if not forecast or not historical_data.values:
                return None
            
            # Get current capacity and utilization
            current_usage = historical_data.values[-1] if historical_data.values else 0.0
            thresholds = self.capacity_thresholds.get(resource_type, {})
            
            if not thresholds:
                return None
            
            # Estimate current capacity (assume current usage is 70% of capacity)
            current_capacity = current_usage / 0.7 if current_usage > 0 else 100.0
            current_utilization = current_usage
            utilization_percentage = (current_utilization / current_capacity) * 100
            
            # Determine capacity status
            warning_threshold = thresholds.get("warning", 70.0)
            critical_threshold = thresholds.get("critical", 85.0)
            maximum_threshold = thresholds.get("maximum", 95.0)
            
            if utilization_percentage >= maximum_threshold:
                capacity_status = CapacityStatus.OVER_CAPACITY
            elif utilization_percentage >= critical_threshold:
                capacity_status = CapacityStatus.AT_CAPACITY
            elif utilization_percentage >= warning_threshold:
                capacity_status = CapacityStatus.APPROACHING_LIMIT
            else:
                capacity_status = CapacityStatus.OPTIMAL
            
            # Calculate time to capacity
            max_forecast = max(forecast) if forecast else current_usage
            if max_forecast > current_capacity:
                time_to_capacity = timedelta(days=1)  # Immediate concern
            else:
                # Find when forecast crosses warning threshold
                warning_capacity = current_capacity * (warning_threshold / 100)
                days_to_warning = None
                
                for i, value in enumerate(forecast):
                    if value > warning_capacity:
                        days_to_warning = i
                        break
                
                if days_to_warning is not None:
                    time_to_capacity = timedelta(days=days_to_warning)
                else:
                    time_to_capacity = None
            
            # Generate recommendations
            recommendations = []
            scaling_recommendations = {}
            
            if capacity_status in [CapacityStatus.AT_CAPACITY, CapacityStatus.OVER_CAPACITY]:
                recommendations.append("Immediate capacity increase required")
                scaling_recommendations["immediate_action"] = True
                scaling_recommendations["recommended_increase"] = "50%"
            elif capacity_status == CapacityStatus.APPROACHING_LIMIT:
                recommendations.append("Plan capacity increase within 2 weeks")
                scaling_recommendations["planning_horizon"] = "2 weeks"
                scaling_recommendations["recommended_increase"] = "25%"
            
            if max_forecast > current_capacity * 0.8:
                recommendations.append("Consider auto-scaling policies")
                scaling_recommendations["auto_scaling"] = True
            
            return CapacityAnalysis(
                resource_type=resource_type,
                current_capacity=current_capacity,
                current_utilization=current_utilization,
                utilization_percentage=utilization_percentage,
                capacity_status=capacity_status,
                time_to_capacity=time_to_capacity,
                recommended_actions=recommendations,
                scaling_recommendations=scaling_recommendations
            )
            
        except Exception:
            return None

    def _generate_forecast_timestamps(
        self, 
        forecast_period_days: int, 
        granularity: ForecastGranularity
    ) -> List[datetime]:
        """Generate forecast timestamps based on period and granularity."""
        timestamps = []
        start_time = datetime.now(UTC)
        
        if granularity == ForecastGranularity.HOURLY:
            for hour in range(forecast_period_days * 24):
                timestamps.append(start_time + timedelta(hours=hour + 1))
        elif granularity == ForecastGranularity.DAILY:
            for day in range(forecast_period_days):
                timestamps.append(start_time + timedelta(days=day + 1))
        elif granularity == ForecastGranularity.WEEKLY:
            weeks = forecast_period_days // 7
            for week in range(weeks):
                timestamps.append(start_time + timedelta(weeks=week + 1))
        elif granularity == ForecastGranularity.MONTHLY:
            months = forecast_period_days // 30
            for month in range(months):
                timestamps.append(start_time + timedelta(days=(month + 1) * 30))
        
        return timestamps

    def _extract_seasonality_patterns(self, data: TimeSeriesData) -> Dict[str, Any]:
        """Extract seasonality patterns from historical data."""
        patterns = {}
        
        if len(data.timestamps) < 30:
            return patterns
        
        # Monthly patterns
        monthly_stats = defaultdict(list)
        for timestamp, value in zip(data.timestamps, data.values):
            if isinstance(value, (int, float)):
                monthly_stats[timestamp.month].append(float(value))
        
        if len(monthly_stats) >= 6:
            monthly_averages = {month: statistics.mean(values) for month, values in monthly_stats.items()}
            patterns["monthly"] = monthly_averages
        
        # Weekly patterns
        weekly_stats = defaultdict(list)
        for timestamp, value in zip(data.timestamps, data.values):
            if isinstance(value, (int, float)):
                weekly_stats[timestamp.weekday()].append(float(value))
        
        if len(weekly_stats) >= 5:
            weekly_averages = {day: statistics.mean(values) for day, values in weekly_stats.items()}
            patterns["weekly"] = weekly_averages
        
        # Hourly patterns
        hourly_stats = defaultdict(list)
        for timestamp, value in zip(data.timestamps, data.values):
            if isinstance(value, (int, float)):
                hourly_stats[timestamp.hour].append(float(value))
        
        if len(hourly_stats) >= 12:
            hourly_averages = {hour: statistics.mean(values) for hour, values in hourly_stats.items()}
            patterns["hourly"] = hourly_averages
        
        return patterns

    async def get_forecasting_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of forecasting capabilities and status."""
        try:
            resource_counts = {
                resource_type.value: len(data_list) 
                for resource_type, data_list in self.resource_data.items()
            }
            
            total_data_points = sum(
                len(data.values) for data_list in self.resource_data.values() 
                for data in data_list
            )
            
            cache_status = {
                "cached_forecasts": len(self.forecast_cache),
                "cache_keys": list(self.forecast_cache.keys())
            }
            
            return {
                "resources_tracked": len(self.resource_data),
                "resource_counts": resource_counts,
                "total_data_points": total_data_points,
                "forecasting_models": {
                    resource_type.value: models 
                    for resource_type, models in self.forecasting_models.items()
                },
                "capacity_thresholds": {
                    resource_type.value: thresholds 
                    for resource_type, thresholds in self.capacity_thresholds.items()
                },
                "cache_status": cache_status,
                "summary_timestamp": datetime.now(UTC).isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Failed to generate forecasting summary: {str(e)}",
                "timestamp": datetime.now(UTC).isoformat()
            }