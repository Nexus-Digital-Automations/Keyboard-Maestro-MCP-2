"""Analytics architecture for comprehensive automation insights and business intelligence.

This module defines the analytics type system, metrics framework, and ML infrastructure
for deep analysis of the complete 48-tool enterprise automation ecosystem.

Security: Enterprise-grade analytics with privacy compliance and data protection.
Performance: <100ms metric collection, <500ms analysis, <2s dashboard generation.
Type Safety: Complete analytics type system with contract-driven development.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, NewType

from .errors import ValidationError
from .types import Permission

# Branded Types for Analytics
MetricId = NewType("MetricId", str)
DashboardId = NewType("DashboardId", str)
ReportId = NewType("ReportId", str)
InsightId = NewType("InsightId", str)
ModelId = NewType("ModelId", str)
AnalyticsSessionId = NewType("AnalyticsSessionId", str)


class MetricType(Enum):
    """Types of metrics collected."""

    PERFORMANCE = "performance"
    USAGE = "usage"
    ROI = "roi"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    SECURITY = "security"
    USER_EXPERIENCE = "user_experience"
    RESOURCE_UTILIZATION = "resource_utilization"


class AnalyticsScope(Enum):
    """Scope of analytics analysis."""

    TOOL = "tool"
    CATEGORY = "category"
    ECOSYSTEM = "ecosystem"
    ENTERPRISE = "enterprise"


class AnalysisDepth(Enum):
    """Depth of analytics analysis."""

    BASIC = "basic"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    ML_ENHANCED = "ml_enhanced"


class VisualizationFormat(Enum):
    """Format for analytics visualization."""

    RAW = "raw"
    TABLE = "table"
    CHART = "chart"
    DASHBOARD = "dashboard"
    REPORT = "report"
    EXECUTIVE_SUMMARY = "executive_summary"


class PrivacyMode(Enum):
    """Privacy protection levels for analytics."""

    NONE = "none"
    BASIC = "basic"
    COMPLIANT = "compliant"
    STRICT = "strict"


class MLModelType(Enum):
    """Types of machine learning models."""

    PATTERN_RECOGNITION = "pattern_recognition"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE_ANALYTICS = "predictive_analytics"
    RECOMMENDATION = "recommendation"
    OPTIMIZATION = "optimization"
    CLASSIFICATION = "classification"


@dataclass(frozen=True)
class MetricDefinition:
    """Definition of a specific metric."""

    metric_id: MetricId
    name: str
    metric_type: MetricType
    scope: AnalyticsScope
    description: str
    unit: str
    category: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        if not self.metric_id or len(self.metric_id) == 0:
            raise ValidationError("metric_id", self.metric_id, "cannot be empty")

        if not self.name or len(self.name.strip()) == 0:
            raise ValidationError("name", self.name, "cannot be empty")

        if len(self.name) > 100:
            raise ValidationError("name", self.name, "must be 100 characters or less")


@dataclass(frozen=True)
class MetricValue:
    """Individual metric measurement."""

    metric_id: MetricId
    value: float | int | str | bool
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metric_id:
            raise ValidationError("metric_id", self.metric_id, "cannot be empty")


@dataclass(frozen=True)
class PerformanceMetrics:
    """Performance-specific metrics collection."""

    response_time_ms: float
    throughput_per_second: int
    error_rate_percent: float
    resource_usage_percent: float
    measurement_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        if self.response_time_ms < 0:
            raise ValidationError(
                "response_time_ms",
                self.response_time_ms,
                "cannot be negative",
            )

        if self.throughput_per_second < 0:
            raise ValidationError(
                "throughput_per_second",
                self.throughput_per_second,
                "cannot be negative",
            )

        if not (0.0 <= self.error_rate_percent <= 100.0):
            raise ValidationError(
                "error_rate_percent",
                self.error_rate_percent,
                "must be between 0.0 and 100.0",
            )


@dataclass(frozen=True)
class ROIMetrics:
    """Return on Investment metrics."""

    investment_amount: Decimal
    savings_amount: Decimal
    time_period_days: int
    efficiency_gain_percent: float
    calculation_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    def calculate_roi(self) -> float:
        """Calculate ROI based on investment and savings."""
        if self.investment_amount == Decimal("0.00"):
            return 0.0

        roi_decimal = (
            (self.savings_amount - self.investment_amount)
            / self.investment_amount
            * 100
        )
        return float(roi_decimal)


@dataclass(frozen=True)
class MLInsight:
    """Machine learning generated insight."""

    insight_id: InsightId
    model_type: MLModelType
    confidence_score: float
    prediction_data: dict[str, Any]
    model_version: str
    feature_importance: dict[str, float] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValidationError(
                "confidence_score",
                self.confidence_score,
                "must be between 0.0 and 1.0",
            )


@dataclass(frozen=True)
class AnalyticsDashboard:
    """Analytics dashboard configuration."""

    dashboard_id: DashboardId
    name: str
    description: str
    metrics: list[MetricId]
    visualizations: list[VisualizationFormat]
    refresh_interval: timedelta
    access_permissions: list[Permission]
    filters: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        if not self.dashboard_id:
            raise ValidationError("dashboard_id", self.dashboard_id, "cannot be empty")

        if not self.metrics:
            raise ValidationError("metrics", self.metrics, "cannot be empty")

        if self.refresh_interval.total_seconds() < 30:
            raise ValidationError(
                "refresh_interval",
                self.refresh_interval,
                "must be at least 30 seconds",
            )


# Alias for backward compatibility
Dashboard = AnalyticsDashboard


@dataclass(frozen=True)
class AnalyticsReport:
    """Generated analytics report."""

    report_id: ReportId
    title: str
    executive_summary: str
    key_insights: list[MLInsight]
    performance_highlights: dict[str, Any]
    roi_analysis: dict[str, ROIMetrics]
    recommendations: list[str]
    data_quality_score: float
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        if not (0.0 <= self.data_quality_score <= 1.0):
            raise ValidationError(
                "data_quality_score",
                self.data_quality_score,
                "must be between 0.0 and 1.0",
            )


@dataclass(frozen=True)
class AnalyticsConfiguration:
    """Configuration for analytics engine."""

    collection_enabled: bool = True
    real_time_monitoring: bool = True
    ml_insights_enabled: bool = True
    anomaly_detection_enabled: bool = True
    predictive_analytics_enabled: bool = True
    privacy_mode: PrivacyMode = PrivacyMode.COMPLIANT
    data_retention_days: int = 365
    alert_thresholds: dict[str, float] = field(default_factory=dict)
    enterprise_integration_enabled: bool = True

    def __post_init__(self):
        if self.data_retention_days < 1:
            raise ValidationError(
                "data_retention_days",
                self.data_retention_days,
                "must be at least 1 day",
            )


# Required permissions for analytics operations
ANALYTICS_PERMISSIONS = {
    MetricType.PERFORMANCE: [Permission.SYSTEM_CONTROL],
    MetricType.USAGE: [Permission.SYSTEM_CONTROL],
    MetricType.ROI: [Permission.SYSTEM_CONTROL, Permission.FILE_ACCESS],
    MetricType.SECURITY: [Permission.SYSTEM_CONTROL, Permission.APPLICATION_CONTROL],
    AnalyticsScope.ENTERPRISE: [Permission.NETWORK_ACCESS, Permission.SYSTEM_CONTROL],
}


# Analytics utility functions
def create_metric_id(metric_name: str, tool_name: str) -> MetricId:
    """Create a unique metric identifier."""
    return MetricId(f"{tool_name}_{metric_name}_{uuid.uuid4().hex[:8]}")


def create_dashboard_id(dashboard_name: str) -> DashboardId:
    """Create a unique dashboard identifier."""
    return DashboardId(f"dashboard_{dashboard_name}_{uuid.uuid4().hex[:8]}")


def create_insight_id(model_type: MLModelType) -> InsightId:
    """Create a unique insight identifier."""
    return InsightId(f"insight_{model_type.value}_{uuid.uuid4().hex[:8]}")


def validate_metric_value(value: Any, expected_type: type) -> bool:
    """Validate metric value type and range."""
    if not isinstance(value, expected_type):
        return False

    if isinstance(value, int | float):
        return not (value < 0 and expected_type in [int, float])

    return True


class TrendDirection(Enum):
    """Direction of trend analysis."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class AlertSeverity(Enum):
    """Severity levels for analytics alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class TrendAnalysis:
    """Basic trend analysis result."""

    metric_id: str
    direction: TrendDirection
    magnitude: float
    confidence: float
    time_period: str


@dataclass(frozen=True)
class AnomalyDetection:
    """Basic anomaly detection result."""

    metric_id: str
    value: float
    expected_range: tuple
    severity: AlertSeverity
    timestamp: datetime


# AnalyticsError imported from .errors module
