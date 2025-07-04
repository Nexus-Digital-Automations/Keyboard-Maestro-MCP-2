"""
Analytics architecture for comprehensive automation insights and business intelligence.

This module defines the analytics type system, metrics framework, and ML infrastructure
for deep analysis of the complete 48-tool enterprise automation ecosystem.

Security: Enterprise-grade analytics with privacy compliance and data protection.
Performance: <100ms metric collection, <500ms analysis, <2s dashboard generation.
Type Safety: Complete analytics type system with contract-driven development.
"""

from __future__ import annotations
from typing import NewType, Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
import uuid

from .types import Permission
from .contracts import require, ensure
from .either import Either
from .errors import ValidationError, AnalyticsError, SecurityError


# Branded Types for Analytics
MetricId = NewType('MetricId', str)
DashboardId = NewType('DashboardId', str)
ReportId = NewType('ReportId', str)
InsightId = NewType('InsightId', str)
ModelId = NewType('ModelId', str)
AnalyticsSessionId = NewType('AnalyticsSessionId', str)


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
    unit: str
    description: str
    collection_frequency: timedelta
    aggregation_methods: List[str]
    privacy_level: PrivacyMode = PrivacyMode.COMPLIANT
    retention_period: timedelta = field(default_factory=lambda: timedelta(days=365))
    
    def __post_init__(self):
        if not self.metric_id or len(self.metric_id) == 0:
            raise ValidationError("metric_id", self.metric_id, "cannot be empty")
        
        if not self.name or len(self.name.strip()) == 0:
            raise ValidationError("name", self.name, "cannot be empty")
        
        if len(self.name) > 100:
            raise ValidationError("name", self.name, "must be 100 characters or less")
        
        if self.collection_frequency.total_seconds() < 1:
            raise ValidationError("collection_frequency", self.collection_frequency, "must be at least 1 second")


@dataclass(frozen=True)
class MetricValue:
    """Individual metric measurement."""
    metric_id: MetricId
    value: Union[float, int, str, bool]
    timestamp: datetime
    source_tool: str
    context: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0  # 0.0 to 1.0
    
    def __post_init__(self):
        if not self.metric_id:
            raise ValidationError("metric_id", self.metric_id, "cannot be empty")
        
        if not self.source_tool:
            raise ValidationError("source_tool", self.source_tool, "cannot be empty")
        
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValidationError("quality_score", self.quality_score, "must be between 0.0 and 1.0")


@dataclass(frozen=True)
class PerformanceMetrics:
    """Performance-specific metrics collection."""
    tool_name: str
    operation: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    success_rate: float
    error_count: int
    throughput: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if self.execution_time_ms < 0:
            raise ValidationError("execution_time_ms", self.execution_time_ms, "cannot be negative")
        
        if self.memory_usage_mb < 0:
            raise ValidationError("memory_usage_mb", self.memory_usage_mb, "cannot be negative")
        
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValidationError("success_rate", self.success_rate, "must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ROIMetrics:
    """Return on Investment metrics."""
    tool_name: str
    time_saved_hours: float
    cost_saved_dollars: float
    efficiency_gain_percent: float
    automation_accuracy: float
    user_satisfaction: float
    implementation_cost: float
    maintenance_cost: float
    calculated_roi: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def calculate_roi(self) -> float:
        """Calculate ROI based on costs and benefits."""
        total_benefits = self.cost_saved_dollars + (self.time_saved_hours * 50)  # $50/hour assumption
        total_costs = self.implementation_cost + self.maintenance_cost
        
        if total_costs == 0:
            return float('inf') if total_benefits > 0 else 0.0
        
        return (total_benefits - total_costs) / total_costs


@dataclass(frozen=True)
class MLInsight:
    """Machine learning generated insight."""
    insight_id: InsightId
    model_type: MLModelType
    confidence: float
    description: str
    recommendation: str
    supporting_data: Dict[str, Any]
    impact_score: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError("confidence", self.confidence, "must be between 0.0 and 1.0")
        
        if not (0.0 <= self.impact_score <= 1.0):
            raise ValidationError("impact_score", self.impact_score, "must be between 0.0 and 1.0")


@dataclass(frozen=True)
class AnalyticsDashboard:
    """Analytics dashboard configuration."""
    dashboard_id: DashboardId
    name: str
    description: str
    metrics: List[MetricId]
    visualizations: List[VisualizationFormat]
    refresh_interval: timedelta
    access_permissions: List[Permission]
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if not self.dashboard_id:
            raise ValidationError("dashboard_id", self.dashboard_id, "cannot be empty")
        
        if not self.metrics:
            raise ValidationError("metrics", self.metrics, "cannot be empty")
        
        if self.refresh_interval.total_seconds() < 30:
            raise ValidationError("refresh_interval", self.refresh_interval, "must be at least 30 seconds")


# Alias for backward compatibility
Dashboard = AnalyticsDashboard


@dataclass(frozen=True)
class AnalyticsReport:
    """Generated analytics report."""
    report_id: ReportId
    title: str
    executive_summary: str
    key_insights: List[MLInsight]
    performance_highlights: Dict[str, Any]
    roi_analysis: Dict[str, ROIMetrics]
    recommendations: List[str]
    data_quality_score: float
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if not (0.0 <= self.data_quality_score <= 1.0):
            raise ValidationError("data_quality_score", self.data_quality_score, "must be between 0.0 and 1.0")


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
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    enterprise_integration_enabled: bool = True
    
    def __post_init__(self):
        if self.data_retention_days < 1:
            raise ValidationError("data_retention_days", self.data_retention_days, "must be at least 1 day")


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
    
    if isinstance(value, (int, float)):
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


class AnalyticsError(Exception):
    """Base exception for analytics-related errors."""
    pass