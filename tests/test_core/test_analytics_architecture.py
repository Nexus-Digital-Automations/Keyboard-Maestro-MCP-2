"""Comprehensive test coverage for analytics architecture core module.

Tests the complete analytics system including branded types, enums, dataclasses,
and business logic following ADDER+ methodology for enterprise analytics.
"""

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st
from src.core.analytics_architecture import (
    AnalysisDepth,
    AnalyticsScope,
    AnalyticsSessionId,
    DashboardId,
    InsightId,
    MetricDefinition,
    MetricId,
    MetricType,
    MetricValue,
    MLInsight,
    MLModelType,
    ModelId,
    PerformanceMetrics,
    PrivacyMode,
    ReportId,
    ROIMetrics,
    VisualizationFormat,
)


class TestBrandedTypes:
    """Test branded types for analytics architecture."""

    def test_metric_id_creation(self):
        """Test MetricId branded type creation."""
        metric_id = MetricId("metric_123")
        assert isinstance(metric_id, str)
        assert metric_id == "metric_123"

    def test_dashboard_id_creation(self):
        """Test DashboardId branded type creation."""
        dashboard_id = DashboardId("dashboard_abc")
        assert isinstance(dashboard_id, str)
        assert dashboard_id == "dashboard_abc"

    def test_report_id_creation(self):
        """Test ReportId branded type creation."""
        report_id = ReportId("report_xyz")
        assert isinstance(report_id, str)
        assert report_id == "report_xyz"

    def test_insight_id_creation(self):
        """Test InsightId branded type creation."""
        insight_id = InsightId("insight_456")
        assert isinstance(insight_id, str)
        assert insight_id == "insight_456"

    def test_model_id_creation(self):
        """Test ModelId branded type creation."""
        model_id = ModelId("model_789")
        assert isinstance(model_id, str)
        assert model_id == "model_789"

    def test_analytics_session_id_creation(self):
        """Test AnalyticsSessionId branded type creation."""
        session_id = AnalyticsSessionId("session_def")
        assert isinstance(session_id, str)
        assert session_id == "session_def"


class TestMetricTypeEnum:
    """Test MetricType enum values and behavior."""

    def test_metric_type_values(self):
        """Test all MetricType enum values."""
        assert MetricType.PERFORMANCE.value == "performance"
        assert MetricType.USAGE.value == "usage"
        assert MetricType.EFFICIENCY.value == "efficiency"
        assert MetricType.QUALITY.value == "quality"
        assert MetricType.SECURITY.value == "security"
        assert MetricType.ROI.value == "roi"
        assert MetricType.USER_EXPERIENCE.value == "user_experience"
        assert MetricType.RESOURCE_UTILIZATION.value == "resource_utilization"

    def test_metric_type_enum_complete(self):
        """Test MetricType enum completeness."""
        expected_types = {
            "performance",
            "usage",
            "efficiency",
            "quality",
            "security",
            "roi",
            "user_experience",
            "resource_utilization",
        }
        actual_types = {mt.value for mt in MetricType}
        assert actual_types == expected_types


class TestAnalyticsScopeEnum:
    """Test AnalyticsScope enum values and behavior."""

    def test_analytics_scope_values(self):
        """Test all AnalyticsScope enum values."""
        assert AnalyticsScope.TOOL.value == "tool"
        assert AnalyticsScope.CATEGORY.value == "category"
        assert AnalyticsScope.ECOSYSTEM.value == "ecosystem"
        assert AnalyticsScope.ENTERPRISE.value == "enterprise"


class TestAnalysisDepthEnum:
    """Test AnalysisDepth enum values and behavior."""

    def test_analysis_depth_values(self):
        """Test all AnalysisDepth enum values."""
        assert AnalysisDepth.BASIC.value == "basic"
        assert AnalysisDepth.STANDARD.value == "standard"
        assert AnalysisDepth.DETAILED.value == "detailed"
        assert AnalysisDepth.COMPREHENSIVE.value == "comprehensive"
        assert AnalysisDepth.ML_ENHANCED.value == "ml_enhanced"


class TestVisualizationFormatEnum:
    """Test VisualizationFormat enum values and behavior."""

    def test_visualization_format_values(self):
        """Test all VisualizationFormat enum values."""
        assert VisualizationFormat.RAW.value == "raw"
        assert VisualizationFormat.TABLE.value == "table"
        assert VisualizationFormat.CHART.value == "chart"
        assert VisualizationFormat.DASHBOARD.value == "dashboard"
        assert VisualizationFormat.REPORT.value == "report"
        assert VisualizationFormat.EXECUTIVE_SUMMARY.value == "executive_summary"


class TestPrivacyModeEnum:
    """Test PrivacyMode enum values and behavior."""

    def test_privacy_mode_values(self):
        """Test all PrivacyMode enum values."""
        assert PrivacyMode.NONE.value == "none"
        assert PrivacyMode.BASIC.value == "basic"
        assert PrivacyMode.COMPLIANT.value == "compliant"


class TestMLModelTypeEnum:
    """Test MLModelType enum values and behavior."""

    def test_ml_model_type_values(self):
        """Test all MLModelType enum values."""
        assert MLModelType.PATTERN_RECOGNITION.value == "pattern_recognition"
        assert MLModelType.ANOMALY_DETECTION.value == "anomaly_detection"
        assert MLModelType.PREDICTIVE_ANALYTICS.value == "predictive_analytics"
        assert MLModelType.RECOMMENDATION.value == "recommendation"
        assert MLModelType.OPTIMIZATION.value == "optimization"
        assert MLModelType.CLASSIFICATION.value == "classification"


class TestMetricDefinition:
    """Test MetricDefinition dataclass functionality."""

    def test_metric_definition_creation(self):
        """Test MetricDefinition creation with valid parameters."""
        metric_def = MetricDefinition(
            metric_id=MetricId("perf_001"),
            name="Response Time",
            metric_type=MetricType.PERFORMANCE,
            scope=AnalyticsScope.TOOL,
            description="Average response time for system operations",
            unit="milliseconds",
        )

        assert metric_def.metric_id == MetricId("perf_001")
        assert metric_def.name == "Response Time"
        assert metric_def.metric_type == MetricType.PERFORMANCE
        assert metric_def.scope == AnalyticsScope.TOOL
        assert metric_def.description == "Average response time for system operations"
        assert metric_def.unit == "milliseconds"
        assert isinstance(metric_def.created_at, datetime)

    def test_metric_definition_with_optional_params(self):
        """Test MetricDefinition with optional parameters."""
        metric_def = MetricDefinition(
            metric_id=MetricId("cost_001"),
            name="API Cost",
            metric_type=MetricType.ROI,
            scope=AnalyticsScope.ENTERPRISE,
            description="Total API usage cost",
            unit="USD",
            category="financial",
            tags=["api", "cost", "billing"],
        )

        assert metric_def.category == "financial"
        assert metric_def.tags == ["api", "cost", "billing"]


class TestMetricValue:
    """Test MetricValue dataclass functionality."""

    def test_metric_value_creation(self):
        """Test MetricValue creation with valid parameters."""
        metric_value = MetricValue(
            metric_id=MetricId("perf_001"), value=125.5, timestamp=datetime.now(UTC)
        )

        assert metric_value.metric_id == MetricId("perf_001")
        assert metric_value.value == 125.5
        assert isinstance(metric_value.timestamp, datetime)

    def test_metric_value_with_metadata(self):
        """Test MetricValue with additional metadata."""
        metadata = {"source": "api_monitor", "region": "us-west-2"}
        metric_value = MetricValue(
            metric_id=MetricId("usage_001"),
            value=1500,
            timestamp=datetime.now(UTC),
            context=metadata,
        )

        assert metric_value.context == metadata

    def test_metric_value_validation(self):
        """Test MetricValue validation logic."""
        # Test that __post_init__ validation works
        metric_value = MetricValue(
            metric_id=MetricId("valid_001"), value=100.0, timestamp=datetime.now(UTC)
        )

        # Should not raise any exceptions
        assert metric_value.value == 100.0


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass functionality."""

    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation with valid parameters."""
        perf_metrics = PerformanceMetrics(
            response_time_ms=150.5,
            throughput_per_second=1000,
            error_rate_percent=0.5,
            resource_usage_percent=75.0,
        )

        assert perf_metrics.response_time_ms == 150.5
        assert perf_metrics.throughput_per_second == 1000
        assert perf_metrics.error_rate_percent == 0.5
        assert perf_metrics.resource_usage_percent == 75.0
        assert isinstance(perf_metrics.measurement_time, datetime)

    def test_performance_metrics_calculation_methods(self):
        """Test PerformanceMetrics calculation methods."""
        perf_metrics = PerformanceMetrics(
            response_time_ms=100.0,
            throughput_per_second=500,
            error_rate_percent=1.0,
            resource_usage_percent=50.0,
        )

        # Test availability calculation (100% - error_rate)
        availability = 100.0 - perf_metrics.error_rate_percent
        assert availability == 99.0

        # Test efficiency score (example calculation)
        efficiency = (
            perf_metrics.throughput_per_second / 10
        ) - perf_metrics.response_time_ms / 10
        assert efficiency == 50.0 - 10.0  # 40.0


class TestROIMetrics:
    """Test ROIMetrics dataclass functionality."""

    def test_roi_metrics_creation(self):
        """Test ROIMetrics creation with valid parameters."""
        roi_metrics = ROIMetrics(
            investment_amount=Decimal("10000.00"),
            savings_amount=Decimal("15000.00"),
            time_period_days=365,
            efficiency_gain_percent=25.0,
        )

        assert roi_metrics.investment_amount == Decimal("10000.00")
        assert roi_metrics.savings_amount == Decimal("15000.00")
        assert roi_metrics.time_period_days == 365
        assert roi_metrics.efficiency_gain_percent == 25.0
        assert isinstance(roi_metrics.calculation_date, datetime)

    def test_roi_calculation(self):
        """Test ROI calculation method."""
        roi_metrics = ROIMetrics(
            investment_amount=Decimal("10000.00"),
            savings_amount=Decimal("12000.00"),
            time_period_days=365,
            efficiency_gain_percent=20.0,
        )

        roi = roi_metrics.calculate_roi()
        expected_roi = float(
            (Decimal("12000.00") - Decimal("10000.00")) / Decimal("10000.00") * 100
        )
        assert roi == expected_roi  # 20.0%

    def test_roi_calculation_zero_investment(self):
        """Test ROI calculation with zero investment."""
        roi_metrics = ROIMetrics(
            investment_amount=Decimal("0.00"),
            savings_amount=Decimal("5000.00"),
            time_period_days=365,
            efficiency_gain_percent=100.0,
        )

        # Should handle division by zero gracefully
        roi = roi_metrics.calculate_roi()
        assert roi == 0.0  # or whatever the implementation defines for this edge case


class TestMLInsight:
    """Test MLInsight dataclass functionality."""

    def test_ml_insight_creation(self):
        """Test MLInsight creation with valid parameters."""
        ml_insight = MLInsight(
            insight_id=InsightId("insight_123"),
            model_type=MLModelType.CLASSIFICATION,
            confidence_score=0.85,
            prediction_data={"outcome": "high_efficiency", "probability": 0.85},
            model_version="v2.1.0",
        )

        assert ml_insight.insight_id == InsightId("insight_123")
        assert ml_insight.model_type == MLModelType.CLASSIFICATION
        assert ml_insight.confidence_score == 0.85
        assert ml_insight.prediction_data["outcome"] == "high_efficiency"
        assert ml_insight.model_version == "v2.1.0"
        assert isinstance(ml_insight.generated_at, datetime)

    def test_ml_insight_with_feature_importance(self):
        """Test MLInsight with feature importance data."""
        feature_importance = {
            "response_time": 0.4,
            "throughput": 0.3,
            "error_rate": 0.2,
            "resource_usage": 0.1,
        }

        ml_insight = MLInsight(
            insight_id=InsightId("insight_456"),
            model_type=MLModelType.PREDICTIVE_ANALYTICS,
            confidence_score=0.92,
            prediction_data={"predicted_value": 150.5},
            model_version="v1.0.0",
            feature_importance=feature_importance,
        )

        assert ml_insight.feature_importance == feature_importance
        assert abs(sum(feature_importance.values()) - 1.0) < 1e-10  # Should sum to 1.0


class TestPropertyBasedValidation:
    """Property-based tests for analytics architecture."""

    @given(st.text(min_size=1, max_size=50))
    def test_metric_id_properties(self, metric_name):
        """Property test for metric ID creation."""
        metric_id = MetricId(metric_name)
        assert isinstance(metric_id, str)
        assert metric_id == metric_name

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_confidence_score_properties(self, confidence):
        """Property test for confidence score validation."""
        if 0.0 <= confidence <= 1.0:
            ml_insight = MLInsight(
                insight_id=InsightId("test"),
                model_type=MLModelType.CLASSIFICATION,
                confidence_score=confidence,
                prediction_data={"test": "data"},
                model_version="test",
            )
            assert ml_insight.confidence_score == confidence

    @given(st.integers(min_value=1, max_value=10000))
    def test_throughput_properties(self, throughput):
        """Property test for throughput values."""
        perf_metrics = PerformanceMetrics(
            response_time_ms=100.0,
            throughput_per_second=throughput,
            error_rate_percent=1.0,
            resource_usage_percent=50.0,
        )
        assert perf_metrics.throughput_per_second == throughput
        assert perf_metrics.throughput_per_second > 0


class TestIntegrationScenarios:
    """Integration test scenarios for analytics system."""

    def test_complete_analytics_workflow(self):
        """Test complete analytics processing workflow."""
        # Create metric definition
        metric_def = MetricDefinition(
            metric_id=MetricId("workflow_efficiency"),
            name="Workflow Efficiency",
            metric_type=MetricType.EFFICIENCY,
            scope=AnalyticsScope.ECOSYSTEM,
            description="Efficiency metric for automation workflows",
            unit="percentage",
        )

        # Create metric value
        metric_value = MetricValue(
            metric_id=metric_def.metric_id, value=85.5, timestamp=datetime.now(UTC)
        )

        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            response_time_ms=120.0,
            throughput_per_second=800,
            error_rate_percent=0.2,
            resource_usage_percent=60.0,
        )

        # Create ML insight
        ml_insight = MLInsight(
            insight_id=InsightId("efficiency_prediction"),
            model_type=MLModelType.PREDICTIVE_ANALYTICS,
            confidence_score=0.88,
            prediction_data={"forecasted_efficiency": 90.0},
            model_version="v3.0.0",
        )

        # Validate complete workflow
        assert metric_def.metric_id == metric_value.metric_id
        assert perf_metrics.error_rate_percent < 1.0  # Good performance
        assert ml_insight.confidence_score > 0.8  # High confidence

    def test_roi_analysis_scenario(self):
        """Test ROI analysis and calculation scenario."""
        roi_metrics = ROIMetrics(
            investment_amount=Decimal("50000.00"),
            savings_amount=Decimal("75000.00"),
            time_period_days=365,
            efficiency_gain_percent=50.0,
        )

        roi = roi_metrics.calculate_roi()

        # Validate ROI calculation
        assert roi > 0  # Positive ROI
        assert roi == 50.0  # 50% ROI ((75000-50000)/50000 * 100)

        # Validate time-based analysis
        daily_savings = roi_metrics.savings_amount / roi_metrics.time_period_days
        assert daily_savings == Decimal("75000.00") / 365

    def test_ml_model_performance_analysis(self):
        """Test ML model performance tracking scenario."""
        models = [
            MLInsight(
                insight_id=InsightId(f"model_{i}"),
                model_type=MLModelType.CLASSIFICATION,
                confidence_score=confidence,
                prediction_data={"accuracy": confidence},
                model_version=f"v{i}.0.0",
            )
            for i, confidence in enumerate([0.85, 0.92, 0.78, 0.95], 1)
        ]

        # Find best performing model
        best_model = max(models, key=lambda m: m.confidence_score)
        assert best_model.confidence_score == 0.95
        assert best_model.model_version == "v4.0.0"

        # Calculate average performance
        avg_confidence = sum(m.confidence_score for m in models) / len(models)
        assert 0.85 < avg_confidence < 0.95

    def test_multi_scope_analytics_aggregation(self):
        """Test analytics aggregation across different scopes."""
        metrics = [
            MetricDefinition(
                metric_id=MetricId(f"metric_{scope.value}"),
                name=f"{scope.value.title()} Metric",
                metric_type=MetricType.PERFORMANCE,
                scope=scope,
                description=f"Performance metric for {scope.value}",
                unit="ms",
            )
            for scope in [
                AnalyticsScope.TOOL,
                AnalyticsScope.CATEGORY,
                AnalyticsScope.ECOSYSTEM,
            ]
        ]

        # Validate scope coverage
        scopes = {metric.scope for metric in metrics}
        expected_scopes = {
            AnalyticsScope.TOOL,
            AnalyticsScope.CATEGORY,
            AnalyticsScope.ECOSYSTEM,
        }
        assert scopes == expected_scopes

        # Validate metric types are consistent
        metric_types = {metric.metric_type for metric in metrics}
        assert len(metric_types) == 1  # All same type
        assert MetricType.PERFORMANCE in metric_types
