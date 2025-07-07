"""Test real ML implementation in ML insights engine.

This test suite verifies that the ML insights engine is using real
machine learning algorithms instead of mock/simulated implementations.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from src.analytics.ml_insights_engine import (
    AnomalyDetectionModel,
    MLInsightsEngine,
    PatternRecognitionModel,
    PredictiveAnalyticsModel,
)
from src.core.analytics_architecture import (
    MetricValue,
    MLModelType,
    PrivacyMode,
)


@pytest.fixture
def sample_metric_data() -> Any:
    """Generate realistic sample metric data for testing."""
    base_time = datetime.now(UTC)
    metrics = []

    # Generate time series data with patterns
    for i in range(100):
        timestamp = base_time + timedelta(hours=i)
        # Add some seasonality and trend
        value = 50 + 10 * np.sin(i * 2 * np.pi / 24) + 0.1 * i + np.random.normal(0, 2)

        metric = MetricValue(
            metric_id=f"test_metric_{i % 5}",
            value=value,
            timestamp=timestamp,
            source_tool="test_tool_main",  # Use same tool for all metrics
        )
        metrics.append(metric)

    return metrics


@pytest.fixture
def anomaly_metric_data() -> Any:
    """Generate metric data with known anomalies."""
    base_time = datetime.now(UTC)
    metrics = []

    # Generate normal data
    for i in range(50):
        timestamp = base_time + timedelta(hours=i)
        value = 50 + np.random.normal(0, 5)  # Normal variation

        metric = MetricValue(
            metric_id="test_metric",
            value=value,
            timestamp=timestamp,
            source_tool="test_tool",
        )
        metrics.append(metric)

    # Add anomalies
    for i in range(50, 55):
        timestamp = base_time + timedelta(hours=i)
        value = 150 + np.random.normal(0, 5)  # Anomalous high values

        metric = MetricValue(
            metric_id="test_metric",
            value=value,
            timestamp=timestamp,
            source_tool="test_tool",
        )
        metrics.append(metric)

    return metrics


class TestPatternRecognitionModel:
    """Test real pattern recognition using scikit-learn."""

    @pytest.mark.asyncio
    async def test_real_pattern_recognition_training(self, sample_metric_data: Any) -> None:
        """Test that pattern recognition uses real ML algorithms."""
        model = PatternRecognitionModel("test_pattern_model")

        # Train the model with real data
        success = await model.train(sample_metric_data)
        assert success is True
        assert model.trained is True
        assert model.training_data_size == len(sample_metric_data)

        # Verify that real scikit-learn models are trained
        assert hasattr(model, "kmeans")
        assert hasattr(model, "dbscan")
        assert hasattr(model, "scaler")
        assert model.kmeans.cluster_centers_ is not None  # KMeans should have centers
        assert len(model.trained_models) > 0  # Should have training results

    @pytest.mark.asyncio
    async def test_pattern_finding_with_real_clustering(self, sample_metric_data: Any) -> None:
        """Test that pattern finding uses real clustering algorithms."""
        model = PatternRecognitionModel("test_pattern_model")
        await model.train(sample_metric_data)

        # Find patterns using trained models
        patterns = await model.find_patterns(sample_metric_data[:50])

        # Should detect patterns using real ML
        assert len(patterns) > 0
        pattern = patterns[0]
        assert "method" in pattern
        assert pattern["method"] == "ml_clustering"  # Should use ML method
        assert "cluster_info" in pattern
        assert "kmeans_clusters" in pattern["cluster_info"]
        assert "seasonality_strength" in pattern["cluster_info"]

    @pytest.mark.asyncio
    async def test_pattern_model_accuracy_calculation(self, sample_metric_data: Any) -> None:
        """Test that model accuracy is calculated using real metrics."""
        model = PatternRecognitionModel("test_pattern_model")
        await model.train(sample_metric_data)

        # Model accuracy should be calculated from silhouette score
        assert 0.0 <= model.model_accuracy <= 1.0
        assert model.model_accuracy > 0.5  # Should be reasonable for synthetic data


class TestAnomalyDetectionModel:
    """Test real anomaly detection using Isolation Forest and One-Class SVM."""

    @pytest.mark.asyncio
    async def test_real_anomaly_detection_training(self, sample_metric_data: Any) -> None:
        """Test that anomaly detection uses real ML algorithms."""
        model = AnomalyDetectionModel("test_anomaly_model")

        # Train the model with real data
        success = await model.train(sample_metric_data)
        assert success is True
        assert model.trained is True

        # Verify that real scikit-learn models are initialized and trained
        assert hasattr(model, "isolation_forest")
        assert hasattr(model, "one_class_svm")
        assert hasattr(model, "scaler")
        assert model.isolation_forest.n_features_in_ is not None  # Should be fitted

    @pytest.mark.asyncio
    async def test_ml_based_anomaly_detection(self, anomaly_metric_data: Any) -> None:
        """Test that anomaly detection uses ML ensemble methods."""
        model = AnomalyDetectionModel("test_anomaly_model")

        # Train on data without anomalies
        normal_data = anomaly_metric_data[:50]
        await model.train(normal_data)

        # Detect anomalies in data with known anomalies
        anomalies = await model.detect_anomalies(anomaly_metric_data)

        # Should detect anomalies using ML methods
        assert len(anomalies) > 0
        anomaly = anomalies[0]
        assert anomaly["anomaly_type"] == "ml_ensemble_detection"
        assert "detection_method" in anomaly
        assert anomaly["detection_method"] == "isolation_forest_svm_ensemble"
        assert "model_stats" in anomaly
        assert "isolation_forest_outliers" in anomaly["model_stats"]

    @pytest.mark.asyncio
    async def test_ensemble_scoring(self, anomaly_metric_data: Any) -> None:
        """Test that ensemble scoring combines multiple ML models."""
        model = AnomalyDetectionModel("test_anomaly_model")
        await model.train(anomaly_metric_data[:50])

        anomalies = await model.detect_anomalies(anomaly_metric_data)

        if anomalies:
            anomaly = anomalies[0]
            # Check that ensemble scores are provided
            for anomalous_value in anomaly["anomalous_values"]:
                assert "isolation_forest_score" in anomalous_value
                assert "svm_score" in anomalous_value
                assert "ensemble_score" in anomalous_value


class TestPredictiveAnalyticsModel:
    """Test real predictive analytics using ARIMA and regression."""

    @pytest.mark.asyncio
    async def test_real_predictive_model_training(self, sample_metric_data: Any) -> None:
        """Test that predictive analytics uses real time series models."""
        model = PredictiveAnalyticsModel("test_prediction_model")

        # Train the model with sufficient data for ARIMA
        success = await model.train(sample_metric_data)
        assert success is True
        assert model.trained is True

        # Verify that real models are available
        assert hasattr(model, "linear_model")
        assert hasattr(model, "arima_models")
        assert len(model.trained_models) > 0

    @pytest.mark.asyncio
    async def test_arima_model_selection(self, sample_metric_data: Any) -> None:
        """Test that ARIMA models are trained and selected based on AIC."""
        model = PredictiveAnalyticsModel("test_prediction_model")
        await model.train(sample_metric_data)

        # Should have trained models for tools with sufficient data
        assert len(model.trained_models) > 0

        for _tool, model_info in model.trained_models.items():
            assert "best_model" in model_info
            assert "performance" in model_info
            assert model_info["best_model"] in ["linear", "arima"]

            # If ARIMA was selected, should have AIC score
            if model_info["best_model"] == "arima":
                assert "arima" in model_info["performance"]
                assert "aic" in model_info["performance"]["arima"]

    @pytest.mark.asyncio
    async def test_real_forecasting_with_confidence_intervals(self, sample_metric_data: Any) -> None:
        """Test that forecasting provides real predictions with confidence intervals."""
        model = PredictiveAnalyticsModel("test_prediction_model")
        await model.train(sample_metric_data)

        # Generate forecasts
        forecast_result = await model.generate_forecast(
            sample_metric_data,
            timedelta(days=7),
        )

        assert "forecasts" in forecast_result
        assert len(forecast_result["forecasts"]) > 0

        # Check a specific forecast
        for _tool, forecast in forecast_result["forecasts"].items():
            assert "forecast_points" in forecast
            assert "model_used" in forecast
            assert forecast["model_used"] in [
                "arima",
                "linear_regression",
                "simple_linear",
            ]

            # If ARIMA was used, should have confidence intervals
            if forecast["model_used"] == "arima":
                for point in forecast["forecast_points"]:
                    assert "prediction_interval_lower" in point
                    assert "prediction_interval_upper" in point


class TestMLInsightsEngine:
    """Test the complete ML insights engine with real implementations."""

    @pytest.mark.asyncio
    async def test_comprehensive_ml_insights_generation(self, sample_metric_data: Any) -> None:
        """Test that the insights engine generates real ML insights."""
        engine = MLInsightsEngine(privacy_mode=PrivacyMode.COMPLIANT)

        # Train models first
        for model in engine.models.values():
            await model.train(sample_metric_data)

        # Generate insights
        insights = await engine.generate_comprehensive_insights(sample_metric_data)

        # Should generate insights from real ML models
        assert len(insights) > 0

        # Check that different types of insights are generated
        insight_types = {insight.model_type for insight in insights}
        assert MLModelType.PATTERN_RECOGNITION in insight_types

        # Verify insights have real ML backing
        for insight in insights:
            assert insight.confidence > 0
            assert insight.description is not None
            assert insight.supporting_data is not None

    @pytest.mark.asyncio
    async def test_model_performance_tracking(self, sample_metric_data: Any) -> None:
        """Test that model performance metrics are tracked."""
        engine = MLInsightsEngine(privacy_mode=PrivacyMode.COMPLIANT)

        # Train models
        for model in engine.models.values():
            await model.train(sample_metric_data)

        # Get performance metrics
        performance = await engine.get_model_performance()

        # Should have performance data for each model type
        for model_type in [
            MLModelType.PATTERN_RECOGNITION,
            MLModelType.ANOMALY_DETECTION,
            MLModelType.PREDICTIVE_ANALYTICS,
        ]:
            assert model_type.value in performance
            model_info = performance[model_type.value]
            assert model_info["trained"] is True
            assert model_info["model_accuracy"] >= 0
            assert model_info["training_data_size"] > 0

    @pytest.mark.asyncio
    async def test_model_retraining_with_new_data(self, sample_metric_data: Any) -> None:
        """Test that models can be retrained with new data."""
        engine = MLInsightsEngine()

        # Initial training
        initial_results = await engine.retrain_models(sample_metric_data[:50])
        assert all(initial_results.values())

        # Retrain with more data
        retrain_results = await engine.retrain_models(sample_metric_data)
        assert all(retrain_results.values())

        # Models should be updated
        for model in engine.models.values():
            assert model.trained is True
            assert model.training_data_size == len(sample_metric_data)


@pytest.mark.property
class TestMLPropertyBasedTests:
    """Property-based tests for ML implementations."""

    @pytest.mark.asyncio
    async def test_pattern_recognition_consistency(self) -> None:
        """Test that pattern recognition produces consistent results."""
        # Create deterministic test data
        base_time = datetime.now(UTC)
        metrics = []
        for i in range(50):
            metric = MetricValue(
                metric_id="test_metric",
                value=50 + 10 * np.sin(i * 0.1),  # Deterministic pattern
                timestamp=base_time + timedelta(hours=i),
                source_tool="test_tool",
            )
            metrics.append(metric)

        model = PatternRecognitionModel("test_model")
        success = await model.train(metrics)

        if success:
            patterns = await model.find_patterns(metrics)
            # Pattern detection should be deterministic for same data
            patterns2 = await model.find_patterns(metrics)
            assert len(patterns) == len(patterns2)
            # Content should be the same for same input
            if patterns:
                assert patterns[0]["confidence"] == patterns2[0]["confidence"]

    @pytest.mark.asyncio
    async def test_anomaly_detection_sensitivity(self) -> None:
        """Test that anomaly detection is appropriately sensitive."""
        # Normal data should not trigger many anomalies
        normal_data = []
        base_time = datetime.now(UTC)
        for i in range(50):
            metric = MetricValue(
                metric_id="test_metric",
                value=50 + np.random.normal(0, 2),  # Small variation
                timestamp=base_time + timedelta(hours=i),
                source_tool="test_tool",
            )
            normal_data.append(metric)

        model = AnomalyDetectionModel("test_model")
        await model.train(normal_data)
        anomalies = await model.detect_anomalies(normal_data)

        # Should not detect many anomalies in training data
        assert len(anomalies) <= 5  # Allow for some false positives
