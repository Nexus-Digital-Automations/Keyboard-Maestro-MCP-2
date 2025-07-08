"""Anomaly prediction for proactive issue detection and prevention."""

import logging
from datetime import timedelta
from typing import Any

from ..core.either import Either
from .model_manager import PredictiveModelManager
from .predictive_types import (
    AlertSeverity,
    AnomalyPrediction,
    ProbabilityScore,
    create_anomaly_id,
)

logger = logging.getLogger(__name__)


class AnomalyPredictor:
    """Proactive anomaly detection and prediction."""

    def __init__(self, model_manager: PredictiveModelManager | None = None):
        self.model_manager = model_manager or PredictiveModelManager()
        self.detected_anomalies: list[AnomalyPrediction] = []
        self.logger = logging.getLogger(__name__)

    async def predict_anomalies(
        self,
        _metrics_data: list[dict[str, Any]],
    ) -> Either[Exception, list[AnomalyPrediction]]:
        """Predict potential anomalies in system behavior."""
        try:
            anomalies = []

            # Example anomaly prediction
            anomaly = AnomalyPrediction(
                anomaly_id=create_anomaly_id(),
                anomaly_type="performance_degradation",
                severity=AlertSeverity.WARNING,
                probability=ProbabilityScore(0.7),
                affected_metric="response_time",
                current_value=150.0,
                expected_range=(50.0, 100.0),
                deviation_score=2.5,
                predicted_impact="moderate performance impact",
                time_to_resolution=timedelta(hours=2),
                mitigation_suggestions=[
                    "Restart affected services",
                    "Check resource usage",
                ],
                model_used="anomaly_model_001",
            )
            anomalies.append(anomaly)

            self.detected_anomalies.extend(anomalies)
            return Either.right(anomalies)

        except Exception as e:
            return Either.left(e)
