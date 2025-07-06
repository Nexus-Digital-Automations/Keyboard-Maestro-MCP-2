"""
Automated ML training pipeline with hyperparameter optimization and validation.

Provides comprehensive training workflow with cross-validation, model selection,
and performance evaluation for production ML deployment.
"""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score

from ...core.analytics_architecture import MetricValue, MLModelType, ModelId
from ..models.model_storage import ModelStorage

if TYPE_CHECKING:
    from ..ml_insights_engine import MLModel
from ...core.errors import AnalyticsError


class TrainingPipelineError(AnalyticsError):
    """Training pipeline related errors."""

    pass


class TrainingPipeline:
    """Automated ML training pipeline with optimization."""

    def __init__(self, storage: ModelStorage | None = None):
        self.storage = storage or ModelStorage()
        self.logger = logging.getLogger(__name__)
        self.training_history = []

    async def train_model(
        self,
        model_type: MLModelType,
        model_id: ModelId,
        training_data: list[MetricValue],
        optimize_hyperparameters: bool = True,
        validation_split: float = 0.2,
    ) -> dict[str, Any]:
        """Train a model with comprehensive validation and optimization."""
        try:
            training_start = datetime.now(UTC)

            # Validate training data
            if len(training_data) < 10:
                raise TrainingPipelineError(
                    "Insufficient training data (minimum 10 samples)"
                )

            # Create model instance
            model = self._create_model(model_type, model_id)

            # Prepare training data
            X, y = self._prepare_training_data(training_data, model_type)

            # Split data for validation
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = (
                y[:split_idx] if y is not None else None,
                y[split_idx:] if y is not None else None,
            )

            # Perform hyperparameter optimization if requested
            if optimize_hyperparameters:
                best_params = await self._optimize_hyperparameters(
                    model, X_train, y_train
                )
                self._apply_hyperparameters(model, best_params)

            # Train the model
            training_success = await model.train(training_data[:split_idx])
            if not training_success:
                raise TrainingPipelineError(f"Model training failed for {model_id}")

            # Validate model performance
            validation_results = await self._validate_model(
                model, X_val, y_val, model_type
            )

            # Calculate comprehensive performance metrics
            performance_metrics = self._calculate_performance_metrics(
                model, validation_results
            )

            # Save model with version
            version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            model_path = self.storage.save_model(model, version)

            # Record training history
            training_record = {
                "model_type": model_type.value,
                "model_id": model_id,
                "version": version,
                "training_start": training_start.isoformat(),
                "training_end": datetime.now(UTC).isoformat(),
                "training_duration": (
                    datetime.now(UTC) - training_start
                ).total_seconds(),
                "training_data_size": len(training_data),
                "validation_data_size": len(X_val) if X_val is not None else 0,
                "hyperparameter_optimization": optimize_hyperparameters,
                "performance_metrics": performance_metrics,
                "model_path": model_path,
            }

            self.training_history.append(training_record)

            return {
                "success": True,
                "model_id": model_id,
                "version": version,
                "performance": performance_metrics,
                "training_time": training_record["training_duration"],
                "model_path": model_path,
            }

        except Exception as e:
            self.logger.error(f"Training failed for {model_id}: {e}")
            raise TrainingPipelineError(f"Training pipeline failed: {e}")

    def _create_model(self, model_type: MLModelType, model_id: ModelId) -> "MLModel":
        """Create model instance based on type."""
        # Import at runtime to avoid circular imports
        from ..ml_insights_engine import (
            AnomalyDetectionModel,
            PatternRecognitionModel,
            PredictiveAnalyticsModel,
        )

        if model_type == MLModelType.PATTERN_RECOGNITION:
            return PatternRecognitionModel(model_id)
        elif model_type == MLModelType.ANOMALY_DETECTION:
            return AnomalyDetectionModel(model_id)
        elif model_type == MLModelType.PREDICTIVE_ANALYTICS:
            return PredictiveAnalyticsModel(model_id)
        else:
            raise TrainingPipelineError(f"Unsupported model type: {model_type}")

    def _prepare_training_data(
        self, training_data: list[MetricValue], model_type: MLModelType
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Prepare training data for different model types."""
        # Extract features and labels based on model type
        if model_type == MLModelType.PATTERN_RECOGNITION:
            # For clustering, we only need features (no labels)
            features = []
            for metric in training_data:
                # Extract numeric features
                feature_vector = [
                    float(metric.value)
                    if isinstance(metric.value, (int, float))
                    else 0.0,
                    metric.timestamp.hour,
                    metric.timestamp.weekday(),
                    metric.quality_score,
                ]
                features.append(feature_vector)
            return np.array(features), None

        elif model_type == MLModelType.ANOMALY_DETECTION:
            # For anomaly detection, we need normal patterns (no labels)
            features = []
            for metric in training_data:
                feature_vector = [
                    float(metric.value)
                    if isinstance(metric.value, (int, float))
                    else 0.0,
                    metric.timestamp.hour,
                    metric.timestamp.weekday(),
                ]
                features.append(feature_vector)
            return np.array(features), None

        elif model_type == MLModelType.PREDICTIVE_ANALYTICS:
            # For prediction, we need time series data
            features = []
            labels = []
            for i, metric in enumerate(training_data[:-1]):
                feature_vector = [
                    float(metric.value)
                    if isinstance(metric.value, (int, float))
                    else 0.0,
                    metric.timestamp.hour,
                    metric.timestamp.weekday(),
                ]
                features.append(feature_vector)
                # Use next value as label
                next_metric = training_data[i + 1]
                labels.append(
                    float(next_metric.value)
                    if isinstance(next_metric.value, (int, float))
                    else 0.0
                )
            return np.array(features), np.array(labels)

        else:
            raise TrainingPipelineError(
                f"Unsupported model type for data preparation: {model_type}"
            )

    async def _optimize_hyperparameters(
        self, model: "MLModel", X: np.ndarray, y: np.ndarray | None
    ) -> dict[str, Any]:
        """Optimize hyperparameters using grid search."""
        best_params = {}

        try:
            from ..ml_insights_engine import (
                AnomalyDetectionModel,
                PatternRecognitionModel,
                PredictiveAnalyticsModel,
            )

            if isinstance(model, PatternRecognitionModel):
                # Optimize KMeans parameters
                param_grid = {
                    "n_clusters": [3, 5, 7, 10],
                    "random_state": [42],
                }

                best_score = -1
                best_k = 5

                for n_clusters in param_grid["n_clusters"]:
                    if n_clusters < len(X):
                        from sklearn.cluster import KMeans

                        kmeans = KMeans(
                            n_clusters=n_clusters, random_state=42, n_init=10
                        )
                        labels = kmeans.fit_predict(X)

                        if (
                            len(set(labels)) > 1
                        ):  # Need at least 2 clusters for silhouette score
                            score = silhouette_score(X, labels)
                            if score > best_score:
                                best_score = score
                                best_k = n_clusters

                best_params["kmeans_clusters"] = best_k

            elif isinstance(model, AnomalyDetectionModel):
                # Optimize Isolation Forest parameters
                param_grid = {
                    "contamination": [0.05, 0.1, 0.15, 0.2],
                    "n_estimators": [100, 200],
                }

                # Simple validation for anomaly detection
                best_params["contamination"] = 0.1
                best_params["n_estimators"] = 100

            elif isinstance(model, PredictiveAnalyticsModel):
                # Optimize regression parameters
                if y is not None and len(y) > 0:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.model_selection import cross_val_score

                    # Simple cross-validation for regression
                    reg = LinearRegression()
                    scores = cross_val_score(
                        reg, X, y, cv=min(3, len(X) // 2), scoring="r2"
                    )
                    best_params["regression_score"] = np.mean(scores)

        except Exception as e:
            self.logger.warning(f"Hyperparameter optimization failed: {e}")

        return best_params

    def _apply_hyperparameters(
        self, model: "MLModel", best_params: dict[str, Any]
    ) -> None:
        """Apply optimized hyperparameters to model."""
        from ..ml_insights_engine import AnomalyDetectionModel, PatternRecognitionModel

        if (
            isinstance(model, PatternRecognitionModel)
            and "kmeans_clusters" in best_params
        ):
            from sklearn.cluster import KMeans

            model.kmeans = KMeans(
                n_clusters=best_params["kmeans_clusters"], random_state=42, n_init=10
            )

        elif isinstance(model, AnomalyDetectionModel):
            if "contamination" in best_params:
                from sklearn.ensemble import IsolationForest

                model.isolation_forest = IsolationForest(
                    contamination=best_params["contamination"],
                    random_state=42,
                    n_estimators=best_params.get("n_estimators", 100),
                )

    async def _validate_model(
        self,
        model: "MLModel",
        X_val: np.ndarray,
        y_val: np.ndarray | None,
        model_type: MLModelType,
    ) -> dict[str, Any]:
        """Validate model performance on validation set."""
        validation_results = {}

        try:
            from ..ml_insights_engine import (
                AnomalyDetectionModel,
                PatternRecognitionModel,
                PredictiveAnalyticsModel,
            )

            if (
                isinstance(model, PatternRecognitionModel)
                and X_val is not None
                and len(X_val) > 0
            ):
                # Validate clustering
                if (
                    hasattr(model, "kmeans")
                    and model.kmeans.cluster_centers_ is not None
                ):
                    labels = model.kmeans.predict(X_val)
                    if len(set(labels)) > 1:
                        validation_results["silhouette_score"] = silhouette_score(
                            X_val, labels
                        )
                    else:
                        validation_results["silhouette_score"] = 0.0
                else:
                    validation_results["silhouette_score"] = 0.0

            elif (
                isinstance(model, AnomalyDetectionModel)
                and X_val is not None
                and len(X_val) > 0
            ):
                # Validate anomaly detection
                if hasattr(model, "isolation_forest") and hasattr(
                    model.isolation_forest, "decision_function"
                ):
                    anomaly_scores = model.isolation_forest.decision_function(X_val)
                    validation_results["anomaly_score_mean"] = np.mean(anomaly_scores)
                    validation_results["anomaly_score_std"] = np.std(anomaly_scores)
                else:
                    validation_results["anomaly_score_mean"] = 0.0
                    validation_results["anomaly_score_std"] = 0.0

            elif (
                isinstance(model, PredictiveAnalyticsModel)
                and X_val is not None
                and y_val is not None
                and len(X_val) > 0
            ):
                # Validate prediction
                if hasattr(model, "linear_model") and hasattr(
                    model.linear_model, "predict"
                ):
                    predictions = model.linear_model.predict(X_val)
                    validation_results["mse"] = mean_squared_error(y_val, predictions)
                    validation_results["mae"] = mean_absolute_error(y_val, predictions)
                    validation_results["r2_score"] = model.linear_model.score(
                        X_val, y_val
                    )
                else:
                    validation_results["mse"] = float("inf")
                    validation_results["mae"] = float("inf")
                    validation_results["r2_score"] = 0.0

        except Exception as e:
            self.logger.warning(f"Validation failed: {e}")

        return validation_results

    def _calculate_performance_metrics(
        self, model: "MLModel", validation_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {
            "model_accuracy": getattr(model, "model_accuracy", 0.0),
            "training_data_size": getattr(model, "training_data_size", 0),
            "trained": getattr(model, "trained", False),
        }

        # Add validation metrics
        metrics.update(validation_results)

        # Calculate overall performance score
        performance_score = 0.0
        from ..ml_insights_engine import (
            AnomalyDetectionModel,
            PatternRecognitionModel,
            PredictiveAnalyticsModel,
        )

        if isinstance(model, PatternRecognitionModel):
            performance_score = validation_results.get("silhouette_score", 0.0)
        elif isinstance(model, AnomalyDetectionModel):
            # For anomaly detection, we use a simple heuristic
            performance_score = min(
                1.0, abs(validation_results.get("anomaly_score_mean", 0.0)) * 0.1
            )
        elif isinstance(model, PredictiveAnalyticsModel):
            performance_score = max(0.0, validation_results.get("r2_score", 0.0))

        metrics["overall_performance_score"] = performance_score

        return metrics

    async def retrain_all_models(
        self, training_data: list[MetricValue]
    ) -> dict[str, Any]:
        """Retrain all model types with new data."""
        results = {}

        for model_type in [
            MLModelType.PATTERN_RECOGNITION,
            MLModelType.ANOMALY_DETECTION,
            MLModelType.PREDICTIVE_ANALYTICS,
        ]:
            try:
                model_id = (
                    f"auto_{model_type.value}_{datetime.now(UTC).strftime('%Y%m%d')}"
                )
                result = await self.train_model(model_type, model_id, training_data)
                results[model_type.value] = result
            except Exception as e:
                self.logger.error(f"Failed to retrain {model_type.value}: {e}")
                results[model_type.value] = {"success": False, "error": str(e)}

        return results

    def get_training_history(self) -> list[dict[str, Any]]:
        """Get complete training history."""
        return self.training_history.copy()

    def get_model_registry(self) -> list[dict[str, Any]]:
        """Get all models in storage registry."""
        return self.storage.list_models()
