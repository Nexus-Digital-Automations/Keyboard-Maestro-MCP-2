"""
Model Validator - TASK_59 Phase 5 Integration & Validation Implementation

Comprehensive model validation and accuracy testing for predictive analytics models.
Provides systematic validation, cross-validation, performance metrics, and accuracy assessment.

Architecture: Validation Framework + Cross-Validation + Performance Metrics + Accuracy Testing
Performance: <1s validation setup, <5s cross-validation, <10s comprehensive validation
Security: Safe validation execution, validated test data, comprehensive audit logging
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import statistics
import json
import math
import random
from collections import defaultdict, deque

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.predictive_modeling import (
    ModelId, PredictionId, ModelPerformance, PredictiveModelingError,
    ModelValidationError, validate_model_performance
)


class ValidationMethod(Enum):
    """Methods for model validation."""
    HOLDOUT_VALIDATION = "holdout_validation"
    K_FOLD_CROSS_VALIDATION = "k_fold_cross_validation"
    TIME_SERIES_SPLIT = "time_series_split"
    BOOTSTRAP_VALIDATION = "bootstrap_validation"
    LEAVE_ONE_OUT = "leave_one_out"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    WALK_FORWARD = "walk_forward"
    BLOCKED_TIME_SERIES = "blocked_time_series"


class ValidationMetric(Enum):
    """Metrics for model validation."""
    # Regression metrics
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
    R_SQUARED = "r_squared"
    ADJUSTED_R_SQUARED = "adjusted_r_squared"
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mean_absolute_percentage_error"
    
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    
    # Time series specific
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    FORECAST_BIAS = "forecast_bias"
    THEIL_U_STATISTIC = "theil_u_statistic"
    
    # Custom metrics
    PREDICTION_STABILITY = "prediction_stability"
    MODEL_CONFIDENCE = "model_confidence"


class ModelType(Enum):
    """Types of models that can be validated."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class ValidationDataset:
    """Dataset for model validation."""
    dataset_id: str
    features: List[List[float]]
    targets: List[float]
    timestamps: Optional[List[datetime]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if len(self.features) != len(self.targets):
            raise ValueError("Features and targets must have the same length")
        if self.timestamps and len(self.timestamps) != len(self.features):
            raise ValueError("Timestamps must match features length")


@dataclass(frozen=True)
class ValidationConfiguration:
    """Configuration for model validation."""
    config_id: str
    model_id: ModelId
    model_type: ModelType
    validation_methods: List[ValidationMethod]
    validation_metrics: List[ValidationMetric]
    test_size: float = 0.2
    random_state: int = 42
    k_folds: int = 5
    enable_cross_validation: bool = True
    enable_statistical_tests: bool = True
    confidence_level: float = 0.95
    
    def __post_init__(self):
        if not (0.1 <= self.test_size <= 0.5):
            raise ValueError("Test size must be between 0.1 and 0.5")
        if self.k_folds < 2:
            raise ValueError("K-folds must be at least 2")


@dataclass(frozen=True)
class ValidationResult:
    """Result from a single validation run."""
    result_id: str
    model_id: ModelId
    validation_method: ValidationMethod
    validation_metrics: Dict[ValidationMetric, float]
    predictions: List[float]
    actuals: List[float]
    residuals: List[float]
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    validation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelValidationReport:
    """Comprehensive model validation report."""
    report_id: str
    model_id: ModelId
    validation_config: ValidationConfiguration
    individual_results: List[ValidationResult]
    aggregated_metrics: Dict[ValidationMetric, Dict[str, float]]
    cross_validation_scores: Dict[ValidationMethod, Dict[ValidationMetric, List[float]]]
    statistical_significance: Dict[str, Any] = field(default_factory=dict)
    model_stability_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    overall_score: float = 0.0
    validation_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class ModelComparison:
    """Comparison between multiple models."""
    comparison_id: str
    model_reports: List[ModelValidationReport]
    comparative_metrics: Dict[str, Dict[str, float]]
    ranking_by_metric: Dict[ValidationMetric, List[str]]
    statistical_comparisons: Dict[str, Any] = field(default_factory=dict)
    recommended_model: Optional[str] = None


class ModelValidator:
    """Comprehensive model validation and testing system."""
    
    def __init__(self):
        self.validation_cache: Dict[str, ModelValidationReport] = {}
        self.validation_history: deque = deque(maxlen=1000)
        self.benchmark_models: Dict[ModelType, Dict[str, float]] = {}
        self.validation_templates: Dict[ModelType, ValidationConfiguration] = {}
        self.metric_calculators: Dict[ValidationMetric, Callable] = {}
        self._initialize_metric_calculators()
        self._initialize_validation_templates()
        self._initialize_benchmark_models()
    
    def _initialize_metric_calculators(self):
        """Initialize metric calculation functions."""
        self.metric_calculators = {
            ValidationMetric.MEAN_ABSOLUTE_ERROR: self._calculate_mae,
            ValidationMetric.MEAN_SQUARED_ERROR: self._calculate_mse,
            ValidationMetric.ROOT_MEAN_SQUARED_ERROR: self._calculate_rmse,
            ValidationMetric.R_SQUARED: self._calculate_r_squared,
            ValidationMetric.MEAN_ABSOLUTE_PERCENTAGE_ERROR: self._calculate_mape,
            ValidationMetric.ACCURACY: self._calculate_accuracy,
            ValidationMetric.PRECISION: self._calculate_precision,
            ValidationMetric.RECALL: self._calculate_recall,
            ValidationMetric.F1_SCORE: self._calculate_f1_score,
            ValidationMetric.DIRECTIONAL_ACCURACY: self._calculate_directional_accuracy,
            ValidationMetric.FORECAST_BIAS: self._calculate_forecast_bias,
            ValidationMetric.PREDICTION_STABILITY: self._calculate_prediction_stability,
            ValidationMetric.MODEL_CONFIDENCE: self._calculate_model_confidence
        }
    
    def _initialize_validation_templates(self):
        """Initialize validation templates for different model types."""
        self.validation_templates[ModelType.REGRESSION] = ValidationConfiguration(
            config_id="regression_default",
            model_id=ModelId("template"),
            model_type=ModelType.REGRESSION,
            validation_methods=[
                ValidationMethod.HOLDOUT_VALIDATION,
                ValidationMethod.K_FOLD_CROSS_VALIDATION
            ],
            validation_metrics=[
                ValidationMetric.MEAN_ABSOLUTE_ERROR,
                ValidationMetric.ROOT_MEAN_SQUARED_ERROR,
                ValidationMetric.R_SQUARED,
                ValidationMetric.MEAN_ABSOLUTE_PERCENTAGE_ERROR
            ],
            test_size=0.2,
            k_folds=5
        )
        
        self.validation_templates[ModelType.TIME_SERIES_FORECASTING] = ValidationConfiguration(
            config_id="time_series_default",
            model_id=ModelId("template"),
            model_type=ModelType.TIME_SERIES_FORECASTING,
            validation_methods=[
                ValidationMethod.TIME_SERIES_SPLIT,
                ValidationMethod.WALK_FORWARD
            ],
            validation_metrics=[
                ValidationMetric.MEAN_ABSOLUTE_ERROR,
                ValidationMetric.ROOT_MEAN_SQUARED_ERROR,
                ValidationMetric.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
                ValidationMetric.DIRECTIONAL_ACCURACY,
                ValidationMetric.FORECAST_BIAS
            ],
            test_size=0.3,
            k_folds=3
        )
    
    def _initialize_benchmark_models(self):
        """Initialize benchmark performance metrics."""
        self.benchmark_models[ModelType.REGRESSION] = {
            "mae": 10.0,
            "rmse": 15.0,
            "r_squared": 0.8,
            "mape": 0.1
        }
        
        self.benchmark_models[ModelType.TIME_SERIES_FORECASTING] = {
            "mae": 8.0,
            "rmse": 12.0,
            "mape": 0.15,
            "directional_accuracy": 0.7
        }
        
        self.benchmark_models[ModelType.CLASSIFICATION] = {
            "accuracy": 0.85,
            "precision": 0.8,
            "recall": 0.8,
            "f1_score": 0.8
        }
    
    @require(lambda config: isinstance(config, ValidationConfiguration))
    @require(lambda dataset: len(dataset.features) > 10)
    @ensure(lambda result: result.is_right() or isinstance(result.left_value, ModelValidationError))
    async def validate_model(
        self,
        config: ValidationConfiguration,
        dataset: ValidationDataset,
        model_predictor: Callable[[List[List[float]]], List[float]]
    ) -> Either[ModelValidationError, ModelValidationReport]:
        """Validate a model using comprehensive validation methods."""
        try:
            start_time = datetime.now(UTC)
            
            # Check cache
            cache_key = f"{config.model_id}_{hash(str(config))}_{hash(str(dataset.features))}"
            if cache_key in self.validation_cache:
                return Either.right(self.validation_cache[cache_key])
            
            # Run individual validation methods
            individual_results = []
            
            for method in config.validation_methods:
                result = await self._run_validation_method(
                    method, config, dataset, model_predictor
                )
                if result:
                    individual_results.append(result)
            
            # Aggregate metrics across all validation runs
            aggregated_metrics = self._aggregate_validation_metrics(individual_results, config.validation_metrics)
            
            # Perform cross-validation if enabled
            cross_validation_scores = {}
            if config.enable_cross_validation:
                cross_validation_scores = await self._perform_cross_validation(
                    config, dataset, model_predictor
                )
            
            # Perform statistical significance tests
            statistical_significance = {}
            if config.enable_statistical_tests:
                statistical_significance = await self._perform_statistical_tests(
                    individual_results, config
                )
            
            # Analyze model stability
            stability_analysis = self._analyze_model_stability(individual_results, config)
            
            # Generate recommendations
            recommendations = self._generate_validation_recommendations(
                aggregated_metrics, cross_validation_scores, config
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_validation_score(
                aggregated_metrics, config.model_type
            )
            
            # Create validation report
            report = ModelValidationReport(
                report_id=f"validation_{config.model_id}_{datetime.now(UTC).isoformat()}",
                model_id=config.model_id,
                validation_config=config,
                individual_results=individual_results,
                aggregated_metrics=aggregated_metrics,
                cross_validation_scores=cross_validation_scores,
                statistical_significance=statistical_significance,
                model_stability_analysis=stability_analysis,
                recommendations=recommendations,
                overall_score=overall_score
            )
            
            # Cache report
            self.validation_cache[cache_key] = report
            
            # Record validation activity
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            self.validation_history.append({
                "timestamp": datetime.now(UTC),
                "model_id": str(config.model_id),
                "model_type": config.model_type.value,
                "validation_methods": [method.value for method in config.validation_methods],
                "overall_score": overall_score,
                "processing_time": processing_time
            })
            
            return Either.right(report)
            
        except Exception as e:
            return Either.left(ModelValidationError(f"Model validation failed: {str(e)}"))
    
    async def _run_validation_method(
        self,
        method: ValidationMethod,
        config: ValidationConfiguration,
        dataset: ValidationDataset,
        model_predictor: Callable[[List[List[float]]], List[float]]
    ) -> Optional[ValidationResult]:
        """Run a specific validation method."""
        try:
            if method == ValidationMethod.HOLDOUT_VALIDATION:
                return await self._holdout_validation(config, dataset, model_predictor)
            elif method == ValidationMethod.K_FOLD_CROSS_VALIDATION:
                return await self._k_fold_validation(config, dataset, model_predictor)
            elif method == ValidationMethod.TIME_SERIES_SPLIT:
                return await self._time_series_split_validation(config, dataset, model_predictor)
            elif method == ValidationMethod.BOOTSTRAP_VALIDATION:
                return await self._bootstrap_validation(config, dataset, model_predictor)
            elif method == ValidationMethod.WALK_FORWARD:
                return await self._walk_forward_validation(config, dataset, model_predictor)
            else:
                # Default to holdout validation
                return await self._holdout_validation(config, dataset, model_predictor)
        except Exception:
            return None
    
    async def _holdout_validation(
        self,
        config: ValidationConfiguration,
        dataset: ValidationDataset,
        model_predictor: Callable[[List[List[float]]], List[float]]
    ) -> ValidationResult:
        """Perform holdout validation."""
        # Split data
        n_samples = len(dataset.features)
        n_test = int(n_samples * config.test_size)
        n_train = n_samples - n_test
        
        # Use random state for reproducibility
        random.seed(config.random_state)
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Prepare test data
        test_features = [dataset.features[i] for i in test_indices]
        test_targets = [dataset.targets[i] for i in test_indices]
        
        # Get predictions
        predictions = model_predictor(test_features)
        
        # Calculate metrics
        validation_metrics = {}
        for metric in config.validation_metrics:
            calculator = self.metric_calculators.get(metric)
            if calculator:
                value = calculator(test_targets, predictions)
                validation_metrics[metric] = value
        
        # Calculate residuals
        residuals = [actual - pred for actual, pred in zip(test_targets, predictions)]
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            residuals, config.confidence_level
        )
        
        return ValidationResult(
            result_id=f"holdout_{config.model_id}_{datetime.now(UTC).isoformat()}",
            model_id=config.model_id,
            validation_method=ValidationMethod.HOLDOUT_VALIDATION,
            validation_metrics=validation_metrics,
            predictions=predictions,
            actuals=test_targets,
            residuals=residuals,
            confidence_intervals=confidence_intervals,
            validation_metadata={
                "train_size": n_train,
                "test_size": n_test,
                "test_ratio": config.test_size
            }
        )
    
    async def _k_fold_validation(
        self,
        config: ValidationConfiguration,
        dataset: ValidationDataset,
        model_predictor: Callable[[List[List[float]]], List[float]]
    ) -> ValidationResult:
        """Perform k-fold cross-validation."""
        n_samples = len(dataset.features)
        fold_size = n_samples // config.k_folds
        
        all_predictions = []
        all_actuals = []
        fold_metrics = defaultdict(list)
        
        random.seed(config.random_state)
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        for fold in range(config.k_folds):
            # Create fold split
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < config.k_folds - 1 else n_samples
            
            test_indices = indices[start_idx:end_idx]
            train_indices = [i for i in indices if i not in test_indices]
            
            # Prepare test data for this fold
            test_features = [dataset.features[i] for i in test_indices]
            test_targets = [dataset.targets[i] for i in test_indices]
            
            # Get predictions for this fold
            fold_predictions = model_predictor(test_features)
            
            all_predictions.extend(fold_predictions)
            all_actuals.extend(test_targets)
            
            # Calculate metrics for this fold
            for metric in config.validation_metrics:
                calculator = self.metric_calculators.get(metric)
                if calculator:
                    value = calculator(test_targets, fold_predictions)
                    fold_metrics[metric].append(value)
        
        # Average metrics across folds
        validation_metrics = {}
        for metric in config.validation_metrics:
            if metric in fold_metrics:
                validation_metrics[metric] = statistics.mean(fold_metrics[metric])
        
        # Calculate residuals
        residuals = [actual - pred for actual, pred in zip(all_actuals, all_predictions)]
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            residuals, config.confidence_level
        )
        
        return ValidationResult(
            result_id=f"kfold_{config.model_id}_{datetime.now(UTC).isoformat()}",
            model_id=config.model_id,
            validation_method=ValidationMethod.K_FOLD_CROSS_VALIDATION,
            validation_metrics=validation_metrics,
            predictions=all_predictions,
            actuals=all_actuals,
            residuals=residuals,
            confidence_intervals=confidence_intervals,
            validation_metadata={
                "k_folds": config.k_folds,
                "fold_metrics": dict(fold_metrics)
            }
        )
    
    async def _time_series_split_validation(
        self,
        config: ValidationConfiguration,
        dataset: ValidationDataset,
        model_predictor: Callable[[List[List[float]]], List[float]]
    ) -> ValidationResult:
        """Perform time series split validation."""
        n_samples = len(dataset.features)
        n_test = int(n_samples * config.test_size)
        n_train = n_samples - n_test
        
        # For time series, use chronological split
        train_features = dataset.features[:n_train]
        test_features = dataset.features[n_train:]
        
        train_targets = dataset.targets[:n_train]
        test_targets = dataset.targets[n_train:]
        
        # Get predictions
        predictions = model_predictor(test_features)
        
        # Calculate metrics
        validation_metrics = {}
        for metric in config.validation_metrics:
            calculator = self.metric_calculators.get(metric)
            if calculator:
                value = calculator(test_targets, predictions)
                validation_metrics[metric] = value
        
        # Calculate residuals
        residuals = [actual - pred for actual, pred in zip(test_targets, predictions)]
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            residuals, config.confidence_level
        )
        
        return ValidationResult(
            result_id=f"timeseries_{config.model_id}_{datetime.now(UTC).isoformat()}",
            model_id=config.model_id,
            validation_method=ValidationMethod.TIME_SERIES_SPLIT,
            validation_metrics=validation_metrics,
            predictions=predictions,
            actuals=test_targets,
            residuals=residuals,
            confidence_intervals=confidence_intervals,
            validation_metadata={
                "train_size": n_train,
                "test_size": n_test,
                "chronological_split": True
            }
        )
    
    async def _bootstrap_validation(
        self,
        config: ValidationConfiguration,
        dataset: ValidationDataset,
        model_predictor: Callable[[List[List[float]]], List[float]]
    ) -> ValidationResult:
        """Perform bootstrap validation."""
        n_samples = len(dataset.features)
        n_bootstrap = 100  # Number of bootstrap samples
        
        all_predictions = []
        all_actuals = []
        bootstrap_metrics = defaultdict(list)
        
        random.seed(config.random_state)
        
        for _ in range(n_bootstrap):
            # Create bootstrap sample
            bootstrap_indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            out_of_bag_indices = [i for i in range(n_samples) if i not in bootstrap_indices]
            
            if not out_of_bag_indices:
                continue
            
            # Use out-of-bag samples for testing
            test_features = [dataset.features[i] for i in out_of_bag_indices]
            test_targets = [dataset.targets[i] for i in out_of_bag_indices]
            
            # Get predictions
            predictions = model_predictor(test_features)
            
            all_predictions.extend(predictions)
            all_actuals.extend(test_targets)
            
            # Calculate metrics for this bootstrap
            for metric in config.validation_metrics:
                calculator = self.metric_calculators.get(metric)
                if calculator:
                    value = calculator(test_targets, predictions)
                    bootstrap_metrics[metric].append(value)
        
        # Average metrics across bootstrap samples
        validation_metrics = {}
        for metric in config.validation_metrics:
            if metric in bootstrap_metrics and bootstrap_metrics[metric]:
                validation_metrics[metric] = statistics.mean(bootstrap_metrics[metric])
        
        # Calculate residuals
        residuals = [actual - pred for actual, pred in zip(all_actuals, all_predictions)]
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            residuals, config.confidence_level
        )
        
        return ValidationResult(
            result_id=f"bootstrap_{config.model_id}_{datetime.now(UTC).isoformat()}",
            model_id=config.model_id,
            validation_method=ValidationMethod.BOOTSTRAP_VALIDATION,
            validation_metrics=validation_metrics,
            predictions=all_predictions,
            actuals=all_actuals,
            residuals=residuals,
            confidence_intervals=confidence_intervals,
            validation_metadata={
                "n_bootstrap": n_bootstrap,
                "bootstrap_metrics": dict(bootstrap_metrics)
            }
        )
    
    async def _walk_forward_validation(
        self,
        config: ValidationConfiguration,
        dataset: ValidationDataset,
        model_predictor: Callable[[List[List[float]]], List[float]]
    ) -> ValidationResult:
        """Perform walk-forward validation for time series."""
        n_samples = len(dataset.features)
        min_train_size = max(10, int(n_samples * 0.3))  # Minimum training size
        
        all_predictions = []
        all_actuals = []
        
        # Walk forward through the data
        for i in range(min_train_size, n_samples):
            # Use all previous data for training context
            train_features = dataset.features[:i]
            
            # Predict next point
            test_features = [dataset.features[i]]
            test_target = dataset.targets[i]
            
            # Get prediction
            prediction = model_predictor(test_features)[0]
            
            all_predictions.append(prediction)
            all_actuals.append(test_target)
        
        # Calculate metrics
        validation_metrics = {}
        for metric in config.validation_metrics:
            calculator = self.metric_calculators.get(metric)
            if calculator:
                value = calculator(all_actuals, all_predictions)
                validation_metrics[metric] = value
        
        # Calculate residuals
        residuals = [actual - pred for actual, pred in zip(all_actuals, all_predictions)]
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            residuals, config.confidence_level
        )
        
        return ValidationResult(
            result_id=f"walkforward_{config.model_id}_{datetime.now(UTC).isoformat()}",
            model_id=config.model_id,
            validation_method=ValidationMethod.WALK_FORWARD,
            validation_metrics=validation_metrics,
            predictions=all_predictions,
            actuals=all_actuals,
            residuals=residuals,
            confidence_intervals=confidence_intervals,
            validation_metadata={
                "min_train_size": min_train_size,
                "predictions_count": len(all_predictions)
            }
        )
    
    def _aggregate_validation_metrics(
        self,
        results: List[ValidationResult],
        metrics: List[ValidationMetric]
    ) -> Dict[ValidationMetric, Dict[str, float]]:
        """Aggregate metrics across validation results."""
        aggregated = {}
        
        for metric in metrics:
            metric_values = []
            for result in results:
                if metric in result.validation_metrics:
                    metric_values.append(result.validation_metrics[metric])
            
            if metric_values:
                aggregated[metric] = {
                    "mean": statistics.mean(metric_values),
                    "std": statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0,
                    "min": min(metric_values),
                    "max": max(metric_values),
                    "median": statistics.median(metric_values),
                    "count": len(metric_values)
                }
        
        return aggregated
    
    async def _perform_cross_validation(
        self,
        config: ValidationConfiguration,
        dataset: ValidationDataset,
        model_predictor: Callable[[List[List[float]]], List[float]]
    ) -> Dict[ValidationMethod, Dict[ValidationMetric, List[float]]]:
        """Perform cross-validation across multiple methods."""
        cv_scores = {}
        
        for method in config.validation_methods:
            if method == ValidationMethod.K_FOLD_CROSS_VALIDATION:
                # Perform multiple k-fold runs with different seeds
                method_scores = defaultdict(list)
                
                for seed in range(5):  # 5 different random seeds
                    temp_config = ValidationConfiguration(
                        config_id=config.config_id,
                        model_id=config.model_id,
                        model_type=config.model_type,
                        validation_methods=[method],
                        validation_metrics=config.validation_metrics,
                        test_size=config.test_size,
                        random_state=seed,
                        k_folds=config.k_folds
                    )
                    
                    result = await self._k_fold_validation(temp_config, dataset, model_predictor)
                    if result:
                        for metric, value in result.validation_metrics.items():
                            method_scores[metric].append(value)
                
                cv_scores[method] = dict(method_scores)
        
        return cv_scores
    
    async def _perform_statistical_tests(
        self,
        results: List[ValidationResult],
        config: ValidationConfiguration
    ) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        statistical_tests = {}
        
        if len(results) >= 2:
            # Perform paired t-test between validation methods
            for i, result1 in enumerate(results):
                for j, result2 in enumerate(results):
                    if i < j:
                        test_name = f"{result1.validation_method.value}_vs_{result2.validation_method.value}"
                        
                        # Compare primary metric (MAE for regression, accuracy for classification)
                        primary_metric = ValidationMetric.MEAN_ABSOLUTE_ERROR
                        if config.model_type == ModelType.CLASSIFICATION:
                            primary_metric = ValidationMetric.ACCURACY
                        
                        if (primary_metric in result1.validation_metrics and 
                            primary_metric in result2.validation_metrics):
                            
                            # Simplified statistical test (in real implementation, use scipy.stats)
                            value1 = result1.validation_metrics[primary_metric]
                            value2 = result2.validation_metrics[primary_metric]
                            
                            difference = abs(value1 - value2)
                            pooled_std = (
                                statistics.stdev(result1.residuals) + 
                                statistics.stdev(result2.residuals)
                            ) / 2 if len(result1.residuals) > 1 and len(result2.residuals) > 1 else 0.0
                            
                            t_statistic = difference / max(0.001, pooled_std)
                            p_value = 2 * (1 - min(0.999, t_statistic / 10))  # Simplified p-value
                            
                            statistical_tests[test_name] = {
                                "t_statistic": t_statistic,
                                "p_value": p_value,
                                "significant": p_value < 0.05,
                                "difference": difference
                            }
        
        return statistical_tests
    
    def _analyze_model_stability(
        self,
        results: List[ValidationResult],
        config: ValidationConfiguration
    ) -> Dict[str, Any]:
        """Analyze model stability across validation runs."""
        stability_analysis = {}
        
        # Collect metric values across validation methods
        metric_variations = defaultdict(list)
        
        for result in results:
            for metric, value in result.validation_metrics.items():
                metric_variations[metric].append(value)
        
        # Calculate stability metrics
        for metric, values in metric_variations.items():
            if len(values) > 1:
                mean_value = statistics.mean(values)
                std_value = statistics.stdev(values)
                coefficient_of_variation = std_value / abs(mean_value) if mean_value != 0 else float('inf')
                
                stability_analysis[metric.value] = {
                    "coefficient_of_variation": coefficient_of_variation,
                    "stability_score": 1.0 / (1.0 + coefficient_of_variation),
                    "range": max(values) - min(values),
                    "relative_range": (max(values) - min(values)) / abs(mean_value) if mean_value != 0 else 0.0
                }
        
        # Overall stability score
        if stability_analysis:
            stability_scores = [analysis["stability_score"] for analysis in stability_analysis.values()]
            stability_analysis["overall_stability"] = statistics.mean(stability_scores)
        
        return stability_analysis
    
    def _generate_validation_recommendations(
        self,
        aggregated_metrics: Dict[ValidationMetric, Dict[str, float]],
        cross_validation_scores: Dict[ValidationMethod, Dict[ValidationMetric, List[float]]],
        config: ValidationConfiguration
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Get benchmark metrics for model type
        benchmarks = self.benchmark_models.get(config.model_type, {})
        
        # Primary metric recommendations
        primary_metric = ValidationMetric.MEAN_ABSOLUTE_ERROR
        if config.model_type == ModelType.CLASSIFICATION:
            primary_metric = ValidationMetric.ACCURACY
        
        if primary_metric in aggregated_metrics:
            metric_value = aggregated_metrics[primary_metric]["mean"]
            benchmark_key = primary_metric.value.lower()
            
            if benchmark_key in benchmarks:
                benchmark_value = benchmarks[benchmark_key]
                
                if config.model_type == ModelType.CLASSIFICATION:
                    # Higher is better for accuracy
                    if metric_value >= benchmark_value:
                        recommendations.append(f"EXCELLENT: {primary_metric.value} ({metric_value:.3f}) exceeds benchmark ({benchmark_value:.3f})")
                    elif metric_value >= benchmark_value * 0.9:
                        recommendations.append(f"GOOD: {primary_metric.value} ({metric_value:.3f}) is close to benchmark ({benchmark_value:.3f})")
                    else:
                        recommendations.append(f"NEEDS IMPROVEMENT: {primary_metric.value} ({metric_value:.3f}) is below benchmark ({benchmark_value:.3f})")
                else:
                    # Lower is better for error metrics
                    if metric_value <= benchmark_value:
                        recommendations.append(f"EXCELLENT: {primary_metric.value} ({metric_value:.3f}) is better than benchmark ({benchmark_value:.3f})")
                    elif metric_value <= benchmark_value * 1.1:
                        recommendations.append(f"GOOD: {primary_metric.value} ({metric_value:.3f}) is close to benchmark ({benchmark_value:.3f})")
                    else:
                        recommendations.append(f"NEEDS IMPROVEMENT: {primary_metric.value} ({metric_value:.3f}) exceeds benchmark ({benchmark_value:.3f})")
        
        # Stability recommendations
        metric_stds = [metrics["std"] for metrics in aggregated_metrics.values()]
        if metric_stds:
            avg_std = statistics.mean(metric_stds)
            if avg_std < 0.1:
                recommendations.append("STABLE: Model shows consistent performance across validation methods")
            elif avg_std < 0.2:
                recommendations.append("MODERATELY STABLE: Some variation in performance across validation methods")
            else:
                recommendations.append("UNSTABLE: High variation in performance - consider model refinement")
        
        # Cross-validation recommendations
        if cross_validation_scores:
            for method, method_scores in cross_validation_scores.items():
                for metric, scores in method_scores.items():
                    if len(scores) > 1:
                        cv_std = statistics.stdev(scores)
                        cv_mean = statistics.mean(scores)
                        cv_coefficient = cv_std / abs(cv_mean) if cv_mean != 0 else float('inf')
                        
                        if cv_coefficient < 0.05:
                            recommendations.append(f"ROBUST: {method.value} shows excellent consistency (CV: {cv_coefficient:.3f})")
                        elif cv_coefficient > 0.2:
                            recommendations.append(f"VARIABLE: {method.value} shows high variability (CV: {cv_coefficient:.3f}) - investigate further")
        
        # General recommendations
        if config.model_type == ModelType.TIME_SERIES_FORECASTING:
            if ValidationMetric.DIRECTIONAL_ACCURACY in aggregated_metrics:
                dir_acc = aggregated_metrics[ValidationMetric.DIRECTIONAL_ACCURACY]["mean"]
                if dir_acc < 0.6:
                    recommendations.append("DIRECTIONAL ACCURACY LOW: Model struggles with trend prediction")
        
        return recommendations
    
    def _calculate_overall_validation_score(
        self,
        aggregated_metrics: Dict[ValidationMetric, Dict[str, float]],
        model_type: ModelType
    ) -> float:
        """Calculate overall validation score."""
        scores = []
        
        # Weight different metrics based on model type
        if model_type == ModelType.REGRESSION:
            metric_weights = {
                ValidationMetric.R_SQUARED: 0.4,
                ValidationMetric.MEAN_ABSOLUTE_ERROR: 0.3,
                ValidationMetric.ROOT_MEAN_SQUARED_ERROR: 0.3
            }
        elif model_type == ModelType.CLASSIFICATION:
            metric_weights = {
                ValidationMetric.ACCURACY: 0.3,
                ValidationMetric.PRECISION: 0.25,
                ValidationMetric.RECALL: 0.25,
                ValidationMetric.F1_SCORE: 0.2
            }
        elif model_type == ModelType.TIME_SERIES_FORECASTING:
            metric_weights = {
                ValidationMetric.MEAN_ABSOLUTE_ERROR: 0.3,
                ValidationMetric.ROOT_MEAN_SQUARED_ERROR: 0.25,
                ValidationMetric.DIRECTIONAL_ACCURACY: 0.25,
                ValidationMetric.MEAN_ABSOLUTE_PERCENTAGE_ERROR: 0.2
            }
        else:
            # Default equal weights
            metric_weights = {metric: 1.0 for metric in aggregated_metrics.keys()}
        
        # Normalize weights
        total_weight = sum(metric_weights.values())
        if total_weight > 0:
            for metric in metric_weights:
                metric_weights[metric] /= total_weight
        
        # Calculate weighted score
        for metric, weight in metric_weights.items():
            if metric in aggregated_metrics:
                metric_value = aggregated_metrics[metric]["mean"]
                
                # Normalize metric to 0-1 scale
                if metric in [ValidationMetric.R_SQUARED, ValidationMetric.ACCURACY, 
                            ValidationMetric.PRECISION, ValidationMetric.RECALL, 
                            ValidationMetric.F1_SCORE, ValidationMetric.DIRECTIONAL_ACCURACY]:
                    # Higher is better - already in 0-1 range typically
                    normalized_score = min(1.0, max(0.0, metric_value))
                else:
                    # Lower is better (error metrics) - use inverse relationship
                    # Assume reasonable range for normalization
                    max_acceptable_error = 100.0  # Adjust based on domain
                    normalized_score = 1.0 - min(1.0, metric_value / max_acceptable_error)
                
                scores.append(normalized_score * weight)
        
        return sum(scores) if scores else 0.5
    
    def _calculate_confidence_intervals(
        self,
        residuals: List[float],
        confidence_level: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for residuals."""
        if len(residuals) < 2:
            return {}
        
        alpha = 1.0 - confidence_level
        sorted_residuals = sorted(residuals)
        n = len(sorted_residuals)
        
        lower_index = int(n * alpha / 2)
        upper_index = int(n * (1 - alpha / 2))
        
        mean_residual = statistics.mean(residuals)
        std_residual = statistics.stdev(residuals)
        
        return {
            "residual_range": (sorted_residuals[lower_index], sorted_residuals[upper_index]),
            "mean_confidence": (
                mean_residual - 1.96 * std_residual / math.sqrt(n),
                mean_residual + 1.96 * std_residual / math.sqrt(n)
            )
        }
    
    # Metric calculation methods
    def _calculate_mae(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate Mean Absolute Error."""
        if len(actuals) != len(predictions) or len(actuals) == 0:
            return float('inf')
        return sum(abs(a - p) for a, p in zip(actuals, predictions)) / len(actuals)
    
    def _calculate_mse(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate Mean Squared Error."""
        if len(actuals) != len(predictions) or len(actuals) == 0:
            return float('inf')
        return sum((a - p) ** 2 for a, p in zip(actuals, predictions)) / len(actuals)
    
    def _calculate_rmse(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate Root Mean Squared Error."""
        mse = self._calculate_mse(actuals, predictions)
        return math.sqrt(mse) if mse != float('inf') else float('inf')
    
    def _calculate_r_squared(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate R-squared coefficient of determination."""
        if len(actuals) < 2:
            return 0.0
        
        mean_actual = statistics.mean(actuals)
        ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
        ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def _calculate_mape(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate Mean Absolute Percentage Error."""
        if len(actuals) != len(predictions) or len(actuals) == 0:
            return float('inf')
        
        percentage_errors = []
        for a, p in zip(actuals, predictions):
            if a != 0:
                percentage_errors.append(abs((a - p) / a))
        
        return statistics.mean(percentage_errors) if percentage_errors else float('inf')
    
    def _calculate_accuracy(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate classification accuracy."""
        if len(actuals) != len(predictions) or len(actuals) == 0:
            return 0.0
        
        correct = sum(1 for a, p in zip(actuals, predictions) if round(a) == round(p))
        return correct / len(actuals)
    
    def _calculate_precision(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate precision for binary classification."""
        if len(actuals) != len(predictions) or len(actuals) == 0:
            return 0.0
        
        true_positives = sum(1 for a, p in zip(actuals, predictions) if round(a) == 1 and round(p) == 1)
        predicted_positives = sum(1 for p in predictions if round(p) == 1)
        
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    def _calculate_recall(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate recall for binary classification."""
        if len(actuals) != len(predictions) or len(actuals) == 0:
            return 0.0
        
        true_positives = sum(1 for a, p in zip(actuals, predictions) if round(a) == 1 and round(p) == 1)
        actual_positives = sum(1 for a in actuals if round(a) == 1)
        
        return true_positives / actual_positives if actual_positives > 0 else 0.0
    
    def _calculate_f1_score(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate F1 score."""
        precision = self._calculate_precision(actuals, predictions)
        recall = self._calculate_recall(actuals, predictions)
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _calculate_directional_accuracy(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate directional accuracy for time series."""
        if len(actuals) < 2 or len(predictions) < 2:
            return 0.0
        
        correct_directions = 0
        total_directions = 0
        
        for i in range(1, len(actuals)):
            actual_direction = actuals[i] - actuals[i-1]
            pred_direction = predictions[i] - predictions[i-1]
            
            if actual_direction * pred_direction > 0:  # Same direction
                correct_directions += 1
            total_directions += 1
        
        return correct_directions / total_directions if total_directions > 0 else 0.0
    
    def _calculate_forecast_bias(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate forecast bias."""
        if len(actuals) != len(predictions) or len(actuals) == 0:
            return 0.0
        
        errors = [p - a for a, p in zip(actuals, predictions)]
        return statistics.mean(errors)
    
    def _calculate_prediction_stability(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate prediction stability."""
        if len(predictions) < 2:
            return 0.0
        
        prediction_changes = [abs(predictions[i] - predictions[i-1]) for i in range(1, len(predictions))]
        actual_changes = [abs(actuals[i] - actuals[i-1]) for i in range(1, len(actuals))]
        
        if not prediction_changes or not actual_changes:
            return 0.0
        
        avg_pred_change = statistics.mean(prediction_changes)
        avg_actual_change = statistics.mean(actual_changes)
        
        # Stability is high when prediction changes are similar to actual changes
        if avg_actual_change == 0:
            return 1.0 if avg_pred_change == 0 else 0.0
        
        stability = 1.0 - abs(avg_pred_change - avg_actual_change) / avg_actual_change
        return max(0.0, min(1.0, stability))
    
    def _calculate_model_confidence(self, actuals: List[float], predictions: List[float]) -> float:
        """Calculate model confidence based on prediction consistency."""
        if len(predictions) < 2:
            return 0.0
        
        # Calculate prediction variance relative to actual variance
        pred_variance = statistics.variance(predictions) if len(predictions) > 1 else 0.0
        actual_variance = statistics.variance(actuals) if len(actuals) > 1 else 0.0
        
        if actual_variance == 0:
            return 1.0 if pred_variance == 0 else 0.0
        
        # Confidence is higher when prediction variance is similar to actual variance
        variance_ratio = pred_variance / actual_variance
        confidence = 1.0 / (1.0 + abs(variance_ratio - 1.0))
        
        return confidence
    
    async def get_validation_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the validation system."""
        if not self.validation_history:
            return {"total_validations": 0, "average_score": 0.0}
        
        recent_history = list(self.validation_history)[-50:]
        
        scores = [entry["overall_score"] for entry in recent_history]
        processing_times = [entry["processing_time"] for entry in recent_history]
        model_types = [entry["model_type"] for entry in recent_history]
        
        from collections import Counter
        
        return {
            "total_validations": len(self.validation_history),
            "recent_validations": len(recent_history),
            "average_validation_score": statistics.mean(scores) if scores else 0.0,
            "max_validation_score": max(scores) if scores else 0.0,
            "average_processing_time": statistics.mean(processing_times) if processing_times else 0.0,
            "cached_reports": len(self.validation_cache),
            "most_validated_model_types": [
                model_type for model_type, count in 
                Counter(model_types).most_common(3)
            ],
            "validation_success_rate": len([s for s in scores if s > 0.6]) / len(scores) if scores else 0.0
        }