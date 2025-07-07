"""Predictive automation module for machine learning-powered optimization and forecasting.

This module provides predictive capabilities including ML model management, performance forecasting,
proactive optimization, resource prediction, and automated system improvement recommendations.
"""

from .anomaly_predictor import AnomalyPredictor
from .capacity_planner import CapacityPlanner
from .model_manager import PredictiveModelManager
from .optimization_engine import OptimizationEngine
from .pattern_recognition import PatternRecognitionEngine
from .performance_predictor import PerformancePredictor
from .predictive_alerts import PredictiveAlertSystem
from .predictive_types import (
    AnomalyPrediction,
    CapacityPlan,
    OptimizationSuggestion,
    PatternAnalysis,
    PerformanceForecast,
    PredictionRequest,
    PredictiveAlert,
    PredictiveModel,
    ResourcePrediction,
    WorkflowOptimization,
)
from .resource_predictor import ResourcePredictor
from .workflow_optimizer import WorkflowOptimizer

__all__ = [
    "AnomalyPrediction",
    "AnomalyPredictor",
    "CapacityPlan",
    "CapacityPlanner",
    "OptimizationEngine",
    "OptimizationSuggestion",
    "PatternAnalysis",
    "PatternRecognitionEngine",
    "PerformanceForecast",
    "PerformancePredictor",
    "PredictionRequest",
    "PredictiveAlert",
    "PredictiveAlertSystem",
    # Core Types
    "PredictiveModel",
    # Core Components
    "PredictiveModelManager",
    "ResourcePrediction",
    "ResourcePredictor",
    "WorkflowOptimization",
    "WorkflowOptimizer",
]
