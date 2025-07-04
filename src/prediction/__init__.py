"""
Predictive automation module for machine learning-powered optimization and forecasting.

This module provides predictive capabilities including ML model management, performance forecasting,
proactive optimization, resource prediction, and automated system improvement recommendations.
"""

from .predictive_types import (
    PredictiveModel,
    PredictionRequest,
    OptimizationSuggestion,
    PerformanceForecast,
    ResourcePrediction,
    PatternAnalysis,
    AnomalyPrediction,
    CapacityPlan,
    WorkflowOptimization,
    PredictiveAlert
)
from .model_manager import PredictiveModelManager
from .performance_predictor import PerformancePredictor
from .optimization_engine import OptimizationEngine
from .resource_predictor import ResourcePredictor
from .pattern_recognition import PatternRecognitionEngine
from .anomaly_predictor import AnomalyPredictor
from .capacity_planner import CapacityPlanner
from .workflow_optimizer import WorkflowOptimizer
from .predictive_alerts import PredictiveAlertSystem

__all__ = [
    # Core Types
    'PredictiveModel',
    'PredictionRequest', 
    'OptimizationSuggestion',
    'PerformanceForecast',
    'ResourcePrediction',
    'PatternAnalysis',
    'AnomalyPrediction',
    'CapacityPlan',
    'WorkflowOptimization',
    'PredictiveAlert',
    
    # Core Components
    'PredictiveModelManager',
    'PerformancePredictor',
    'OptimizationEngine',
    'ResourcePredictor',
    'PatternRecognitionEngine',
    'AnomalyPredictor',
    'CapacityPlanner',
    'WorkflowOptimizer',
    'PredictiveAlertSystem'
]