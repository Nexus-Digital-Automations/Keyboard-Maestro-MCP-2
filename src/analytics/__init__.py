"""Analytics module for comprehensive automation insights and business intelligence.

This module provides analytics capabilities including metrics collection, performance analysis,
ROI calculation, ML-powered insights, and real-time monitoring across the ecosystem.
"""

from .anomaly_detector import AnomalyDetector

# from .roi_calculator import ROICalculator  # TODO: Implement ROICalculator
# from .ml_insights_engine import MLInsightsEngine  # TODO: Add numpy dependency first
from .dashboard_generator import DashboardGenerator
from .metrics_collector import MetricsCollector
from .performance_analyzer import PerformanceAnalyzer
from .recommendation_engine import RecommendationEngine
from .report_automation import ReportAutomation

__all__ = [
    "AnomalyDetector",
    # 'ROICalculator',  # TODO: Implement ROICalculator
    # 'MLInsightsEngine',  # TODO: Add numpy dependency first
    "DashboardGenerator",
    "MetricsCollector",
    "PerformanceAnalyzer",
    "RecommendationEngine",
    "ReportAutomation",
]
