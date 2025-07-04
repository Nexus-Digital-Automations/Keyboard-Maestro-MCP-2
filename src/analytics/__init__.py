"""
Analytics module for comprehensive automation insights and business intelligence.

This module provides analytics capabilities including metrics collection, performance analysis,
ROI calculation, ML-powered insights, and real-time monitoring across the ecosystem.
"""

from .metrics_collector import MetricsCollector
from .performance_analyzer import PerformanceAnalyzer
# from .roi_calculator import ROICalculator  # TODO: Implement ROICalculator
# from .ml_insights_engine import MLInsightsEngine  # TODO: Add numpy dependency first
from .dashboard_generator import DashboardGenerator
from .report_automation import ReportAutomation
from .anomaly_detector import AnomalyDetector
from .recommendation_engine import RecommendationEngine

__all__ = [
    'MetricsCollector',
    'PerformanceAnalyzer',
    # 'ROICalculator',  # TODO: Implement ROICalculator
    # 'MLInsightsEngine',  # TODO: Add numpy dependency first
    'DashboardGenerator',
    'ReportAutomation',
    'AnomalyDetector',
    'RecommendationEngine'
]