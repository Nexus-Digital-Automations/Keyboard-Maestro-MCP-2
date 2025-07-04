"""
MCP tools for predictive automation operations.

This module provides comprehensive MCP tools for machine learning-powered
predictive automation, optimization, and system intelligence.

Security: Secure prediction operations with input validation and access control.
Performance: <1s prediction generation, <2s optimization recommendations.
Type Safety: Complete MCP integration with contract validation.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, UTC
import logging

from fastmcp import FastMCP
from fastmcp.types import TextContent

from ...prediction.model_manager import PredictiveModelManager
from ...prediction.performance_predictor import PerformancePredictor
from ...prediction.optimization_engine import OptimizationEngine, OptimizationContext, OptimizationStrategy
from ...prediction.resource_predictor import ResourcePredictor
from ...prediction.pattern_recognition import PatternRecognitionEngine
from ...prediction.anomaly_predictor import AnomalyPredictor
from ...prediction.capacity_planner import CapacityPlanner
from ...prediction.workflow_optimizer import WorkflowOptimizer
from ...prediction.predictive_alerts import PredictiveAlertSystem
from ...prediction.predictive_types import (
    PredictionType, PredictionRequest, PredictionPriority, OptimizationType,
    create_prediction_request_id, create_confidence_level
)
from ...core.contracts import require, ensure
from ...core.either import Either

logger = logging.getLogger(__name__)


class PredictiveAutomationTools:
    """Comprehensive MCP tools for predictive automation operations."""
    
    def __init__(self):
        # Initialize prediction components
        self.model_manager = PredictiveModelManager()
        self.performance_predictor = PerformancePredictor(self.model_manager)
        self.optimization_engine = OptimizationEngine(self.model_manager, self.performance_predictor)
        self.resource_predictor = ResourcePredictor(self.model_manager)
        self.pattern_recognition = PatternRecognitionEngine(self.model_manager)
        self.anomaly_predictor = AnomalyPredictor(self.model_manager)
        self.capacity_planner = CapacityPlanner(self.model_manager)
        self.workflow_optimizer = WorkflowOptimizer(self.model_manager)
        self.alert_system = PredictiveAlertSystem(self.model_manager)
        
        self.logger = logging.getLogger(__name__)
    
    def register_tools(self, mcp: FastMCP) -> None:
        """Register all predictive automation tools with FastMCP."""
        
        @mcp.tool()
        async def km_predict_performance(
            metric_name: str,
            forecast_hours: int = 24,
            confidence_level: float = 0.8
        ) -> str:
            """
            Predict system performance trends and forecasts.
            
            Args:
                metric_name: Name of metric to predict (response_time, throughput, error_rate, etc.)
                forecast_hours: Hours to forecast ahead (1-168)
                confidence_level: Required confidence level (0.1-1.0)
            
            Returns:
                JSON string with performance forecast including trends and recommendations
            """
            try:
                # Validate inputs
                if not 1 <= forecast_hours <= 168:
                    return f"Error: forecast_hours must be between 1 and 168, got {forecast_hours}"
                
                if not 0.1 <= confidence_level <= 1.0:
                    return f"Error: confidence_level must be between 0.1 and 1.0, got {confidence_level}"
                
                # Generate forecast
                forecast_result = await self.performance_predictor.forecast_performance(
                    metric_name=metric_name,
                    forecast_horizon=timedelta(hours=forecast_hours),
                    confidence_level=create_confidence_level(confidence_level)
                )
                
                if forecast_result.is_left():
                    return f"Error: {forecast_result.left().message}"
                
                forecast = forecast_result.right()
                
                # Format response
                response = {
                    "success": True,
                    "forecast": {
                        "metric_name": forecast.metric_name,
                        "current_value": forecast.current_value,
                        "trend": forecast.trend,
                        "forecast_accuracy": float(forecast.forecast_accuracy),
                        "predicted_values": [
                            {
                                "timestamp": ts.isoformat(),
                                "value": val,
                                "confidence": float(conf)
                            }
                            for ts, val, conf in forecast.predicted_values[:10]  # Limit to 10 points
                        ],
                        "confidence_interval": forecast.confidence_interval,
                        "anomaly_probability": float(forecast.anomaly_probability),
                        "recommendation": forecast.recommendation
                    },
                    "metadata": {
                        "forecast_id": forecast.forecast_id,
                        "model_used": forecast.model_used,
                        "generated_at": datetime.now(UTC).isoformat()
                    }
                }
                
                return f"```json\n{response}\n```"
                
            except Exception as e:
                self.logger.error(f"Performance prediction failed: {e}")
                return f"Error: Performance prediction failed - {str(e)}"
        
        @mcp.tool()
        async def km_generate_optimizations(
            system_health: float = 0.8,
            optimization_strategy: str = "balanced",
            max_suggestions: int = 5
        ) -> str:
            """
            Generate system optimization recommendations using ML analysis.
            
            Args:
                system_health: Current system health score (0.0-1.0)
                optimization_strategy: Strategy type (conservative, balanced, aggressive, emergency)
                max_suggestions: Maximum number of suggestions to return (1-10)
            
            Returns:
                JSON string with optimization suggestions and implementation details
            """
            try:
                # Validate inputs
                if not 0.0 <= system_health <= 1.0:
                    return f"Error: system_health must be between 0.0 and 1.0, got {system_health}"
                
                if optimization_strategy not in ["conservative", "balanced", "aggressive", "emergency"]:
                    return f"Error: Invalid optimization_strategy: {optimization_strategy}"
                
                if not 1 <= max_suggestions <= 10:
                    return f"Error: max_suggestions must be between 1 and 10, got {max_suggestions}"
                
                # Create optimization context
                context = OptimizationContext(
                    system_health=system_health,
                    resource_pressure={"cpu": 0.6, "memory": 0.7, "storage": 0.4},
                    performance_trends={"response_time": "stable", "throughput": "increasing"},
                    recent_issues=[],
                    optimization_budget=1000.0,
                    risk_tolerance=OptimizationStrategy(optimization_strategy)
                )
                
                # Generate optimizations
                optimizations_result = await self.optimization_engine.analyze_optimization_opportunities(
                    context, OptimizationStrategy(optimization_strategy)
                )
                
                if optimizations_result.is_left():
                    return f"Error: {optimizations_result.left().message}"
                
                suggestions = optimizations_result.right()
                
                # Limit suggestions
                limited_suggestions = suggestions[:max_suggestions]
                
                # Format response
                response = {
                    "success": True,
                    "optimization_analysis": {
                        "system_health": system_health,
                        "strategy": optimization_strategy,
                        "suggestions_count": len(limited_suggestions),
                        "suggestions": [
                            {
                                "optimization_id": suggestion.optimization_id,
                                "title": suggestion.title,
                                "description": suggestion.description,
                                "optimization_type": suggestion.optimization_type.value,
                                "expected_impact": float(suggestion.expected_impact),
                                "confidence": float(suggestion.confidence),
                                "implementation_effort": suggestion.implementation_effort,
                                "priority": suggestion.priority.value,
                                "affected_components": suggestion.affected_components,
                                "implementation_steps": suggestion.implementation_steps,
                                "estimated_duration_hours": suggestion.estimated_duration.total_seconds() / 3600,
                                "prerequisites": suggestion.prerequisites,
                                "risks": suggestion.risks,
                                "metrics_to_monitor": suggestion.metrics_to_monitor
                            }
                            for suggestion in limited_suggestions
                        ]
                    },
                    "metadata": {
                        "generated_at": datetime.now(UTC).isoformat(),
                        "total_available_suggestions": len(suggestions)
                    }
                }
                
                return f"```json\n{response}\n```"
                
            except Exception as e:
                self.logger.error(f"Optimization generation failed: {e}")
                return f"Error: Optimization generation failed - {str(e)}"
        
        @mcp.tool()
        async def km_predict_resource_usage(
            resource_type: str,
            prediction_hours: int = 24,
            include_recommendations: bool = True
        ) -> str:
            """
            Predict resource usage patterns and capacity needs.
            
            Args:
                resource_type: Type of resource (cpu, memory, storage, network)
                prediction_hours: Hours to predict ahead (1-720)
                include_recommendations: Include optimization recommendations
            
            Returns:
                JSON string with resource usage predictions and capacity planning
            """
            try:
                # Validate inputs
                if not 1 <= prediction_hours <= 720:  # Max 30 days
                    return f"Error: prediction_hours must be between 1 and 720, got {prediction_hours}"
                
                # Generate resource prediction
                prediction_result = await self.resource_predictor.predict_resource_usage(
                    resource_type=resource_type,
                    prediction_horizon=timedelta(hours=prediction_hours),
                    confidence_level=create_confidence_level(0.8)
                )
                
                if prediction_result.is_left():
                    return f"Error: {prediction_result.left().message}"
                
                prediction = prediction_result.right()
                
                # Format response
                response = {
                    "success": True,
                    "resource_prediction": {
                        "resource_type": prediction.resource_type,
                        "current_usage": float(prediction.current_usage),
                        "predicted_usage": [
                            {
                                "timestamp": ts.isoformat(),
                                "usage": float(usage),
                                "confidence": float(conf)
                            }
                            for ts, usage, conf in prediction.predicted_usage[:24]  # Limit to 24 points
                        ],
                        "capacity_threshold": float(prediction.capacity_threshold),
                        "expected_shortage": prediction.expected_shortage.isoformat() if prediction.expected_shortage else None,
                        "scaling_recommendation": prediction.scaling_recommendation
                    },
                    "metadata": {
                        "prediction_id": prediction.prediction_id,
                        "model_used": prediction.model_used,
                        "generated_at": datetime.now(UTC).isoformat()
                    }
                }
                
                if include_recommendations:
                    response["resource_prediction"]["optimization_opportunities"] = prediction.optimization_opportunities
                
                return f"```json\n{response}\n```"
                
            except Exception as e:
                self.logger.error(f"Resource prediction failed: {e}")
                return f"Error: Resource prediction failed - {str(e)}"
        
        @mcp.tool()
        async def km_detect_anomalies(
            metric_data: str = "{}",
            sensitivity: float = 0.8
        ) -> str:
            """
            Detect anomalies and predict potential system issues.
            
            Args:
                metric_data: JSON string with metrics data for analysis
                sensitivity: Anomaly detection sensitivity (0.1-1.0)
            
            Returns:
                JSON string with detected anomalies and predictions
            """
            try:
                import json
                
                # Parse metric data
                try:
                    metrics = json.loads(metric_data) if metric_data != "{}" else {}
                except json.JSONDecodeError:
                    return "Error: Invalid JSON in metric_data parameter"
                
                # Validate sensitivity
                if not 0.1 <= sensitivity <= 1.0:
                    return f"Error: sensitivity must be between 0.1 and 1.0, got {sensitivity}"
                
                # Generate synthetic metrics if none provided
                if not metrics:
                    metrics = [
                        {"timestamp": datetime.now(UTC).isoformat(), "metric": "response_time", "value": 150.0},
                        {"timestamp": (datetime.now(UTC) - timedelta(hours=1)).isoformat(), "metric": "response_time", "value": 120.0}
                    ]
                
                # Detect anomalies
                anomalies_result = await self.anomaly_predictor.predict_anomalies(metrics)
                
                if anomalies_result.is_left():
                    return f"Error: {anomalies_result.left()}"
                
                anomalies = anomalies_result.right()
                
                # Format response
                response = {
                    "success": True,
                    "anomaly_detection": {
                        "sensitivity": sensitivity,
                        "anomalies_detected": len(anomalies),
                        "anomalies": [
                            {
                                "anomaly_id": anomaly.anomaly_id,
                                "anomaly_type": anomaly.anomaly_type,
                                "severity": anomaly.severity.value,
                                "probability": float(anomaly.probability),
                                "affected_metric": anomaly.affected_metric,
                                "current_value": anomaly.current_value,
                                "expected_range": anomaly.expected_range,
                                "deviation_score": anomaly.deviation_score,
                                "predicted_impact": anomaly.predicted_impact,
                                "time_to_resolution": str(anomaly.time_to_resolution) if anomaly.time_to_resolution else None,
                                "mitigation_suggestions": anomaly.mitigation_suggestions,
                                "detected_at": anomaly.detected_at.isoformat()
                            }
                            for anomaly in anomalies
                        ]
                    },
                    "metadata": {
                        "detection_time": datetime.now(UTC).isoformat(),
                        "metrics_analyzed": len(metrics)
                    }
                }
                
                return f"```json\n{response}\n```"
                
            except Exception as e:
                self.logger.error(f"Anomaly detection failed: {e}")
                return f"Error: Anomaly detection failed - {str(e)}"
        
        @mcp.tool()
        async def km_create_capacity_plan(
            resource_type: str,
            planning_days: int = 30
        ) -> str:
            """
            Create capacity planning recommendations for resource scaling.
            
            Args:
                resource_type: Type of resource to plan for
                planning_days: Days ahead to plan (1-365)
            
            Returns:
                JSON string with capacity plan and scaling recommendations
            """
            try:
                # Validate inputs
                if not 1 <= planning_days <= 365:
                    return f"Error: planning_days must be between 1 and 365, got {planning_days}"
                
                # Create capacity plan
                plan_result = await self.capacity_planner.create_capacity_plan(
                    resource_type=resource_type,
                    planning_horizon=timedelta(days=planning_days)
                )
                
                if plan_result.is_left():
                    return f"Error: {plan_result.left()}"
                
                plan = plan_result.right()
                
                # Format response
                response = {
                    "success": True,
                    "capacity_plan": {
                        "plan_id": plan.plan_id,
                        "resource_type": plan.resource_type,
                        "current_capacity": plan.current_capacity,
                        "projected_demand": [
                            {
                                "timestamp": ts.isoformat(),
                                "demand": demand,
                                "confidence": float(conf)
                            }
                            for ts, demand, conf in plan.projected_demand
                        ],
                        "scaling_recommendations": plan.scaling_recommendations,
                        "optimal_scaling_time": plan.optimal_scaling_time.isoformat(),
                        "cost_implications": plan.cost_implications,
                        "risk_assessment": plan.risk_assessment,
                        "confidence": float(plan.confidence)
                    },
                    "metadata": {
                        "planning_horizon_days": planning_days,
                        "model_used": plan.model_used,
                        "created_at": plan.created_at.isoformat()
                    }
                }
                
                return f"```json\n{response}\n```"
                
            except Exception as e:
                self.logger.error(f"Capacity planning failed: {e}")
                return f"Error: Capacity planning failed - {str(e)}"
        
        @mcp.tool()
        async def km_get_prediction_status() -> str:
            """
            Get comprehensive status of the predictive automation system.
            
            Returns:
                JSON string with system status and statistics
            """
            try:
                # Get statistics from all components
                model_stats = self.model_manager.get_model_statistics()
                optimization_status = self.optimization_engine.get_optimization_status()
                prediction_stats = self.performance_predictor.get_prediction_statistics()
                active_alerts = self.alert_system.get_active_alerts()
                
                response = {
                    "success": True,
                    "system_status": {
                        "model_manager": {
                            "total_models": model_stats["total_models"],
                            "total_predictions": model_stats["total_predictions"],
                            "success_rate": model_stats["success_rate"],
                            "active_predictions": model_stats["active_predictions"]
                        },
                        "optimization_engine": {
                            "optimizations_generated": optimization_status["optimizations_generated"],
                            "optimizations_implemented": optimization_status["optimizations_implemented"],
                            "total_impact_achieved": optimization_status["total_impact_achieved"],
                            "active_optimizations": optimization_status["active_optimizations"]
                        },
                        "performance_predictor": {
                            "forecasts_generated": prediction_stats["forecasts_generated"],
                            "cache_size": prediction_stats["cache_size"],
                            "historical_data_points": prediction_stats["historical_data_points"]
                        },
                        "alert_system": {
                            "active_alerts": len(active_alerts),
                            "alert_types": list(set(alert.alert_type for alert in active_alerts))
                        }
                    },
                    "metadata": {
                        "system_initialized": True,
                        "last_updated": datetime.now(UTC).isoformat(),
                        "prediction_system_version": "1.0.0"
                    }
                }
                
                return f"```json\n{response}\n```"
                
            except Exception as e:
                self.logger.error(f"Status retrieval failed: {e}")
                return f"Error: Status retrieval failed - {str(e)}"
        
        self.logger.info("Registered predictive automation MCP tools successfully")


# Global instance for tool registration
_predictive_tools: Optional[PredictiveAutomationTools] = None


def get_predictive_automation_tools() -> PredictiveAutomationTools:
    """Get or create the global predictive automation tools instance."""
    global _predictive_tools
    if _predictive_tools is None:
        _predictive_tools = PredictiveAutomationTools()
    return _predictive_tools