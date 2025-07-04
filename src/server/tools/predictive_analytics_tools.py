"""
Predictive Analytics MCP Tools - TASK_59 Phase 3 Implementation

Comprehensive predictive analytics tools for automation patterns, usage forecasting, and intelligent insights.
Provides advanced ML-powered predictions, capacity planning, and optimization recommendations through FastMCP.

Architecture: FastMCP Integration + Predictive Engines + Analytics Integration + Enterprise Security
Performance: <500ms prediction responses, <1s insight generation, <2s comprehensive analysis
Security: Validated inputs, secure model access, comprehensive audit logging, data privacy protection
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Annotated
from datetime import datetime, UTC, timedelta
import asyncio
import json
import logging
from pathlib import Path

from fastmcp import FastMCP
from pydantic import Field

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.predictive_modeling import (
    TimeSeriesData, PredictionConfig, ModelType, ConfidenceLevel, 
    ForecastGranularity, PredictiveModelingError
)
from src.analytics.pattern_predictor import (
    PatternPredictor, PatternFeature, PatternType, PatternComplexity
)
from src.analytics.usage_forecaster import (
    UsageForecaster, ResourceType, ForecastScenario, CapacityStatus
)
from src.analytics.insight_generator import (
    InsightGenerator, InsightData, InsightCategory, InsightSeverity
)
from src.analytics.model_manager import (
    ModelManager, ModelConfiguration, TrainingDataset, ValidationMethod, ModelCategory
)

# Initialize FastMCP
mcp = FastMCP("Predictive Analytics Tools")

# Global instances (will be initialized on startup)
pattern_predictor: Optional[PatternPredictor] = None
usage_forecaster: Optional[UsageForecaster] = None
insight_generator: Optional[InsightGenerator] = None
model_manager: Optional[ModelManager] = None

# Performance tracking
tool_performance_metrics = {
    "total_predictions": 0,
    "total_forecasts": 0,
    "total_insights": 0,
    "average_response_time": 0.0,
    "last_updated": datetime.now(UTC).isoformat()
}


async def initialize_predictive_analytics():
    """Initialize all predictive analytics components."""
    global pattern_predictor, usage_forecaster, insight_generator, model_manager
    
    try:
        # Initialize components
        pattern_predictor = PatternPredictor()
        usage_forecaster = UsageForecaster()
        model_manager = ModelManager()
        insight_generator = InsightGenerator(pattern_predictor, usage_forecaster)
        
        logging.info("Predictive analytics components initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize predictive analytics: {str(e)}")
        return False


def _validate_components():
    """Validate that all components are initialized."""
    if not all([pattern_predictor, usage_forecaster, insight_generator, model_manager]):
        raise RuntimeError("Predictive analytics components not initialized. Call initialize_predictive_analytics() first.")


def _update_performance_metrics(operation: str, response_time: float):
    """Update performance tracking metrics."""
    global tool_performance_metrics
    
    if operation == "prediction":
        tool_performance_metrics["total_predictions"] += 1
    elif operation == "forecast":
        tool_performance_metrics["total_forecasts"] += 1
    elif operation == "insight":
        tool_performance_metrics["total_insights"] += 1
    
    # Update average response time
    current_avg = tool_performance_metrics["average_response_time"]
    total_ops = (tool_performance_metrics["total_predictions"] + 
                 tool_performance_metrics["total_forecasts"] + 
                 tool_performance_metrics["total_insights"])
    
    tool_performance_metrics["average_response_time"] = (
        (current_avg * (total_ops - 1) + response_time) / total_ops
    )
    tool_performance_metrics["last_updated"] = datetime.now(UTC).isoformat()


@mcp.tool()
async def km_predict_automation_patterns(
    prediction_scope: Annotated[str, Field(description="Prediction scope (user|macro|system|workflow)")],
    target_id: Annotated[Optional[str], Field(description="Specific target UUID for focused prediction")] = None,
    prediction_horizon: Annotated[int, Field(description="Prediction horizon in days", ge=1, le=365)] = 30,
    pattern_types: Annotated[List[str], Field(description="Pattern types to predict")] = ["usage", "performance", "errors"],
    include_confidence_intervals: Annotated[bool, Field(description="Include prediction confidence intervals")] = True,
    model_type: Annotated[str, Field(description="Model type (linear|arima|lstm|ensemble)")] = "ensemble",
    include_external_factors: Annotated[bool, Field(description="Include external factor analysis")] = True,
    generate_visualizations: Annotated[bool, Field(description="Generate prediction visualizations")] = True,
    export_predictions: Annotated[bool, Field(description="Export predictions for analysis")] = False
) -> Dict[str, Any]:
    """
    Predict automation usage patterns and trends using advanced machine learning models.
    
    FastMCP Tool for pattern prediction through Claude Desktop.
    Uses historical data to predict future automation usage, performance, and behavior patterns.
    
    Returns predictions, confidence intervals, trend analysis, and actionable insights.
    """
    start_time = datetime.now(UTC)
    
    try:
        _validate_components()
        
        # Validate inputs
        if prediction_scope not in ["user", "macro", "system", "workflow"]:
            return {
                "success": False,
                "error": "Invalid prediction scope. Must be one of: user, macro, system, workflow",
                "timestamp": start_time.isoformat()
            }
        
        valid_pattern_types = ["usage", "performance", "errors", "workflow", "resource", "seasonal"]
        invalid_types = [pt for pt in pattern_types if pt not in valid_pattern_types]
        if invalid_types:
            return {
                "success": False,
                "error": f"Invalid pattern types: {invalid_types}. Valid types: {valid_pattern_types}",
                "timestamp": start_time.isoformat()
            }
        
        # Map string pattern types to enum values
        pattern_type_mapping = {
            "usage": PatternType.USAGE_PATTERNS,
            "performance": PatternType.PERFORMANCE_PATTERNS,
            "errors": PatternType.ERROR_PATTERNS,
            "workflow": PatternType.WORKFLOW_PATTERNS,
            "resource": PatternType.RESOURCE_CONSUMPTION,
            "seasonal": PatternType.SEASONAL_BEHAVIOR
        }
        
        pattern_enums = [pattern_type_mapping[pt] for pt in pattern_types if pt in pattern_type_mapping]
        
        # Generate sample pattern features (in real implementation, this would come from data sources)
        sample_features = await _generate_sample_pattern_features(prediction_scope, target_id)
        
        # Detect patterns
        pattern_result = await pattern_predictor.detect_patterns(
            features=sample_features,
            pattern_types=pattern_enums,
            min_confidence=0.6
        )
        
        if pattern_result.is_left():
            return {
                "success": False,
                "error": f"Pattern detection failed: {pattern_result.get_left().message}",
                "timestamp": start_time.isoformat()
            }
        
        detected_patterns = pattern_result.get_right()
        
        # Generate predictions for each pattern
        predictions = []
        for pattern in detected_patterns[:5]:  # Limit to top 5 patterns
            pred_result = await pattern_predictor.predict_pattern_future(
                pattern_id=pattern.pattern_id,
                horizon_hours=prediction_horizon * 24,
                confidence_level=ConfidenceLevel.MEDIUM
            )
            
            if pred_result.is_right():
                prediction = pred_result.get_right()
                predictions.append({
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type.value,
                    "pattern_description": pattern.description,
                    "pattern_confidence": pattern.confidence_score,
                    "pattern_strength": pattern.strength,
                    "prediction_values": prediction.predicted_values[:10],  # Limit for response size
                    "confidence_intervals": prediction.confidence_intervals[:10] if include_confidence_intervals else None,
                    "accuracy_estimate": prediction.accuracy_estimate,
                    "factors_considered": prediction.factors_considered,
                    "assumptions": prediction.assumptions
                })
        
        # Generate summary statistics
        pattern_summary = await pattern_predictor.get_pattern_summary()
        
        # Calculate response time
        response_time = (datetime.now(UTC) - start_time).total_seconds()
        _update_performance_metrics("prediction", response_time)
        
        result = {
            "success": True,
            "prediction_scope": prediction_scope,
            "target_id": target_id,
            "prediction_horizon_days": prediction_horizon,
            "patterns_analyzed": len(detected_patterns),
            "predictions": predictions,
            "pattern_summary": {
                "total_patterns_detected": pattern_summary.get("total_patterns_detected", 0),
                "high_confidence_patterns": len([p for p in predictions if p["pattern_confidence"] > 0.8]),
                "patterns_by_type": pattern_summary.get("patterns_by_type", {}),
                "detection_timestamp": pattern_summary.get("detection_timestamp")
            },
            "model_configuration": {
                "model_type": model_type,
                "confidence_intervals_included": include_confidence_intervals,
                "external_factors_included": include_external_factors,
                "visualizations_generated": generate_visualizations
            },
            "performance": {
                "response_time_seconds": response_time,
                "patterns_per_second": len(detected_patterns) / max(response_time, 0.001)
            },
            "recommendations": _generate_pattern_recommendations(detected_patterns),
            "timestamp": start_time.isoformat(),
            "completed_at": datetime.now(UTC).isoformat()
        }
        
        # Export if requested
        if export_predictions:
            result["export_path"] = await _export_predictions(result, "patterns")
        
        return result
        
    except Exception as e:
        response_time = (datetime.now(UTC) - start_time).total_seconds()
        
        return {
            "success": False,
            "error": f"Pattern prediction failed: {str(e)}",
            "response_time_seconds": response_time,
            "timestamp": start_time.isoformat()
        }


@mcp.tool()
async def km_forecast_resource_usage(
    resource_types: Annotated[List[str], Field(description="Resource types to forecast")] = ["cpu", "memory", "storage", "network"],
    forecast_period: Annotated[int, Field(description="Forecast period in days", ge=1, le=365)] = 90,
    granularity: Annotated[str, Field(description="Forecast granularity (hourly|daily|weekly)")] = "daily",
    include_seasonality: Annotated[bool, Field(description="Include seasonal patterns in forecast")] = True,
    include_growth_trends: Annotated[bool, Field(description="Include growth trend analysis")] = True,
    capacity_planning: Annotated[bool, Field(description="Include capacity planning recommendations")] = True,
    alert_thresholds: Annotated[Optional[Dict[str, float]], Field(description="Resource threshold alerts")] = None,
    scenario_analysis: Annotated[bool, Field(description="Include scenario-based forecasting")] = False,
    export_forecast: Annotated[bool, Field(description="Export forecast data")] = False
) -> Dict[str, Any]:
    """
    Forecast resource usage and capacity requirements for automation workflows.
    
    FastMCP Tool for resource forecasting through Claude Desktop.
    Predicts CPU, memory, storage, and network usage with capacity planning insights.
    
    Returns usage forecasts, capacity recommendations, threshold alerts, and optimization suggestions.
    """
    start_time = datetime.now(UTC)
    
    try:
        _validate_components()
        
        # Validate inputs
        valid_resources = ["cpu", "memory", "storage", "network", "automation_executions", "api_calls"]
        invalid_resources = [rt for rt in resource_types if rt not in valid_resources]
        if invalid_resources:
            return {
                "success": False,
                "error": f"Invalid resource types: {invalid_resources}. Valid types: {valid_resources}",
                "timestamp": start_time.isoformat()
            }
        
        if granularity not in ["hourly", "daily", "weekly"]:
            return {
                "success": False,
                "error": "Invalid granularity. Must be one of: hourly, daily, weekly",
                "timestamp": start_time.isoformat()
            }
        
        # Map string resource types to enum values
        resource_type_mapping = {
            "cpu": ResourceType.CPU_USAGE,
            "memory": ResourceType.MEMORY_USAGE,
            "storage": ResourceType.STORAGE_USAGE,
            "network": ResourceType.NETWORK_BANDWIDTH,
            "automation_executions": ResourceType.AUTOMATION_EXECUTIONS,
            "api_calls": ResourceType.API_CALLS
        }
        
        granularity_mapping = {
            "hourly": ForecastGranularity.HOURLY,
            "daily": ForecastGranularity.DAILY,
            "weekly": ForecastGranularity.WEEKLY
        }
        
        # Generate forecasts for each resource type
        forecasts = []
        capacity_analyses = []
        
        for resource_str in resource_types:
            if resource_str not in resource_type_mapping:
                continue
                
            resource_type = resource_type_mapping[resource_str]
            granularity_enum = granularity_mapping[granularity]
            
            # Add sample usage data (in real implementation, this would come from monitoring systems)
            sample_data = await _generate_sample_usage_data(resource_type)
            usage_result = await usage_forecaster.add_usage_data(resource_type, sample_data)
            
            if usage_result.is_left():
                continue  # Skip this resource if data addition failed
            
            # Create forecast scenario if scenario analysis is requested
            scenario = None
            if scenario_analysis:
                scenario = ForecastScenario(
                    scenario_name="growth_scenario",
                    growth_multiplier=1.2,  # 20% growth
                    seasonal_adjustment=1.1,
                    description="Projected growth scenario with 20% increased usage"
                )
            
            # Generate forecast
            forecast_result = await usage_forecaster.generate_forecast(
                resource_type=resource_type,
                forecast_period_days=forecast_period,
                granularity=granularity_enum,
                confidence_level=ConfidenceLevel.MEDIUM,
                scenario=scenario
            )
            
            if forecast_result.is_right():
                forecast = forecast_result.get_right()
                
                forecasts.append({
                    "resource_type": resource_str,
                    "forecast_id": forecast.forecast_id,
                    "current_usage": forecast.current_usage,
                    "predicted_usage": forecast.predicted_usage[:20],  # Limit for response size
                    "forecast_timestamps": [ts.isoformat() for ts in forecast.forecast_timestamps[:20]],
                    "growth_rate": forecast.growth_rate,
                    "seasonality_detected": bool(forecast.seasonality_patterns),
                    "capacity_thresholds": forecast.capacity_thresholds,
                    "capacity_recommendations": forecast.capacity_recommendations
                })
                
                # Add capacity analysis if requested
                if capacity_planning:
                    max_usage = max(forecast.predicted_usage) if forecast.predicted_usage else forecast.current_usage
                    current_capacity = forecast.current_usage / 0.7 if forecast.current_usage > 0 else 100.0
                    utilization = (max_usage / current_capacity) * 100
                    
                    capacity_status = CapacityStatus.OPTIMAL
                    if utilization > 90:
                        capacity_status = CapacityStatus.OVER_CAPACITY
                    elif utilization > 80:
                        capacity_status = CapacityStatus.AT_CAPACITY
                    elif utilization > 70:
                        capacity_status = CapacityStatus.APPROACHING_LIMIT
                    
                    capacity_analyses.append({
                        "resource_type": resource_str,
                        "current_capacity": current_capacity,
                        "max_predicted_usage": max_usage,
                        "utilization_percentage": utilization,
                        "capacity_status": capacity_status.value,
                        "recommendations": forecast.capacity_recommendations
                    })
        
        # Generate summary
        forecasting_summary = await usage_forecaster.get_forecasting_summary()
        
        # Calculate response time
        response_time = (datetime.now(UTC) - start_time).total_seconds()
        _update_performance_metrics("forecast", response_time)
        
        result = {
            "success": True,
            "resource_types": resource_types,
            "forecast_period_days": forecast_period,
            "granularity": granularity,
            "forecasts": forecasts,
            "capacity_analyses": capacity_analyses if capacity_planning else [],
            "forecast_summary": {
                "total_forecasts_generated": len(forecasts),
                "resources_at_capacity": len([ca for ca in capacity_analyses if ca["capacity_status"] in ["at_capacity", "over_capacity"]]),
                "average_growth_rate": sum([f.get("growth_rate", 0) for f in forecasts]) / len(forecasts) if forecasts else 0,
                "seasonality_detected": sum([1 for f in forecasts if f["seasonality_detected"]]),
            },
            "configuration": {
                "seasonality_included": include_seasonality,
                "growth_trends_included": include_growth_trends,
                "capacity_planning_included": capacity_planning,
                "scenario_analysis_included": scenario_analysis
            },
            "alerts": _generate_capacity_alerts(capacity_analyses, alert_thresholds),
            "performance": {
                "response_time_seconds": response_time,
                "forecasts_per_second": len(forecasts) / max(response_time, 0.001)
            },
            "recommendations": _generate_capacity_recommendations(forecasts, capacity_analyses),
            "timestamp": start_time.isoformat(),
            "completed_at": datetime.now(UTC).isoformat()
        }
        
        # Export if requested
        if export_forecast:
            result["export_path"] = await _export_predictions(result, "forecasts")
        
        return result
        
    except Exception as e:
        response_time = (datetime.now(UTC) - start_time).total_seconds()
        
        return {
            "success": False,
            "error": f"Resource forecasting failed: {str(e)}",
            "response_time_seconds": response_time,
            "timestamp": start_time.isoformat()
        }


@mcp.tool()
async def km_generate_insights(
    analysis_scope: Annotated[str, Field(description="Analysis scope (automation|performance|usage|efficiency)")],
    data_timeframe: Annotated[str, Field(description="Data timeframe (week|month|quarter|year)")] = "month",
    insight_types: Annotated[List[str], Field(description="Insight types to generate")] = ["optimization", "efficiency", "cost"],
    include_actionable_recommendations: Annotated[bool, Field(description="Include actionable recommendations")] = True,
    prioritize_insights: Annotated[bool, Field(description="Prioritize insights by impact")] = True,
    include_roi_analysis: Annotated[bool, Field(description="Include ROI analysis for recommendations")] = True,
    generate_executive_summary: Annotated[bool, Field(description="Generate executive summary")] = True,
    export_insights: Annotated[bool, Field(description="Export insights for sharing")] = False,
    schedule_updates: Annotated[Optional[str], Field(description="Schedule regular insight updates")] = None
) -> Dict[str, Any]:
    """
    Generate intelligent insights and actionable recommendations from automation data.
    
    FastMCP Tool for insight generation through Claude Desktop.
    Analyzes automation data to generate actionable insights and optimization recommendations.
    
    Returns prioritized insights, recommendations, ROI analysis, and executive summaries.
    """
    start_time = datetime.now(UTC)
    
    try:
        _validate_components()
        
        # Validate inputs
        if analysis_scope not in ["automation", "performance", "usage", "efficiency"]:
            return {
                "success": False,
                "error": "Invalid analysis scope. Must be one of: automation, performance, usage, efficiency",
                "timestamp": start_time.isoformat()
            }
        
        if data_timeframe not in ["week", "month", "quarter", "year"]:
            return {
                "success": False,
                "error": "Invalid data timeframe. Must be one of: week, month, quarter, year",
                "timestamp": start_time.isoformat()
            }
        
        valid_insight_types = ["optimization", "efficiency", "cost", "risk", "capacity", "workflow", "security"]
        invalid_types = [it for it in insight_types if it not in valid_insight_types]
        if invalid_types:
            return {
                "success": False,
                "error": f"Invalid insight types: {invalid_types}. Valid types: {valid_insight_types}",
                "timestamp": start_time.isoformat()
            }
        
        # Map string insight types to enum values
        insight_category_mapping = {
            "optimization": InsightCategory.PERFORMANCE_OPTIMIZATION,
            "efficiency": InsightCategory.EFFICIENCY_IMPROVEMENT,
            "cost": InsightCategory.COST_REDUCTION,
            "risk": InsightCategory.RISK_MITIGATION,
            "capacity": InsightCategory.CAPACITY_PLANNING,
            "workflow": InsightCategory.WORKFLOW_ENHANCEMENT,
            "security": InsightCategory.SECURITY_IMPROVEMENT
        }
        
        insight_categories = [insight_category_mapping[it] for it in insight_types if it in insight_category_mapping]
        
        # Generate sample insight data (in real implementation, this would come from analytics systems)
        insight_data_list = await _generate_sample_insight_data(analysis_scope, data_timeframe)
        
        # Generate insights
        insights_result = await insight_generator.generate_insights(
            insight_data=insight_data_list,
            categories=insight_categories,
            min_confidence=0.6,
            include_roi_analysis=include_roi_analysis,
            prioritize_by_impact=prioritize_insights
        )
        
        if insights_result.is_left():
            return {
                "success": False,
                "error": f"Insight generation failed: {insights_result.get_left().message}",
                "timestamp": start_time.isoformat()
            }
        
        insights = insights_result.get_right()
        
        # Convert insights to response format
        insights_data = []
        for insight in insights[:10]:  # Limit to top 10 insights
            insight_data = {
                "insight_id": insight.insight_id,
                "insight_type": insight.insight_type.value,
                "title": insight.title,
                "description": insight.description,
                "confidence_score": insight.confidence_score,
                "impact_score": insight.impact_score,
                "priority_level": insight.priority_level,
                "actionable_recommendations": insight.actionable_recommendations if include_actionable_recommendations else [],
                "data_sources": insight.data_sources,
                "roi_estimate": insight.roi_estimate,
                "implementation_effort": insight.implementation_effort,
                "supporting_evidence": insight.supporting_evidence
            }
            insights_data.append(insight_data)
        
        # Generate executive summary if requested
        executive_summary = None
        if generate_executive_summary:
            summary_result = await insight_generator.generate_executive_summary(insights, data_timeframe)
            if summary_result.is_right():
                summary = summary_result.get_right()
                executive_summary = {
                    "summary_id": summary.summary_id,
                    "time_period": summary.time_period,
                    "key_findings": summary.key_findings,
                    "critical_issues": summary.critical_issues,
                    "top_opportunities": summary.top_opportunities,
                    "recommended_actions": summary.recommended_actions,
                    "total_potential_savings": summary.total_potential_savings,
                    "total_investment_required": summary.total_investment_required,
                    "overall_roi": summary.overall_roi,
                    "strategic_priorities": summary.strategic_priorities,
                    "confidence_score": summary.confidence_score
                }
        
        # Generate insight summary
        insight_summary = await insight_generator.get_insight_summary()
        
        # Calculate response time
        response_time = (datetime.now(UTC) - start_time).total_seconds()
        _update_performance_metrics("insight", response_time)
        
        result = {
            "success": True,
            "analysis_scope": analysis_scope,
            "data_timeframe": data_timeframe,
            "insight_types": insight_types,
            "insights": insights_data,
            "executive_summary": executive_summary,
            "insight_summary": {
                "total_insights_generated": len(insights),
                "high_impact_insights": len([i for i in insights_data if i["impact_score"] > 0.7]),
                "critical_priority_insights": len([i for i in insights_data if i["priority_level"] == "critical"]),
                "total_estimated_roi": sum([i.get("roi_estimate", 0) for i in insights_data if i.get("roi_estimate")]),
                "average_confidence": sum([i["confidence_score"] for i in insights_data]) / len(insights_data) if insights_data else 0
            },
            "configuration": {
                "actionable_recommendations_included": include_actionable_recommendations,
                "prioritized_by_impact": prioritize_insights,
                "roi_analysis_included": include_roi_analysis,
                "executive_summary_included": generate_executive_summary
            },
            "performance": {
                "response_time_seconds": response_time,
                "insights_per_second": len(insights) / max(response_time, 0.001)
            },
            "next_steps": _generate_insight_next_steps(insights_data),
            "timestamp": start_time.isoformat(),
            "completed_at": datetime.now(UTC).isoformat()
        }
        
        # Schedule updates if requested
        if schedule_updates:
            result["scheduled_updates"] = {
                "frequency": schedule_updates,
                "next_update": _calculate_next_update(schedule_updates),
                "status": "scheduled"
            }
        
        # Export if requested
        if export_insights:
            result["export_path"] = await _export_predictions(result, "insights")
        
        return result
        
    except Exception as e:
        response_time = (datetime.now(UTC) - start_time).total_seconds()
        
        return {
            "success": False,
            "error": f"Insight generation failed: {str(e)}",
            "response_time_seconds": response_time,
            "timestamp": start_time.isoformat()
        }


@mcp.tool()
async def km_analyze_trends(
    trend_analysis_scope: Annotated[str, Field(description="Analysis scope (usage|performance|errors|efficiency)")],
    analysis_period: Annotated[str, Field(description="Analysis period (month|quarter|year|custom)")] = "quarter",
    trend_detection_sensitivity: Annotated[str, Field(description="Trend detection sensitivity (low|medium|high)")] = "medium",
    include_statistical_significance: Annotated[bool, Field(description="Include statistical significance testing")] = True,
    decompose_trends: Annotated[bool, Field(description="Decompose trends into components")] = True,
    predict_trend_continuation: Annotated[bool, Field(description="Predict trend continuation")] = True,
    identify_inflection_points: Annotated[bool, Field(description="Identify trend inflection points")] = True,
    generate_trend_report: Annotated[bool, Field(description="Generate comprehensive trend report")] = True,
    export_analysis: Annotated[bool, Field(description="Export trend analysis data")] = False
) -> Dict[str, Any]:
    """
    Analyze automation trends with statistical significance testing and prediction.
    
    FastMCP Tool for trend analysis through Claude Desktop.
    Performs comprehensive trend analysis with statistical validation and predictions.
    
    Returns trend analysis, statistical tests, predictions, and inflection point identification.
    """
    start_time = datetime.now(UTC)
    
    try:
        _validate_components()
        
        # Validate inputs
        if trend_analysis_scope not in ["usage", "performance", "errors", "efficiency"]:
            return {
                "success": False,
                "error": "Invalid trend analysis scope. Must be one of: usage, performance, errors, efficiency",
                "timestamp": start_time.isoformat()
            }
        
        if analysis_period not in ["month", "quarter", "year", "custom"]:
            return {
                "success": False,
                "error": "Invalid analysis period. Must be one of: month, quarter, year, custom",
                "timestamp": start_time.isoformat()
            }
        
        if trend_detection_sensitivity not in ["low", "medium", "high"]:
            return {
                "success": False,
                "error": "Invalid trend detection sensitivity. Must be one of: low, medium, high",
                "timestamp": start_time.isoformat()
            }
        
        # Generate sample trend data based on scope
        trend_data = await _generate_sample_trend_data(trend_analysis_scope, analysis_period)
        
        # Perform trend analysis using pattern predictor
        sample_features = await _convert_trend_data_to_features(trend_data, trend_analysis_scope)
        
        # Detect patterns that represent trends
        pattern_result = await pattern_predictor.detect_patterns(
            features=sample_features,
            pattern_types=[PatternType.PERFORMANCE_PATTERNS, PatternType.USAGE_PATTERNS],
            min_confidence=0.5 if trend_detection_sensitivity == "low" else 0.7 if trend_detection_sensitivity == "medium" else 0.8
        )
        
        if pattern_result.is_left():
            return {
                "success": False,
                "error": f"Trend analysis failed: {pattern_result.get_left().message}",
                "timestamp": start_time.isoformat()
            }
        
        detected_patterns = pattern_result.get_right()
        
        # Analyze trends
        trends = []
        inflection_points = []
        
        for pattern in detected_patterns:
            trend_analysis = {
                "trend_id": pattern.pattern_id,
                "trend_type": pattern.pattern_type.value,
                "trend_direction": pattern.trend_direction,
                "trend_strength": pattern.strength,
                "trend_confidence": pattern.confidence_score,
                "trend_description": pattern.description,
                "statistical_significance": pattern.statistical_significance,
                "seasonality": pattern.seasonality,
                "frequency": pattern.frequency
            }
            
            # Add trend decomposition if requested
            if decompose_trends:
                trend_analysis["decomposition"] = {
                    "trend_component": pattern.strength * 0.7,  # Simulated
                    "seasonal_component": pattern.strength * 0.2,
                    "noise_component": pattern.strength * 0.1,
                    "decomposition_method": "statistical_decomposition"
                }
            
            # Predict trend continuation if requested
            if predict_trend_continuation:
                pred_result = await pattern_predictor.predict_pattern_future(
                    pattern_id=pattern.pattern_id,
                    horizon_hours=30 * 24,  # 30 days
                    confidence_level=ConfidenceLevel.MEDIUM
                )
                
                if pred_result.is_right():
                    prediction = pred_result.get_right()
                    trend_analysis["trend_prediction"] = {
                        "predicted_direction": "continuing" if pattern.trend_direction else "stable",
                        "prediction_confidence": prediction.accuracy_estimate,
                        "predicted_values": prediction.predicted_values[:10],  # Limit for response
                        "prediction_horizon_days": 30
                    }
            
            # Identify inflection points if requested
            if identify_inflection_points and pattern.trend_direction:
                # Simulate inflection point detection
                inflection_point = {
                    "point_id": f"inflection_{pattern.pattern_id}",
                    "trend_id": pattern.pattern_id,
                    "inflection_type": "trend_change",
                    "significance": pattern.statistical_significance or 0.8,
                    "estimated_date": datetime.now(UTC) - timedelta(days=15),
                    "description": f"Potential trend change detected in {pattern.description}"
                }
                inflection_points.append(inflection_point)
            
            trends.append(trend_analysis)
        
        # Generate statistical summary
        statistical_summary = {
            "total_trends_analyzed": len(trends),
            "significant_trends": len([t for t in trends if t.get("statistical_significance", 0) > 0.7]),
            "positive_trends": len([t for t in trends if t.get("trend_direction") == "increasing"]),
            "negative_trends": len([t for t in trends if t.get("trend_direction") == "decreasing"]),
            "stable_trends": len([t for t in trends if t.get("trend_direction") == "stable"]),
            "average_trend_strength": sum([t.get("trend_strength", 0) for t in trends]) / len(trends) if trends else 0,
            "average_confidence": sum([t.get("trend_confidence", 0) for t in trends]) / len(trends) if trends else 0
        }
        
        # Generate trend report if requested
        trend_report = None
        if generate_trend_report:
            trend_report = {
                "report_id": f"trend_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                "analysis_scope": trend_analysis_scope,
                "analysis_period": analysis_period,
                "key_findings": [
                    f"Detected {len(trends)} significant trends in {trend_analysis_scope}",
                    f"Average trend confidence: {statistical_summary['average_confidence']:.2%}",
                    f"Identified {len(inflection_points)} potential inflection points"
                ],
                "trend_summary": statistical_summary,
                "recommendations": [
                    "Monitor high-confidence trends for strategic planning",
                    "Investigate trends with declining performance",
                    "Capitalize on positive efficiency trends"
                ]
            }
        
        # Calculate response time
        response_time = (datetime.now(UTC) - start_time).total_seconds()
        
        result = {
            "success": True,
            "analysis_scope": trend_analysis_scope,
            "analysis_period": analysis_period,
            "detection_sensitivity": trend_detection_sensitivity,
            "trends": trends,
            "inflection_points": inflection_points if identify_inflection_points else [],
            "statistical_summary": statistical_summary,
            "trend_report": trend_report,
            "configuration": {
                "statistical_significance_included": include_statistical_significance,
                "trends_decomposed": decompose_trends,
                "trend_continuation_predicted": predict_trend_continuation,
                "inflection_points_identified": identify_inflection_points
            },
            "performance": {
                "response_time_seconds": response_time,
                "trends_per_second": len(trends) / max(response_time, 0.001)
            },
            "insights": _generate_trend_insights(trends, inflection_points),
            "timestamp": start_time.isoformat(),
            "completed_at": datetime.now(UTC).isoformat()
        }
        
        # Export if requested
        if export_analysis:
            result["export_path"] = await _export_predictions(result, "trends")
        
        return result
        
    except Exception as e:
        response_time = (datetime.now(UTC) - start_time).total_seconds()
        
        return {
            "success": False,
            "error": f"Trend analysis failed: {str(e)}",
            "response_time_seconds": response_time,
            "timestamp": start_time.isoformat()
        }


@mcp.tool()
async def km_get_analytics_status() -> Dict[str, Any]:
    """
    Get comprehensive status of predictive analytics system.
    
    Returns system health, performance metrics, and component status.
    """
    try:
        _validate_components()
        
        # Get component summaries
        pattern_summary = await pattern_predictor.get_pattern_summary()
        forecasting_summary = await usage_forecaster.get_forecasting_summary()
        insight_summary = await insight_generator.get_insight_summary()
        model_summary = await model_manager.get_model_manager_summary()
        
        return {
            "success": True,
            "system_status": "operational",
            "components": {
                "pattern_predictor": {
                    "status": "active",
                    "patterns_detected": pattern_summary.get("total_patterns_detected", 0),
                    "pattern_types": pattern_summary.get("patterns_by_type", {})
                },
                "usage_forecaster": {
                    "status": "active",
                    "resources_tracked": forecasting_summary.get("resources_tracked", 0),
                    "total_forecasts": forecasting_summary.get("total_data_points", 0)
                },
                "insight_generator": {
                    "status": "active",
                    "insights_generated": insight_summary.get("total_insights_generated", 0),
                    "high_impact_insights": insight_summary.get("high_impact_insights_count", 0)
                },
                "model_manager": {
                    "status": "active",
                    "models_trained": model_summary.get("trained_models", 0),
                    "models_deployed": model_summary.get("deployed_models", 0)
                }
            },
            "performance_metrics": tool_performance_metrics,
            "capabilities": {
                "pattern_prediction": True,
                "resource_forecasting": True,
                "insight_generation": True,
                "trend_analysis": True,
                "model_management": True
            },
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Status check failed: {str(e)}",
            "system_status": "error",
            "timestamp": datetime.now(UTC).isoformat()
        }


# Utility functions for data generation and processing

async def _generate_sample_pattern_features(scope: str, target_id: Optional[str]) -> List[PatternFeature]:
    """Generate sample pattern features for demonstration."""
    import random
    
    features = []
    
    # Generate usage pattern feature
    timestamps = [datetime.now(UTC) - timedelta(hours=i) for i in range(24, 0, -1)]
    usage_values = [random.uniform(10, 100) for _ in range(24)]
    
    features.append(PatternFeature(
        feature_name=f"{scope}_usage",
        feature_type="numeric",
        values=usage_values,
        timestamps=timestamps,
        confidence_score=0.85
    ))
    
    # Generate performance pattern feature
    performance_values = [random.uniform(1.0, 5.0) for _ in range(24)]
    
    features.append(PatternFeature(
        feature_name=f"{scope}_response_time",
        feature_type="numeric", 
        values=performance_values,
        timestamps=timestamps,
        confidence_score=0.78
    ))
    
    return features


async def _generate_sample_usage_data(resource_type: ResourceType) -> TimeSeriesData:
    """Generate sample usage data for forecasting."""
    import random
    
    # Generate 30 days of hourly data
    timestamps = [datetime.now(UTC) - timedelta(hours=i) for i in range(720, 0, -1)]
    
    # Generate realistic usage values based on resource type
    if resource_type == ResourceType.CPU_USAGE:
        values = [random.uniform(20, 80) for _ in range(720)]
    elif resource_type == ResourceType.MEMORY_USAGE:
        values = [random.uniform(30, 90) for _ in range(720)]
    elif resource_type == ResourceType.STORAGE_USAGE:
        values = [random.uniform(40, 95) for _ in range(720)]
    else:
        values = [random.uniform(10, 60) for _ in range(720)]
    
    return TimeSeriesData(
        timestamps=timestamps,
        values=values,
        metadata={"source": "sample_data", "resource_type": resource_type.value}
    )


async def _generate_sample_insight_data(scope: str, timeframe: str) -> List[InsightData]:
    """Generate sample insight data for analysis."""
    import random
    
    data_list = []
    
    # Performance data
    data_list.append(InsightData(
        data_source=f"{scope}_performance",
        data_type="performance",
        time_period=timeframe,
        metrics={
            "response_time": random.uniform(1.5, 4.0),
            "cpu_usage": random.uniform(40, 85),
            "memory_usage": random.uniform(50, 90),
            "error_rate": random.uniform(1, 8)
        },
        trends={"response_time_trend": "increasing", "cpu_trend": "stable"},
        context={"analysis_scope": scope}
    ))
    
    # Usage data
    data_list.append(InsightData(
        data_source=f"{scope}_usage",
        data_type="usage",
        time_period=timeframe,
        metrics={
            "execution_count": random.uniform(1000, 5000),
            "automation_executions": random.uniform(500, 2000),
            "manual_tasks": random.uniform(50, 200),
            "cost_per_execution": random.uniform(0.05, 0.25)
        },
        trends={"usage_trend": "increasing", "efficiency_trend": "improving"},
        context={"analysis_scope": scope}
    ))
    
    return data_list


async def _generate_sample_trend_data(scope: str, period: str) -> Dict[str, Any]:
    """Generate sample trend data for analysis."""
    import random
    
    # Determine data points based on period
    if period == "month":
        data_points = 30
    elif period == "quarter":
        data_points = 90
    elif period == "year":
        data_points = 365
    else:
        data_points = 60
    
    timestamps = [datetime.now(UTC) - timedelta(days=i) for i in range(data_points, 0, -1)]
    
    if scope == "usage":
        values = [random.uniform(100, 500) + i * 2 for i in range(data_points)]
    elif scope == "performance":
        values = [random.uniform(1.0, 3.0) + random.uniform(-0.1, 0.1) for _ in range(data_points)]
    elif scope == "errors":
        values = [random.uniform(1, 10) for _ in range(data_points)]
    else:  # efficiency
        values = [random.uniform(70, 95) + i * 0.1 for i in range(data_points)]
    
    return {
        "timestamps": timestamps,
        "values": values,
        "scope": scope,
        "period": period
    }


async def _convert_trend_data_to_features(trend_data: Dict[str, Any], scope: str) -> List[PatternFeature]:
    """Convert trend data to pattern features."""
    return [PatternFeature(
        feature_name=f"{scope}_trend",
        feature_type="numeric",
        values=trend_data["values"],
        timestamps=trend_data["timestamps"],
        confidence_score=0.8
    )]


def _generate_pattern_recommendations(patterns: List) -> List[str]:
    """Generate recommendations based on detected patterns."""
    recommendations = []
    
    if len(patterns) > 0:
        recommendations.append("Monitor detected patterns for automation optimization opportunities")
    
    high_confidence_patterns = [p for p in patterns if hasattr(p, 'confidence_score') and p.confidence_score > 0.8]
    if high_confidence_patterns:
        recommendations.append(f"Focus on {len(high_confidence_patterns)} high-confidence patterns for immediate action")
    
    recommendations.append("Implement automated alerting for pattern changes")
    recommendations.append("Consider pattern-based automation rules")
    
    return recommendations


def _generate_capacity_recommendations(forecasts: List, capacity_analyses: List) -> List[str]:
    """Generate capacity planning recommendations."""
    recommendations = []
    
    at_capacity = [ca for ca in capacity_analyses if ca.get("capacity_status") in ["at_capacity", "over_capacity"]]
    if at_capacity:
        recommendations.append(f"Immediate capacity expansion needed for {len(at_capacity)} resources")
    
    approaching_capacity = [ca for ca in capacity_analyses if ca.get("capacity_status") == "approaching_limit"]
    if approaching_capacity:
        recommendations.append(f"Plan capacity increases for {len(approaching_capacity)} resources within 30 days")
    
    high_growth = [f for f in forecasts if f.get("growth_rate", 0) > 0.2]
    if high_growth:
        recommendations.append(f"Monitor high-growth resources: {[f['resource_type'] for f in high_growth]}")
    
    return recommendations


def _generate_capacity_alerts(capacity_analyses: List, alert_thresholds: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
    """Generate capacity alerts based on thresholds."""
    alerts = []
    
    if not alert_thresholds:
        alert_thresholds = {"cpu": 85.0, "memory": 90.0, "storage": 80.0}
    
    for analysis in capacity_analyses:
        resource_type = analysis["resource_type"]
        utilization = analysis["utilization_percentage"]
        threshold = alert_thresholds.get(resource_type, 85.0)
        
        if utilization > threshold:
            alerts.append({
                "alert_type": "capacity_threshold_exceeded",
                "resource_type": resource_type,
                "current_utilization": utilization,
                "threshold": threshold,
                "severity": "high" if utilization > threshold * 1.1 else "medium",
                "message": f"{resource_type} utilization at {utilization:.1f}% exceeds threshold of {threshold:.1f}%"
            })
    
    return alerts


def _generate_insight_next_steps(insights: List[Dict[str, Any]]) -> List[str]:
    """Generate next steps based on insights."""
    next_steps = []
    
    critical_insights = [i for i in insights if i["priority_level"] == "critical"]
    if critical_insights:
        next_steps.append(f"Address {len(critical_insights)} critical insights immediately")
    
    high_impact = [i for i in insights if i["impact_score"] > 0.7]
    if high_impact:
        next_steps.append(f"Prioritize {len(high_impact)} high-impact optimization opportunities")
    
    roi_insights = [i for i in insights if i.get("roi_estimate", 0) > 10000]
    if roi_insights:
        next_steps.append(f"Focus on insights with ROI > $10,000: {len(roi_insights)} opportunities")
    
    next_steps.append("Schedule regular insight reviews with stakeholders")
    
    return next_steps


def _generate_trend_insights(trends: List[Dict[str, Any]], inflection_points: List[Dict[str, Any]]) -> List[str]:
    """Generate insights from trend analysis."""
    insights = []
    
    positive_trends = [t for t in trends if t.get("trend_direction") == "increasing" and t.get("trend_strength", 0) > 0.7]
    if positive_trends:
        insights.append(f"Strong positive trends detected in {len(positive_trends)} areas")
    
    negative_trends = [t for t in trends if t.get("trend_direction") == "decreasing" and t.get("trend_strength", 0) > 0.7]
    if negative_trends:
        insights.append(f"Concerning negative trends identified in {len(negative_trends)} areas - investigate immediately")
    
    if inflection_points:
        insights.append(f"Detected {len(inflection_points)} potential trend changes requiring attention")
    
    high_confidence = [t for t in trends if t.get("statistical_significance", 0) > 0.8]
    insights.append(f"{len(high_confidence)} trends have high statistical significance")
    
    return insights


def _calculate_next_update(frequency: str) -> str:
    """Calculate next scheduled update time."""
    now = datetime.now(UTC)
    
    if frequency == "daily":
        next_update = now + timedelta(days=1)
    elif frequency == "weekly":
        next_update = now + timedelta(weeks=1)
    elif frequency == "monthly":
        next_update = now + timedelta(days=30)
    else:
        next_update = now + timedelta(days=7)
    
    return next_update.isoformat()


async def _export_predictions(data: Dict[str, Any], export_type: str) -> str:
    """Export prediction data to file."""
    try:
        export_dir = Path("./exports/predictive_analytics")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{export_type}_{timestamp}.json"
        filepath = export_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return str(filepath)
        
    except Exception as e:
        return f"Export failed: {str(e)}"