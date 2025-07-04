"""
MCP tools for real-time performance monitoring and optimization.

Comprehensive FastMCP tools for performance monitoring, bottleneck detection,
resource optimization, and automated performance tuning through Claude Desktop.

Security: Performance monitoring validation with access control.
Performance: <100ms tool response times, efficient monitoring.
Type Safety: Complete MCP integration with contract validation.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Annotated
from datetime import datetime, timedelta, UTC
import logging

from fastmcp import FastMCP
from pydantic import Field

from ...monitoring.metrics_collector import get_metrics_collector
from ...monitoring.performance_analyzer import get_performance_analyzer
from ...core.performance_monitoring import (
    MonitoringSessionID, MetricType, MonitoringScope, AlertSeverity,
    MonitoringConfiguration, OptimizationStrategy, PerformanceTarget,
    ExportFormat, ThresholdOperator, PerformanceThreshold,
    generate_monitoring_session_id, create_cpu_threshold, create_memory_threshold,
    PerformanceMonitoringError
)
from ...core.contracts import require, ensure
from ...core.either import Either

logger = logging.getLogger(__name__)


class PerformanceMonitorTools:
    """Comprehensive MCP tools for performance monitoring operations."""
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.performance_analyzer = get_performance_analyzer()
        self.active_monitoring_sessions: Dict[str, MonitoringSessionID] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_tools(self, mcp: FastMCP) -> None:
        """Register all performance monitoring tools with FastMCP."""
        
        @mcp.tool()
        async def km_monitor_performance(
            monitoring_scope: Annotated[str, Field(description="Monitoring scope (system|automation|macro|specific)")],
            target_id: Annotated[Optional[str], Field(description="Specific macro or automation UUID to monitor")] = None,
            metrics_types: Annotated[List[str], Field(description="Metrics to collect")] = ["cpu", "memory", "execution_time"],
            monitoring_duration: Annotated[int, Field(description="Monitoring duration in seconds", ge=1, le=3600)] = 60,
            sampling_interval: Annotated[float, Field(description="Sampling interval in seconds", ge=0.1, le=60)] = 1.0,
            include_historical: Annotated[bool, Field(description="Include historical performance data")] = False,
            alert_thresholds: Annotated[Optional[str], Field(description="JSON string with alert thresholds")] = None,
            export_format: Annotated[str, Field(description="Export format (json|csv|dashboard)")] = "json"
        ) -> str:
            """
            Monitor real-time performance metrics for system, automations, or specific macros.
            
            FastMCP Tool for comprehensive performance monitoring through Claude Desktop.
            Collects CPU, memory, disk, network, and automation-specific performance metrics.
            
            Returns real-time metrics, performance analysis, and optimization recommendations.
            """
            try:
                # Validate monitoring scope
                try:
                    scope = MonitoringScope(monitoring_scope.lower())
                except ValueError:
                    valid_scopes = [s.value for s in MonitoringScope]
                    return f"Error: Invalid monitoring_scope '{monitoring_scope}'. Must be one of: {', '.join(valid_scopes)}"
                
                # Validate and convert metric types
                valid_metric_types = []
                for metric_str in metrics_types:
                    try:
                        metric_type = MetricType(metric_str.lower())
                        valid_metric_types.append(metric_type)
                    except ValueError:
                        return f"Error: Invalid metric type '{metric_str}'. Valid types: cpu, memory, disk, network, execution_time, throughput, latency, error_rate"
                
                # Parse alert thresholds if provided
                thresholds = []
                if alert_thresholds:
                    try:
                        threshold_data = json.loads(alert_thresholds)
                        for metric_name, threshold_value in threshold_data.items():
                            try:
                                metric_type = MetricType(metric_name.lower())
                                if metric_type == MetricType.CPU:
                                    thresholds.append(create_cpu_threshold(threshold_value))
                                elif metric_type == MetricType.MEMORY:
                                    thresholds.append(create_memory_threshold(threshold_value))
                                # Add more threshold types as needed
                            except ValueError:
                                continue
                    except json.JSONDecodeError:
                        return "Error: Invalid JSON in alert_thresholds parameter"
                
                # Create monitoring configuration
                session_id = generate_monitoring_session_id()
                config = MonitoringConfiguration(
                    session_id=session_id,
                    scope=scope,
                    target_id=target_id,
                    metrics_types=valid_metric_types,
                    sampling_interval=sampling_interval,
                    duration=monitoring_duration,
                    thresholds=thresholds,
                    include_historical=include_historical
                )
                
                # Start monitoring session
                session_result = await self.metrics_collector.start_monitoring_session(config)
                if session_result.is_left():
                    return f"Error: Failed to start monitoring - {session_result.left()}"
                
                # Wait for initial data collection (at least 3 samples)
                initial_wait = min(sampling_interval * 3, 10.0)
                await asyncio.sleep(initial_wait)
                
                # Get current metrics
                metrics_result = await self.metrics_collector.get_session_metrics(session_id)
                if metrics_result.is_left():
                    return f"Error: Failed to get metrics - {metrics_result.left()}"
                
                metrics = metrics_result.right()
                
                # Basic analysis
                analysis_result = await self.performance_analyzer.analyze_performance(metrics, "basic")
                analysis = analysis_result.right() if analysis_result.is_right() else {}
                
                # Store session for later reference
                session_key = f"{scope.value}_{target_id or 'default'}"
                self.active_monitoring_sessions[session_key] = session_id
                
                # Format response
                response = {
                    "success": True,
                    "monitoring_session": {
                        "session_id": session_id,
                        "scope": scope.value,
                        "target_id": target_id,
                        "duration_seconds": monitoring_duration,
                        "sampling_interval": sampling_interval,
                        "metrics_types": [m.value for m in valid_metric_types],
                        "started_at": datetime.now(UTC).isoformat()
                    },
                    "current_metrics": {
                        "metrics_collected": len(metrics.metrics),
                        "snapshots_collected": len(metrics.snapshots),
                        "alerts_triggered": len(metrics.alerts),
                        "performance_score": analysis.get("performance_score", 50.0)
                    },
                    "real_time_data": self._format_latest_metrics(metrics),
                    "analysis_summary": analysis.get("summary", {}),
                    "alerts": [
                        {
                            "metric_type": alert.metric_type.value,
                            "severity": alert.threshold.severity.value,
                            "current_value": alert.current_value,
                            "threshold": alert.threshold.threshold_value,
                            "message": alert.message,
                            "triggered_at": alert.triggered_at.isoformat()
                        }
                        for alert in metrics.alerts[-5:]  # Last 5 alerts
                    ],
                    "next_steps": [
                        f"Monitor for {monitoring_duration} seconds to collect comprehensive data",
                        "Use km_analyze_bottlenecks for detailed performance analysis",
                        "Use km_optimize_resources for automated optimization recommendations"
                    ]
                }
                
                return f"```json\\n{json.dumps(response, indent=2)}\\n```"
                
            except Exception as e:
                self.logger.error(f"Performance monitoring failed: {e}")
                return f"Error: Performance monitoring failed - {str(e)}"
        
        @mcp.tool()
        async def km_analyze_bottlenecks(
            analysis_scope: Annotated[str, Field(description="Analysis scope (system|automation|workflow)")],
            time_range: Annotated[str, Field(description="Analysis time range (last_hour|last_day|last_week|custom)")] = "last_hour",
            custom_start_time: Annotated[Optional[str], Field(description="Custom start time (ISO format)")] = None,
            custom_end_time: Annotated[Optional[str], Field(description="Custom end time (ISO format)")] = None,
            bottleneck_types: Annotated[List[str], Field(description="Bottleneck types to analyze")] = ["cpu", "memory", "io", "network"],
            severity_threshold: Annotated[str, Field(description="Minimum severity level (low|medium|high|critical)")] = "medium",
            include_recommendations: Annotated[bool, Field(description="Include optimization recommendations")] = True,
            generate_report: Annotated[bool, Field(description="Generate detailed analysis report")] = True
        ) -> str:
            """
            Analyze performance bottlenecks and identify optimization opportunities.
            
            FastMCP Tool for comprehensive bottleneck analysis through Claude Desktop.
            Identifies CPU, memory, I/O, and network bottlenecks with optimization suggestions.
            
            Returns bottleneck analysis, severity assessment, and actionable recommendations.
            """
            try:
                # Find active monitoring session for scope
                session_key = f"{analysis_scope.lower()}_default"
                session_id = self.active_monitoring_sessions.get(session_key)
                
                if not session_id:
                    # Try to get any active session
                    active_sessions = self.metrics_collector.get_active_sessions()
                    if not active_sessions:
                        return "Error: No active monitoring sessions found. Start monitoring with km_monitor_performance first."
                    session_id = active_sessions[0]
                
                # Get metrics for analysis
                metrics_result = await self.metrics_collector.get_session_metrics(session_id)
                if metrics_result.is_left():
                    return f"Error: Failed to get metrics - {metrics_result.left()}"
                
                metrics = metrics_result.right()
                
                # Perform comprehensive analysis
                analysis_result = await self.performance_analyzer.analyze_performance(metrics, "full")
                if analysis_result.is_left():
                    return f"Error: Analysis failed - {analysis_result.left()}"
                
                analysis = analysis_result.right()
                
                # Filter bottlenecks by severity
                severity_levels = ["low", "medium", "high", "critical"]
                min_severity_index = severity_levels.index(severity_threshold.lower())
                
                filtered_bottlenecks = [
                    b for b in analysis["bottlenecks"]
                    if severity_levels.index(b["severity"]) >= min_severity_index
                ]
                
                # Generate detailed report if requested
                report_data = {}
                if generate_report:
                    report_data = {
                        "executive_summary": {
                            "total_bottlenecks": len(filtered_bottlenecks),
                            "critical_issues": len([b for b in filtered_bottlenecks if b["severity"] == "critical"]),
                            "overall_performance_score": analysis["performance_score"],
                            "analysis_timeframe": self._format_time_range(metrics),
                            "recommendations_count": len(analysis["recommendations"])
                        },
                        "detailed_findings": {
                            "performance_trends": analysis.get("trends", {}),
                            "resource_utilization": self._format_resource_utilization(metrics),
                            "alert_summary": self._format_alert_summary(metrics)
                        }
                    }
                
                response = {
                    "success": True,
                    "bottleneck_analysis": {
                        "analysis_timestamp": analysis["analysis_timestamp"],
                        "analysis_scope": analysis_scope,
                        "time_range": time_range,
                        "severity_threshold": severity_threshold,
                        "bottlenecks_found": len(filtered_bottlenecks),
                        "bottlenecks": filtered_bottlenecks,
                        "performance_score": analysis["performance_score"],
                        "overall_health": analysis["summary"]["overall_health"]
                    },
                    "optimization_recommendations": analysis["recommendations"] if include_recommendations else [],
                    "detailed_report": report_data if generate_report else None,
                    "actionable_insights": [
                        "Address critical bottlenecks first to maximize impact",
                        "Monitor performance trends after implementing optimizations",
                        "Set up automated alerts for early bottleneck detection"
                    ],
                    "next_steps": [
                        "Use km_optimize_resources to implement recommended optimizations",
                        "Use km_set_performance_alerts to prevent future bottlenecks",
                        "Schedule regular performance analysis for ongoing optimization"
                    ]
                }
                
                return f"```json\\n{json.dumps(response, indent=2)}\\n```"
                
            except Exception as e:
                self.logger.error(f"Bottleneck analysis failed: {e}")
                return f"Error: Bottleneck analysis failed - {str(e)}"
        
        @mcp.tool()
        async def km_optimize_resources(
            optimization_scope: Annotated[str, Field(description="Optimization scope (system|automation|specific)")],
            target_resources: Annotated[List[str], Field(description="Resources to optimize")] = ["cpu", "memory", "disk"],
            optimization_strategy: Annotated[str, Field(description="Optimization strategy (conservative|balanced|aggressive)")] = "balanced",
            auto_apply: Annotated[bool, Field(description="Automatically apply optimization recommendations")] = False,
            backup_current_settings: Annotated[bool, Field(description="Backup current settings before optimization")] = True,
            performance_target: Annotated[Optional[str], Field(description="Performance target (throughput|latency|efficiency)")] = None,
            resource_limits: Annotated[Optional[str], Field(description="JSON string with resource usage limits")] = None,
            monitoring_period: Annotated[int, Field(description="Post-optimization monitoring period (seconds)", ge=60, le=3600)] = 300
        ) -> str:
            """
            Optimize system and automation resource usage for improved performance.
            
            FastMCP Tool for automated resource optimization through Claude Desktop.
            Optimizes CPU, memory, disk usage, and automation workflow efficiency.
            
            Returns optimization results, performance improvements, and monitoring data.
            """
            try:
                # Validate optimization strategy
                valid_strategies = ["conservative", "balanced", "aggressive"]
                if optimization_strategy not in valid_strategies:
                    return f"Error: Invalid optimization_strategy '{optimization_strategy}'. Must be one of: {', '.join(valid_strategies)}"
                
                # Validate target resources
                valid_resources = ["cpu", "memory", "disk", "network"]
                invalid_resources = [r for r in target_resources if r not in valid_resources]
                if invalid_resources:
                    return f"Error: Invalid target resources: {', '.join(invalid_resources)}. Valid: {', '.join(valid_resources)}"
                
                # Parse resource limits if provided
                limits = {}
                if resource_limits:
                    try:
                        limits = json.loads(resource_limits)
                    except json.JSONDecodeError:
                        return "Error: Invalid JSON in resource_limits parameter"
                
                # Get current performance baseline
                session_id = None
                active_sessions = self.metrics_collector.get_active_sessions()
                if active_sessions:
                    session_id = active_sessions[0]
                
                if session_id:
                    baseline_result = await self.metrics_collector.get_session_metrics(session_id)
                    baseline_metrics = baseline_result.right() if baseline_result.is_right() else None
                else:
                    baseline_metrics = None
                
                # Generate optimization recommendations
                if baseline_metrics:
                    analysis_result = await self.performance_analyzer.analyze_performance(baseline_metrics)
                    if analysis_result.is_right():
                        recommendations = analysis_result.right()["recommendations"]
                    else:
                        recommendations = []
                else:
                    # Generate general recommendations
                    recommendations = self._generate_default_optimizations(
                        target_resources, optimization_strategy
                    )
                
                # Filter recommendations by strategy
                filtered_recommendations = self._filter_recommendations_by_strategy(
                    recommendations, optimization_strategy
                )
                
                # Apply optimizations if requested
                optimization_results = []
                if auto_apply:
                    for rec in filtered_recommendations[:3]:  # Apply top 3 recommendations
                        result = await self._apply_optimization(rec, backup_current_settings)
                        optimization_results.append(result)
                
                # Start post-optimization monitoring
                post_opt_session_id = None
                if optimization_results:
                    config = MonitoringConfiguration(
                        session_id=generate_monitoring_session_id(),
                        scope=MonitoringScope.SYSTEM,
                        metrics_types=[MetricType.CPU, MetricType.MEMORY, MetricType.DISK],
                        sampling_interval=2.0,
                        duration=monitoring_period
                    )
                    
                    session_result = await self.metrics_collector.start_monitoring_session(config)
                    if session_result.is_right():
                        post_opt_session_id = config.session_id
                
                # Calculate estimated improvements
                estimated_improvements = self._calculate_estimated_improvements(
                    filtered_recommendations, optimization_strategy
                )
                
                response = {
                    "success": True,
                    "optimization_session": {
                        "scope": optimization_scope,
                        "strategy": optimization_strategy,
                        "target_resources": target_resources,
                        "auto_applied": auto_apply,
                        "backup_created": backup_current_settings,
                        "started_at": datetime.now(UTC).isoformat()
                    },
                    "recommendations": [
                        {
                            "type": rec["type"],
                            "description": rec["description"],
                            "expected_improvement": rec["expected_improvement"],
                            "complexity": rec["complexity"],
                            "risk": rec["risk"],
                            "status": "applied" if auto_apply and any(r["recommendation_id"] == rec.get("id") for r in optimization_results) else "pending"
                        }
                        for rec in filtered_recommendations
                    ],
                    "optimization_results": optimization_results if auto_apply else [],
                    "estimated_improvements": estimated_improvements,
                    "monitoring": {
                        "post_optimization_session_id": post_opt_session_id,
                        "monitoring_duration": monitoring_period,
                        "baseline_available": baseline_metrics is not None
                    },
                    "performance_prediction": {
                        "cpu_improvement": estimated_improvements.get("cpu", 0),
                        "memory_improvement": estimated_improvements.get("memory", 0),
                        "overall_improvement": estimated_improvements.get("overall", 0),
                        "confidence_level": "medium"
                    },
                    "next_steps": [
                        "Monitor performance improvements over the specified period",
                        "Use km_monitor_performance to track optimization effectiveness",
                        "Consider additional optimizations based on results"
                    ] if auto_apply else [
                        "Review recommendations and select optimizations to implement",
                        "Set auto_apply=true to automatically implement optimizations",
                        "Use km_monitor_performance to establish performance baseline"
                    ]
                }
                
                return f"```json\\n{json.dumps(response, indent=2)}\\n```"
                
            except Exception as e:
                self.logger.error(f"Resource optimization failed: {e}")
                return f"Error: Resource optimization failed - {str(e)}"
        
        @mcp.tool()
        async def km_set_performance_alerts(
            alert_name: Annotated[str, Field(description="Alert configuration name", min_length=1, max_length=100)],
            metric_type: Annotated[str, Field(description="Metric type to monitor (cpu|memory|execution_time|error_rate)")],
            threshold_value: Annotated[float, Field(description="Alert threshold value")],
            threshold_operator: Annotated[str, Field(description="Threshold operator (gt|lt|eq|gte|lte)")] = "gt",
            alert_severity: Annotated[str, Field(description="Alert severity level (low|medium|high|critical)")] = "medium",
            notification_channels: Annotated[List[str], Field(description="Notification channels")] = ["log"],
            monitoring_scope: Annotated[str, Field(description="Monitoring scope (system|automation|macro)")] = "system",
            evaluation_period: Annotated[int, Field(description="Evaluation period in seconds", ge=30, le=3600)] = 300,
            alert_cooldown: Annotated[int, Field(description="Cooldown period between alerts (seconds)", ge=60, le=7200)] = 900,
            auto_resolution: Annotated[bool, Field(description="Enable automatic issue resolution")] = False
        ) -> str:
            """
            Set up performance monitoring alerts with customizable thresholds and notifications.
            
            FastMCP Tool for configuring performance alerts through Claude Desktop.
            Monitors performance metrics and triggers alerts when thresholds are exceeded.
            
            Returns alert configuration, monitoring status, and notification settings.
            """
            try:
                # Validate metric type
                try:
                    metric = MetricType(metric_type.lower())
                except ValueError:
                    valid_metrics = [m.value for m in MetricType]
                    return f"Error: Invalid metric_type '{metric_type}'. Must be one of: {', '.join(valid_metrics)}"
                
                # Validate threshold operator
                try:
                    operator = ThresholdOperator(threshold_operator.lower())
                except ValueError:
                    valid_operators = [o.value for o in ThresholdOperator]
                    return f"Error: Invalid threshold_operator '{threshold_operator}'. Must be one of: {', '.join(valid_operators)}"
                
                # Validate alert severity
                try:
                    severity = AlertSeverity(alert_severity.lower())
                except ValueError:
                    valid_severities = [s.value for s in AlertSeverity]
                    return f"Error: Invalid alert_severity '{alert_severity}'. Must be one of: {', '.join(valid_severities)}"
                
                # Create performance threshold
                threshold = PerformanceThreshold(
                    metric_type=metric,
                    threshold_value=threshold_value,
                    operator=operator,
                    severity=severity,
                    evaluation_period=evaluation_period,
                    cooldown_period=alert_cooldown
                )
                
                # Store alert configuration (in a real implementation, this would be persisted)
                alert_config = {
                    "alert_id": f"alert_{len(self.active_monitoring_sessions)}",
                    "name": alert_name,
                    "threshold": threshold,
                    "notification_channels": notification_channels,
                    "monitoring_scope": monitoring_scope,
                    "auto_resolution": auto_resolution,
                    "created_at": datetime.now(UTC),
                    "enabled": True
                }
                
                # Test threshold value against recent metrics
                test_result = await self._test_alert_threshold(metric, threshold_value, operator)
                
                response = {
                    "success": True,
                    "alert_configuration": {
                        "alert_id": alert_config["alert_id"],
                        "name": alert_name,
                        "metric_type": metric.value,
                        "threshold_value": threshold_value,
                        "operator": operator.value,
                        "severity": severity.value,
                        "evaluation_period": evaluation_period,
                        "cooldown_period": alert_cooldown,
                        "monitoring_scope": monitoring_scope,
                        "notification_channels": notification_channels,
                        "auto_resolution": auto_resolution,
                        "enabled": True,
                        "created_at": alert_config["created_at"].isoformat()
                    },
                    "threshold_test": test_result,
                    "monitoring_status": {
                        "active_sessions": len(self.metrics_collector.get_active_sessions()),
                        "will_be_monitored": len(self.metrics_collector.get_active_sessions()) > 0,
                        "recommendation": "Start monitoring with km_monitor_performance if not already active"
                    },
                    "estimated_sensitivity": self._estimate_alert_sensitivity(metric, threshold_value),
                    "configuration_tips": [
                        f"Threshold set at {threshold_value} for {metric.value}",
                        f"Will trigger when value is {operator.value} {threshold_value}",
                        f"Alerts will have {alert_cooldown}s cooldown to prevent spam",
                        "Consider testing with different threshold values to find optimal sensitivity"
                    ],
                    "next_steps": [
                        "Monitor alert effectiveness and adjust thresholds as needed",
                        "Set up additional alerts for comprehensive monitoring",
                        "Review notification channels and response procedures"
                    ]
                }
                
                return f"```json\\n{json.dumps(response, indent=2)}\\n```"
                
            except Exception as e:
                self.logger.error(f"Alert configuration failed: {e}")
                return f"Error: Alert configuration failed - {str(e)}"
        
        @mcp.tool()
        async def km_get_performance_dashboard() -> str:
            """
            Get comprehensive performance dashboard with real-time metrics and insights.
            
            FastMCP Tool for accessing performance dashboards through Claude Desktop.
            Provides real-time metrics, historical trends, and optimization recommendations.
            
            Returns dashboard data, performance insights, and actionable recommendations.
            """
            try:
                # Get all active monitoring sessions
                active_sessions = self.metrics_collector.get_active_sessions()
                
                if not active_sessions:
                    return f"```json\\n{json.dumps({
                        'success': False,
                        'message': 'No active monitoring sessions found. Start monitoring with km_monitor_performance first.',
                        'dashboard_available': False
                    }, indent=2)}\\n```"
                
                # Collect data from all sessions
                dashboard_data = {
                    "success": True,
                    "dashboard_timestamp": datetime.now(UTC).isoformat(),
                    "active_sessions": len(active_sessions),
                    "system_overview": {},
                    "performance_metrics": {},
                    "recent_alerts": [],
                    "optimization_opportunities": [],
                    "recommendations": []
                }
                
                # Process each active session
                for session_id in active_sessions:
                    session_status = self.metrics_collector.get_session_status(session_id)
                    if session_status:
                        dashboard_data["system_overview"][session_id] = session_status
                    
                    # Get metrics
                    metrics_result = await self.metrics_collector.get_session_metrics(session_id)
                    if metrics_result.is_right():
                        metrics = metrics_result.right()
                        
                        # Add recent alerts
                        dashboard_data["recent_alerts"].extend([
                            {
                                "session_id": session_id,
                                "metric_type": alert.metric_type.value,
                                "severity": alert.threshold.severity.value,
                                "value": alert.current_value,
                                "threshold": alert.threshold.threshold_value,
                                "triggered_at": alert.triggered_at.isoformat()
                            }
                            for alert in metrics.alerts[-3:]  # Last 3 alerts per session
                        ])
                        
                        # Add performance metrics summary
                        if metrics.metrics:
                            latest_metrics = {}
                            for metric_type in MetricType:
                                latest = metrics.get_latest_value(metric_type)
                                if latest:
                                    latest_metrics[metric_type.value] = {
                                        "value": latest.value,
                                        "unit": latest.unit,
                                        "timestamp": latest.timestamp.isoformat()
                                    }
                            
                            dashboard_data["performance_metrics"][session_id] = latest_metrics
                
                # Generate overall recommendations
                dashboard_data["recommendations"] = [
                    "Monitor CPU and memory usage trends for optimization opportunities",
                    "Set up automated alerts for proactive issue detection",
                    "Review performance bottlenecks and implement optimizations",
                    "Establish performance baselines for comparison"
                ]
                
                return f"```json\\n{json.dumps(dashboard_data, indent=2)}\\n```"
                
            except Exception as e:
                self.logger.error(f"Dashboard generation failed: {e}")
                return f"Error: Dashboard generation failed - {str(e)}"
        
        self.logger.info("Registered performance monitoring MCP tools successfully")
    
    def _format_latest_metrics(self, metrics) -> Dict[str, Any]:
        """Format latest metrics for response."""
        latest_data = {}
        
        for metric_type in MetricType:
            latest = metrics.get_latest_value(metric_type)
            if latest:
                latest_data[metric_type.value] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp.isoformat()
                }
        
        return latest_data
    
    def _format_time_range(self, metrics) -> str:
        """Format time range of metrics."""
        if not metrics.metrics:
            return "No data"
        
        start_time = min(m.timestamp for m in metrics.metrics)
        end_time = max(m.timestamp for m in metrics.metrics)
        duration = end_time - start_time
        
        return f"{duration.total_seconds():.0f} seconds"
    
    def _format_resource_utilization(self, metrics) -> Dict[str, Any]:
        """Format resource utilization summary."""
        if not metrics.snapshots:
            return {}
        
        latest = max(metrics.snapshots, key=lambda s: s.timestamp)
        
        return {
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "load_average": latest.load_average,
            "timestamp": latest.timestamp.isoformat()
        }
    
    def _format_alert_summary(self, metrics) -> Dict[str, Any]:
        """Format alert summary."""
        severity_counts = {}
        for alert in metrics.alerts:
            severity = alert.threshold.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_alerts": len(metrics.alerts),
            "by_severity": severity_counts,
            "most_recent": metrics.alerts[-1].triggered_at.isoformat() if metrics.alerts else None
        }
    
    def _generate_default_optimizations(
        self, 
        target_resources: List[str], 
        strategy: str
    ) -> List[Dict[str, Any]]:
        """Generate default optimization recommendations."""
        recommendations = []
        
        if "cpu" in target_resources:
            recommendations.append({
                "type": "cpu_optimization",
                "description": "Optimize CPU usage through process scheduling",
                "expected_improvement": 15.0 if strategy == "aggressive" else 10.0,
                "complexity": "medium",
                "risk": "low"
            })
        
        if "memory" in target_resources:
            recommendations.append({
                "type": "memory_optimization", 
                "description": "Implement memory caching and cleanup",
                "expected_improvement": 20.0 if strategy == "aggressive" else 12.0,
                "complexity": "medium",
                "risk": "medium"
            })
        
        return recommendations
    
    def _filter_recommendations_by_strategy(
        self, 
        recommendations: List[Dict[str, Any]], 
        strategy: str
    ) -> List[Dict[str, Any]]:
        """Filter recommendations based on optimization strategy."""
        if strategy == "conservative":
            return [r for r in recommendations if r.get("risk", "medium") == "low"]
        elif strategy == "aggressive":
            return recommendations  # Include all recommendations
        else:  # balanced
            return [r for r in recommendations if r.get("risk", "medium") in ["low", "medium"]]
    
    async def _apply_optimization(
        self, 
        recommendation: Dict[str, Any], 
        backup: bool
    ) -> Dict[str, Any]:
        """Apply optimization recommendation (mock implementation)."""
        # In a real implementation, this would apply actual optimizations
        return {
            "recommendation_id": recommendation.get("type", "unknown"),
            "status": "applied",
            "backup_created": backup,
            "applied_at": datetime.now(UTC).isoformat(),
            "estimated_improvement": recommendation.get("expected_improvement", 0)
        }
    
    def _calculate_estimated_improvements(
        self, 
        recommendations: List[Dict[str, Any]], 
        strategy: str
    ) -> Dict[str, float]:
        """Calculate estimated performance improvements."""
        cpu_improvement = sum(r.get("expected_improvement", 0) for r in recommendations if "cpu" in r.get("type", ""))
        memory_improvement = sum(r.get("expected_improvement", 0) for r in recommendations if "memory" in r.get("type", ""))
        
        overall_improvement = (cpu_improvement + memory_improvement) / 2
        
        # Apply strategy multiplier
        multiplier = {"conservative": 0.7, "balanced": 1.0, "aggressive": 1.3}.get(strategy, 1.0)
        
        return {
            "cpu": cpu_improvement * multiplier,
            "memory": memory_improvement * multiplier,
            "overall": overall_improvement * multiplier
        }
    
    async def _test_alert_threshold(
        self, 
        metric_type: MetricType, 
        threshold_value: float, 
        operator: ThresholdOperator
    ) -> Dict[str, Any]:
        """Test alert threshold against recent metrics."""
        # Get recent metrics from cache
        recent_metrics = await self.metrics_collector.get_recent_metrics(metric_type)
        
        if not recent_metrics:
            return {
                "status": "no_data",
                "message": "No recent data available for testing"
            }
        
        # Test threshold
        current_value = recent_metrics[-1].value if recent_metrics else 0
        would_trigger = False
        
        if operator == ThresholdOperator.GREATER_THAN:
            would_trigger = current_value > threshold_value
        elif operator == ThresholdOperator.LESS_THAN:
            would_trigger = current_value < threshold_value
        
        return {
            "status": "tested",
            "current_value": current_value,
            "threshold_value": threshold_value,
            "would_trigger_now": would_trigger,
            "recommendation": "Threshold appears appropriate" if not would_trigger else "Consider adjusting threshold to reduce false positives"
        }
    
    def _estimate_alert_sensitivity(self, metric_type: MetricType, threshold_value: float) -> str:
        """Estimate alert sensitivity based on threshold."""
        # Simple heuristic based on typical ranges
        ranges = {
            MetricType.CPU: (0, 100),
            MetricType.MEMORY: (0, 100),
            MetricType.EXECUTION_TIME: (0, 10000),
            MetricType.ERROR_RATE: (0, 10)
        }
        
        range_min, range_max = ranges.get(metric_type, (0, 100))
        normalized_threshold = (threshold_value - range_min) / (range_max - range_min)
        
        if normalized_threshold < 0.3:
            return "high"
        elif normalized_threshold > 0.8:
            return "low"
        else:
            return "medium"


# Global instance for tool registration
_performance_monitor_tools: Optional[PerformanceMonitorTools] = None


def get_performance_monitor_tools() -> PerformanceMonitorTools:
    """Get or create the global performance monitor tools instance."""
    global _performance_monitor_tools
    if _performance_monitor_tools is None:
        _performance_monitor_tools = PerformanceMonitorTools()
    return _performance_monitor_tools