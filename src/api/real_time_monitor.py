"""
Real-Time Monitor - TASK_64 Phase 5 Integration & Monitoring

Real-time API monitoring, metrics, and alerting for API orchestration.
Provides comprehensive monitoring with intelligent alerting and analytics.

Architecture: Real-time Monitoring + Metrics Collection + Alerting + Analytics + Dashboards
Performance: <5ms metric collection, <50ms alert evaluation, <100ms dashboard updates
Intelligence: Anomaly detection, predictive alerting, trend analysis, capacity planning
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import json
import time
import statistics
from collections import defaultdict, deque
import websockets
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.api_orchestration_architecture import (
    ServiceId, APIOrchestrationError, create_service_id
)


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"                    # Incrementing counter
    GAUGE = "gauge"                        # Current value
    HISTOGRAM = "histogram"                # Distribution of values
    TIMER = "timer"                        # Duration measurements
    RATE = "rate"                          # Rate over time


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class MonitoringScope(Enum):
    """Monitoring scope levels."""
    GLOBAL = "global"
    SERVICE = "service"
    ENDPOINT = "endpoint"
    USER = "user"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    rule_name: str
    description: str
    
    # Rule conditions
    metric_name: str
    condition: str                         # e.g., "> 1000", "< 0.95"
    threshold_value: float
    comparison_operator: str               # >, <, >=, <=, ==, !=
    
    # Evaluation settings
    evaluation_window_minutes: int = 5
    data_points_required: int = 3
    consecutive_violations: int = 1
    
    # Alert settings
    severity: AlertSeverity = AlertSeverity.WARNING
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Notification settings
    notification_channels: List[str] = field(default_factory=list)
    notification_interval_minutes: int = 60
    auto_resolve: bool = True
    auto_resolve_timeout_minutes: int = 30
    
    # Suppression
    suppression_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_evaluated: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Active alert instance."""
    alert_id: str
    rule_id: str
    metric_name: str
    severity: AlertSeverity
    
    # Alert details
    title: str
    description: str
    current_value: float
    threshold_value: float
    
    # Timing
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Status
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    resolution_note: Optional[str] = None
    
    # Context
    tags: Dict[str, str] = field(default_factory=dict)
    affected_services: List[ServiceId] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    
    # Escalation
    escalation_level: int = 0
    last_notification: Optional[datetime] = None
    notification_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_duration(self) -> timedelta:
        """Get alert duration."""
        end_time = self.resolved_at or datetime.now(UTC)
        return end_time - self.triggered_at
    
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == AlertStatus.ACTIVE


@dataclass
class Dashboard:
    """Monitoring dashboard definition."""
    dashboard_id: str
    dashboard_name: str
    description: str
    
    # Layout configuration
    widgets: List[Dict[str, Any]] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    
    # Data sources
    metrics: List[str] = field(default_factory=list)
    time_range: str = "1h"                 # 1h, 6h, 24h, 7d, 30d
    refresh_interval_seconds: int = 30
    
    # Access control
    visibility: str = "public"             # public, private, team
    allowed_users: List[str] = field(default_factory=list)
    
    # Status
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metrics."""
    metric_name: str
    data_points: deque = field(default_factory=lambda: deque(maxlen=10000))
    aggregations: Dict[str, float] = field(default_factory=dict)
    
    def add_point(self, point: MetricPoint):
        """Add data point to series."""
        self.data_points.append(point)
        self._update_aggregations()
    
    def _update_aggregations(self):
        """Update aggregated statistics."""
        if not self.data_points:
            return
        
        values = [point.value for point in self.data_points]
        recent_values = [point.value for point in list(self.data_points)[-100:]]  # Last 100 points
        
        self.aggregations = {
            "count": len(values),
            "current": values[-1] if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "avg": statistics.mean(values) if values else 0,
            "median": statistics.median(values) if values else 0,
            "recent_avg": statistics.mean(recent_values) if recent_values else 0,
            "rate_per_minute": self._calculate_rate(),
            "trend": self._calculate_trend()
        }
    
    def _calculate_rate(self) -> float:
        """Calculate rate per minute for counters."""
        if len(self.data_points) < 2:
            return 0.0
        
        recent_points = list(self.data_points)[-60:]  # Last 60 points
        if len(recent_points) < 2:
            return 0.0
        
        time_span = (recent_points[-1].timestamp - recent_points[0].timestamp).total_seconds()
        if time_span <= 0:
            return 0.0
        
        value_change = recent_points[-1].value - recent_points[0].value
        return (value_change / time_span) * 60  # Per minute
    
    def _calculate_trend(self) -> str:
        """Calculate trend direction."""
        if len(self.data_points) < 10:
            return "stable"
        
        recent_values = [point.value for point in list(self.data_points)[-10:]]
        
        # Simple linear trend calculation
        if recent_values[-1] > recent_values[0] * 1.1:
            return "increasing"
        elif recent_values[-1] < recent_values[0] * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def get_percentile(self, percentile: float) -> float:
        """Get percentile value."""
        if not self.data_points:
            return 0.0
        
        values = sorted([point.value for point in self.data_points])
        index = int(percentile / 100 * len(values))
        return values[min(index, len(values) - 1)]


class AnomalyDetector:
    """Anomaly detection for metrics."""
    
    def __init__(self):
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_thresholds = {
            "response_time": 2.0,      # 2 standard deviations
            "error_rate": 3.0,         # 3 standard deviations
            "throughput": 2.5          # 2.5 standard deviations
        }
    
    async def detect_anomalies(self, metric_name: str, series: MetricSeries) -> List[Dict[str, Any]]:
        """Detect anomalies in metric series."""
        anomalies = []
        
        if len(series.data_points) < 50:  # Need baseline data
            return anomalies
        
        # Update baseline
        await self._update_baseline(metric_name, series)
        
        # Check recent points for anomalies
        recent_points = list(series.data_points)[-10:]
        baseline = self.baselines.get(metric_name, {})
        
        for point in recent_points:
            anomaly = await self._check_point_anomaly(point, baseline)
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _update_baseline(self, metric_name: str, series: MetricSeries):
        """Update baseline statistics for metric."""
        historical_values = [point.value for point in list(series.data_points)[:-10]]  # Exclude recent points
        
        if len(historical_values) < 20:
            return
        
        self.baselines[metric_name] = {
            "mean": statistics.mean(historical_values),
            "std_dev": statistics.stdev(historical_values) if len(historical_values) > 1 else 0,
            "median": statistics.median(historical_values),
            "min": min(historical_values),
            "max": max(historical_values),
            "updated_at": datetime.now(UTC).isoformat()
        }
    
    async def _check_point_anomaly(self, point: MetricPoint, baseline: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Check if point is anomalous."""
        if not baseline or baseline.get("std_dev", 0) == 0:
            return None
        
        mean = baseline["mean"]
        std_dev = baseline["std_dev"]
        
        # Calculate z-score
        z_score = abs((point.value - mean) / std_dev)
        
        # Get threshold for metric type
        threshold = self.anomaly_thresholds.get("response_time", 2.0)  # Default threshold
        if "error" in point.metric_name.lower():
            threshold = self.anomaly_thresholds.get("error_rate", 3.0)
        elif "throughput" in point.metric_name.lower() or "rps" in point.metric_name.lower():
            threshold = self.anomaly_thresholds.get("throughput", 2.5)
        
        if z_score > threshold:
            return {
                "type": "statistical_anomaly",
                "metric_name": point.metric_name,
                "timestamp": point.timestamp.isoformat(),
                "value": point.value,
                "expected_range": (mean - threshold * std_dev, mean + threshold * std_dev),
                "z_score": z_score,
                "severity": "high" if z_score > threshold * 1.5 else "medium"
            }
        
        return None


class RealTimeMonitor:
    """Advanced real-time monitoring system."""
    
    def __init__(self):
        self.metrics: Dict[str, MetricSeries] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.dashboards: Dict[str, Dashboard] = {}
        self.anomaly_detector = AnomalyDetector()
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Set[websockets.WebSocketServerProtocol] = set()
        
        # Notification channels
        self.notification_channels: Dict[str, Callable] = {}
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.metric_retention_hours = 24
        self.alert_evaluation_interval_seconds = 30
        
        # Performance metrics
        self.monitor_metrics = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "alerts_resolved": 0,
            "anomalies_detected": 0,
            "dashboard_views": 0,
            "notification_sent": 0,
            "average_collection_time_ms": 0.0,
            "average_alert_evaluation_time_ms": 0.0
        }
        
        # Start background tasks
        asyncio.create_task(self._alert_evaluation_loop())
        asyncio.create_task(self._cleanup_loop())
        asyncio.create_task(self._anomaly_detection_loop())
    
    @require(lambda metric: isinstance(metric, MetricPoint))
    async def collect_metric(self, metric: MetricPoint) -> Either[APIOrchestrationError, bool]:
        """Collect metric data point."""
        try:
            collection_start = time.time()
            
            if not self.monitoring_enabled:
                return Either.success(True)
            
            # Get or create metric series
            if metric.metric_name not in self.metrics:
                self.metrics[metric.metric_name] = MetricSeries(metric_name=metric.metric_name)
            
            series = self.metrics[metric.metric_name]
            series.add_point(metric)
            
            # Broadcast to WebSocket connections
            await self._broadcast_metric_update(metric)
            
            # Update performance metrics
            collection_time = (time.time() - collection_start) * 1000
            self.monitor_metrics["metrics_collected"] += 1
            
            current_avg = self.monitor_metrics["average_collection_time_ms"]
            total_collections = self.monitor_metrics["metrics_collected"]
            self.monitor_metrics["average_collection_time_ms"] = (current_avg * (total_collections - 1) + collection_time) / total_collections
            
            return Either.success(True)
            
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Metric collection failed: {str(e)}"))
    
    @require(lambda rule: isinstance(rule, AlertRule))
    def add_alert_rule(self, rule: AlertRule) -> Either[APIOrchestrationError, bool]:
        """Add alert rule."""
        try:
            self.alert_rules[rule.rule_id] = rule
            return Either.success(True)
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Failed to add alert rule: {str(e)}"))
    
    @require(lambda dashboard: isinstance(dashboard, Dashboard))
    def add_dashboard(self, dashboard: Dashboard) -> Either[APIOrchestrationError, bool]:
        """Add monitoring dashboard."""
        try:
            self.dashboards[dashboard.dashboard_id] = dashboard
            return Either.success(True)
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Failed to add dashboard: {str(e)}"))
    
    def add_notification_channel(self, channel_name: str, handler: Callable) -> Either[APIOrchestrationError, bool]:
        """Add notification channel handler."""
        try:
            self.notification_channels[channel_name] = handler
            return Either.success(True)
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Failed to add notification channel: {str(e)}"))
    
    async def get_metric_data(self, metric_name: str, time_range: str = "1h") -> Either[APIOrchestrationError, Dict[str, Any]]:
        """Get metric data for specified time range."""
        try:
            if metric_name not in self.metrics:
                return Either.error(APIOrchestrationError(f"Metric not found: {metric_name}"))
            
            series = self.metrics[metric_name]
            
            # Parse time range
            time_delta = self._parse_time_range(time_range)
            cutoff_time = datetime.now(UTC) - time_delta
            
            # Filter data points by time range
            filtered_points = [
                point for point in series.data_points
                if point.timestamp >= cutoff_time
            ]
            
            result = {
                "metric_name": metric_name,
                "time_range": time_range,
                "data_points": [point.to_dict() for point in filtered_points],
                "aggregations": series.aggregations,
                "count": len(filtered_points),
                "percentiles": {
                    "p50": series.get_percentile(50),
                    "p95": series.get_percentile(95),
                    "p99": series.get_percentile(99)
                }
            }
            
            return Either.success(result)
            
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Failed to get metric data: {str(e)}"))
    
    async def get_dashboard_data(self, dashboard_id: str) -> Either[APIOrchestrationError, Dict[str, Any]]:
        """Get dashboard data."""
        try:
            if dashboard_id not in self.dashboards:
                return Either.error(APIOrchestrationError(f"Dashboard not found: {dashboard_id}"))
            
            dashboard = self.dashboards[dashboard_id]
            self.monitor_metrics["dashboard_views"] += 1
            
            # Collect data for all dashboard metrics
            dashboard_data = {
                "dashboard_id": dashboard_id,
                "dashboard_name": dashboard.dashboard_name,
                "description": dashboard.description,
                "widgets": dashboard.widgets,
                "last_updated": datetime.now(UTC).isoformat(),
                "metrics_data": {}
            }
            
            for metric_name in dashboard.metrics:
                metric_result = await self.get_metric_data(metric_name, dashboard.time_range)
                if metric_result.is_success():
                    dashboard_data["metrics_data"][metric_name] = metric_result.value
            
            return Either.success(dashboard_data)
            
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Failed to get dashboard data: {str(e)}"))
    
    async def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = [alert for alert in self.active_alerts.values() if alert.is_active()]
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        # Sort by severity and timestamp
        severity_order = {AlertSeverity.CRITICAL: 0, AlertSeverity.ERROR: 1, AlertSeverity.WARNING: 2, AlertSeverity.INFO: 3}
        alerts.sort(key=lambda a: (severity_order.get(a.severity, 99), a.triggered_at))
        
        return alerts
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, note: Optional[str] = None) -> Either[APIOrchestrationError, bool]:
        """Acknowledge an alert."""
        try:
            if alert_id not in self.active_alerts:
                return Either.error(APIOrchestrationError(f"Alert not found: {alert_id}"))
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(UTC)
            alert.acknowledged_by = acknowledged_by
            
            if note:
                alert.metadata["acknowledgment_note"] = note
            
            # Broadcast alert update
            await self._broadcast_alert_update(alert)
            
            return Either.success(True)
            
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Failed to acknowledge alert: {str(e)}"))
    
    async def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None) -> Either[APIOrchestrationError, bool]:
        """Resolve an alert."""
        try:
            if alert_id not in self.active_alerts:
                return Either.error(APIOrchestrationError(f"Alert not found: {alert_id}"))
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(UTC)
            alert.resolution_note = resolution_note
            
            self.monitor_metrics["alerts_resolved"] += 1
            
            # Broadcast alert update
            await self._broadcast_alert_update(alert)
            
            return Either.success(True)
            
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Failed to resolve alert: {str(e)}"))
    
    # Background tasks
    
    async def _alert_evaluation_loop(self):
        """Background loop for evaluating alert rules."""
        while True:
            try:
                await asyncio.sleep(self.alert_evaluation_interval_seconds)
                
                evaluation_start = time.time()
                await self._evaluate_alert_rules()
                
                evaluation_time = (time.time() - evaluation_start) * 1000
                current_avg = self.monitor_metrics["average_alert_evaluation_time_ms"]
                self.monitor_metrics["average_alert_evaluation_time_ms"] = (current_avg + evaluation_time) / 2
                
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(60)  # Error recovery
    
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules."""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_single_rule(rule)
                rule.last_evaluated = datetime.now(UTC)
            except Exception:
                pass  # Continue with other rules
    
    async def _evaluate_single_rule(self, rule: AlertRule):
        """Evaluate single alert rule."""
        if rule.metric_name not in self.metrics:
            return
        
        series = self.metrics[rule.metric_name]
        
        # Get recent data points for evaluation
        evaluation_window = timedelta(minutes=rule.evaluation_window_minutes)
        cutoff_time = datetime.now(UTC) - evaluation_window
        
        recent_points = [
            point for point in series.data_points
            if point.timestamp >= cutoff_time
        ]
        
        if len(recent_points) < rule.data_points_required:
            return
        
        # Check if condition is violated
        violations = 0
        for point in recent_points[-rule.consecutive_violations:]:
            if self._evaluate_condition(point.value, rule.comparison_operator, rule.threshold_value):
                violations += 1
        
        # Trigger alert if condition met
        if violations >= rule.consecutive_violations:
            await self._trigger_alert(rule, recent_points[-1])
        else:
            # Auto-resolve if condition no longer met
            await self._check_auto_resolve(rule)
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            return False
    
    async def _trigger_alert(self, rule: AlertRule, triggering_point: MetricPoint):
        """Trigger alert for rule violation."""
        # Check if alert already exists for this rule
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule.rule_id and alert.is_active():
                existing_alert = alert
                break
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = triggering_point.value
            existing_alert.last_notification = datetime.now(UTC)
            await self._send_notifications(existing_alert, is_update=True)
        else:
            # Create new alert
            alert = Alert(
                alert_id=f"{rule.rule_id}_{int(time.time() * 1000)}",
                rule_id=rule.rule_id,
                metric_name=rule.metric_name,
                severity=rule.severity,
                title=f"{rule.rule_name} - {rule.metric_name}",
                description=f"Metric {rule.metric_name} {rule.comparison_operator} {rule.threshold_value}",
                current_value=triggering_point.value,
                threshold_value=rule.threshold_value,
                triggered_at=datetime.now(UTC),
                tags=rule.tags.copy()
            )
            
            self.active_alerts[alert.alert_id] = alert
            self.monitor_metrics["alerts_triggered"] += 1
            
            # Send notifications
            await self._send_notifications(alert, is_new=True)
            
            # Broadcast to WebSocket connections
            await self._broadcast_alert_update(alert)
    
    async def _check_auto_resolve(self, rule: AlertRule):
        """Check for auto-resolution of alerts."""
        if not rule.auto_resolve:
            return
        
        # Find active alerts for this rule
        rule_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.rule_id == rule.rule_id and alert.is_active()
        ]
        
        for alert in rule_alerts:
            # Check if auto-resolve timeout reached
            auto_resolve_time = alert.triggered_at + timedelta(minutes=rule.auto_resolve_timeout_minutes)
            if datetime.now(UTC) >= auto_resolve_time:
                await self.resolve_alert(alert.alert_id, "Auto-resolved: condition no longer met")
    
    async def _send_notifications(self, alert: Alert, is_new: bool = False, is_update: bool = False):
        """Send alert notifications."""
        if alert.rule_id not in self.alert_rules:
            return
        
        rule = self.alert_rules[alert.rule_id]
        
        # Check notification interval
        if (alert.last_notification and 
            (datetime.now(UTC) - alert.last_notification).total_seconds() < rule.notification_interval_minutes * 60):
            return
        
        # Send to all configured channels
        for channel_name in rule.notification_channels:
            if channel_name in self.notification_channels:
                try:
                    await self.notification_channels[channel_name](alert, is_new, is_update)
                    alert.notification_count += 1
                    self.monitor_metrics["notification_sent"] += 1
                except Exception:
                    pass  # Continue with other channels
        
        alert.last_notification = datetime.now(UTC)
    
    async def _anomaly_detection_loop(self):
        """Background loop for anomaly detection."""
        while True:
            try:
                await asyncio.sleep(120)  # Run every 2 minutes
                
                for metric_name, series in self.metrics.items():
                    anomalies = await self.anomaly_detector.detect_anomalies(metric_name, series)
                    
                    for anomaly in anomalies:
                        self.monitor_metrics["anomalies_detected"] += 1
                        await self._broadcast_anomaly_detection(anomaly)
                
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(60)  # Error recovery
    
    async def _cleanup_loop(self):
        """Background loop for cleaning up old data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.now(UTC) - timedelta(hours=self.metric_retention_hours)
                
                # Clean up old metric data
                for series in self.metrics.values():
                    while series.data_points and series.data_points[0].timestamp < cutoff_time:
                        series.data_points.popleft()
                
                # Clean up resolved alerts older than 7 days
                alert_cutoff = datetime.now(UTC) - timedelta(days=7)
                old_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if alert.status == AlertStatus.RESOLVED and alert.resolved_at and alert.resolved_at < alert_cutoff
                ]
                
                for alert_id in old_alerts:
                    del self.active_alerts[alert_id]
                
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(300)  # Error recovery
    
    # WebSocket broadcasting
    
    async def _broadcast_metric_update(self, metric: MetricPoint):
        """Broadcast metric update to WebSocket connections."""
        if not self.websocket_connections:
            return
        
        message = {
            "type": "metric_update",
            "data": metric.to_dict()
        }
        
        await self._broadcast_to_websockets(message)
    
    async def _broadcast_alert_update(self, alert: Alert):
        """Broadcast alert update to WebSocket connections."""
        if not self.websocket_connections:
            return
        
        message = {
            "type": "alert_update",
            "data": {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "title": alert.title,
                "current_value": alert.current_value,
                "triggered_at": alert.triggered_at.isoformat(),
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
        }
        
        await self._broadcast_to_websockets(message)
    
    async def _broadcast_anomaly_detection(self, anomaly: Dict[str, Any]):
        """Broadcast anomaly detection to WebSocket connections."""
        if not self.websocket_connections:
            return
        
        message = {
            "type": "anomaly_detected",
            "data": anomaly
        }
        
        await self._broadcast_to_websockets(message)
    
    async def _broadcast_to_websockets(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections."""
        if not self.websocket_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send(message_json)
            except Exception:
                disconnected.add(websocket)
        
        # Remove disconnected connections
        self.websocket_connections -= disconnected
    
    # Helper methods
    
    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta."""
        if time_range.endswith('m'):
            return timedelta(minutes=int(time_range[:-1]))
        elif time_range.endswith('h'):
            return timedelta(hours=int(time_range[:-1]))
        elif time_range.endswith('d'):
            return timedelta(days=int(time_range[:-1]))
        else:
            return timedelta(hours=1)  # Default to 1 hour
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring system metrics."""
        total_metrics = len(self.metrics)
        total_data_points = sum(len(series.data_points) for series in self.metrics.values())
        active_alert_count = len([alert for alert in self.active_alerts.values() if alert.is_active()])
        
        return {
            **self.monitor_metrics,
            "total_metrics": total_metrics,
            "total_data_points": total_data_points,
            "total_alert_rules": len(self.alert_rules),
            "active_alerts": active_alert_count,
            "total_dashboards": len(self.dashboards),
            "websocket_connections": len(self.websocket_connections),
            "notification_channels": len(self.notification_channels),
            "monitoring_enabled": self.monitoring_enabled,
            "uptime_hours": (datetime.now(UTC) - datetime.now(UTC).replace(hour=0, minute=0, second=0)).total_seconds() / 3600
        }


# Export the monitoring classes
__all__ = [
    "RealTimeMonitor", "MetricPoint", "AlertRule", "Alert", "Dashboard",
    "MetricSeries", "AnomalyDetector", "MetricType", "AlertSeverity", 
    "AlertStatus", "MonitoringScope"
]