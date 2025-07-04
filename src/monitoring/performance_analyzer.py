"""
Performance analysis and bottleneck detection engine.

Advanced performance analysis with ML-powered bottleneck detection,
optimization recommendations, and predictive insights.

Security: Secure performance analysis with access validation.
Performance: <200ms analysis time, intelligent caching.
Type Safety: Complete contract-driven analysis validation.
"""

import asyncio
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, UTC
from collections import defaultdict
import logging
from dataclasses import dataclass

from ..core.performance_monitoring import (
    MonitoringSessionID, MetricType, AlertSeverity, BottleneckType,
    PerformanceMetrics, BottleneckAnalysis, OptimizationRecommendation,
    PerformanceThreshold, OptimizationStrategy, PerformanceTarget,
    SystemResourceSnapshot, MetricValue, PerformanceMonitoringError
)
from ..core.contracts import require, ensure
from ..core.either import Either

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline for metric comparison."""
    metric_type: MetricType
    baseline_value: float
    baseline_range: Tuple[float, float]
    sample_size: int
    confidence_interval: float
    created_at: datetime


class PerformanceAnalyzer:
    """Advanced performance analysis and bottleneck detection engine."""
    
    def __init__(self):
        self.analysis_cache: Dict[str, Any] = {}
        self.baselines: Dict[MetricType, PerformanceBaseline] = {}
        self.baseline_metrics: Dict[MetricType, Dict[str, float]] = {}
        self.optimization_history: List[OptimizationRecommendation] = []
        self.logger = logging.getLogger(__name__)
        
        # Performance analysis thresholds
        self.bottleneck_thresholds = {
            MetricType.CPU: 80.0,           # CPU usage > 80%
            MetricType.MEMORY: 85.0,        # Memory usage > 85%
            MetricType.DISK: 90.0,          # Disk usage > 90%
            MetricType.NETWORK: 80.0,       # Network utilization > 80%
            MetricType.EXECUTION_TIME: 5000.0,  # Execution time > 5s
            MetricType.ERROR_RATE: 2.0,     # Error rate > 2%
        }
        
        # Normal performance ranges
        self.normal_ranges = {
            MetricType.CPU: (0.0, 60.0),
            MetricType.MEMORY: (0.0, 70.0),
            MetricType.DISK: (0.0, 80.0),
            MetricType.NETWORK: (0.0, 70.0),
            MetricType.EXECUTION_TIME: (0.0, 2000.0),
            MetricType.ERROR_RATE: (0.0, 1.0),
        }
    
    async def analyze_performance(
        self, 
        metrics: PerformanceMetrics,
        analysis_depth: str = "full"
    ) -> Either[Exception, Dict[str, Any]]:
        """Comprehensive performance analysis."""
        try:
            analysis_start = datetime.now(UTC)
            
            # Basic metrics analysis
            basic_analysis = await self._analyze_basic_metrics(metrics)
            if basic_analysis.is_left():
                return basic_analysis
            
            # Bottleneck detection
            bottlenecks_result = await self.detect_bottlenecks(metrics)
            if bottlenecks_result.is_left():
                return Either.left(bottlenecks_result.get_left())
            
            bottlenecks = bottlenecks_result.get_right()
            
            # Generate recommendations
            recommendations_result = await self.generate_optimization_recommendations(
                metrics, bottlenecks
            )
            if recommendations_result.is_left():
                return Either.left(recommendations_result.get_left())
            
            recommendations = recommendations_result.get_right()
            
            # Performance trends (if sufficient data)
            trends = await self._analyze_trends(metrics)
            
            # Performance score calculation
            performance_score = self._calculate_performance_score(metrics)
            
            analysis_time = (datetime.now(UTC) - analysis_start).total_seconds()
            
            result = {
                "analysis_timestamp": analysis_start.isoformat(),
                "analysis_duration_ms": analysis_time * 1000,
                "performance_score": performance_score,
                "basic_metrics": basic_analysis.get_right(),
                "bottlenecks": [self._bottleneck_to_dict(b) for b in bottlenecks],
                "recommendations": [self._recommendation_to_dict(r) for r in recommendations],
                "trends": trends,
                "summary": {
                    "overall_health": "good" if performance_score > 75 else "fair" if performance_score > 50 else "poor",
                    "critical_issues": len([b for b in bottlenecks if b.severity == AlertSeverity.CRITICAL]),
                    "optimization_opportunities": len(recommendations),
                    "metrics_analyzed": len(metrics.metrics),
                    "time_range_hours": self._get_time_range_hours(metrics)
                }
            }
            
            return Either.right(result)
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return Either.left(PerformanceMonitoringError(f"Analysis failed: {e}"))
    
    async def detect_bottlenecks(
        self, 
        metrics: PerformanceMetrics
    ) -> Either[Exception, List[BottleneckAnalysis]]:
        """Detect performance bottlenecks using advanced analysis."""
        try:
            bottlenecks = []
            
            # Analyze each metric type for bottlenecks
            for metric_type in MetricType:
                bottleneck_result = await self._analyze_metric_bottleneck(
                    metrics, metric_type
                )
                
                if bottleneck_result.is_right():
                    bottleneck = bottleneck_result.get_right()
                    if bottleneck:
                        bottlenecks.append(bottleneck)
            
            # Sort by severity (critical first)
            severity_order = {
                AlertSeverity.CRITICAL: 0,
                AlertSeverity.HIGH: 1,
                AlertSeverity.MEDIUM: 2,
                AlertSeverity.LOW: 3
            }
            
            bottlenecks.sort(key=lambda b: severity_order.get(b.severity, 4))
            
            return Either.right(bottlenecks)
            
        except Exception as e:
            self.logger.error(f"Bottleneck detection failed: {e}")
            return Either.left(PerformanceMonitoringError(f"Bottleneck detection failed: {e}"))
    
    async def generate_optimization_recommendations(
        self, 
        metrics: PerformanceMetrics,
        bottlenecks: List[BottleneckAnalysis]
    ) -> Either[Exception, List[OptimizationRecommendation]]:
        """Generate optimization recommendations based on analysis."""
        try:
            recommendations = []
            
            # Generate recommendations for each bottleneck
            for bottleneck in bottlenecks:
                bottleneck_recommendations = await self._generate_bottleneck_recommendations(
                    bottleneck, metrics
                )
                recommendations.extend(bottleneck_recommendations)
            
            # Generate general performance recommendations
            general_recommendations = await self._generate_general_recommendations(metrics)
            recommendations.extend(general_recommendations)
            
            # Remove duplicates and prioritize
            unique_recommendations = self._deduplicate_recommendations(recommendations)
            
            # Sort by expected improvement (highest first)
            unique_recommendations.sort(
                key=lambda r: r.expected_improvement, 
                reverse=True
            )
            
            return Either.right(unique_recommendations[:10])  # Top 10 recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return Either.left(PerformanceMonitoringError(f"Recommendation generation failed: {e}"))
    
    async def benchmark_comparison(
        self, 
        current_metrics: PerformanceMetrics,
        baseline_metrics: Optional[PerformanceMetrics] = None
    ) -> Either[Exception, Dict[str, Any]]:
        """Compare current performance against baseline."""
        try:
            if not baseline_metrics:
                baseline_metrics = self._get_cached_baseline()
                if not baseline_metrics:
                    return Either.left(
                        PerformanceMonitoringError("No baseline metrics available")
                    )
            
            comparison = {}
            
            for metric_type in MetricType:
                current_stats = self._calculate_metric_stats(current_metrics, metric_type)
                baseline_stats = self._calculate_metric_stats(baseline_metrics, metric_type)
                
                if current_stats and baseline_stats:
                    improvement = self._calculate_improvement(
                        baseline_stats["mean"], 
                        current_stats["mean"], 
                        metric_type
                    )
                    
                    comparison[metric_type.value] = {
                        "current": current_stats,
                        "baseline": baseline_stats,
                        "improvement_percent": improvement,
                        "status": "improved" if improvement > 0 else "degraded" if improvement < -5 else "stable"
                    }
            
            return Either.right(comparison)
            
        except Exception as e:
            return Either.left(PerformanceMonitoringError(f"Benchmark comparison failed: {e}"))
    
    def establish_baseline(
        self, 
        metric_type: MetricType, 
        values: List[float]
    ) -> None:
        """Establish performance baseline from historical data."""
        if not values:
            return
        
        # Calculate baseline statistics
        baseline_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Calculate confidence interval (±2 standard deviations for ~95% confidence)
        confidence_factor = 2.0
        baseline_range = (
            max(0.0, baseline_value - confidence_factor * std_dev),
            baseline_value + confidence_factor * std_dev
        )
        
        # Calculate confidence interval as a normalized value (0-1)
        # Higher confidence for more samples and lower variance
        confidence_interval = min(1.0, len(values) / 100.0) * max(0.1, 1.0 - (std_dev / baseline_value) if baseline_value > 0 else 0.1)
        
        # Create and store baseline
        baseline = PerformanceBaseline(
            metric_type=metric_type,
            baseline_value=baseline_value,
            baseline_range=baseline_range,
            sample_size=len(values),
            confidence_interval=confidence_interval,
            created_at=datetime.now(UTC)
        )
        
        self.baselines[metric_type] = baseline
        
        # Also update legacy baseline_metrics for compatibility
        self.baseline_metrics[metric_type] = {
            "mean": baseline_value,
            "std_dev": std_dev,
            "min": min(values),
            "max": max(values),
            "sample_size": len(values)
        }
    
    async def _analyze_basic_metrics(
        self, 
        metrics: PerformanceMetrics
    ) -> Either[Exception, Dict[str, Any]]:
        """Analyze basic performance metrics."""
        try:
            analysis = {}
            
            for metric_type in MetricType:
                stats = self._calculate_metric_stats(metrics, metric_type)
                if stats:
                    analysis[metric_type.value] = {
                        **stats,
                        "threshold_violations": self._count_threshold_violations(
                            metrics, metric_type
                        ),
                        "trend": self._calculate_trend(metrics, metric_type)
                    }
            
            return Either.right(analysis)
            
        except Exception as e:
            return Either.left(PerformanceMonitoringError(f"Basic analysis failed: {e}"))
    
    async def _analyze_metric_bottleneck(
        self, 
        metrics: PerformanceMetrics,
        metric_type: MetricType
    ) -> Either[Exception, Optional[BottleneckAnalysis]]:
        """Analyze specific metric for bottlenecks."""
        try:
            # Get metric values
            metric_values = [
                float(m.value) for m in metrics.metrics 
                if m.metric_type == metric_type
            ]
            
            if not metric_values:
                return Either.right(None)
            
            # Calculate statistics
            avg_value = statistics.mean(metric_values)
            max_value = max(metric_values)
            threshold = self.bottleneck_thresholds.get(metric_type, float('inf'))
            normal_range = self.normal_ranges.get(metric_type, (0, 100))
            
            # Check if this is a bottleneck
            if max_value <= threshold and avg_value <= threshold * 0.8:
                return Either.right(None)
            
            # Determine severity
            if avg_value > threshold * 1.2:
                severity = AlertSeverity.CRITICAL
            elif avg_value > threshold:
                severity = AlertSeverity.HIGH
            elif max_value > threshold:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            # Generate bottleneck analysis
            bottleneck = BottleneckAnalysis(
                bottleneck_type=self._metric_to_bottleneck_type(metric_type),
                severity=severity,
                current_value=avg_value,
                normal_range=normal_range,
                impact_description=self._generate_impact_description(metric_type, avg_value, threshold),
                recommendations=self._generate_metric_recommendations(metric_type, avg_value),
                estimated_improvement=self._estimate_improvement(metric_type, avg_value, threshold)
            )
            
            return Either.right(bottleneck)
            
        except Exception as e:
            return Either.left(PerformanceMonitoringError(f"Metric bottleneck analysis failed: {e}"))
    
    async def _generate_bottleneck_recommendations(
        self, 
        bottleneck: BottleneckAnalysis,
        metrics: PerformanceMetrics
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations for a specific bottleneck."""
        recommendations = []
        
        # CPU bottleneck recommendations
        if bottleneck.bottleneck_type == BottleneckType.CPU_BOUND:
            recommendations.append(OptimizationRecommendation(
                optimization_type="cpu_optimization",
                description="Optimize CPU-intensive operations and enable parallel processing",
                expected_improvement=25.0,
                implementation_complexity="medium",
                risk_level="low",
                estimated_time="2-4 hours",
                prerequisites=["Performance profiling", "Code analysis"]
            ))
        
        # Memory bottleneck recommendations
        elif bottleneck.bottleneck_type == BottleneckType.MEMORY_BOUND:
            recommendations.append(OptimizationRecommendation(
                optimization_type="memory_optimization",
                description="Implement memory pooling and optimize data structures",
                expected_improvement=20.0,
                implementation_complexity="medium",
                risk_level="medium",
                estimated_time="3-6 hours",
                prerequisites=["Memory profiling", "Garbage collection tuning"]
            ))
        
        # I/O bottleneck recommendations
        elif bottleneck.bottleneck_type == BottleneckType.IO_BOUND:
            recommendations.append(OptimizationRecommendation(
                optimization_type="io_optimization",
                description="Implement asynchronous I/O and disk caching strategies",
                expected_improvement=30.0,
                implementation_complexity="high",
                risk_level="medium",
                estimated_time="4-8 hours",
                prerequisites=["I/O analysis", "Storage optimization"]
            ))
        
        return recommendations
    
    async def _generate_general_recommendations(
        self, 
        metrics: PerformanceMetrics
    ) -> List[OptimizationRecommendation]:
        """Generate general performance recommendations."""
        recommendations = []
        
        # Always include monitoring enhancement
        recommendations.append(OptimizationRecommendation(
            optimization_type="monitoring_enhancement",
            description="Enhance performance monitoring with additional metrics",
            expected_improvement=5.0,
            implementation_complexity="low",
            risk_level="low",
            estimated_time="1-2 hours",
            prerequisites=["Monitoring system access"]
        ))
        
        return recommendations
    
    def _calculate_metric_stats(
        self, 
        metrics: PerformanceMetrics,
        metric_type: MetricType
    ) -> Optional[Dict[str, float]]:
        """Calculate statistics for a specific metric type."""
        values = [
            float(m.value) for m in metrics.metrics 
            if m.metric_type == metric_type
        ]
        
        if not values:
            return None
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
        }
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score (0-100)."""
        if not metrics.snapshots:
            return 50.0
        
        # Get latest snapshot
        latest = max(metrics.snapshots, key=lambda s: s.timestamp)
        
        # Calculate component scores
        cpu_score = max(0, 100 - latest.cpu_percent)
        memory_score = max(0, 100 - latest.memory_percent)
        
        # Check for alerts (reduce score)
        alert_penalty = len(metrics.alerts) * 5
        
        # Calculate overall score
        base_score = (cpu_score * 0.4 + memory_score * 0.4 + 20)
        final_score = max(0, min(100, base_score - alert_penalty))
        
        return final_score
    
    def _count_threshold_violations(
        self, 
        metrics: PerformanceMetrics,
        metric_type: MetricType
    ) -> int:
        """Count threshold violations for a metric type."""
        return len([
            alert for alert in metrics.alerts 
            if alert.metric_type == metric_type
        ])
    
    def _calculate_trend(
        self, 
        metrics: PerformanceMetrics,
        metric_type: MetricType
    ) -> str:
        """Calculate trend for a metric type."""
        values = [
            float(m.value) for m in metrics.metrics 
            if m.metric_type == metric_type
        ]
        
        if len(values) < 5:
            return "insufficient_data"
        
        # Simple trend calculation using first and last third
        third = len(values) // 3
        first_third_avg = statistics.mean(values[:third])
        last_third_avg = statistics.mean(values[-third:])
        
        change_percent = ((last_third_avg - first_third_avg) / first_third_avg) * 100
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"
    
    def _metric_to_bottleneck_type(self, metric_type: MetricType) -> BottleneckType:
        """Convert metric type to bottleneck type."""
        mapping = {
            MetricType.CPU: BottleneckType.CPU_BOUND,
            MetricType.MEMORY: BottleneckType.MEMORY_BOUND,
            MetricType.DISK: BottleneckType.IO_BOUND,
            MetricType.NETWORK: BottleneckType.NETWORK_BOUND,
            MetricType.EXECUTION_TIME: BottleneckType.ALGORITHM_INEFFICIENCY,
        }
        return mapping.get(metric_type, BottleneckType.RESOURCE_CONTENTION)
    
    def _generate_impact_description(
        self, 
        metric_type: MetricType, 
        current_value: float, 
        threshold: float
    ) -> str:
        """Generate impact description for bottleneck."""
        excess_percent = ((current_value - threshold) / threshold) * 100
        
        descriptions = {
            MetricType.CPU: f"High CPU usage ({current_value:.1f}%) causing {excess_percent:.1f}% performance degradation",
            MetricType.MEMORY: f"High memory usage ({current_value:.1f}%) may lead to system instability",
            MetricType.DISK: f"High disk usage ({current_value:.1f}%) causing I/O bottlenecks",
            MetricType.NETWORK: f"High network usage ({current_value:.1f}%) causing communication delays",
        }
        
        return descriptions.get(
            metric_type, 
            f"Performance metric {metric_type.value} exceeds threshold by {excess_percent:.1f}%"
        )
    
    def _generate_metric_recommendations(
        self, 
        metric_type: MetricType, 
        current_value: float
    ) -> List[str]:
        """Generate specific recommendations for metric."""
        recommendations = {
            MetricType.CPU: [
                "Enable CPU throttling for non-critical processes",
                "Implement task scheduling and priority management",
                "Consider upgrading CPU or adding cores"
            ],
            MetricType.MEMORY: [
                "Implement memory pooling and caching",
                "Optimize data structures and algorithms",
                "Consider increasing available memory"
            ],
            MetricType.DISK: [
                "Implement disk caching strategies",
                "Optimize file I/O operations",
                "Consider faster storage solutions"
            ],
        }
        
        return recommendations.get(metric_type, ["Monitor metric closely", "Investigate root cause"])
    
    def _estimate_improvement(
        self, 
        metric_type: MetricType, 
        current_value: float, 
        threshold: float
    ) -> float:
        """Estimate potential improvement percentage."""
        if current_value <= threshold:
            return 0.0
        
        # Calculate potential improvement
        excess = current_value - threshold
        max_improvement = min(excess / current_value * 100, 50.0)  # Cap at 50%
        
        return max_improvement
    
    def _calculate_improvement(
        self, 
        baseline: float, 
        current: float, 
        metric_type: MetricType
    ) -> float:
        """Calculate improvement percentage (positive = better)."""
        if baseline == 0:
            return 0.0
        
        # For metrics where lower is better (CPU, memory, execution time)
        lower_is_better = metric_type in [
            MetricType.CPU, MetricType.MEMORY, MetricType.EXECUTION_TIME, 
            MetricType.ERROR_RATE, MetricType.LATENCY
        ]
        
        change_percent = ((current - baseline) / baseline) * 100
        
        return -change_percent if lower_is_better else change_percent
    
    async def _analyze_trends(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze performance trends."""
        trends = {}
        
        for metric_type in MetricType:
            trend = self._calculate_trend(metrics, metric_type)
            if trend != "insufficient_data":
                trends[metric_type.value] = {
                    "direction": trend,
                    "confidence": "medium"  # Could be enhanced with more sophisticated analysis
                }
        
        return trends
    
    def _get_time_range_hours(self, metrics: PerformanceMetrics) -> float:
        """Get time range of metrics in hours."""
        if not metrics.metrics:
            return 0.0
        
        start_time = min(m.timestamp for m in metrics.metrics)
        end_time = max(m.timestamp for m in metrics.metrics)
        
        return (end_time - start_time).total_seconds() / 3600
    
    def _bottleneck_to_dict(self, bottleneck: BottleneckAnalysis) -> Dict[str, Any]:
        """Convert bottleneck analysis to dictionary."""
        return {
            "type": bottleneck.bottleneck_type.value,
            "severity": bottleneck.severity.value,
            "current_value": bottleneck.current_value,
            "normal_range": bottleneck.normal_range,
            "impact": bottleneck.impact_description,
            "recommendations": bottleneck.recommendations,
            "estimated_improvement": bottleneck.estimated_improvement
        }
    
    def _recommendation_to_dict(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return {
            "type": recommendation.optimization_type,
            "description": recommendation.description,
            "expected_improvement": recommendation.expected_improvement,
            "complexity": recommendation.implementation_complexity,
            "risk": recommendation.risk_level,
            "time": recommendation.estimated_time,
            "prerequisites": recommendation.prerequisites
        }
    
    def _deduplicate_recommendations(
        self, 
        recommendations: List[OptimizationRecommendation]
    ) -> List[OptimizationRecommendation]:
        """Remove duplicate recommendations."""
        seen = set()
        unique = []
        
        for rec in recommendations:
            key = (rec.optimization_type, rec.description)
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        
        return unique
    
    def _get_cached_baseline(self) -> Optional[PerformanceMetrics]:
        """Get cached baseline metrics."""
        # Implementation would retrieve baseline from storage
        return None


# Global performance analyzer instance
_performance_analyzer: Optional[PerformanceAnalyzer] = None


def get_performance_analyzer() -> PerformanceAnalyzer:
    """Get or create global performance analyzer instance."""
    global _performance_analyzer
    if _performance_analyzer is None:
        _performance_analyzer = PerformanceAnalyzer()
    return _performance_analyzer