"""
Performance Optimizer for Automation Intelligence Analysis.

This module provides comprehensive performance analysis, optimization recommendations,
and efficiency tracking for automation workflows with intelligent insights and
actionable improvement suggestions based on behavioral pattern analysis.

Security: Secure performance analysis with privacy-protected data processing.
Performance: Optimized algorithms with intelligent caching and efficient analysis.
Type Safety: Complete branded type system with contract-driven validation.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import statistics
from datetime import datetime, timedelta, UTC

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger
from src.core.suggestion_system import UserBehaviorPattern, AutomationPerformanceMetrics
from src.core.errors import IntelligenceError

logger = get_logger(__name__)


class OptimizationTarget(Enum):
    """Performance optimization target metrics."""
    EFFICIENCY = "efficiency"               # Overall workflow efficiency
    SPEED = "speed"                        # Execution speed optimization
    ACCURACY = "accuracy"                  # Success rate improvement
    RESOURCE_USAGE = "resource_usage"      # Resource consumption optimization
    USER_SATISFACTION = "user_satisfaction" # User experience optimization
    ERROR_REDUCTION = "error_reduction"    # Error rate minimization


class PerformanceCategory(Enum):
    """Performance analysis categories."""
    EXECUTION_TIME = "execution_time"      # Time-based performance metrics
    SUCCESS_RATE = "success_rate"          # Success/failure rate analysis
    RESOURCE_CONSUMPTION = "resource_consumption"  # Resource usage analysis
    USER_EXPERIENCE = "user_experience"    # User interaction metrics
    SYSTEM_EFFICIENCY = "system_efficiency"  # Overall system performance


@dataclass(frozen=True)
class PerformanceInsight:
    """Comprehensive performance insight with actionable recommendations."""
    insight_id: str
    category: PerformanceCategory
    title: str
    description: str
    impact_level: str  # low|medium|high|critical
    confidence: float
    affected_patterns: List[str]
    metrics: Dict[str, float]
    recommendations: List[str]
    potential_improvement: float
    implementation_effort: str  # low|medium|high
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: self.potential_improvement >= 0.0)
    @require(lambda self: self.impact_level in ["low", "medium", "high", "critical"])
    @require(lambda self: self.implementation_effort in ["low", "medium", "high"])
    def __post_init__(self):
        pass


@dataclass(frozen=True)
class OptimizationRecommendation:
    """Detailed optimization recommendation with implementation guidance."""
    recommendation_id: str
    target: OptimizationTarget
    title: str
    description: str
    priority: str  # low|medium|high|critical
    confidence: float
    expected_improvement: float
    implementation_steps: List[str]
    affected_workflows: List[str]
    success_criteria: List[str]
    estimated_timeline: str
    resource_requirements: List[str]
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: self.expected_improvement >= 0.0)
    @require(lambda self: self.priority in ["low", "medium", "high", "critical"])
    def __post_init__(self):
        pass


class PerformanceOptimizer:
    """Comprehensive performance optimization with intelligent analysis."""
    
    def __init__(self):
        self.optimization_algorithms: Dict[OptimizationTarget, Any] = {}
        self.performance_cache: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Performance analysis configuration
        self.min_patterns_for_analysis = 3
        self.confidence_threshold = 0.6
        self.improvement_threshold = 0.1  # 10% improvement minimum
        
        # Performance thresholds
        self.performance_thresholds = {
            'execution_time_slow': 30.0,      # Seconds
            'success_rate_low': 0.8,           # 80%
            'efficiency_poor': 0.6,            # 60%
            'error_rate_high': 0.15            # 15%
        }
        
        self._initialize_optimization_algorithms()
    
    async def initialize(self) -> Either[IntelligenceError, None]:
        """Initialize performance optimizer with analysis algorithms."""
        try:
            # Configure optimization algorithms
            self._configure_performance_analysis()
            
            logger.info("Performance optimizer initialized successfully")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Performance optimizer initialization failed: {str(e)}")
            return Either.left(IntelligenceError.initialization_failed(str(e)))
    
    @require(lambda self, patterns: len(patterns) >= 1)
    async def analyze_performance(
        self,
        patterns: List[UserBehaviorPattern],
        target: OptimizationTarget = OptimizationTarget.EFFICIENCY
    ) -> Either[IntelligenceError, List[PerformanceInsight]]:
        """
        Analyze performance patterns and generate comprehensive insights.
        
        Performs sophisticated performance analysis on behavioral patterns to
        identify optimization opportunities, performance bottlenecks, and
        efficiency improvements with actionable recommendations.
        
        Args:
            patterns: Behavioral patterns for performance analysis
            target: Optimization target focus for analysis
            
        Returns:
            Either error or list of performance insights with recommendations
            
        Security:
            - Privacy-protected performance analysis
            - Secure pattern processing with no sensitive data exposure
            - Validated insights with confidence scoring
        """
        try:
            if len(patterns) < self.min_patterns_for_analysis:
                return Either.left(IntelligenceError.optimization_failed(
                    f"Insufficient patterns for analysis: {len(patterns)} < {self.min_patterns_for_analysis}"
                ))
            
            # Check cache for recent analysis
            cache_key = self._generate_performance_cache_key(patterns, target)
            if cache_key in self.performance_cache:
                cached_insights = self.performance_cache[cache_key]['insights']
                logger.debug(f"Retrieved {len(cached_insights)} performance insights from cache")
                return Either.right(cached_insights)
            
            # Perform comprehensive performance analysis
            insights = []
            
            # Execution time analysis
            execution_insights = await self._analyze_execution_performance(patterns, target)
            insights.extend(execution_insights)
            
            # Success rate analysis
            success_insights = await self._analyze_success_rate_performance(patterns, target)
            insights.extend(success_insights)
            
            # Efficiency analysis
            efficiency_insights = await self._analyze_efficiency_performance(patterns, target)
            insights.extend(efficiency_insights)
            
            # Resource usage analysis
            resource_insights = await self._analyze_resource_performance(patterns, target)
            insights.extend(resource_insights)
            
            # Filter insights by confidence and impact
            qualified_insights = [
                insight for insight in insights
                if insight.confidence >= self.confidence_threshold and
                   insight.potential_improvement >= self.improvement_threshold
            ]
            
            # Sort by impact and confidence
            sorted_insights = sorted(
                qualified_insights,
                key=lambda i: (self._get_impact_score(i.impact_level), i.confidence),
                reverse=True
            )
            
            # Cache results
            self.performance_cache[cache_key] = {
                'insights': sorted_insights,
                'timestamp': datetime.now(UTC),
                'target': target.value
            }
            
            # Track analysis
            self._track_performance_analysis(patterns, sorted_insights, target)
            
            logger.info(f"Generated {len(sorted_insights)} performance insights for {len(patterns)} patterns")
            return Either.right(sorted_insights)
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            return Either.left(IntelligenceError.optimization_failed(str(e)))
    
    @require(lambda self, insights: len(insights) >= 1)
    async def generate_optimization_recommendations(
        self,
        insights: List[PerformanceInsight],
        target: OptimizationTarget = OptimizationTarget.EFFICIENCY
    ) -> Either[IntelligenceError, List[OptimizationRecommendation]]:
        """
        Generate detailed optimization recommendations from performance insights.
        
        Converts performance insights into actionable optimization recommendations
        with implementation guidance, priority scoring, and success criteria
        for systematic performance improvement.
        
        Args:
            insights: Performance insights for recommendation generation
            target: Optimization target for focused recommendations
            
        Returns:
            Either error or list of optimization recommendations
        """
        try:
            recommendations = []
            
            # Group insights by category for comprehensive recommendations
            categorized_insights = self._categorize_insights_by_performance_area(insights)
            
            # Generate recommendations for each category
            for category, category_insights in categorized_insights.items():
                category_recommendations = await self._generate_category_recommendations(
                    category_insights, category, target
                )
                recommendations.extend(category_recommendations)
            
            # Generate cross-cutting optimization recommendations
            cross_cutting_recommendations = await self._generate_cross_cutting_recommendations(insights, target)
            recommendations.extend(cross_cutting_recommendations)
            
            # Prioritize recommendations
            prioritized_recommendations = self._prioritize_recommendations(recommendations)
            
            logger.info(f"Generated {len(prioritized_recommendations)} optimization recommendations")
            return Either.right(prioritized_recommendations)
            
        except Exception as e:
            logger.error(f"Optimization recommendation generation failed: {str(e)}")
            return Either.left(IntelligenceError.optimization_failed(str(e)))
    
    async def _analyze_execution_performance(
        self,
        patterns: List[UserBehaviorPattern],
        target: OptimizationTarget
    ) -> List[PerformanceInsight]:
        """Analyze execution time performance patterns."""
        insights = []
        
        # Calculate execution time statistics
        execution_times = [p.average_completion_time for p in patterns]
        avg_execution_time = statistics.mean(execution_times)
        
        # Identify slow patterns
        slow_patterns = [
            p for p in patterns
            if p.average_completion_time > self.performance_thresholds['execution_time_slow']
        ]
        
        if slow_patterns:
            total_slow_time = sum(p.average_completion_time * p.frequency for p in slow_patterns)
            potential_savings = total_slow_time * 0.3  # Assume 30% improvement possible
            
            insight = PerformanceInsight(
                insight_id=f"execution_time_{datetime.now(UTC).timestamp()}",
                category=PerformanceCategory.EXECUTION_TIME,
                title=f"Slow Execution Performance Detected",
                description=f"Found {len(slow_patterns)} patterns with execution times > {self.performance_thresholds['execution_time_slow']}s",
                impact_level=self._calculate_impact_level(len(slow_patterns), len(patterns)),
                confidence=0.8,
                affected_patterns=[p.pattern_id for p in slow_patterns],
                metrics={
                    'average_execution_time': avg_execution_time,
                    'slow_patterns_count': len(slow_patterns),
                    'potential_time_savings': potential_savings
                },
                recommendations=[
                    "Optimize slow automation workflows",
                    "Consider parallel processing for independent tasks",
                    "Review and streamline complex action sequences"
                ],
                potential_improvement=potential_savings / max(1, total_slow_time),
                implementation_effort="medium"
            )
            insights.append(insight)
        
        # Analyze execution time consistency
        if len(execution_times) > 1:
            time_variance = statistics.stdev(execution_times)
            consistency_score = 1.0 - min(1.0, time_variance / avg_execution_time)
            
            if consistency_score < 0.7:  # Poor consistency
                insight = PerformanceInsight(
                    insight_id=f"execution_consistency_{datetime.now(UTC).timestamp()}",
                    category=PerformanceCategory.EXECUTION_TIME,
                    title="Inconsistent Execution Times",
                    description=f"Execution times vary significantly (consistency: {consistency_score:.1%})",
                    impact_level="medium",
                    confidence=0.7,
                    affected_patterns=[p.pattern_id for p in patterns],
                    metrics={
                        'consistency_score': consistency_score,
                        'time_variance': time_variance,
                        'average_time': avg_execution_time
                    },
                    recommendations=[
                        "Investigate causes of timing variability",
                        "Standardize workflow execution environments",
                        "Add performance monitoring to workflows"
                    ],
                    potential_improvement=0.2,
                    implementation_effort="medium"
                )
                insights.append(insight)
        
        return insights
    
    async def _analyze_success_rate_performance(
        self,
        patterns: List[UserBehaviorPattern],
        target: OptimizationTarget
    ) -> List[PerformanceInsight]:
        """Analyze success rate performance patterns."""
        insights = []
        
        # Calculate success rate statistics
        success_rates = [p.success_rate for p in patterns]
        avg_success_rate = statistics.mean(success_rates)
        
        # Identify low success rate patterns
        low_success_patterns = [
            p for p in patterns
            if p.success_rate < self.performance_thresholds['success_rate_low']
        ]
        
        if low_success_patterns:
            total_failures = sum((1 - p.success_rate) * p.frequency for p in low_success_patterns)
            
            insight = PerformanceInsight(
                insight_id=f"success_rate_{datetime.now(UTC).timestamp()}",
                category=PerformanceCategory.SUCCESS_RATE,
                title="Low Success Rate Patterns Detected",
                description=f"Found {len(low_success_patterns)} patterns with success rates < {self.performance_thresholds['success_rate_low']:.0%}",
                impact_level=self._calculate_impact_level(len(low_success_patterns), len(patterns)),
                confidence=0.85,
                affected_patterns=[p.pattern_id for p in low_success_patterns],
                metrics={
                    'average_success_rate': avg_success_rate,
                    'low_success_patterns_count': len(low_success_patterns),
                    'total_failure_instances': total_failures
                },
                recommendations=[
                    "Add error handling and validation to workflows",
                    "Review and fix common failure points",
                    "Implement retry mechanisms for transient failures"
                ],
                potential_improvement=0.15,  # 15% success rate improvement
                implementation_effort="medium"
            )
            insights.append(insight)
        
        return insights
    
    async def _analyze_efficiency_performance(
        self,
        patterns: List[UserBehaviorPattern],
        target: OptimizationTarget
    ) -> List[PerformanceInsight]:
        """Analyze workflow efficiency patterns."""
        insights = []
        
        # Calculate efficiency scores
        efficiency_scores = [p.get_efficiency_score() for p in patterns]
        avg_efficiency = statistics.mean(efficiency_scores)
        
        # Identify low efficiency patterns
        low_efficiency_patterns = [
            p for p in patterns
            if p.get_efficiency_score() < self.performance_thresholds['efficiency_poor']
        ]
        
        if low_efficiency_patterns:
            insight = PerformanceInsight(
                insight_id=f"efficiency_{datetime.now(UTC).timestamp()}",
                category=PerformanceCategory.SYSTEM_EFFICIENCY,
                title="Low Efficiency Workflows Identified",
                description=f"Found {len(low_efficiency_patterns)} workflows with efficiency < {self.performance_thresholds['efficiency_poor']:.0%}",
                impact_level=self._calculate_impact_level(len(low_efficiency_patterns), len(patterns)),
                confidence=0.75,
                affected_patterns=[p.pattern_id for p in low_efficiency_patterns],
                metrics={
                    'average_efficiency': avg_efficiency,
                    'low_efficiency_count': len(low_efficiency_patterns),
                    'efficiency_gap': self.performance_thresholds['efficiency_poor'] - avg_efficiency
                },
                recommendations=[
                    "Streamline workflow steps and remove redundancies",
                    "Optimize tool usage and action sequences",
                    "Consider automation for repetitive manual steps"
                ],
                potential_improvement=0.25,
                implementation_effort="high"
            )
            insights.append(insight)
        
        return insights
    
    async def _analyze_resource_performance(
        self,
        patterns: List[UserBehaviorPattern],
        target: OptimizationTarget
    ) -> List[PerformanceInsight]:
        """Analyze resource usage performance patterns."""
        insights = []
        
        # Analyze frequency and resource consumption
        high_frequency_patterns = [p for p in patterns if p.frequency >= 10]
        
        if high_frequency_patterns:
            total_resource_consumption = sum(
                p.frequency * p.average_completion_time for p in high_frequency_patterns
            )
            
            insight = PerformanceInsight(
                insight_id=f"resource_usage_{datetime.now(UTC).timestamp()}",
                category=PerformanceCategory.RESOURCE_CONSUMPTION,
                title="High Resource Consumption Patterns",
                description=f"Found {len(high_frequency_patterns)} high-frequency patterns consuming significant resources",
                impact_level="medium",
                confidence=0.7,
                affected_patterns=[p.pattern_id for p in high_frequency_patterns],
                metrics={
                    'high_frequency_patterns': len(high_frequency_patterns),
                    'total_resource_consumption': total_resource_consumption,
                    'average_frequency': statistics.mean([p.frequency for p in high_frequency_patterns])
                },
                recommendations=[
                    "Optimize high-frequency workflows for resource efficiency",
                    "Consider caching for repeated operations",
                    "Implement batch processing where applicable"
                ],
                potential_improvement=0.2,
                implementation_effort="medium"
            )
            insights.append(insight)
        
        return insights
    
    def _calculate_impact_level(self, affected_count: int, total_count: int) -> str:
        """Calculate impact level based on affected patterns ratio."""
        ratio = affected_count / max(1, total_count)
        
        if ratio >= 0.5:
            return "critical"
        elif ratio >= 0.3:
            return "high"
        elif ratio >= 0.1:
            return "medium"
        else:
            return "low"
    
    def _get_impact_score(self, impact_level: str) -> float:
        """Convert impact level to numerical score for sorting."""
        impact_scores = {
            "critical": 4.0,
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0
        }
        return impact_scores.get(impact_level, 1.0)
    
    def _categorize_insights_by_performance_area(
        self,
        insights: List[PerformanceInsight]
    ) -> Dict[PerformanceCategory, List[PerformanceInsight]]:
        """Categorize insights by performance area."""
        categorized = {}
        
        for insight in insights:
            if insight.category not in categorized:
                categorized[insight.category] = []
            categorized[insight.category].append(insight)
        
        return categorized
    
    async def _generate_category_recommendations(
        self,
        insights: List[PerformanceInsight],
        category: PerformanceCategory,
        target: OptimizationTarget
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations for specific performance category."""
        recommendations = []
        
        if category == PerformanceCategory.EXECUTION_TIME:
            recommendation = OptimizationRecommendation(
                recommendation_id=f"optimize_execution_{datetime.now(UTC).timestamp()}",
                target=target,
                title="Optimize Execution Time Performance",
                description="Systematic approach to reducing workflow execution times",
                priority="high",
                confidence=0.8,
                expected_improvement=0.25,
                implementation_steps=[
                    "Profile slow workflows to identify bottlenecks",
                    "Optimize critical path operations",
                    "Implement parallel processing where possible",
                    "Cache frequently accessed data"
                ],
                affected_workflows=[insight.insight_id for insight in insights],
                success_criteria=[
                    "Average execution time reduced by 20%+",
                    "No workflows exceeding 30-second threshold",
                    "Improved execution time consistency"
                ],
                estimated_timeline="2-4 weeks",
                resource_requirements=["Development time", "Performance testing", "Monitoring tools"]
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_cross_cutting_recommendations(
        self,
        insights: List[PerformanceInsight],
        target: OptimizationTarget
    ) -> List[OptimizationRecommendation]:
        """Generate cross-cutting optimization recommendations."""
        recommendations = []
        
        # General performance monitoring recommendation
        recommendation = OptimizationRecommendation(
            recommendation_id=f"performance_monitoring_{datetime.now(UTC).timestamp()}",
            target=target,
            title="Implement Comprehensive Performance Monitoring",
            description="Establish continuous performance monitoring and alerting",
            priority="medium",
            confidence=0.9,
            expected_improvement=0.15,
            implementation_steps=[
                "Set up performance metrics collection",
                "Create performance dashboards",
                "Implement automated alerting for degradation",
                "Establish performance baseline measurements"
            ],
            affected_workflows=["all_workflows"],
            success_criteria=[
                "Real-time performance visibility",
                "Automated performance alerts",
                "Historical performance trends",
                "Performance regression detection"
            ],
            estimated_timeline="1-2 weeks",
            resource_requirements=["Monitoring infrastructure", "Dashboard tools", "Alert configuration"]
        )
        recommendations.append(recommendation)
        
        return recommendations
    
    def _prioritize_recommendations(
        self,
        recommendations: List[OptimizationRecommendation]
    ) -> List[OptimizationRecommendation]:
        """Prioritize recommendations by impact and feasibility."""
        priority_scores = {
            "critical": 4.0,
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0
        }
        
        def recommendation_score(rec: OptimizationRecommendation) -> float:
            priority_score = priority_scores.get(rec.priority, 1.0)
            return priority_score * rec.confidence * rec.expected_improvement
        
        return sorted(recommendations, key=recommendation_score, reverse=True)
    
    def _generate_performance_cache_key(
        self,
        patterns: List[UserBehaviorPattern],
        target: OptimizationTarget
    ) -> str:
        """Generate cache key for performance analysis."""
        pattern_ids = sorted([p.pattern_id for p in patterns])
        return f"{target.value}_{hash(tuple(pattern_ids[:10]))}"
    
    def _track_performance_analysis(
        self,
        patterns: List[UserBehaviorPattern],
        insights: List[PerformanceInsight],
        target: OptimizationTarget
    ) -> None:
        """Track performance analysis for historical trending."""
        analysis_record = {
            'timestamp': datetime.now(UTC),
            'patterns_analyzed': len(patterns),
            'insights_generated': len(insights),
            'target': target.value,
            'high_impact_insights': len([i for i in insights if i.impact_level in ["high", "critical"]]),
            'average_confidence': statistics.mean([i.confidence for i in insights]) if insights else 0.0
        }
        
        self.optimization_history.append(analysis_record)
        
        # Limit history size
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-50:]
    
    def _initialize_optimization_algorithms(self) -> None:
        """Initialize optimization algorithms for different targets."""
        self.optimization_algorithms = {
            OptimizationTarget.EFFICIENCY: self._optimize_for_efficiency,
            OptimizationTarget.SPEED: self._optimize_for_speed,
            OptimizationTarget.ACCURACY: self._optimize_for_accuracy,
            OptimizationTarget.RESOURCE_USAGE: self._optimize_for_resource_usage,
            OptimizationTarget.USER_SATISFACTION: self._optimize_for_user_satisfaction,
            OptimizationTarget.ERROR_REDUCTION: self._optimize_for_error_reduction
        }
    
    def _configure_performance_analysis(self) -> None:
        """Configure performance analysis parameters."""
        # Configuration for different analysis modes
        pass
    
    # Placeholder optimization algorithms
    def _optimize_for_efficiency(self, patterns: List[UserBehaviorPattern]) -> Dict[str, Any]:
        """Optimize patterns for efficiency."""
        return {}
    
    def _optimize_for_speed(self, patterns: List[UserBehaviorPattern]) -> Dict[str, Any]:
        """Optimize patterns for speed."""
        return {}
    
    def _optimize_for_accuracy(self, patterns: List[UserBehaviorPattern]) -> Dict[str, Any]:
        """Optimize patterns for accuracy."""
        return {}
    
    def _optimize_for_resource_usage(self, patterns: List[UserBehaviorPattern]) -> Dict[str, Any]:
        """Optimize patterns for resource usage."""
        return {}
    
    def _optimize_for_user_satisfaction(self, patterns: List[UserBehaviorPattern]) -> Dict[str, Any]:
        """Optimize patterns for user satisfaction."""
        return {}
    
    def _optimize_for_error_reduction(self, patterns: List[UserBehaviorPattern]) -> Dict[str, Any]:
        """Optimize patterns for error reduction."""
        return {}