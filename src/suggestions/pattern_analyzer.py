"""
Pattern recognition and analysis system for intelligent automation insights.

This module implements advanced pattern recognition algorithms to identify
optimization opportunities, workflow inefficiencies, and automation improvement
suggestions based on user behavior and performance data.

Security: All analysis includes privacy protection and secure pattern detection.
Performance: Optimized algorithms for real-time pattern analysis.
Type Safety: Complete integration with behavior tracking and suggestion systems.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta, UTC
from collections import defaultdict, Counter
import statistics
import math

from src.core.suggestion_system import (
    UserBehaviorPattern, AutomationPerformanceMetrics, SuggestionType,
    PriorityLevel, SuggestionError
)
from src.suggestions.behavior_tracker import BehaviorTracker
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger

logger = get_logger(__name__)


class PatternInsight:
    """Represents an insight discovered through pattern analysis."""
    
    def __init__(self, insight_type: str, message: str, confidence: float = 1.0,
                 priority: PriorityLevel = PriorityLevel.MEDIUM, 
                 patterns: Optional[List[str]] = None,
                 automations: Optional[List[str]] = None,
                 metrics: Optional[Dict[str, Any]] = None):
        self.insight_type = insight_type
        self.message = message
        self.confidence = confidence
        self.priority = priority
        self.patterns = patterns or []
        self.automations = automations or []
        self.metrics = metrics or {}
        self.timestamp = datetime.now(UTC)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary format."""
        return {
            'type': self.insight_type,
            'message': self.message,
            'confidence': self.confidence,
            'priority': self.priority.value,
            'patterns': self.patterns,
            'automations': self.automations,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat()
        }


class OptimizationOpportunity:
    """Represents an optimization opportunity discovered through analysis."""
    
    def __init__(self, opportunity_type: str, description: str, 
                 potential_impact: str, implementation_effort: str,
                 priority: PriorityLevel = PriorityLevel.MEDIUM,
                 affected_items: Optional[List[str]] = None,
                 estimated_savings: Optional[Dict[str, Any]] = None):
        self.opportunity_type = opportunity_type
        self.description = description
        self.potential_impact = potential_impact
        self.implementation_effort = implementation_effort
        self.priority = priority
        self.affected_items = affected_items or []
        self.estimated_savings = estimated_savings or {}
        self.confidence = 0.8  # Default confidence
        self.timestamp = datetime.now(UTC)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert opportunity to dictionary format."""
        return {
            'type': self.opportunity_type,
            'description': self.description,
            'potential_impact': self.potential_impact,
            'implementation_effort': self.implementation_effort,
            'priority': self.priority.value,
            'confidence': self.confidence,
            'affected_items': self.affected_items,
            'estimated_savings': self.estimated_savings,
            'timestamp': self.timestamp.isoformat()
        }


class PatternAnalyzer:
    """Advanced pattern recognition and analysis system for automation optimization."""
    
    def __init__(self, behavior_tracker: BehaviorTracker):
        self.behavior_tracker = behavior_tracker
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=30)  # Cache results for 30 minutes
    
    @require(lambda self, user_id: len(user_id) > 0)
    async def analyze_user_patterns(self, user_id: str, depth: str = "standard") -> List[PatternInsight]:
        """
        Analyze user behavior patterns to generate actionable insights.
        
        Args:
            user_id: User identifier for pattern analysis
            depth: Analysis depth (quick, standard, deep, comprehensive)
            
        Returns:
            List of pattern insights with optimization recommendations
        """
        try:
            # Check cache first
            cache_key = f"patterns_{user_id}_{depth}"
            if self._is_cache_valid(cache_key):
                cached_results = self.analysis_cache[cache_key]['results']
                return [self._dict_to_insight(insight) for insight in cached_results]
            
            patterns = self.behavior_tracker.get_user_patterns(user_id, recent_only=True)
            insights = []
            
            if not patterns:
                insights.append(PatternInsight(
                    insight_type="no_data",
                    message="No recent behavior patterns found for analysis",
                    confidence=1.0,
                    priority=PriorityLevel.LOW
                ))
                return insights
            
            # Analyze efficiency patterns
            efficiency_insights = await self._analyze_efficiency_patterns(patterns)
            insights.extend(efficiency_insights)
            
            # Analyze reliability patterns
            reliability_insights = await self._analyze_reliability_patterns(patterns)
            insights.extend(reliability_insights)
            
            # Analyze frequency patterns
            frequency_insights = await self._analyze_frequency_patterns(patterns)
            insights.extend(frequency_insights)
            
            # Advanced analysis for deeper modes
            if depth in ["deep", "comprehensive"]:
                correlation_insights = await self._analyze_pattern_correlations(patterns)
                insights.extend(correlation_insights)
                
                temporal_insights = await self._analyze_temporal_patterns(user_id, patterns)
                insights.extend(temporal_insights)
            
            # Comprehensive analysis includes cross-user comparisons
            if depth == "comprehensive":
                comparative_insights = await self._analyze_comparative_patterns(user_id, patterns)
                insights.extend(comparative_insights)
            
            # Cache results
            self._cache_results(cache_key, [insight.to_dict() for insight in insights])
            
            logger.info(f"Generated {len(insights)} pattern insights for user {user_id}")
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing user patterns: {str(e)}")
            return [PatternInsight(
                insight_type="analysis_error",
                message=f"Pattern analysis failed: {str(e)}",
                confidence=0.0,
                priority=PriorityLevel.LOW
            )]
    
    @require(lambda self, user_id: len(user_id) > 0)
    async def identify_optimization_opportunities(self, user_id: str) -> List[OptimizationOpportunity]:
        """
        Identify specific optimization opportunities based on performance data.
        
        Args:
            user_id: User identifier for optimization analysis
            
        Returns:
            List of optimization opportunities with implementation details
        """
        try:
            opportunities = []
            
            # Get performance data
            performance_data = self.behavior_tracker.get_automation_metrics()
            user_patterns = self.behavior_tracker.get_user_patterns(user_id)
            
            # Analyze slow automations
            slow_opportunities = await self._identify_performance_opportunities(performance_data)
            opportunities.extend(slow_opportunities)
            
            # Analyze unreliable automations
            reliability_opportunities = await self._identify_reliability_opportunities(performance_data)
            opportunities.extend(reliability_opportunities)
            
            # Analyze workflow duplication
            duplication_opportunities = await self._identify_duplication_opportunities(user_patterns)
            opportunities.extend(duplication_opportunities)
            
            # Analyze tool usage optimization
            tool_opportunities = await self._identify_tool_optimization_opportunities(user_id)
            opportunities.extend(tool_opportunities)
            
            # Sort by priority and potential impact
            opportunities.sort(key=lambda o: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[o.priority.value],
                o.confidence
            ), reverse=True)
            
            logger.info(f"Identified {len(opportunities)} optimization opportunities for user {user_id}")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {str(e)}")
            return []
    
    async def analyze_automation_trends(self, time_window_days: int = 30) -> Dict[str, Any]:
        """
        Analyze automation trends across all users and automations.
        
        Args:
            time_window_days: Number of days to analyze
            
        Returns:
            Dictionary containing trend analysis and insights
        """
        try:
            performance_data = self.behavior_tracker.get_automation_metrics()
            cutoff_date = datetime.now(UTC) - timedelta(days=time_window_days)
            
            # Filter recent data
            recent_metrics = {
                aid: metrics for aid, metrics in performance_data.items()
                if metrics.last_execution >= cutoff_date
            }
            
            if not recent_metrics:
                return {'error': 'No recent automation data available'}
            
            # Calculate overall trends
            total_executions = sum(m.execution_count for m in recent_metrics.values())
            average_success_rate = statistics.mean(m.success_rate for m in recent_metrics.values())
            average_execution_time = statistics.mean(m.average_execution_time for m in recent_metrics.values())
            average_error_rate = statistics.mean(m.error_frequency for m in recent_metrics.values())
            
            # Identify top performers and problem areas
            top_performers = [
                (aid, metrics) for aid, metrics in recent_metrics.items()
                if metrics.get_performance_score() > 0.8
            ]
            
            problem_automations = [
                (aid, metrics) for aid, metrics in recent_metrics.items()
                if metrics.needs_optimization()
            ]
            
            # Analyze trends by category
            trend_analysis = {
                'analysis_period_days': time_window_days,
                'total_automations': len(recent_metrics),
                'total_executions': total_executions,
                'average_success_rate': round(average_success_rate, 3),
                'average_execution_time': round(average_execution_time, 2),
                'average_error_rate': round(average_error_rate, 3),
                'top_performers_count': len(top_performers),
                'problem_automations_count': len(problem_automations),
                'performance_distribution': self._calculate_performance_distribution(recent_metrics),
                'improvement_trends': self._calculate_improvement_trends(recent_metrics),
                'bottlenecks': self._identify_system_bottlenecks(recent_metrics)
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing automation trends: {str(e)}")
            return {'error': str(e)}
    
    async def _analyze_efficiency_patterns(self, patterns: List[UserBehaviorPattern]) -> List[PatternInsight]:
        """Analyze patterns for efficiency insights."""
        insights = []
        
        # Find highly efficient patterns
        efficient_patterns = [p for p in patterns if p.get_efficiency_score() > 0.8]
        if efficient_patterns:
            insights.append(PatternInsight(
                insight_type="high_efficiency",
                message=f"Found {len(efficient_patterns)} highly efficient automation patterns. "
                       f"These represent your most optimized workflows.",
                confidence=0.9,
                priority=PriorityLevel.LOW,
                patterns=[p.pattern_id for p in efficient_patterns[:3]],
                metrics={
                    'average_efficiency': statistics.mean(p.get_efficiency_score() for p in efficient_patterns),
                    'total_patterns': len(efficient_patterns)
                }
            ))
        
        # Find inefficient patterns that could be optimized
        inefficient_patterns = [p for p in patterns if p.get_efficiency_score() < 0.5 and p.frequency > 3]
        if inefficient_patterns:
            avg_time = statistics.mean(p.average_completion_time for p in inefficient_patterns)
            insights.append(PatternInsight(
                insight_type="low_efficiency",
                message=f"Found {len(inefficient_patterns)} automation patterns with low efficiency. "
                       f"Average completion time is {avg_time:.1f} seconds. Consider optimization.",
                confidence=0.8,
                priority=PriorityLevel.MEDIUM,
                patterns=[p.pattern_id for p in inefficient_patterns[:3]],
                metrics={
                    'average_efficiency': statistics.mean(p.get_efficiency_score() for p in inefficient_patterns),
                    'average_completion_time': avg_time,
                    'total_patterns': len(inefficient_patterns)
                }
            ))
        
        return insights
    
    async def _analyze_reliability_patterns(self, patterns: List[UserBehaviorPattern]) -> List[PatternInsight]:
        """Analyze patterns for reliability insights."""
        insights = []
        
        # Find patterns with low success rates
        unreliable_patterns = [p for p in patterns if p.success_rate < 0.7 and p.frequency > 2]
        if unreliable_patterns:
            avg_success = statistics.mean(p.success_rate for p in unreliable_patterns)
            insights.append(PatternInsight(
                insight_type="low_reliability",
                message=f"Found {len(unreliable_patterns)} automation patterns with success rates below 70%. "
                       f"Average success rate is {avg_success:.1%}. Review error handling and conditions.",
                confidence=0.9,
                priority=PriorityLevel.HIGH,
                patterns=[p.pattern_id for p in unreliable_patterns[:3]],
                metrics={
                    'average_success_rate': avg_success,
                    'total_patterns': len(unreliable_patterns),
                    'lowest_success_rate': min(p.success_rate for p in unreliable_patterns)
                }
            ))
        
        # Find highly reliable patterns to use as examples
        reliable_patterns = [p for p in patterns if p.success_rate > 0.95 and p.frequency > 5]
        if reliable_patterns:
            insights.append(PatternInsight(
                insight_type="high_reliability",
                message=f"Found {len(reliable_patterns)} highly reliable automation patterns. "
                       f"These can serve as templates for improving other workflows.",
                confidence=0.8,
                priority=PriorityLevel.LOW,
                patterns=[p.pattern_id for p in reliable_patterns[:3]],
                metrics={
                    'average_success_rate': statistics.mean(p.success_rate for p in reliable_patterns),
                    'total_patterns': len(reliable_patterns)
                }
            ))
        
        return insights
    
    async def _analyze_frequency_patterns(self, patterns: List[UserBehaviorPattern]) -> List[PatternInsight]:
        """Analyze patterns for frequency-based insights."""
        insights = []
        
        # Find most frequent patterns
        frequent_patterns = sorted(patterns, key=lambda p: p.frequency, reverse=True)[:5]
        if frequent_patterns and frequent_patterns[0].frequency > 10:
            total_usage = sum(p.frequency for p in frequent_patterns)
            insights.append(PatternInsight(
                insight_type="high_frequency",
                message=f"Your top 5 automation patterns account for {total_usage} executions. "
                       f"Optimizing these will have the biggest impact.",
                confidence=0.9,
                priority=PriorityLevel.MEDIUM,
                patterns=[p.pattern_id for p in frequent_patterns],
                metrics={
                    'total_executions': total_usage,
                    'top_pattern_frequency': frequent_patterns[0].frequency,
                    'patterns_analyzed': len(frequent_patterns)
                }
            ))
        
        # Find underutilized patterns that might indicate setup issues
        underutilized = [p for p in patterns if p.frequency == 1 and 
                        (datetime.now(UTC) - p.last_observed).days < 7]
        if len(underutilized) > 5:
            insights.append(PatternInsight(
                insight_type="underutilized",
                message=f"Found {len(underutilized)} automation patterns used only once recently. "
                       f"Consider whether these are set up correctly or need adjustment.",
                confidence=0.7,
                priority=PriorityLevel.LOW,
                patterns=[p.pattern_id for p in underutilized[:3]],
                metrics={
                    'total_underutilized': len(underutilized),
                    'recent_days': 7
                }
            ))
        
        return insights
    
    async def _analyze_pattern_correlations(self, patterns: List[UserBehaviorPattern]) -> List[PatternInsight]:
        """Analyze correlations between patterns for advanced insights."""
        insights = []
        
        try:
            # Group patterns by context tags to find correlations
            tag_groups = defaultdict(list)
            for pattern in patterns:
                for tag in pattern.context_tags:
                    tag_groups[tag].append(pattern)
            
            # Find tags with multiple high-performing patterns
            for tag, tag_patterns in tag_groups.items():
                if len(tag_patterns) >= 3:
                    avg_efficiency = statistics.mean(p.get_efficiency_score() for p in tag_patterns)
                    if avg_efficiency > 0.8:
                        insights.append(PatternInsight(
                            insight_type="context_correlation",
                            message=f"Automations with context '{tag}' show consistently high performance "
                                   f"({avg_efficiency:.1%} average efficiency). This context appears optimal.",
                            confidence=0.8,
                            priority=PriorityLevel.LOW,
                            patterns=[p.pattern_id for p in tag_patterns[:3]],
                            metrics={
                                'context_tag': tag,
                                'average_efficiency': avg_efficiency,
                                'pattern_count': len(tag_patterns)
                            }
                        ))
                    elif avg_efficiency < 0.5:
                        insights.append(PatternInsight(
                            insight_type="context_problem",
                            message=f"Automations with context '{tag}' show poor performance "
                                   f"({avg_efficiency:.1%} average efficiency). This context may need attention.",
                            confidence=0.8,
                            priority=PriorityLevel.MEDIUM,
                            patterns=[p.pattern_id for p in tag_patterns[:3]],
                            metrics={
                                'context_tag': tag,
                                'average_efficiency': avg_efficiency,
                                'pattern_count': len(tag_patterns)
                            }
                        ))
        
        except Exception as e:
            logger.error(f"Error analyzing pattern correlations: {str(e)}")
        
        return insights
    
    async def _analyze_temporal_patterns(self, user_id: str, patterns: List[UserBehaviorPattern]) -> List[PatternInsight]:
        """Analyze temporal patterns in user behavior."""
        insights = []
        
        try:
            # Get user activity summary for temporal analysis
            activity_summary = self.behavior_tracker.get_user_activity_summary(user_id, days=14)
            
            # Analyze activity trends
            trend = activity_summary.get('activity_trend', 'unknown')
            if trend == "increasing":
                insights.append(PatternInsight(
                    insight_type="increasing_activity",
                    message="Your automation usage is increasing over time. "
                           "Consider setting up more advanced workflows to handle the growing volume.",
                    confidence=0.8,
                    priority=PriorityLevel.LOW,
                    metrics={'trend': trend}
                ))
            elif trend == "decreasing":
                insights.append(PatternInsight(
                    insight_type="decreasing_activity",
                    message="Your automation usage is decreasing. "
                           "This might indicate workflow efficiency improvements or potential issues.",
                    confidence=0.7,
                    priority=PriorityLevel.MEDIUM,
                    metrics={'trend': trend}
                ))
            
            # Analyze peak activity hours
            peak_hours = activity_summary.get('peak_activity_hours', [])
            if peak_hours:
                insights.append(PatternInsight(
                    insight_type="peak_hours",
                    message=f"Your peak automation hours are {', '.join(map(str, peak_hours))}. "
                           f"Consider scheduling resource-intensive automations during off-peak times.",
                    confidence=0.9,
                    priority=PriorityLevel.LOW,
                    metrics={'peak_hours': peak_hours}
                ))
        
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {str(e)}")
        
        return insights
    
    async def _analyze_comparative_patterns(self, user_id: str, patterns: List[UserBehaviorPattern]) -> List[PatternInsight]:
        """Analyze patterns compared to other users (anonymized)."""
        insights = []
        
        try:
            # This would involve comparing user patterns to aggregated anonymized data
            # For now, provide general comparative insights based on pattern characteristics
            
            user_avg_efficiency = statistics.mean(p.get_efficiency_score() for p in patterns) if patterns else 0
            user_avg_success = statistics.mean(p.success_rate for p in patterns) if patterns else 0
            
            # Compare to hypothetical benchmarks (in production, these would be real aggregated data)
            benchmark_efficiency = 0.7  # Example benchmark
            benchmark_success = 0.85    # Example benchmark
            
            if user_avg_efficiency > benchmark_efficiency + 0.1:
                insights.append(PatternInsight(
                    insight_type="above_average_performance",
                    message=f"Your automation efficiency ({user_avg_efficiency:.1%}) is significantly above average. "
                           f"You're using automation very effectively.",
                    confidence=0.8,
                    priority=PriorityLevel.LOW,
                    metrics={
                        'user_efficiency': user_avg_efficiency,
                        'benchmark': benchmark_efficiency
                    }
                ))
            elif user_avg_efficiency < benchmark_efficiency - 0.1:
                insights.append(PatternInsight(
                    insight_type="below_average_performance",
                    message=f"Your automation efficiency ({user_avg_efficiency:.1%}) could be improved. "
                           f"Consider reviewing your most frequent workflows for optimization opportunities.",
                    confidence=0.8,
                    priority=PriorityLevel.MEDIUM,
                    metrics={
                        'user_efficiency': user_avg_efficiency,
                        'benchmark': benchmark_efficiency
                    }
                ))
        
        except Exception as e:
            logger.error(f"Error analyzing comparative patterns: {str(e)}")
        
        return insights
    
    async def _identify_performance_opportunities(self, performance_data: Dict[str, AutomationPerformanceMetrics]) -> List[OptimizationOpportunity]:
        """Identify performance optimization opportunities."""
        opportunities = []
        
        # Find slow automations
        slow_automations = [
            (aid, metrics) for aid, metrics in performance_data.items()
            if metrics.average_execution_time > 5.0 and metrics.execution_count > 3
        ]
        
        if slow_automations:
            total_time_savings = sum(
                (metrics.average_execution_time - 2.0) * metrics.execution_count 
                for _, metrics in slow_automations
            )
            
            opportunities.append(OptimizationOpportunity(
                opportunity_type="performance",
                description="Several automations are running slower than optimal",
                potential_impact=f"Could save approximately {total_time_savings:.1f} seconds in total execution time",
                implementation_effort="Medium",
                priority=PriorityLevel.MEDIUM,
                affected_items=[aid for aid, _ in slow_automations[:5]],
                estimated_savings={
                    'time_seconds': total_time_savings,
                    'automations_affected': len(slow_automations)
                }
            ))
        
        return opportunities
    
    async def _identify_reliability_opportunities(self, performance_data: Dict[str, AutomationPerformanceMetrics]) -> List[OptimizationOpportunity]:
        """Identify reliability improvement opportunities."""
        opportunities = []
        
        # Find unreliable automations
        unreliable_automations = [
            (aid, metrics) for aid, metrics in performance_data.items()
            if metrics.error_frequency > 0.2 and metrics.execution_count > 5
        ]
        
        if unreliable_automations:
            avg_error_rate = statistics.mean(metrics.error_frequency for _, metrics in unreliable_automations)
            
            opportunities.append(OptimizationOpportunity(
                opportunity_type="reliability",
                description="Multiple automations have high error rates that could be improved",
                potential_impact=f"Reducing errors could improve success rate by {(1-avg_error_rate):.1%}",
                implementation_effort="Medium to High",
                priority=PriorityLevel.HIGH,
                affected_items=[aid for aid, _ in unreliable_automations[:5]],
                estimated_savings={
                    'error_reduction': avg_error_rate,
                    'automations_affected': len(unreliable_automations)
                }
            ))
        
        return opportunities
    
    async def _identify_duplication_opportunities(self, patterns: List[UserBehaviorPattern]) -> List[OptimizationOpportunity]:
        """Identify workflow duplication opportunities."""
        opportunities = []
        
        # Find similar patterns that might be consolidated
        similar_groups = self._find_similar_patterns(patterns)
        
        for group in similar_groups:
            if len(group) >= 3:  # Multiple similar patterns
                total_frequency = sum(p.frequency for p in group)
                
                opportunities.append(OptimizationOpportunity(
                    opportunity_type="consolidation",
                    description=f"Found {len(group)} similar automation patterns that could be consolidated",
                    potential_impact="Simplify workflow management and reduce maintenance overhead",
                    implementation_effort="Low to Medium",
                    priority=PriorityLevel.LOW,
                    affected_items=[p.pattern_id for p in group],
                    estimated_savings={
                        'patterns_consolidated': len(group),
                        'total_frequency': total_frequency
                    }
                ))
        
        return opportunities
    
    async def _identify_tool_optimization_opportunities(self, user_id: str) -> List[OptimizationOpportunity]:
        """Identify tool usage optimization opportunities."""
        opportunities = []
        
        try:
            # Get user preferences to analyze tool usage
            user_prefs = self.behavior_tracker.user_preferences.get(user_id, {})
            preferred_tools = user_prefs.get('preferred_tools', {})
            
            if preferred_tools:
                # Find underutilized tools that might be beneficial
                top_tools = sorted(preferred_tools.items(), key=lambda x: x[1], reverse=True)
                
                if len(top_tools) > 1:
                    # Check if user is heavily relying on just a few tools
                    total_usage = sum(preferred_tools.values())
                    top_tool_usage = top_tools[0][1]
                    
                    if top_tool_usage / total_usage > 0.7:  # Using one tool >70% of the time
                        opportunities.append(OptimizationOpportunity(
                            opportunity_type="tool_diversification",
                            description=f"Heavy reliance on '{top_tools[0][0]}' tool. "
                                       f"Consider exploring other tools for varied workflows",
                            potential_impact="Discover more efficient tools for specific tasks",
                            implementation_effort="Low",
                            priority=PriorityLevel.LOW,
                            affected_items=[top_tools[0][0]],
                            estimated_savings={
                                'current_tool_dominance': top_tool_usage / total_usage,
                                'suggested_tools': [tool for tool, _ in top_tools[1:4]]
                            }
                        ))
        
        except Exception as e:
            logger.error(f"Error identifying tool optimization opportunities: {str(e)}")
        
        return opportunities
    
    def _find_similar_patterns(self, patterns: List[UserBehaviorPattern]) -> List[List[UserBehaviorPattern]]:
        """Find groups of similar patterns that might be consolidated."""
        similar_groups = []
        processed = set()
        
        for i, pattern1 in enumerate(patterns):
            if pattern1.pattern_id in processed:
                continue
            
            similar_group = [pattern1]
            processed.add(pattern1.pattern_id)
            
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                if pattern2.pattern_id in processed:
                    continue
                
                # Calculate similarity based on context tags and action sequences
                similarity = self._calculate_pattern_similarity(pattern1, pattern2)
                
                if similarity > 0.7:  # 70% similarity threshold
                    similar_group.append(pattern2)
                    processed.add(pattern2.pattern_id)
            
            if len(similar_group) > 1:
                similar_groups.append(similar_group)
        
        return similar_groups
    
    def _calculate_pattern_similarity(self, pattern1: UserBehaviorPattern, pattern2: UserBehaviorPattern) -> float:
        """Calculate similarity score between two patterns."""
        try:
            # Compare context tags
            tag_similarity = 0.0
            if pattern1.context_tags or pattern2.context_tags:
                common_tags = pattern1.context_tags & pattern2.context_tags
                all_tags = pattern1.context_tags | pattern2.context_tags
                tag_similarity = len(common_tags) / len(all_tags) if all_tags else 0.0
            
            # Compare action sequences
            action_similarity = 0.0
            if pattern1.action_sequence and pattern2.action_sequence:
                common_actions = set(pattern1.action_sequence) & set(pattern2.action_sequence)
                all_actions = set(pattern1.action_sequence) | set(pattern2.action_sequence)
                action_similarity = len(common_actions) / len(all_actions) if all_actions else 0.0
            
            # Compare performance characteristics
            performance_similarity = 1.0 - abs(pattern1.get_efficiency_score() - pattern2.get_efficiency_score())
            
            # Weighted average
            return (tag_similarity * 0.4 + action_similarity * 0.4 + performance_similarity * 0.2)
            
        except Exception:
            return 0.0
    
    def _calculate_performance_distribution(self, metrics_data: Dict[str, AutomationPerformanceMetrics]) -> Dict[str, int]:
        """Calculate distribution of automation performance levels."""
        distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
        for metrics in metrics_data.values():
            score = metrics.get_performance_score()
            if score >= 0.9:
                distribution["excellent"] += 1
            elif score >= 0.7:
                distribution["good"] += 1
            elif score >= 0.5:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    def _calculate_improvement_trends(self, metrics_data: Dict[str, AutomationPerformanceMetrics]) -> Dict[str, int]:
        """Calculate automation improvement trends."""
        trends = {"improving": 0, "stable": 0, "declining": 0}
        
        for metrics in metrics_data.values():
            trends[metrics.trend_direction] += 1
        
        return trends
    
    def _identify_system_bottlenecks(self, metrics_data: Dict[str, AutomationPerformanceMetrics]) -> List[Dict[str, Any]]:
        """Identify potential system bottlenecks."""
        bottlenecks = []
        
        # Find automations with consistently high execution times
        slow_automations = [
            (aid, metrics) for aid, metrics in metrics_data.items()
            if metrics.average_execution_time > 10.0 and metrics.execution_count > 10
        ]
        
        if slow_automations:
            bottlenecks.append({
                "type": "execution_time",
                "description": f"{len(slow_automations)} automations with consistently slow execution",
                "affected_automations": [aid for aid, _ in slow_automations[:3]],
                "average_time": statistics.mean(metrics.average_execution_time for _, metrics in slow_automations)
            })
        
        # Find automations with high resource usage patterns
        # This would be expanded based on actual resource usage data
        
        return bottlenecks
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached analysis is still valid."""
        if cache_key not in self.analysis_cache:
            return False
        
        cache_entry = self.analysis_cache[cache_key]
        return datetime.now(UTC) - cache_entry['timestamp'] < self.cache_ttl
    
    def _cache_results(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """Cache analysis results."""
        self.analysis_cache[cache_key] = {
            'results': results,
            'timestamp': datetime.now(UTC)
        }
    
    def _dict_to_insight(self, insight_dict: Dict[str, Any]) -> PatternInsight:
        """Convert dictionary back to PatternInsight object."""
        return PatternInsight(
            insight_type=insight_dict['type'],
            message=insight_dict['message'],
            confidence=insight_dict['confidence'],
            priority=PriorityLevel(insight_dict['priority']),
            patterns=insight_dict['patterns'],
            automations=insight_dict.get('automations', []),
            metrics=insight_dict['metrics']
        )