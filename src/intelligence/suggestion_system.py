"""
Intelligent Suggestion System for Adaptive Automation Enhancement.

This module provides sophisticated automation suggestion generation based on
behavioral patterns, performance analysis, and machine learning insights while
maintaining strict privacy protection and security validation.

Security: Privacy-preserving suggestion generation with secure pattern analysis.
Performance: Optimized suggestion algorithms with intelligent ranking and caching.
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
from src.core.suggestion_system import (
    UserBehaviorPattern, IntelligentSuggestion, SuggestionType, PriorityLevel
)
from src.core.errors import IntelligenceError

logger = get_logger(__name__)


class SuggestionCategory(Enum):
    """Categories of intelligent automation suggestions."""
    AUTOMATION_OPPORTUNITY = "automation_opportunity"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    TOOL_RECOMMENDATION = "tool_recommendation"
    PROCESS_IMPROVEMENT = "process_improvement"
    ERROR_PREVENTION = "error_prevention"
    EFFICIENCY_ENHANCEMENT = "efficiency_enhancement"


@dataclass(frozen=True)
class AutomationSuggestion:
    """Comprehensive automation suggestion with detailed analysis."""
    suggestion_id: str
    suggestion_type: str
    category: SuggestionCategory
    title: str
    description: str
    confidence: float
    potential_time_saved: float
    implementation_complexity: str  # low|medium|high
    tools_involved: List[str]
    trigger_conditions: Dict[str, Any]
    estimated_success_rate: float
    rationale: str
    supporting_patterns: List[str]
    priority_score: float = 0.0
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: self.potential_time_saved >= 0.0)
    @require(lambda self: 0.0 <= self.estimated_success_rate <= 1.0)
    @require(lambda self: self.implementation_complexity in ["low", "medium", "high"])
    def __post_init__(self):
        pass
    
    def get_roi_estimate(self) -> float:
        """Estimate return on investment for implementing suggestion."""
        complexity_weights = {"low": 1.0, "medium": 3.0, "high": 8.0}
        implementation_cost = complexity_weights[self.implementation_complexity]
        
        # ROI = (time saved * success rate * confidence) / implementation cost
        roi = (self.potential_time_saved * self.estimated_success_rate * self.confidence) / implementation_cost
        return roi
    
    def is_high_impact(self, time_threshold: float = 300.0) -> bool:
        """Check if suggestion provides high impact time savings."""
        return self.potential_time_saved >= time_threshold and self.confidence >= 0.7


class SuggestionRanker:
    """Intelligent ranking system for automation suggestions."""
    
    def __init__(self):
        self.ranking_weights = {
            'confidence': 0.25,
            'potential_savings': 0.25,
            'success_rate': 0.20,
            'implementation_ease': 0.15,
            'pattern_support': 0.15
        }
    
    def rank_suggestions(self, suggestions: List[AutomationSuggestion]) -> List[AutomationSuggestion]:
        """Rank suggestions by priority and potential impact."""
        # Calculate priority scores
        scored_suggestions = []
        
        for suggestion in suggestions:
            priority_score = self._calculate_priority_score(suggestion)
            
            # Create new suggestion with priority score
            scored_suggestion = AutomationSuggestion(
                suggestion_id=suggestion.suggestion_id,
                suggestion_type=suggestion.suggestion_type,
                category=suggestion.category,
                title=suggestion.title,
                description=suggestion.description,
                confidence=suggestion.confidence,
                potential_time_saved=suggestion.potential_time_saved,
                implementation_complexity=suggestion.implementation_complexity,
                tools_involved=suggestion.tools_involved,
                trigger_conditions=suggestion.trigger_conditions,
                estimated_success_rate=suggestion.estimated_success_rate,
                rationale=suggestion.rationale,
                supporting_patterns=suggestion.supporting_patterns,
                priority_score=priority_score
            )
            
            scored_suggestions.append(scored_suggestion)
        
        # Sort by priority score (descending)
        return sorted(scored_suggestions, key=lambda s: s.priority_score, reverse=True)
    
    def _calculate_priority_score(self, suggestion: AutomationSuggestion) -> float:
        """Calculate priority score based on multiple factors."""
        # Normalize factors to 0-1 range
        confidence_score = suggestion.confidence
        
        # Normalize time savings (assume 1 hour = 3600 seconds as high value)
        savings_score = min(1.0, suggestion.potential_time_saved / 3600.0)
        
        success_score = suggestion.estimated_success_rate
        
        # Implementation ease (inverse of complexity)
        complexity_scores = {"low": 1.0, "medium": 0.6, "high": 0.3}
        ease_score = complexity_scores[suggestion.implementation_complexity]
        
        # Pattern support (more supporting patterns = higher score)
        pattern_score = min(1.0, len(suggestion.supporting_patterns) / 5.0)
        
        # Weighted combination
        priority_score = (
            confidence_score * self.ranking_weights['confidence'] +
            savings_score * self.ranking_weights['potential_savings'] +
            success_score * self.ranking_weights['success_rate'] +
            ease_score * self.ranking_weights['implementation_ease'] +
            pattern_score * self.ranking_weights['pattern_support']
        )
        
        return priority_score


class IntelligentSuggestionSystem:
    """Comprehensive intelligent suggestion generation with privacy protection."""
    
    def __init__(self):
        self.suggestion_generators: Dict[str, Any] = {}
        self.suggestion_ranker = SuggestionRanker()
        self.suggestion_cache: Dict[str, List[AutomationSuggestion]] = {}
        self.suggestion_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_suggestions_per_category = 3
        self.min_confidence_for_suggestions = 0.6
        self.min_time_savings_threshold = 30.0  # 30 seconds
        
        self._initialize_suggestion_generators()
    
    async def initialize(self) -> Either[IntelligenceError, None]:
        """Initialize intelligent suggestion system."""
        try:
            # Configure suggestion generation algorithms
            self._configure_suggestion_algorithms()
            
            logger.info("Intelligent suggestion system initialized successfully")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Suggestion system initialization failed: {str(e)}")
            return Either.left(IntelligenceError.initialization_failed(str(e)))
    
    @require(lambda self, patterns: len(patterns) >= 1)
    @require(lambda self, suggestion_count: 1 <= suggestion_count <= 20)
    @require(lambda self, confidence_threshold: 0.0 <= confidence_threshold <= 1.0)
    async def generate_suggestions(
        self,
        patterns: List[UserBehaviorPattern],
        suggestion_count: int = 5,
        confidence_threshold: float = 0.7,
        optimization_target: str = "efficiency"
    ) -> Either[IntelligenceError, List[AutomationSuggestion]]:
        """
        Generate intelligent automation suggestions based on behavioral patterns.
        
        Analyzes behavioral patterns to identify automation opportunities,
        workflow optimizations, and process improvements while maintaining
        privacy protection and security validation throughout generation.
        
        Args:
            patterns: Validated behavioral patterns for analysis
            suggestion_count: Maximum number of suggestions to generate
            confidence_threshold: Minimum confidence for actionable suggestions
            optimization_target: Target metric for optimization focus
            
        Returns:
            Either error or ranked list of intelligent automation suggestions
            
        Security:
            - Privacy-preserving pattern analysis
            - Secure suggestion generation with no sensitive data exposure
            - Validated suggestions with confidence scoring
        """
        try:
            # Check cache for recent suggestions
            cache_key = self._generate_cache_key(patterns, optimization_target)
            if cache_key in self.suggestion_cache:
                cached_suggestions = self.suggestion_cache[cache_key]
                logger.debug(f"Retrieved {len(cached_suggestions)} suggestions from cache")
                return Either.right(cached_suggestions[:suggestion_count])
            
            # Generate suggestions from different sources
            all_suggestions = []
            
            # Pattern-based automation suggestions
            automation_suggestions = await self._generate_automation_suggestions(patterns)
            all_suggestions.extend(automation_suggestions)
            
            # Efficiency optimization suggestions
            efficiency_suggestions = await self._generate_efficiency_suggestions(patterns, optimization_target)
            all_suggestions.extend(efficiency_suggestions)
            
            # Tool recommendation suggestions
            tool_suggestions = await self._generate_tool_recommendations(patterns)
            all_suggestions.extend(tool_suggestions)
            
            # Workflow improvement suggestions
            workflow_suggestions = await self._generate_workflow_improvements(patterns)
            all_suggestions.extend(workflow_suggestions)
            
            # Error prevention suggestions
            error_prevention_suggestions = await self._generate_error_prevention_suggestions(patterns)
            all_suggestions.extend(error_prevention_suggestions)
            
            # Filter by confidence threshold
            qualified_suggestions = [
                suggestion for suggestion in all_suggestions
                if suggestion.confidence >= confidence_threshold
            ]
            
            # Rank suggestions by priority and impact
            ranked_suggestions = self.suggestion_ranker.rank_suggestions(qualified_suggestions)
            
            # Select top suggestions
            final_suggestions = ranked_suggestions[:suggestion_count]
            
            # Cache results for performance
            self.suggestion_cache[cache_key] = final_suggestions
            
            # Track suggestion generation
            self._track_suggestion_generation(patterns, final_suggestions, optimization_target)
            
            logger.info(f"Generated {len(final_suggestions)} intelligent suggestions from {len(patterns)} patterns")
            return Either.right(final_suggestions)
            
        except Exception as e:
            logger.error(f"Suggestion generation failed: {str(e)}")
            return Either.left(IntelligenceError.suggestion_generation_failed(str(e)))
    
    async def _generate_automation_suggestions(self, patterns: List[UserBehaviorPattern]) -> List[AutomationSuggestion]:
        """Generate suggestions for new automation opportunities."""
        suggestions = []
        
        # Identify high-frequency, consistent patterns suitable for automation
        automation_candidates = [
            pattern for pattern in patterns
            if (pattern.frequency >= 5 and 
                pattern.confidence_score >= 0.8 and
                pattern.success_rate >= 0.8 and
                len(pattern.action_sequence) >= 2)
        ]
        
        for pattern in automation_candidates[:self.max_suggestions_per_category]:
            # Calculate potential time savings
            time_saved_per_occurrence = pattern.average_completion_time * 0.8  # 80% automation efficiency
            total_potential_savings = time_saved_per_occurrence * pattern.frequency
            
            if total_potential_savings >= self.min_time_savings_threshold:
                suggestion = AutomationSuggestion(
                    suggestion_id=f"automation_{pattern.pattern_id}",
                    suggestion_type="automation",
                    category=SuggestionCategory.AUTOMATION_OPPORTUNITY,
                    title=f"Automate {' → '.join(pattern.action_sequence[:3])}... workflow",
                    description=f"Create automation for {pattern.frequency}-time repeated workflow with {pattern.success_rate:.1%} success rate",
                    confidence=min(0.95, pattern.confidence_score + 0.1),
                    potential_time_saved=total_potential_savings,
                    implementation_complexity=self._assess_automation_complexity(pattern),
                    tools_involved=self._extract_tools_from_pattern(pattern),
                    trigger_conditions=self._suggest_trigger_conditions(pattern),
                    estimated_success_rate=pattern.success_rate * 0.9,  # Slightly lower for automation
                    rationale=f"Pattern occurs {pattern.frequency} times with {pattern.confidence_score:.1%} confidence and {pattern.success_rate:.1%} success rate",
                    supporting_patterns=[pattern.pattern_id]
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_efficiency_suggestions(self, patterns: List[UserBehaviorPattern], target: str) -> List[AutomationSuggestion]:
        """Generate suggestions for efficiency improvements."""
        suggestions = []
        
        # Identify patterns with optimization potential
        optimization_candidates = [
            pattern for pattern in patterns
            if (pattern.get_efficiency_score() < 0.7 and
                pattern.frequency >= 3 and
                pattern.average_completion_time > 10.0)
        ]
        
        for pattern in optimization_candidates[:self.max_suggestions_per_category]:
            # Estimate efficiency improvement potential
            current_efficiency = pattern.get_efficiency_score()
            potential_improvement = (0.9 - current_efficiency) * pattern.average_completion_time
            total_savings = potential_improvement * pattern.frequency
            
            if total_savings >= self.min_time_savings_threshold:
                suggestion = AutomationSuggestion(
                    suggestion_id=f"efficiency_{pattern.pattern_id}",
                    suggestion_type="optimization",
                    category=SuggestionCategory.EFFICIENCY_ENHANCEMENT,
                    title=f"Optimize {' → '.join(pattern.action_sequence[:2])}... workflow efficiency",
                    description=f"Improve workflow efficiency from {current_efficiency:.1%} to target 90%+",
                    confidence=0.75,
                    potential_time_saved=total_savings,
                    implementation_complexity="medium",
                    tools_involved=self._extract_tools_from_pattern(pattern),
                    trigger_conditions={"efficiency_threshold": current_efficiency},
                    estimated_success_rate=0.8,
                    rationale=f"Low efficiency pattern ({current_efficiency:.1%}) with {total_savings:.0f}s potential savings",
                    supporting_patterns=[pattern.pattern_id]
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_tool_recommendations(self, patterns: List[UserBehaviorPattern]) -> List[AutomationSuggestion]:
        """Generate tool usage recommendations."""
        suggestions = []
        
        # Analyze tool usage patterns
        tool_usage = {}
        for pattern in patterns:
            tools = self._extract_tools_from_pattern(pattern)
            for tool in tools:
                if tool not in tool_usage:
                    tool_usage[tool] = {'frequency': 0, 'efficiency': [], 'patterns': []}
                tool_usage[tool]['frequency'] += pattern.frequency
                tool_usage[tool]['efficiency'].append(pattern.get_efficiency_score())
                tool_usage[tool]['patterns'].append(pattern)
        
        # Identify underutilized tools or tool combinations
        for tool, usage_data in tool_usage.items():
            avg_efficiency = statistics.mean(usage_data['efficiency'])
            if usage_data['frequency'] >= 5 and avg_efficiency < 0.8:
                suggestion = AutomationSuggestion(
                    suggestion_id=f"tool_rec_{tool}",
                    suggestion_type="tool_recommendation",
                    category=SuggestionCategory.TOOL_RECOMMENDATION,
                    title=f"Optimize {tool} usage patterns",
                    description=f"Improve {tool} workflow efficiency through better usage patterns",
                    confidence=0.7,
                    potential_time_saved=usage_data['frequency'] * 30.0,  # Estimate 30s savings per use
                    implementation_complexity="low",
                    tools_involved=[tool],
                    trigger_conditions={"tool": tool, "efficiency_below": avg_efficiency},
                    estimated_success_rate=0.75,
                    rationale=f"Tool used {usage_data['frequency']} times with {avg_efficiency:.1%} average efficiency",
                    supporting_patterns=[p.pattern_id for p in usage_data['patterns'][:3]]
                )
                suggestions.append(suggestion)
        
        return suggestions[:self.max_suggestions_per_category]
    
    async def _generate_workflow_improvements(self, patterns: List[UserBehaviorPattern]) -> List[AutomationSuggestion]:
        """Generate workflow improvement suggestions."""
        suggestions = []
        
        # Identify patterns with long sequences that could be streamlined
        long_sequence_patterns = [
            pattern for pattern in patterns
            if (len(pattern.action_sequence) >= 4 and
                pattern.average_completion_time > 60.0 and
                pattern.frequency >= 3)
        ]
        
        for pattern in long_sequence_patterns[:self.max_suggestions_per_category]:
            # Estimate workflow simplification potential
            sequence_length = len(pattern.action_sequence)
            potential_reduction = min(sequence_length * 0.3, 5)  # Up to 30% reduction, max 5 steps
            time_savings_per_use = pattern.average_completion_time * (potential_reduction / sequence_length)
            total_savings = time_savings_per_use * pattern.frequency
            
            if total_savings >= self.min_time_savings_threshold:
                suggestion = AutomationSuggestion(
                    suggestion_id=f"workflow_{pattern.pattern_id}",
                    suggestion_type="workflow_improvement",
                    category=SuggestionCategory.WORKFLOW_OPTIMIZATION,
                    title=f"Streamline {sequence_length}-step workflow",
                    description=f"Reduce workflow steps from {sequence_length} to ~{sequence_length - int(potential_reduction)} for efficiency",
                    confidence=0.65,
                    potential_time_saved=total_savings,
                    implementation_complexity="medium",
                    tools_involved=self._extract_tools_from_pattern(pattern),
                    trigger_conditions={"sequence_length": sequence_length},
                    estimated_success_rate=0.7,
                    rationale=f"Complex {sequence_length}-step workflow with {total_savings:.0f}s optimization potential",
                    supporting_patterns=[pattern.pattern_id]
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_error_prevention_suggestions(self, patterns: List[UserBehaviorPattern]) -> List[AutomationSuggestion]:
        """Generate error prevention suggestions."""
        suggestions = []
        
        # Identify patterns with lower success rates
        error_prone_patterns = [
            pattern for pattern in patterns
            if (pattern.success_rate < 0.9 and
                pattern.frequency >= 3 and
                pattern.confidence_score >= 0.6)
        ]
        
        for pattern in error_prone_patterns[:self.max_suggestions_per_category]:
            error_rate = 1.0 - pattern.success_rate
            potential_error_reduction = error_rate * 0.7  # Assume 70% error reduction possible
            
            suggestion = AutomationSuggestion(
                suggestion_id=f"error_prev_{pattern.pattern_id}",
                suggestion_type="error_prevention",
                category=SuggestionCategory.ERROR_PREVENTION,
                title=f"Prevent errors in workflow with {error_rate:.1%} failure rate",
                description=f"Add validation and error handling to reduce failure rate from {error_rate:.1%} to {error_rate - potential_error_reduction:.1%}",
                confidence=0.8,
                potential_time_saved=pattern.frequency * pattern.average_completion_time * potential_error_reduction,
                implementation_complexity="medium",
                tools_involved=self._extract_tools_from_pattern(pattern),
                trigger_conditions={"error_rate_above": error_rate},
                estimated_success_rate=0.85,
                rationale=f"Pattern has {error_rate:.1%} failure rate with {pattern.frequency} occurrences",
                supporting_patterns=[pattern.pattern_id]
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _assess_automation_complexity(self, pattern: UserBehaviorPattern) -> str:
        """Assess implementation complexity for automating a pattern."""
        sequence_length = len(pattern.action_sequence)
        unique_tools = len(self._extract_tools_from_pattern(pattern))
        
        if sequence_length <= 3 and unique_tools <= 2:
            return "low"
        elif sequence_length <= 6 and unique_tools <= 4:
            return "medium"
        else:
            return "high"
    
    def _extract_tools_from_pattern(self, pattern: UserBehaviorPattern) -> List[str]:
        """Extract tool names from pattern context tags."""
        tools = []
        for tag in pattern.context_tags:
            if tag.startswith('tool:'):
                tools.append(tag[5:])  # Remove 'tool:' prefix
        return tools
    
    def _suggest_trigger_conditions(self, pattern: UserBehaviorPattern) -> Dict[str, Any]:
        """Suggest trigger conditions for automation."""
        return {
            "frequency": pattern.frequency,
            "success_rate_threshold": pattern.success_rate,
            "context_tags": list(pattern.context_tags)[:3]  # Top 3 context tags
        }
    
    def _generate_cache_key(self, patterns: List[UserBehaviorPattern], target: str) -> str:
        """Generate cache key for suggestion results."""
        pattern_ids = sorted([p.pattern_id for p in patterns])
        return f"{target}_{hash(tuple(pattern_ids[:10]))}"  # Use first 10 pattern IDs
    
    def _track_suggestion_generation(
        self,
        patterns: List[UserBehaviorPattern],
        suggestions: List[AutomationSuggestion],
        target: str
    ) -> None:
        """Track suggestion generation for analytics."""
        self.suggestion_history.append({
            'timestamp': datetime.now(UTC),
            'patterns_analyzed': len(patterns),
            'suggestions_generated': len(suggestions),
            'optimization_target': target,
            'average_confidence': statistics.mean([s.confidence for s in suggestions]) if suggestions else 0.0,
            'total_potential_savings': sum(s.potential_time_saved for s in suggestions)
        })
        
        # Limit history size
        if len(self.suggestion_history) > 100:
            self.suggestion_history = self.suggestion_history[-50:]
    
    def _initialize_suggestion_generators(self) -> None:
        """Initialize suggestion generation functions."""
        self.suggestion_generators = {
            'automation': self._generate_automation_suggestions,
            'efficiency': self._generate_efficiency_suggestions,
            'tools': self._generate_tool_recommendations,
            'workflow': self._generate_workflow_improvements,
            'error_prevention': self._generate_error_prevention_suggestions
        }
    
    def _configure_suggestion_algorithms(self) -> None:
        """Configure suggestion generation algorithms."""
        # Configuration for different suggestion types
        pass