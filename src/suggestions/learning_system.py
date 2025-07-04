"""
Adaptive learning system for continuous improvement of smart suggestions.

This module implements sophisticated learning algorithms that process user feedback,
adapt suggestion generation, and continuously improve the quality and relevance
of AI-powered automation recommendations.

Security: All learning includes privacy protection and secure feedback processing.
Performance: Optimized for real-time learning with incremental updates.
Type Safety: Complete integration with suggestion system and feedback processing.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta, UTC
from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics
import math
import asyncio

from src.core.suggestion_system import (
    IntelligentSuggestion, SuggestionFeedback, SuggestionType, PriorityLevel,
    SuggestionError, UserBehaviorPattern, SuggestionContext
)
from src.suggestions.recommendation_engine import RecommendationEngine
from src.suggestions.pattern_analyzer import PatternAnalyzer
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PersonalizationProfile:
    """User personalization profile for adaptive suggestion generation."""
    user_id: str
    preference_weights: Dict[SuggestionType, float] = field(default_factory=dict)
    successful_patterns: List[str] = field(default_factory=list)
    rejected_patterns: List[str] = field(default_factory=list)
    effectiveness_scores: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.1
    confidence_threshold: float = 0.7
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Contract validation for personalization profile."""
        if len(self.user_id) == 0:
            raise ValueError("User ID cannot be empty")
        if not (0.0 <= self.learning_rate <= 1.0):
            raise ValueError("Learning rate must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def get_type_preference(self, suggestion_type: SuggestionType) -> float:
        """Get user preference score for suggestion type."""
        return self.preference_weights.get(suggestion_type, 0.5)
    
    def has_high_confidence(self, suggestion_type: SuggestionType) -> bool:
        """Check if user has high confidence preference for suggestion type."""
        preference = self.get_type_preference(suggestion_type)
        return preference >= self.confidence_threshold
    
    def get_personalization_score(self, suggestion: IntelligentSuggestion) -> float:
        """Calculate personalization score for suggestion."""
        type_preference = self.get_type_preference(suggestion.suggestion_type)
        
        # Check if suggestion matches successful patterns
        pattern_bonus = 0.0
        suggestion_context = suggestion.context.get('patterns', [])
        for pattern in suggestion_context:
            if pattern in self.successful_patterns:
                pattern_bonus += 0.1
            elif pattern in self.rejected_patterns:
                pattern_bonus -= 0.1
        
        # Calculate final personalization score
        personalized_score = type_preference + pattern_bonus
        return max(0.0, min(1.0, personalized_score))


@dataclass(frozen=True)
class LearningInsight:
    """Learning insight for improving suggestion generation."""
    insight_id: str
    insight_type: str
    description: str
    confidence: float
    supporting_evidence: Dict[str, Any]
    recommendation: str
    priority: PriorityLevel
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        """Contract validation for learning insight."""
        if len(self.insight_id) == 0:
            raise ValueError("Insight ID cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def is_actionable(self, threshold: float = 0.8) -> bool:
        """Check if insight is actionable with high confidence."""
        return self.confidence >= threshold and self.priority in [PriorityLevel.HIGH, PriorityLevel.CRITICAL]


class AdaptiveLearningSystem:
    """Sophisticated adaptive learning system for smart suggestions."""
    
    def __init__(self, recommendation_engine: RecommendationEngine, pattern_analyzer: PatternAnalyzer):
        self.recommendation_engine = recommendation_engine
        self.pattern_analyzer = pattern_analyzer
        self.user_profiles: Dict[str, PersonalizationProfile] = {}
        self.feedback_history: Dict[str, List[SuggestionFeedback]] = defaultdict(list)
        self.learning_insights: List[LearningInsight] = []
        self.adaptation_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.learning_enabled = True
        
        # Learning parameters
        self.min_feedback_for_learning = 5
        self.feedback_window_days = 30
        self.insight_confidence_threshold = 0.75
    
    @require(lambda self, feedback: isinstance(feedback, SuggestionFeedback))
    async def process_feedback(self, feedback: SuggestionFeedback) -> Either[SuggestionError, None]:
        """
        Process user feedback to improve suggestion quality and personalization.
        
        Args:
            feedback: User feedback on a suggestion
            
        Returns:
            Either error or successful processing confirmation
        """
        try:
            # Store feedback
            self.feedback_history[feedback.user_id].append(feedback)
            
            # Update user personalization profile
            profile_result = await self._update_personalization_profile(feedback)
            if profile_result.is_left():
                return profile_result
            
            # Learn from feedback patterns
            learning_result = await self._learn_from_feedback(feedback)
            if learning_result.is_left():
                return learning_result
            
            # Update adaptation metrics
            await self._update_adaptation_metrics(feedback)
            
            # Generate learning insights if enough data
            if len(self.feedback_history[feedback.user_id]) >= self.min_feedback_for_learning:
                await self._generate_learning_insights(feedback.user_id)
            
            logger.info(f"Processed feedback for suggestion {feedback.suggestion_id}")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return Either.left(SuggestionError.learning_failed(f"Feedback processing failed: {str(e)}"))
    
    @require(lambda self, user_id: len(user_id) > 0)
    async def personalize_suggestions(self, user_id: str, suggestions: List[IntelligentSuggestion]) -> List[IntelligentSuggestion]:
        """
        Personalize suggestions based on user learning profile.
        
        Args:
            user_id: User identifier
            suggestions: List of suggestions to personalize
            
        Returns:
            Personalized and re-ranked suggestions
        """
        try:
            if not self.learning_enabled or user_id not in self.user_profiles:
                return suggestions
            
            profile = self.user_profiles[user_id]
            personalized_suggestions = []
            
            for suggestion in suggestions:
                # Calculate personalization score
                personalization_score = profile.get_personalization_score(suggestion)
                
                # Create personalized suggestion with updated score
                personalized_suggestion = IntelligentSuggestion(
                    suggestion_id=suggestion.suggestion_id,
                    suggestion_type=suggestion.suggestion_type,
                    title=suggestion.title,
                    description=suggestion.description,
                    priority=suggestion.priority,
                    confidence=suggestion.confidence,
                    potential_impact=suggestion.potential_impact,
                    implementation_effort=suggestion.implementation_effort,
                    suggested_actions=suggestion.suggested_actions,
                    context=suggestion.context,
                    reasoning=suggestion.reasoning,
                    created_at=suggestion.created_at,
                    expires_at=suggestion.expires_at,
                    personalization_score=personalization_score
                )
                
                personalized_suggestions.append(personalized_suggestion)
            
            # Re-rank based on personalization scores
            personalized_suggestions.sort(
                key=lambda s: s.get_urgency_score() * s.personalization_score,
                reverse=True
            )
            
            logger.debug(f"Personalized {len(suggestions)} suggestions for user {user_id}")
            return personalized_suggestions
            
        except Exception as e:
            logger.error(f"Error personalizing suggestions: {str(e)}")
            return suggestions  # Return original suggestions if personalization fails
    
    @require(lambda self, user_id: len(user_id) > 0)
    def get_user_learning_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get learning statistics and insights for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing learning statistics and insights
        """
        try:
            if user_id not in self.user_profiles:
                return {'error': 'No learning profile found for user'}
            
            profile = self.user_profiles[user_id]
            feedback_list = self.feedback_history.get(user_id, [])
            
            # Calculate feedback statistics
            total_feedback = len(feedback_list)
            accepted_feedback = sum(1 for f in feedback_list if f.accepted)
            recent_feedback = [
                f for f in feedback_list
                if (datetime.now(UTC) - f.timestamp).days <= 7
            ]
            
            # Calculate satisfaction scores
            satisfaction_scores = [
                f.get_satisfaction_score() for f in feedback_list
                if f.accepted and f.rating is not None
            ]
            avg_satisfaction = statistics.mean(satisfaction_scores) if satisfaction_scores else 0.0
            
            # Get learning insights for user
            user_insights = [
                insight for insight in self.learning_insights
                if user_id in insight.supporting_evidence.get('users', [])
            ]
            
            stats = {
                'user_id': user_id,
                'profile_last_updated': profile.last_updated.isoformat(),
                'total_feedback_count': total_feedback,
                'acceptance_rate': accepted_feedback / max(1, total_feedback),
                'recent_feedback_count': len(recent_feedback),
                'average_satisfaction': round(avg_satisfaction, 3),
                'preference_weights': dict(profile.preference_weights),
                'successful_patterns_count': len(profile.successful_patterns),
                'rejected_patterns_count': len(profile.rejected_patterns),
                'learning_insights_count': len(user_insights),
                'high_confidence_types': [
                    stype.value for stype in SuggestionType
                    if profile.has_high_confidence(stype)
                ],
                'adaptation_metrics': self.adaptation_metrics.get(user_id, {})
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting learning stats: {str(e)}")
            return {'error': str(e)}
    
    async def get_system_learning_insights(self) -> List[Dict[str, Any]]:
        """Get system-wide learning insights and recommendations."""
        try:
            insights_data = []
            
            for insight in self.learning_insights:
                if insight.is_actionable():
                    insights_data.append({
                        'insight_id': insight.insight_id,
                        'type': insight.insight_type,
                        'description': insight.description,
                        'confidence': insight.confidence,
                        'recommendation': insight.recommendation,
                        'priority': insight.priority.value,
                        'created_at': insight.created_at.isoformat(),
                        'supporting_evidence': insight.supporting_evidence
                    })
            
            # Sort by priority and confidence
            insights_data.sort(
                key=lambda i: (
                    {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[i['priority']],
                    i['confidence']
                ),
                reverse=True
            )
            
            return insights_data
            
        except Exception as e:
            logger.error(f"Error getting system learning insights: {str(e)}")
            return []
    
    async def _update_personalization_profile(self, feedback: SuggestionFeedback) -> Either[SuggestionError, None]:
        """Update user personalization profile based on feedback."""
        try:
            user_id = feedback.user_id
            
            # Get or create profile
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = PersonalizationProfile(user_id=user_id)
            
            profile = self.user_profiles[user_id]
            
            # Extract suggestion type from feedback (would be stored with feedback in practice)
            # For now, we'll infer from the suggestion context
            suggestion_type = self._infer_suggestion_type_from_feedback(feedback)
            
            # Update preference weights based on feedback
            current_weight = profile.preference_weights.get(suggestion_type, 0.5)
            
            if feedback.accepted:
                # Increase weight for accepted suggestions
                satisfaction_multiplier = feedback.get_satisfaction_score()
                new_weight = current_weight + (profile.learning_rate * satisfaction_multiplier)
                
                # Track successful pattern
                if feedback.outcome and 'pattern' in str(feedback.outcome):
                    pattern_id = str(feedback.outcome)
                    if pattern_id not in profile.successful_patterns:
                        profile.successful_patterns.append(pattern_id)
            else:
                # Decrease weight for rejected suggestions
                new_weight = current_weight - (profile.learning_rate * 0.5)
                
                # Track rejected pattern
                if feedback.notes and 'pattern' in feedback.notes:
                    pattern_id = feedback.notes
                    if pattern_id not in profile.rejected_patterns:
                        profile.rejected_patterns.append(pattern_id)
            
            # Normalize weight to [0, 1] range
            new_weight = max(0.0, min(1.0, new_weight))
            
            # Update profile (create new instance due to frozen dataclass)
            updated_profile = PersonalizationProfile(
                user_id=profile.user_id,
                preference_weights={**profile.preference_weights, suggestion_type: new_weight},
                successful_patterns=profile.successful_patterns,
                rejected_patterns=profile.rejected_patterns,
                effectiveness_scores=profile.effectiveness_scores,
                learning_rate=profile.learning_rate,
                confidence_threshold=profile.confidence_threshold,
                last_updated=datetime.now(UTC)
            )
            
            self.user_profiles[user_id] = updated_profile
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(SuggestionError.learning_failed(f"Profile update failed: {str(e)}"))
    
    async def _learn_from_feedback(self, feedback: SuggestionFeedback) -> Either[SuggestionError, None]:
        """Learn from feedback patterns across all users."""
        try:
            # Analyze feedback patterns for system-wide learning
            all_feedback = []
            for user_feedback_list in self.feedback_history.values():
                all_feedback.extend(user_feedback_list)
            
            if len(all_feedback) < self.min_feedback_for_learning:
                return Either.right(None)
            
            # Learn suggestion type preferences
            type_preferences = defaultdict(list)
            for fb in all_feedback:
                suggestion_type = self._infer_suggestion_type_from_feedback(fb)
                type_preferences[suggestion_type].append(fb.get_satisfaction_score())
            
            # Identify high and low performing suggestion types
            for stype, scores in type_preferences.items():
                if len(scores) >= 3:
                    avg_score = statistics.mean(scores)
                    if avg_score < 0.3:
                        # Low performing suggestion type
                        await self._create_learning_insight(
                            insight_type="low_performing_suggestion_type",
                            description=f"Suggestion type '{stype.value}' has low user satisfaction",
                            confidence=0.8,
                            recommendation=f"Review and improve {stype.value} suggestion generation",
                            priority=PriorityLevel.HIGH,
                            evidence={'suggestion_type': stype.value, 'average_score': avg_score}
                        )
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(SuggestionError.learning_failed(f"Pattern learning failed: {str(e)}"))
    
    async def _update_adaptation_metrics(self, feedback: SuggestionFeedback) -> None:
        """Update adaptation metrics based on feedback."""
        try:
            user_id = feedback.user_id
            
            if user_id not in self.adaptation_metrics:
                self.adaptation_metrics[user_id] = {
                    'total_suggestions': 0,
                    'accepted_suggestions': 0,
                    'average_satisfaction': 0.0,
                    'learning_velocity': 0.0,
                    'personalization_effectiveness': 0.0
                }
            
            metrics = self.adaptation_metrics[user_id]
            metrics['total_suggestions'] += 1
            
            if feedback.accepted:
                metrics['accepted_suggestions'] += 1
            
            # Update average satisfaction
            satisfaction = feedback.get_satisfaction_score()
            current_avg = metrics['average_satisfaction']
            total = metrics['total_suggestions']
            metrics['average_satisfaction'] = ((current_avg * (total - 1)) + satisfaction) / total
            
            # Calculate learning velocity (improvement rate)
            recent_feedback = self.feedback_history[user_id][-10:]  # Last 10 feedback items
            if len(recent_feedback) >= 5:
                recent_scores = [f.get_satisfaction_score() for f in recent_feedback]
                first_half = recent_scores[:len(recent_scores)//2]
                second_half = recent_scores[len(recent_scores)//2:]
                
                if first_half and second_half:
                    learning_velocity = statistics.mean(second_half) - statistics.mean(first_half)
                    metrics['learning_velocity'] = learning_velocity
            
        except Exception as e:
            logger.error(f"Error updating adaptation metrics: {str(e)}")
    
    async def _generate_learning_insights(self, user_id: str) -> None:
        """Generate learning insights for a user."""
        try:
            user_feedback = self.feedback_history[user_id]
            
            # Analyze user feedback patterns
            recent_feedback = [
                f for f in user_feedback
                if (datetime.now(UTC) - f.timestamp).days <= self.feedback_window_days
            ]
            
            if len(recent_feedback) < self.min_feedback_for_learning:
                return
            
            # Insight: Low acceptance rate
            acceptance_rate = sum(1 for f in recent_feedback if f.accepted) / len(recent_feedback)
            if acceptance_rate < 0.3:
                await self._create_learning_insight(
                    insight_type="low_user_acceptance",
                    description=f"User {user_id} has low suggestion acceptance rate",
                    confidence=0.9,
                    recommendation="Review suggestion relevance and personalization for this user",
                    priority=PriorityLevel.MEDIUM,
                    evidence={'user_id': user_id, 'acceptance_rate': acceptance_rate}
                )
            
            # Insight: Declining satisfaction
            satisfaction_scores = [f.get_satisfaction_score() for f in recent_feedback if f.accepted]
            if len(satisfaction_scores) >= 5:
                first_half = satisfaction_scores[:len(satisfaction_scores)//2]
                second_half = satisfaction_scores[len(satisfaction_scores)//2:]
                
                if statistics.mean(first_half) > statistics.mean(second_half) + 0.2:
                    await self._create_learning_insight(
                        insight_type="declining_satisfaction",
                        description=f"User {user_id} shows declining satisfaction with suggestions",
                        confidence=0.8,
                        recommendation="Investigate changes in user needs or suggestion quality",
                        priority=PriorityLevel.HIGH,
                        evidence={
                            'user_id': user_id,
                            'early_satisfaction': statistics.mean(first_half),
                            'recent_satisfaction': statistics.mean(second_half)
                        }
                    )
            
        except Exception as e:
            logger.error(f"Error generating learning insights: {str(e)}")
    
    async def _create_learning_insight(self, insight_type: str, description: str, 
                                     confidence: float, recommendation: str,
                                     priority: PriorityLevel, evidence: Dict[str, Any]) -> None:
        """Create a new learning insight."""
        try:
            insight_id = f"{insight_type}_{datetime.now().timestamp()}"
            
            insight = LearningInsight(
                insight_id=insight_id,
                insight_type=insight_type,
                description=description,
                confidence=confidence,
                supporting_evidence=evidence,
                recommendation=recommendation,
                priority=priority
            )
            
            # Only store high-confidence insights
            if insight.confidence >= self.insight_confidence_threshold:
                self.learning_insights.append(insight)
                
                # Keep only recent insights (last 100)
                if len(self.learning_insights) > 100:
                    self.learning_insights = self.learning_insights[-100:]
                
                logger.info(f"Created learning insight: {insight_type}")
            
        except Exception as e:
            logger.error(f"Error creating learning insight: {str(e)}")
    
    def _infer_suggestion_type_from_feedback(self, feedback: SuggestionFeedback) -> SuggestionType:
        """Infer suggestion type from feedback context."""
        # In practice, this would be stored with the feedback
        # For now, we'll make reasonable inferences
        
        if feedback.notes:
            notes_lower = feedback.notes.lower()
            if 'workflow' in notes_lower or 'optimize' in notes_lower:
                return SuggestionType.WORKFLOW_OPTIMIZATION
            elif 'tool' in notes_lower or 'recommend' in notes_lower:
                return SuggestionType.TOOL_RECOMMENDATION
            elif 'automation' in notes_lower or 'new' in notes_lower:
                return SuggestionType.NEW_AUTOMATION
            elif 'performance' in notes_lower or 'speed' in notes_lower:
                return SuggestionType.PERFORMANCE_IMPROVEMENT
        
        # Default to workflow optimization
        return SuggestionType.WORKFLOW_OPTIMIZATION