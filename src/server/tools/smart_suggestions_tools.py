"""
Smart suggestions MCP tool implementation for AI-powered automation optimization.

This tool provides intelligent automation suggestions, workflow optimization 
recommendations, and adaptive learning capabilities that improve over time
based on user behavior and feedback.

Security: All suggestion data includes privacy protection and secure processing.
Performance: Optimized for real-time suggestion generation and learning.
Type Safety: Complete integration with suggestion system architecture.
"""

import mcp.types as mcp
from typing import Dict, List, Optional, Any, Set, Union
import asyncio
import json
from datetime import datetime, UTC

from src.core.suggestion_system import (
    SuggestionType, PriorityLevel, AnalysisDepth, PrivacyLevel,
    IntelligentSuggestion, SuggestionContext, SuggestionFeedback,
    SuggestionError, create_suggestion_context
)
from src.suggestions.behavior_tracker import BehaviorTracker
from src.suggestions.pattern_analyzer import PatternAnalyzer
from src.suggestions.recommendation_engine import RecommendationEngine
from src.suggestions.learning_system import AdaptiveLearningSystem
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger

logger = get_logger(__name__)


class SmartSuggestionsManager:
    """Comprehensive smart suggestions management system."""
    
    def __init__(self):
        self.behavior_tracker = BehaviorTracker()
        self.pattern_analyzer = PatternAnalyzer(self.behavior_tracker)
        self.recommendation_engine: Optional[RecommendationEngine] = None
        self.learning_system = AdaptiveLearningSystem(None, self.pattern_analyzer)
        self.initialized = False
        self.ai_processor = None
    
    async def initialize(self, ai_processor=None) -> Either[SuggestionError, None]:
        """Initialize smart suggestions system with optional AI processor."""
        try:
            self.ai_processor = ai_processor
            self.recommendation_engine = RecommendationEngine(ai_processor, self.pattern_analyzer)
            self.learning_system = AdaptiveLearningSystem(self.recommendation_engine, self.pattern_analyzer)
            self.initialized = True
            
            logger.info("Smart suggestions system initialized successfully")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Smart suggestions initialization failed: {str(e)}")
            return Either.left(SuggestionError.initialization_failed(str(e)))
    
    async def get_suggestions(self, context: SuggestionContext, 
                            suggestion_types: Optional[Set[SuggestionType]] = None,
                            max_suggestions: int = 5) -> Either[SuggestionError, List[IntelligentSuggestion]]:
        """Get intelligent suggestions for user."""
        try:
            if not self.initialized or not self.recommendation_engine:
                return Either.left(SuggestionError.not_initialized())
            
            # Generate suggestions
            suggestions = await self.recommendation_engine.generate_suggestions(
                context, suggestion_types, max_suggestions
            )
            
            # Apply personalization if user has learning profile
            personalized_suggestions = await self.learning_system.personalize_suggestions(
                context.user_id, suggestions
            )
            
            logger.info(f"Generated {len(personalized_suggestions)} suggestions for user {context.user_id}")
            return Either.right(personalized_suggestions)
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return Either.left(SuggestionError.generation_failed(str(e)))
    
    async def process_feedback(self, feedback: SuggestionFeedback) -> Either[SuggestionError, None]:
        """Process user feedback for adaptive learning."""
        try:
            if not self.initialized:
                return Either.left(SuggestionError.not_initialized())
            
            return await self.learning_system.process_feedback(feedback)
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return Either.left(SuggestionError.learning_failed(str(e)))
    
    async def track_user_action(self, user_id: str, action: str, 
                              context: Dict[str, Any]) -> Either[SuggestionError, None]:
        """Track user action for behavior learning."""
        try:
            return await self.behavior_tracker.track_user_action(user_id, action, context)
            
        except Exception as e:
            logger.error(f"Error tracking user action: {str(e)}")
            return Either.left(SuggestionError.learning_failed(str(e)))
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user learning statistics and insights."""
        if not self.initialized:
            return {'error': 'System not initialized'}
        
        return self.learning_system.get_user_learning_stats(user_id)


# Global smart suggestions manager instance
_smart_suggestions_manager = SmartSuggestionsManager()


async def _ensure_initialized():
    """Ensure smart suggestions system is initialized."""
    if not _smart_suggestions_manager.initialized:
        init_result = await _smart_suggestions_manager.initialize()
        if init_result.is_left():
            raise RuntimeError(f"Smart suggestions initialization failed: {init_result.get_left().message}")


@mcp.tool()
async def km_smart_suggestions(
    operation: str,                             # suggest|analyze|optimize|learn|configure|feedback
    context: Optional[Dict] = None,             # Current automation context
    user_id: Optional[str] = None,              # User identifier for personalization
    suggestion_type: str = "all",               # workflow|tools|performance|new_automation|all
    priority_level: str = "medium",             # low|medium|high|critical
    include_experimental: bool = False,         # Include experimental suggestions
    max_suggestions: int = 5,                   # Maximum number of suggestions
    analysis_depth: str = "standard",           # quick|standard|deep|comprehensive
    time_horizon: str = "immediate",            # immediate|short_term|long_term
    learning_mode: bool = True,                 # Enable learning from interaction
    privacy_level: str = "high",                # low|medium|high privacy protection
    ctx = None
) -> Dict[str, Any]:
    """
    AI-powered smart suggestions for automation optimization and workflow improvement.
    
    This tool provides intelligent automation suggestions, learns from user behavior,
    and continuously improves recommendations based on feedback and usage patterns.
    
    Args:
        operation: Type of operation (suggest, analyze, optimize, learn, configure, feedback)
        context: Current automation context and environment
        user_id: User identifier for personalized suggestions
        suggestion_type: Type of suggestions to generate
        priority_level: Minimum priority level for suggestions
        include_experimental: Whether to include experimental suggestions
        max_suggestions: Maximum number of suggestions to return
        analysis_depth: Depth of analysis for suggestion generation
        time_horizon: Time horizon for suggestions (immediate, short_term, long_term)
        learning_mode: Whether to enable learning from this interaction
        privacy_level: Level of privacy protection for user data
        
    Returns:
        Dictionary containing suggestions, analysis, or configuration results
    """
    try:
        await _ensure_initialized()
        
        # Validate required parameters
        if operation not in ['suggest', 'analyze', 'optimize', 'learn', 'configure', 'feedback']:
            return {
                'success': False,
                'error': f'Invalid operation: {operation}',
                'valid_operations': ['suggest', 'analyze', 'optimize', 'learn', 'configure', 'feedback']
            }
        
        # Default user_id if not provided
        if not user_id:
            user_id = "anonymous_user"
        
        # Parse suggestion types
        suggestion_types = None
        if suggestion_type != "all":
            type_mapping = {
                'workflow': SuggestionType.WORKFLOW_OPTIMIZATION,
                'tools': SuggestionType.TOOL_RECOMMENDATION,
                'performance': SuggestionType.PERFORMANCE_IMPROVEMENT,
                'new_automation': SuggestionType.NEW_AUTOMATION,
                'error_prevention': SuggestionType.ERROR_PREVENTION,
                'best_practice': SuggestionType.BEST_PRACTICE
            }
            
            if suggestion_type in type_mapping:
                suggestion_types = {type_mapping[suggestion_type]}
            else:
                return {
                    'success': False,
                    'error': f'Invalid suggestion type: {suggestion_type}',
                    'valid_types': list(type_mapping.keys()) + ['all']
                }
        
        # Parse priority level
        priority_mapping = {
            'low': PriorityLevel.LOW,
            'medium': PriorityLevel.MEDIUM,
            'high': PriorityLevel.HIGH,
            'critical': PriorityLevel.CRITICAL
        }
        
        if priority_level not in priority_mapping:
            return {
                'success': False,
                'error': f'Invalid priority level: {priority_level}',
                'valid_priorities': list(priority_mapping.keys())
            }
        
        # Parse privacy level
        privacy_mapping = {
            'low': PrivacyLevel.LOW,
            'medium': PrivacyLevel.MEDIUM,
            'high': PrivacyLevel.HIGH,
            'maximum': PrivacyLevel.MAXIMUM
        }
        
        privacy_enum = privacy_mapping.get(privacy_level, PrivacyLevel.HIGH)
        
        # Handle different operations
        if operation == 'suggest':
            return await _handle_suggest_operation(
                user_id, context, suggestion_types, max_suggestions, 
                analysis_depth, learning_mode, privacy_enum
            )
        
        elif operation == 'analyze':
            return await _handle_analyze_operation(
                user_id, context, analysis_depth, privacy_enum
            )
        
        elif operation == 'optimize':
            return await _handle_optimize_operation(
                user_id, context, priority_mapping[priority_level], 
                max_suggestions, privacy_enum
            )
        
        elif operation == 'learn':
            return await _handle_learn_operation(
                user_id, context, learning_mode, privacy_enum
            )
        
        elif operation == 'configure':
            return await _handle_configure_operation(
                user_id, context, privacy_enum
            )
        
        elif operation == 'feedback':
            return await _handle_feedback_operation(
                user_id, context, privacy_enum
            )
        
        else:
            return {
                'success': False,
                'error': f'Operation {operation} not implemented'
            }
    
    except Exception as e:
        logger.error(f"Smart suggestions tool error: {str(e)}")
        return {
            'success': False,
            'error': f'Smart suggestions failed: {str(e)}',
            'operation': operation,
            'timestamp': datetime.now(UTC).isoformat()
        }


async def _handle_suggest_operation(
    user_id: str, context: Optional[Dict], suggestion_types: Optional[Set[SuggestionType]],
    max_suggestions: int, analysis_depth: str, learning_mode: bool, privacy_level: PrivacyLevel
) -> Dict[str, Any]:
    """Handle suggestion generation operation."""
    try:
        # Create suggestion context
        suggestion_context = create_suggestion_context(
            user_id=user_id,
            current_automation=context.get('current_automation') if context else None,
            recent_actions=context.get('recent_actions', []) if context else []
        )
        
        # Add additional context data
        if context:
            suggestion_context = SuggestionContext(
                user_id=suggestion_context.user_id,
                current_automation=suggestion_context.current_automation,
                recent_actions=suggestion_context.recent_actions,
                active_tools=set(context.get('active_tools', [])),
                performance_data=context.get('performance_data', {}),
                user_preferences=context.get('user_preferences', {}),
                time_of_day=suggestion_context.time_of_day,
                day_of_week=suggestion_context.day_of_week,
                context_tags=set(context.get('context_tags', [])),
                session_id=suggestion_context.session_id
            )
        
        # Generate suggestions
        suggestions_result = await _smart_suggestions_manager.get_suggestions(
            suggestion_context, suggestion_types, max_suggestions
        )
        
        if suggestions_result.is_left():
            return {
                'success': False,
                'error': suggestions_result.get_left().message,
                'user_id': user_id
            }
        
        suggestions = suggestions_result.get_right()
        
        # Track interaction if learning enabled
        if learning_mode:
            await _smart_suggestions_manager.track_user_action(
                user_id, 'request_suggestions', 
                {'suggestion_types': [t.value for t in suggestion_types] if suggestion_types else ['all']}
            )
        
        # Convert suggestions to serializable format
        suggestions_data = []
        for suggestion in suggestions:
            suggestions_data.append({
                'suggestion_id': suggestion.suggestion_id,
                'type': suggestion.suggestion_type.value,
                'title': suggestion.title,
                'description': suggestion.description,
                'priority': suggestion.priority.value,
                'confidence': suggestion.confidence,
                'potential_impact': suggestion.potential_impact,
                'implementation_effort': suggestion.implementation_effort,
                'suggested_actions': suggestion.suggested_actions,
                'reasoning': suggestion.reasoning,
                'urgency_score': suggestion.get_urgency_score(),
                'personalization_score': suggestion.personalization_score,
                'created_at': suggestion.created_at.isoformat()
            })
        
        return {
            'success': True,
            'operation': 'suggest',
            'user_id': user_id,
            'suggestions': suggestions_data,
            'suggestion_count': len(suggestions_data),
            'context_summary': suggestion_context.get_context_summary(),
            'analysis_depth': analysis_depth,
            'privacy_level': privacy_level.value,
            'timestamp': datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in suggest operation: {str(e)}")
        return {
            'success': False,
            'error': f'Suggestion generation failed: {str(e)}',
            'user_id': user_id
        }


async def _handle_analyze_operation(
    user_id: str, context: Optional[Dict], analysis_depth: str, privacy_level: PrivacyLevel
) -> Dict[str, Any]:
    """Handle user behavior analysis operation."""
    try:
        # Get user patterns and insights
        insights = await _smart_suggestions_manager.pattern_analyzer.analyze_user_patterns(
            user_id, depth=analysis_depth
        )
        
        # Get optimization opportunities
        opportunities = await _smart_suggestions_manager.pattern_analyzer.identify_optimization_opportunities(
            user_id
        )
        
        # Get user learning statistics
        user_stats = _smart_suggestions_manager.get_user_stats(user_id)
        
        return {
            'success': True,
            'operation': 'analyze',
            'user_id': user_id,
            'analysis_depth': analysis_depth,
            'insights': insights,
            'optimization_opportunities': opportunities,
            'user_statistics': user_stats,
            'privacy_level': privacy_level.value,
            'timestamp': datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in analyze operation: {str(e)}")
        return {
            'success': False,
            'error': f'Analysis failed: {str(e)}',
            'user_id': user_id
        }


async def _handle_optimize_operation(
    user_id: str, context: Optional[Dict], priority_level: PriorityLevel,
    max_suggestions: int, privacy_level: PrivacyLevel
) -> Dict[str, Any]:
    """Handle optimization-focused suggestions operation."""
    try:
        # Focus on optimization suggestions
        optimization_types = {
            SuggestionType.WORKFLOW_OPTIMIZATION,
            SuggestionType.PERFORMANCE_IMPROVEMENT,
            SuggestionType.ERROR_PREVENTION
        }
        
        # Create context for optimization
        suggestion_context = create_suggestion_context(
            user_id=user_id,
            current_automation=context.get('current_automation') if context else None,
            recent_actions=context.get('recent_actions', []) if context else []
        )
        
        # Generate optimization-focused suggestions
        suggestions_result = await _smart_suggestions_manager.get_suggestions(
            suggestion_context, optimization_types, max_suggestions
        )
        
        if suggestions_result.is_left():
            return {
                'success': False,
                'error': suggestions_result.get_left().message,
                'user_id': user_id
            }
        
        suggestions = suggestions_result.get_right()
        
        # Filter by priority level
        priority_filtered_suggestions = [
            s for s in suggestions 
            if s.priority.value in ['high', 'critical'] or s.priority == priority_level
        ]
        
        # Convert to serializable format
        optimization_data = []
        for suggestion in priority_filtered_suggestions:
            optimization_data.append({
                'suggestion_id': suggestion.suggestion_id,
                'type': suggestion.suggestion_type.value,
                'title': suggestion.title,
                'description': suggestion.description,
                'priority': suggestion.priority.value,
                'confidence': suggestion.confidence,
                'potential_impact': suggestion.potential_impact,
                'implementation_effort': suggestion.implementation_effort,
                'suggested_actions': suggestion.suggested_actions,
                'urgency_score': suggestion.get_urgency_score()
            })
        
        return {
            'success': True,
            'operation': 'optimize',
            'user_id': user_id,
            'optimizations': optimization_data,
            'optimization_count': len(optimization_data),
            'priority_filter': priority_level.value,
            'privacy_level': privacy_level.value,
            'timestamp': datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in optimize operation: {str(e)}")
        return {
            'success': False,
            'error': f'Optimization failed: {str(e)}',
            'user_id': user_id
        }


async def _handle_learn_operation(
    user_id: str, context: Optional[Dict], learning_mode: bool, privacy_level: PrivacyLevel
) -> Dict[str, Any]:
    """Handle learning and training operation."""
    try:
        if not learning_mode:
            return {
                'success': False,
                'error': 'Learning mode is disabled',
                'user_id': user_id
            }
        
        # Get system learning insights
        learning_insights = await _smart_suggestions_manager.learning_system.get_system_learning_insights()
        
        # Get user learning statistics
        user_stats = _smart_suggestions_manager.get_user_stats(user_id)
        
        # Track learning interaction
        if context:
            await _smart_suggestions_manager.track_user_action(
                user_id, 'learning_interaction', context
            )
        
        return {
            'success': True,
            'operation': 'learn',
            'user_id': user_id,
            'learning_insights': learning_insights,
            'user_learning_stats': user_stats,
            'learning_mode': learning_mode,
            'privacy_level': privacy_level.value,
            'timestamp': datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in learn operation: {str(e)}")
        return {
            'success': False,
            'error': f'Learning operation failed: {str(e)}',
            'user_id': user_id
        }


async def _handle_configure_operation(
    user_id: str, context: Optional[Dict], privacy_level: PrivacyLevel
) -> Dict[str, Any]:
    """Handle system configuration operation."""
    try:
        # Get system configuration
        config = {
            'initialized': _smart_suggestions_manager.initialized,
            'ai_processor_available': _smart_suggestions_manager.ai_processor is not None,
            'privacy_level': privacy_level.value,
            'supported_suggestion_types': [t.value for t in SuggestionType],
            'supported_priority_levels': [p.value for p in PriorityLevel],
            'supported_analysis_depths': ['quick', 'standard', 'deep', 'comprehensive'],
            'max_suggestions_limit': 20,
            'learning_enabled': True
        }
        
        return {
            'success': True,
            'operation': 'configure',
            'user_id': user_id,
            'configuration': config,
            'timestamp': datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in configure operation: {str(e)}")
        return {
            'success': False,
            'error': f'Configuration failed: {str(e)}',
            'user_id': user_id
        }


async def _handle_feedback_operation(
    user_id: str, context: Optional[Dict], privacy_level: PrivacyLevel
) -> Dict[str, Any]:
    """Handle user feedback processing operation."""
    try:
        if not context:
            return {
                'success': False,
                'error': 'Feedback context required',
                'user_id': user_id
            }
        
        # Extract feedback information
        suggestion_id = context.get('suggestion_id')
        accepted = context.get('accepted', False)
        rating = context.get('rating')
        outcome = context.get('outcome')
        notes = context.get('notes')
        
        if not suggestion_id:
            return {
                'success': False,
                'error': 'suggestion_id required for feedback',
                'user_id': user_id
            }
        
        # Create feedback object
        feedback = SuggestionFeedback(
            feedback_id=f"feedback_{datetime.now().timestamp()}",
            suggestion_id=suggestion_id,
            user_id=user_id,
            accepted=accepted,
            rating=rating,
            outcome=outcome,
            notes=notes
        )
        
        # Process feedback
        feedback_result = await _smart_suggestions_manager.process_feedback(feedback)
        
        if feedback_result.is_left():
            return {
                'success': False,
                'error': feedback_result.get_left().message,
                'user_id': user_id
            }
        
        return {
            'success': True,
            'operation': 'feedback',
            'user_id': user_id,
            'suggestion_id': suggestion_id,
            'accepted': accepted,
            'rating': rating,
            'feedback_processed': True,
            'privacy_level': privacy_level.value,
            'timestamp': datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in feedback operation: {str(e)}")
        return {
            'success': False,
            'error': f'Feedback processing failed: {str(e)}',
            'user_id': user_id
        }