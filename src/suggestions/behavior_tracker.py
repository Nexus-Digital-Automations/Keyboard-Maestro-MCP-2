"""
User behavior tracking and analysis system for intelligent automation optimization.

This module implements comprehensive user behavior tracking, pattern recognition,
and performance analysis to enable AI-powered suggestions and continuous learning
from user automation workflows.

Security: All tracking includes privacy protection and secure data handling.
Performance: Optimized for real-time tracking with minimal overhead.
Type Safety: Complete integration with suggestion system architecture.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta, UTC
from collections import defaultdict, deque
import asyncio
import json
from pathlib import Path

from src.core.suggestion_system import (
    UserBehaviorPattern, AutomationPerformanceMetrics, SuggestionContext,
    SuggestionError, PrivacyLevel, create_behavior_pattern
)
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger

logger = get_logger(__name__)


class BehaviorTracker:
    """Comprehensive user behavior tracking and analysis system."""
    
    def __init__(self, max_patterns_per_user: int = 1000, privacy_level: PrivacyLevel = PrivacyLevel.HIGH):
        self.behavior_patterns: Dict[str, List[UserBehaviorPattern]] = {}
        self.performance_metrics: Dict[str, AutomationPerformanceMetrics] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.action_sequences: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.session_data: Dict[str, Dict[str, Any]] = {}
        self.max_patterns_per_user = max_patterns_per_user
        self.privacy_level = privacy_level
        self._tracking_enabled = True
    
    @require(lambda self, user_id: len(user_id) > 0)
    async def track_user_action(self, user_id: str, action: str, context: Dict[str, Any],
                              session_id: Optional[str] = None) -> Either[SuggestionError, None]:
        """
        Track user action for pattern recognition and behavior analysis.
        
        Args:
            user_id: User identifier for tracking
            action: Action performed by user
            context: Action context and metadata
            session_id: Optional session identifier
            
        Returns:
            Either error or successful tracking confirmation
        """
        try:
            if not self._tracking_enabled:
                return Either.right(None)
            
            # Privacy validation
            if self.privacy_level == PrivacyLevel.MAXIMUM:
                # Maximum privacy: hash user ID and minimal context
                user_id = self._hash_user_id(user_id)
                context = self._minimal_context(context)
            
            # Update action sequence
            self.action_sequences[user_id].append({
                'action': action,
                'timestamp': datetime.now(UTC),
                'context': context,
                'session_id': session_id
            })
            
            # Track session data if provided
            if session_id:
                await self._update_session_data(user_id, session_id, action, context)
            
            # Find or create behavior pattern
            pattern_result = await self._process_behavior_pattern(user_id, action, context)
            if pattern_result.is_left():
                return pattern_result
            
            # Update user preferences based on action
            await self._update_user_preferences(user_id, action, context)
            
            # Clean up old patterns if necessary
            await self._cleanup_old_patterns(user_id)
            
            logger.debug(f"Tracked action '{action}' for user {user_id}")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Error tracking user action: {str(e)}")
            return Either.left(SuggestionError.learning_failed(f"Action tracking failed: {str(e)}"))
    
    @require(lambda self, automation_id: len(automation_id) > 0)
    async def track_automation_performance(self, automation_id: str, execution_result: Dict[str, Any],
                                         user_id: Optional[str] = None) -> Either[SuggestionError, None]:
        """
        Track automation performance metrics for optimization analysis.
        
        Args:
            automation_id: Unique automation identifier
            execution_result: Execution result and performance data
            user_id: Optional user identifier for personalized tracking
            
        Returns:
            Either error or successful tracking confirmation
        """
        try:
            success = execution_result.get('success', False)
            execution_time = execution_result.get('execution_time_ms', 0.0) / 1000.0  # Convert to seconds
            error_occurred = execution_result.get('error') is not None
            timestamp = datetime.now(UTC)
            
            if automation_id in self.performance_metrics:
                # Update existing metrics
                existing = self.performance_metrics[automation_id]
                updated_metrics = self._update_performance_metrics(
                    existing, success, execution_time, error_occurred, timestamp
                )
                self.performance_metrics[automation_id] = updated_metrics
            else:
                # Create new metrics
                resource_usage = execution_result.get('resource_usage', {})
                
                self.performance_metrics[automation_id] = AutomationPerformanceMetrics(
                    automation_id=automation_id,
                    execution_count=1,
                    success_rate=1.0 if success else 0.0,
                    average_execution_time=execution_time,
                    error_frequency=1.0 if error_occurred else 0.0,
                    resource_usage=resource_usage,
                    last_execution=timestamp
                )
            
            # Link performance to user behavior if user provided
            if user_id:
                await self._link_performance_to_behavior(user_id, automation_id, execution_result)
            
            logger.debug(f"Tracked performance for automation {automation_id}")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Error tracking automation performance: {str(e)}")
            return Either.left(SuggestionError.learning_failed(f"Performance tracking failed: {str(e)}"))
    
    @require(lambda self, user_id: len(user_id) > 0)
    def get_user_patterns(self, user_id: str, recent_only: bool = True, 
                         min_frequency: int = 1) -> List[UserBehaviorPattern]:
        """
        Get user behavior patterns with filtering options.
        
        Args:
            user_id: User identifier
            recent_only: Only return patterns from last 7 days
            min_frequency: Minimum pattern frequency threshold
            
        Returns:
            List of filtered behavior patterns sorted by frequency
        """
        try:
            patterns = self.behavior_patterns.get(user_id, [])
            
            # Apply filters
            filtered_patterns = []
            for pattern in patterns:
                if min_frequency > 1 and pattern.frequency < min_frequency:
                    continue
                if recent_only and not pattern.is_recent():
                    continue
                filtered_patterns.append(pattern)
            
            # Sort by frequency and efficiency
            filtered_patterns.sort(
                key=lambda p: (p.frequency, p.get_efficiency_score()), 
                reverse=True
            )
            
            return filtered_patterns
            
        except Exception as e:
            logger.error(f"Error getting user patterns: {str(e)}")
            return []
    
    def get_automation_metrics(self, automation_id: Optional[str] = None) -> Dict[str, AutomationPerformanceMetrics]:
        """
        Get automation performance metrics.
        
        Args:
            automation_id: Optional specific automation ID
            
        Returns:
            Dictionary of automation metrics
        """
        try:
            if automation_id:
                if automation_id in self.performance_metrics:
                    return {automation_id: self.performance_metrics[automation_id]}
                return {}
            
            return self.performance_metrics.copy()
            
        except Exception as e:
            logger.error(f"Error getting automation metrics: {str(e)}")
            return {}
    
    @require(lambda self, user_id: len(user_id) > 0)
    def get_user_activity_summary(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive user activity summary for analysis.
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
            
        Returns:
            Dictionary containing activity summary and insights
        """
        try:
            patterns = self.get_user_patterns(user_id, recent_only=True)
            cutoff_date = datetime.now(UTC) - timedelta(days=days)
            
            # Calculate activity metrics
            total_actions = sum(p.frequency for p in patterns)
            unique_patterns = len(patterns)
            average_success_rate = sum(p.success_rate for p in patterns) / max(1, len(patterns))
            average_completion_time = sum(p.average_completion_time for p in patterns) / max(1, len(patterns))
            
            # Analyze recent action sequences
            recent_actions = []
            if user_id in self.action_sequences:
                for action_data in self.action_sequences[user_id]:
                    if action_data['timestamp'] >= cutoff_date:
                        recent_actions.append(action_data['action'])
            
            # Identify most efficient patterns
            efficient_patterns = [p for p in patterns if p.get_efficiency_score() > 0.8]
            
            # Identify problematic patterns
            problematic_patterns = [p for p in patterns if p.success_rate < 0.7]
            
            summary = {
                'user_id': user_id,
                'analysis_period_days': days,
                'total_actions': total_actions,
                'unique_patterns': unique_patterns,
                'average_success_rate': round(average_success_rate, 3),
                'average_completion_time': round(average_completion_time, 2),
                'recent_actions_count': len(recent_actions),
                'efficient_patterns_count': len(efficient_patterns),
                'problematic_patterns_count': len(problematic_patterns),
                'most_frequent_pattern': patterns[0].pattern_id if patterns else None,
                'activity_trend': self._calculate_activity_trend(user_id, days),
                'preferred_tools': self._get_preferred_tools(user_id),
                'peak_activity_hours': self._get_peak_activity_hours(user_id)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating activity summary: {str(e)}")
            return {'error': str(e)}
    
    async def _process_behavior_pattern(self, user_id: str, action: str, 
                                      context: Dict[str, Any]) -> Either[SuggestionError, None]:
        """Process and update behavior patterns."""
        try:
            if user_id not in self.behavior_patterns:
                self.behavior_patterns[user_id] = []
            
            # Find matching existing pattern
            existing_pattern = self._find_matching_pattern(user_id, action, context)
            
            if existing_pattern:
                # Update existing pattern
                updated_pattern = self._update_pattern(existing_pattern, action, context)
                self._replace_pattern(user_id, existing_pattern, updated_pattern)
            else:
                # Create new pattern
                new_pattern = create_behavior_pattern(
                    user_id=user_id,
                    actions=[action],
                    success_rate=context.get('success', True) and 1.0 or 0.0,
                    completion_time=context.get('execution_time', 0.0)
                )
                
                # Add context tags
                context_tags = set(context.get('tags', []))
                if context.get('tool_name'):
                    context_tags.add(f"tool:{context['tool_name']}")
                
                new_pattern = UserBehaviorPattern(
                    pattern_id=new_pattern.pattern_id,
                    user_id=new_pattern.user_id,
                    action_sequence=new_pattern.action_sequence,
                    frequency=new_pattern.frequency,
                    success_rate=new_pattern.success_rate,
                    average_completion_time=new_pattern.average_completion_time,
                    context_tags=context_tags,
                    last_observed=datetime.now(UTC)
                )
                
                self.behavior_patterns[user_id].append(new_pattern)
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(SuggestionError.learning_failed(f"Pattern processing failed: {str(e)}"))
    
    def _find_matching_pattern(self, user_id: str, action: str, 
                             context: Dict[str, Any]) -> Optional[UserBehaviorPattern]:
        """Find existing pattern that matches current action and context."""
        patterns = self.behavior_patterns.get(user_id, [])
        context_tags = set(context.get('tags', []))
        
        if context.get('tool_name'):
            context_tags.add(f"tool:{context['tool_name']}")
        
        for pattern in patterns:
            # Check if action matches and context is similar
            if (action in pattern.action_sequence and 
                len(pattern.context_tags & context_tags) > 0):
                return pattern
            
            # Also check for exact action sequence match
            if (len(pattern.action_sequence) == 1 and 
                pattern.action_sequence[0] == action):
                return pattern
        
        return None
    
    def _update_pattern(self, existing_pattern: UserBehaviorPattern, action: str, 
                       context: Dict[str, Any]) -> UserBehaviorPattern:
        """Update existing pattern with new observation."""
        # Calculate updated metrics
        new_frequency = existing_pattern.frequency + 1
        
        # Update success rate (moving average)
        action_success = context.get('success', True) and 1.0 or 0.0
        new_success_rate = ((existing_pattern.success_rate * existing_pattern.frequency + action_success) / 
                           new_frequency)
        
        # Update completion time (moving average)
        action_time = context.get('execution_time', existing_pattern.average_completion_time)
        new_completion_time = ((existing_pattern.average_completion_time * existing_pattern.frequency + action_time) / 
                              new_frequency)
        
        # Update context tags
        new_context_tags = existing_pattern.context_tags.copy()
        new_context_tags.update(context.get('tags', []))
        if context.get('tool_name'):
            new_context_tags.add(f"tool:{context['tool_name']}")
        
        # Create updated pattern
        return UserBehaviorPattern(
            pattern_id=existing_pattern.pattern_id,
            user_id=existing_pattern.user_id,
            action_sequence=existing_pattern.action_sequence,
            frequency=new_frequency,
            success_rate=new_success_rate,
            average_completion_time=new_completion_time,
            context_tags=new_context_tags,
            last_observed=datetime.now(UTC),
            confidence_score=min(1.0, existing_pattern.confidence_score + 0.01)  # Increase confidence
        )
    
    def _replace_pattern(self, user_id: str, old_pattern: UserBehaviorPattern, 
                        new_pattern: UserBehaviorPattern) -> None:
        """Replace old pattern with updated pattern."""
        patterns = self.behavior_patterns[user_id]
        for i, pattern in enumerate(patterns):
            if pattern.pattern_id == old_pattern.pattern_id:
                patterns[i] = new_pattern
                break
    
    def _update_performance_metrics(self, existing: AutomationPerformanceMetrics, success: bool,
                                  execution_time: float, error_occurred: bool,
                                  timestamp: datetime) -> AutomationPerformanceMetrics:
        """Update existing performance metrics with new data."""
        new_count = existing.execution_count + 1
        
        # Update success rate (moving average)
        new_success_rate = ((existing.success_rate * existing.execution_count + (1.0 if success else 0.0)) / 
                           new_count)
        
        # Update execution time (moving average)
        new_execution_time = ((existing.average_execution_time * existing.execution_count + execution_time) / 
                             new_count)
        
        # Update error frequency (moving average)
        new_error_frequency = ((existing.error_frequency * existing.execution_count + (1.0 if error_occurred else 0.0)) / 
                              new_count)
        
        # Determine trend direction
        trend_direction = "stable"
        if new_count >= 5:  # Need minimum data for trend analysis
            if new_success_rate > existing.success_rate + 0.1:
                trend_direction = "improving"
            elif new_success_rate < existing.success_rate - 0.1:
                trend_direction = "declining"
        
        return AutomationPerformanceMetrics(
            automation_id=existing.automation_id,
            execution_count=new_count,
            success_rate=new_success_rate,
            average_execution_time=new_execution_time,
            error_frequency=new_error_frequency,
            resource_usage=existing.resource_usage,
            user_satisfaction_score=existing.user_satisfaction_score,
            last_execution=timestamp,
            trend_direction=trend_direction
        )
    
    async def _update_session_data(self, user_id: str, session_id: str, action: str, 
                                 context: Dict[str, Any]) -> None:
        """Update session-specific tracking data."""
        try:
            session_key = f"{user_id}:{session_id}"
            
            if session_key not in self.session_data:
                self.session_data[session_key] = {
                    'start_time': datetime.now(UTC),
                    'actions': [],
                    'tools_used': set(),
                    'success_count': 0,
                    'error_count': 0
                }
            
            session = self.session_data[session_key]
            session['actions'].append({
                'action': action,
                'timestamp': datetime.now(UTC),
                'success': context.get('success', True)
            })
            
            if context.get('tool_name'):
                session['tools_used'].add(context['tool_name'])
            
            if context.get('success', True):
                session['success_count'] += 1
            else:
                session['error_count'] += 1
                
        except Exception as e:
            logger.error(f"Error updating session data: {str(e)}")
    
    async def _update_user_preferences(self, user_id: str, action: str, context: Dict[str, Any]) -> None:
        """Update user preferences based on actions."""
        try:
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {
                    'preferred_tools': defaultdict(int),
                    'preferred_actions': defaultdict(int),
                    'activity_hours': defaultdict(int),
                    'success_patterns': []
                }
            
            prefs = self.user_preferences[user_id]
            
            # Track tool preferences
            if context.get('tool_name'):
                prefs['preferred_tools'][context['tool_name']] += 1
            
            # Track action preferences
            prefs['preferred_actions'][action] += 1
            
            # Track activity time patterns
            current_hour = datetime.now(UTC).hour
            prefs['activity_hours'][current_hour] += 1
            
            # Track successful patterns
            if context.get('success', True):
                prefs['success_patterns'].append({
                    'action': action,
                    'context': context.get('tags', []),
                    'timestamp': datetime.now(UTC)
                })
                
                # Keep only recent success patterns
                cutoff = datetime.now(UTC) - timedelta(days=30)
                prefs['success_patterns'] = [
                    p for p in prefs['success_patterns'] 
                    if p['timestamp'] > cutoff
                ]
                
        except Exception as e:
            logger.error(f"Error updating user preferences: {str(e)}")
    
    async def _cleanup_old_patterns(self, user_id: str) -> None:
        """Clean up old patterns to maintain performance."""
        try:
            if user_id not in self.behavior_patterns:
                return
            
            patterns = self.behavior_patterns[user_id]
            
            # Remove patterns that haven't been observed recently and have low frequency
            cutoff_date = datetime.now(UTC) - timedelta(days=30)
            filtered_patterns = []
            
            for pattern in patterns:
                # Keep pattern if it's recent, frequent, or highly efficient
                if (pattern.last_observed > cutoff_date or 
                    pattern.frequency >= 5 or 
                    pattern.get_efficiency_score() > 0.8):
                    filtered_patterns.append(pattern)
            
            # Limit total patterns per user
            if len(filtered_patterns) > self.max_patterns_per_user:
                # Sort by importance (frequency * efficiency * recency)
                filtered_patterns.sort(
                    key=lambda p: (
                        p.frequency * 
                        p.get_efficiency_score() * 
                        (1.0 if p.is_recent() else 0.5)
                    ), 
                    reverse=True
                )
                filtered_patterns = filtered_patterns[:self.max_patterns_per_user]
            
            self.behavior_patterns[user_id] = filtered_patterns
            
        except Exception as e:
            logger.error(f"Error cleaning up patterns: {str(e)}")
    
    async def _link_performance_to_behavior(self, user_id: str, automation_id: str, 
                                          execution_result: Dict[str, Any]) -> None:
        """Link automation performance to user behavior patterns."""
        try:
            # This could be used to correlate automation performance with user behavior
            # For now, we'll track it in user preferences
            if user_id in self.user_preferences:
                prefs = self.user_preferences[user_id]
                
                if 'automation_performance' not in prefs:
                    prefs['automation_performance'] = {}
                
                prefs['automation_performance'][automation_id] = {
                    'last_result': execution_result,
                    'timestamp': datetime.now(UTC)
                }
                
        except Exception as e:
            logger.error(f"Error linking performance to behavior: {str(e)}")
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for maximum privacy protection."""
        import hashlib
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def _minimal_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract minimal context for maximum privacy."""
        return {
            'success': context.get('success', True),
            'execution_time': context.get('execution_time', 0.0),
            'category': context.get('category', 'unknown')
        }
    
    def _calculate_activity_trend(self, user_id: str, days: int) -> str:
        """Calculate user activity trend over specified days."""
        try:
            if user_id not in self.action_sequences:
                return "no_data"
            
            cutoff_date = datetime.now(UTC) - timedelta(days=days)
            recent_actions = [
                action for action in self.action_sequences[user_id]
                if action['timestamp'] >= cutoff_date
            ]
            
            if len(recent_actions) < 2:
                return "insufficient_data"
            
            # Split into two halves and compare activity levels
            mid_point = len(recent_actions) // 2
            first_half = recent_actions[:mid_point]
            second_half = recent_actions[mid_point:]
            
            first_half_rate = len(first_half) / (days / 2)
            second_half_rate = len(second_half) / (days / 2)
            
            if second_half_rate > first_half_rate * 1.2:
                return "increasing"
            elif second_half_rate < first_half_rate * 0.8:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "unknown"
    
    def _get_preferred_tools(self, user_id: str) -> List[str]:
        """Get user's preferred tools based on usage frequency."""
        try:
            if user_id not in self.user_preferences:
                return []
            
            tool_prefs = self.user_preferences[user_id].get('preferred_tools', {})
            sorted_tools = sorted(tool_prefs.items(), key=lambda x: x[1], reverse=True)
            return [tool for tool, count in sorted_tools[:5]]
            
        except Exception:
            return []
    
    def _get_peak_activity_hours(self, user_id: str) -> List[int]:
        """Get user's peak activity hours."""
        try:
            if user_id not in self.user_preferences:
                return []
            
            hour_prefs = self.user_preferences[user_id].get('activity_hours', {})
            sorted_hours = sorted(hour_prefs.items(), key=lambda x: x[1], reverse=True)
            return [hour for hour, count in sorted_hours[:3]]
            
        except Exception:
            return []