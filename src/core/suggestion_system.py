"""
Smart suggestions system architecture for AI-powered automation optimization.

This module provides comprehensive type definitions and core architecture for the
intelligent suggestion system that learns from user behavior and provides AI-powered
optimization recommendations for automation workflows.

Security: All suggestion data includes privacy protection and secure processing.
Performance: Optimized for real-time suggestion generation and behavior analysis.
Type Safety: Complete branded type system with contract-driven development.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Set, Callable, Type
from enum import Enum
from datetime import datetime, timedelta, UTC
import json
import re
import uuid

from src.core.either import Either
from src.core.errors import ValidationError, SecurityError
from src.core.contracts import require, ensure
from src.core.logging import get_logger

logger = get_logger(__name__)


class SuggestionType(Enum):
    """Types of intelligent suggestions for automation optimization."""
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    NEW_AUTOMATION = "new_automation"
    TOOL_RECOMMENDATION = "tool_recommendation"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    ERROR_PREVENTION = "error_prevention"
    INTEGRATION_OPPORTUNITY = "integration_opportunity"
    BEST_PRACTICE = "best_practice"
    SECURITY_ENHANCEMENT = "security_enhancement"


class PriorityLevel(Enum):
    """Suggestion priority levels for intelligent ranking."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisDepth(Enum):
    """Analysis depth levels for suggestion generation."""
    QUICK = "quick"                # Basic pattern analysis
    STANDARD = "standard"          # Standard analysis with AI
    DEEP = "deep"                  # Comprehensive analysis
    COMPREHENSIVE = "comprehensive" # Full system analysis


class PrivacyLevel(Enum):
    """Privacy protection levels for user data."""
    LOW = "low"                    # Basic anonymization
    MEDIUM = "medium"              # Standard privacy protection
    HIGH = "high"                  # Enhanced privacy with encryption
    MAXIMUM = "maximum"            # Complete data isolation


# Custom errors for suggestion system
class SuggestionError(ValidationError):
    """Base error for suggestion system operations."""
    pass

    @classmethod
    def initialization_failed(cls, message: str) -> 'SuggestionError':
        return cls("SUGGESTION_INIT_FAILED", f"Suggestion system initialization failed: {message}")
    
    @classmethod
    def not_initialized(cls) -> 'SuggestionError':
        return cls("SUGGESTION_NOT_INITIALIZED", "Suggestion system not properly initialized")
    
    @classmethod
    def generation_failed(cls, message: str) -> 'SuggestionError':
        return cls("SUGGESTION_GENERATION_FAILED", f"Suggestion generation failed: {message}")
    
    @classmethod
    def invalid_user_id(cls) -> 'SuggestionError':
        return cls("INVALID_USER_ID", "User ID contains invalid characters")
    
    @classmethod
    def sensitive_data_detected(cls) -> 'SuggestionError':
        return cls("SENSITIVE_DATA_DETECTED", "Sensitive data detected in suggestion context")
    
    @classmethod
    def learning_failed(cls, message: str) -> 'SuggestionError':
        return cls("LEARNING_FAILED", f"Adaptive learning failed: {message}")


@dataclass(frozen=True)
class UserBehaviorPattern:
    """User behavior pattern for intelligent learning and optimization."""
    pattern_id: str
    user_id: str
    action_sequence: List[str]
    frequency: int
    success_rate: float
    average_completion_time: float
    context_tags: Set[str] = field(default_factory=set)
    last_observed: datetime = field(default_factory=lambda: datetime.now(UTC))
    confidence_score: float = 1.0
    
    def __post_init__(self):
        """Contract validation for behavior pattern parameters."""
        if len(self.pattern_id) == 0:
            raise ValueError("Pattern ID cannot be empty")
        if len(self.user_id) == 0:
            raise ValueError("User ID cannot be empty")
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValueError("Success rate must be between 0.0 and 1.0")
        if self.average_completion_time < 0.0:
            raise ValueError("Average completion time cannot be negative")
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
    
    def get_efficiency_score(self) -> float:
        """Calculate pattern efficiency score based on success rate and timing."""
        # Normalize completion time (assuming 60 seconds is baseline)
        time_factor = 1.0 / (1.0 + self.average_completion_time / 60.0)
        efficiency = self.success_rate * time_factor * self.confidence_score
        return min(1.0, efficiency)
    
    def is_recent(self, days: int = 7) -> bool:
        """Check if pattern was observed within specified days."""
        return (datetime.now(UTC) - self.last_observed).days <= days
    
    def get_reliability_score(self) -> float:
        """Calculate reliability score based on frequency and success rate."""
        # More frequent patterns with high success rates are more reliable
        frequency_factor = min(1.0, self.frequency / 50.0)  # Normalize to 50 uses
        return self.success_rate * frequency_factor


@dataclass(frozen=True)
class AutomationPerformanceMetrics:
    """Performance metrics for automation analysis and optimization."""
    automation_id: str
    execution_count: int
    success_rate: float
    average_execution_time: float
    error_frequency: float
    resource_usage: Dict[str, float]
    user_satisfaction_score: Optional[float] = None
    last_execution: datetime = field(default_factory=lambda: datetime.now(UTC))
    trend_direction: str = "stable"  # improving, declining, stable
    
    def __post_init__(self):
        """Contract validation for performance metrics."""
        if len(self.automation_id) == 0:
            raise ValueError("Automation ID cannot be empty")
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValueError("Success rate must be between 0.0 and 1.0")
        if self.average_execution_time < 0.0:
            raise ValueError("Average execution time cannot be negative")
        if not (0.0 <= self.error_frequency <= 1.0):
            raise ValueError("Error frequency must be between 0.0 and 1.0")
        if self.execution_count < 0:
            raise ValueError("Execution count cannot be negative")
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score with weighted metrics."""
        success_weight = 0.4
        speed_weight = 0.3
        reliability_weight = 0.3
        
        # Normalize execution time (assuming 5 seconds is baseline good performance)
        speed_score = max(0.0, 1.0 - (self.average_execution_time / 5.0))
        speed_score = min(1.0, speed_score)
        
        reliability_score = 1.0 - self.error_frequency
        
        performance = (self.success_rate * success_weight + 
                      speed_score * speed_weight + 
                      reliability_score * reliability_weight)
        
        return min(1.0, max(0.0, performance))
    
    def needs_optimization(self, threshold: float = 0.7) -> bool:
        """Check if automation needs optimization based on performance threshold."""
        return self.get_performance_score() < threshold or self.trend_direction == "declining"
    
    def get_optimization_priority(self) -> PriorityLevel:
        """Get optimization priority based on performance metrics."""
        score = self.get_performance_score()
        
        if score < 0.3 or self.error_frequency > 0.5:
            return PriorityLevel.CRITICAL
        elif score < 0.5 or self.error_frequency > 0.2:
            return PriorityLevel.HIGH
        elif score < 0.7 or self.trend_direction == "declining":
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW


@dataclass(frozen=True)
class IntelligentSuggestion:
    """AI-generated intelligent suggestion for automation optimization."""
    suggestion_id: str
    suggestion_type: SuggestionType
    title: str
    description: str
    priority: PriorityLevel
    confidence: float
    potential_impact: str
    implementation_effort: str
    suggested_actions: List[Dict[str, Any]]
    context: Dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = None
    personalization_score: float = 1.0
    
    def __post_init__(self):
        """Contract validation for intelligent suggestion parameters."""
        if len(self.title) == 0:
            raise ValueError("Suggestion title cannot be empty")
        if len(self.description) == 0:
            raise ValueError("Suggestion description cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if len(self.suggested_actions) == 0:
            raise ValueError("Suggested actions cannot be empty")
        if not (0.0 <= self.personalization_score <= 1.0):
            raise ValueError("Personalization score must be between 0.0 and 1.0")
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if suggestion has high confidence score."""
        return self.confidence >= threshold
    
    def get_urgency_score(self) -> float:
        """Calculate urgency score based on priority, confidence, and personalization."""
        priority_weights = {
            PriorityLevel.LOW: 0.25,
            PriorityLevel.MEDIUM: 0.5,
            PriorityLevel.HIGH: 0.75,
            PriorityLevel.CRITICAL: 1.0
        }
        
        base_urgency = priority_weights[self.priority] * self.confidence
        personalized_urgency = base_urgency * self.personalization_score
        
        # Add time decay if suggestion has expiration
        if self.expires_at:
            time_remaining = (self.expires_at - datetime.now(UTC)).total_seconds()
            if time_remaining <= 0:
                return 0.0  # Expired suggestion
            # Increase urgency as expiration approaches
            decay_factor = max(0.5, time_remaining / (24 * 3600))  # 24 hour baseline
            personalized_urgency *= (2.0 - decay_factor)
        
        return min(1.0, personalized_urgency)
    
    def is_expired(self) -> bool:
        """Check if suggestion has expired."""
        if not self.expires_at:
            return False
        return datetime.now(UTC) > self.expires_at
    
    def matches_context(self, context_tags: Set[str]) -> bool:
        """Check if suggestion matches current context tags."""
        suggestion_tags = set(self.context.get('tags', []))
        if not suggestion_tags:
            return True  # Generic suggestions match all contexts
        
        # Suggestion matches if there's overlap in context tags
        return len(suggestion_tags & context_tags) > 0


@dataclass(frozen=True)
class SuggestionContext:
    """Context information for intelligent suggestion generation."""
    user_id: str
    current_automation: Optional[str] = None
    recent_actions: List[str] = field(default_factory=list)
    active_tools: Set[str] = field(default_factory=set)
    performance_data: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    time_of_day: str = ""
    day_of_week: str = ""
    context_tags: Set[str] = field(default_factory=set)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Initialize context with current time information."""
        if len(self.user_id) == 0:
            raise ValueError("User ID cannot be empty")
        
        # Auto-populate time information if not provided
        now = datetime.now()
        if not self.time_of_day:
            object.__setattr__(self, 'time_of_day', now.strftime("%H:%M"))
        if not self.day_of_week:
            object.__setattr__(self, 'day_of_week', now.strftime("%A"))
    
    def get_context_summary(self) -> str:
        """Get a summary of the current context for AI processing."""
        summary_parts = [
            f"User: {self.user_id}",
            f"Time: {self.day_of_week} {self.time_of_day}",
            f"Recent actions: {len(self.recent_actions)}",
            f"Active tools: {len(self.active_tools)}",
        ]
        
        if self.current_automation:
            summary_parts.append(f"Current: {self.current_automation}")
        
        if self.context_tags:
            summary_parts.append(f"Context: {', '.join(list(self.context_tags)[:3])}")
        
        return " | ".join(summary_parts)
    
    def is_work_hours(self) -> bool:
        """Check if current time is during typical work hours."""
        try:
            hour = int(self.time_of_day.split(':')[0])
            return 9 <= hour <= 17 and self.day_of_week not in ['Saturday', 'Sunday']
        except (ValueError, IndexError):
            return False


@dataclass(frozen=True)
class SuggestionFeedback:
    """User feedback on suggestions for adaptive learning."""
    feedback_id: str
    suggestion_id: str
    user_id: str
    accepted: bool
    rating: Optional[int] = None  # 1-5 scale
    outcome: Optional[str] = None
    implementation_time: Optional[float] = None
    effectiveness_score: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Contract validation for suggestion feedback."""
        if len(self.feedback_id) == 0:
            raise ValueError("Feedback ID cannot be empty")
        if len(self.suggestion_id) == 0:
            raise ValueError("Suggestion ID cannot be empty")
        if len(self.user_id) == 0:
            raise ValueError("User ID cannot be empty")
        if self.rating is not None and not (1 <= self.rating <= 5):
            raise ValueError("Rating must be between 1 and 5")
        if self.effectiveness_score is not None and not (0.0 <= self.effectiveness_score <= 1.0):
            raise ValueError("Effectiveness score must be between 0.0 and 1.0")
    
    def get_satisfaction_score(self) -> float:
        """Calculate overall satisfaction score from feedback."""
        if not self.accepted:
            return 0.0
        
        score = 0.5  # Base score for acceptance
        
        if self.rating:
            score += (self.rating / 5.0) * 0.3
        
        if self.effectiveness_score:
            score += self.effectiveness_score * 0.2
        
        return min(1.0, score)


class SuggestionSecurityValidator:
    """Security validation for smart suggestions system with privacy protection."""
    
    SENSITIVE_PATTERNS = [
        r'(?i)(password|secret|token|key|auth)',
        r'(?i)(credit[_\s]*card|ssn|social[_\s]*security)',
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
        r'\b\d{4}[_\s-]*\d{4}[_\s-]*\d{4}[_\s-]*\d{4}\b',  # Credit card format
        r'(?i)(api[_\s]*key|access[_\s]*token)',
        r'(?i)(private[_\s]*key|confidential)',
    ]
    
    def validate_suggestion_context(self, context: SuggestionContext) -> Either[SuggestionError, None]:
        """Validate suggestion context for security and privacy."""
        try:
            # Validate user ID format
            if not self._is_safe_user_id(context.user_id):
                return Either.left(SuggestionError.invalid_user_id())
            
            # Check for sensitive information in context
            if self._contains_sensitive_data(context.performance_data):
                return Either.left(SuggestionError.sensitive_data_detected())
            
            # Validate user preferences for sensitive data
            if self._contains_sensitive_data(context.user_preferences):
                return Either.left(SuggestionError.sensitive_data_detected())
            
            # Check recent actions for sensitive commands
            if self._contains_sensitive_actions(context.recent_actions):
                return Either.left(SuggestionError.sensitive_data_detected())
            
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Context validation error: {str(e)}")
            return Either.left(SuggestionError.generation_failed(f"Context validation failed: {str(e)}"))
    
    def sanitize_suggestion_content(self, suggestion: IntelligentSuggestion) -> IntelligentSuggestion:
        """Sanitize suggestion content to remove potential sensitive information."""
        try:
            # Sanitize title and description
            clean_title = self._sanitize_text(suggestion.title)
            clean_description = self._sanitize_text(suggestion.description)
            
            # Sanitize suggested actions
            clean_actions = []
            for action in suggestion.suggested_actions:
                clean_action = {}
                for key, value in action.items():
                    if isinstance(value, str):
                        clean_action[key] = self._sanitize_text(value)
                    else:
                        clean_action[key] = value
                clean_actions.append(clean_action)
            
            # Sanitize reasoning if present
            clean_reasoning = None
            if suggestion.reasoning:
                clean_reasoning = self._sanitize_text(suggestion.reasoning)
            
            # Create sanitized suggestion
            return IntelligentSuggestion(
                suggestion_id=suggestion.suggestion_id,
                suggestion_type=suggestion.suggestion_type,
                title=clean_title,
                description=clean_description,
                priority=suggestion.priority,
                confidence=suggestion.confidence,
                potential_impact=suggestion.potential_impact,
                implementation_effort=suggestion.implementation_effort,
                suggested_actions=clean_actions,
                context=suggestion.context,
                reasoning=clean_reasoning,
                created_at=suggestion.created_at,
                expires_at=suggestion.expires_at,
                personalization_score=suggestion.personalization_score
            )
            
        except Exception as e:
            logger.error(f"Suggestion sanitization error: {str(e)}")
            return suggestion  # Return original if sanitization fails
    
    def _is_safe_user_id(self, user_id: str) -> bool:
        """Validate user ID format for security."""
        # Allow alphanumeric characters, hyphens, and underscores
        return re.match(r'^[a-zA-Z0-9_-]+$', user_id) is not None and len(user_id) <= 100
    
    def _contains_sensitive_data(self, data: Dict[str, Any]) -> bool:
        """Check for sensitive information in data dictionary."""
        try:
            data_str = json.dumps(data, default=str).lower()
            return any(re.search(pattern, data_str) for pattern in self.SENSITIVE_PATTERNS)
        except Exception:
            return True  # Err on the side of caution
    
    def _contains_sensitive_actions(self, actions: List[str]) -> bool:
        """Check for sensitive actions in action list."""
        sensitive_actions = {
            'delete_all', 'format_disk', 'rm_rf', 'sudo',
            'password', 'login', 'auth', 'secret'
        }
        
        actions_str = ' '.join(actions).lower()
        return any(sensitive in actions_str for sensitive in sensitive_actions)
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text content by masking sensitive patterns."""
        sanitized = text
        
        for pattern in self.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized


# Utility functions for suggestion system
def create_behavior_pattern(user_id: str, actions: List[str], success_rate: float = 1.0, 
                          completion_time: float = 0.0) -> UserBehaviorPattern:
    """Create a validated user behavior pattern."""
    pattern_id = f"{user_id}_{datetime.now().timestamp()}"
    
    return UserBehaviorPattern(
        pattern_id=pattern_id,
        user_id=user_id,
        action_sequence=actions,
        frequency=1,
        success_rate=success_rate,
        average_completion_time=completion_time
    )


def create_suggestion_context(user_id: str, current_automation: Optional[str] = None,
                            recent_actions: Optional[List[str]] = None) -> SuggestionContext:
    """Create a validated suggestion context."""
    return SuggestionContext(
        user_id=user_id,
        current_automation=current_automation,
        recent_actions=recent_actions or [],
        active_tools=set(),
        performance_data={},
        user_preferences={}
    )


def create_performance_metrics(automation_id: str, success_rate: float = 1.0,
                             execution_time: float = 1.0, error_rate: float = 0.0) -> AutomationPerformanceMetrics:
    """Create validated automation performance metrics."""
    return AutomationPerformanceMetrics(
        automation_id=automation_id,
        execution_count=1,
        success_rate=success_rate,
        average_execution_time=execution_time,
        error_frequency=error_rate,
        resource_usage={}
    )