"""
Workflow intelligence type definitions and contracts for AI-powered workflow analysis.

This module defines the comprehensive type system for intelligent workflow analysis,
natural language processing, pattern recognition, and optimization capabilities.

Security: Enterprise-grade workflow analysis with privacy compliance and secure processing.
Performance: <500ms analysis time, <2s NLP processing, optimized intelligence algorithms.
Type Safety: Complete workflow intelligence framework with contract-driven development.
"""

from __future__ import annotations
from typing import NewType, Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
import uuid

from .types import ToolId, UserId, Permission
from .contracts import require, ensure
from .either import Either
from .errors import ValidationError, AnalyticsError, SecurityError


# Branded Types for Workflow Intelligence
WorkflowIntelligenceId = NewType('WorkflowIntelligenceId', str)
PatternId = NewType('PatternId', str)
OptimizationId = NewType('OptimizationId', str)
RecommendationId = NewType('RecommendationId', str)
AnalysisSessionId = NewType('AnalysisSessionId', str)
WorkflowTemplateId = NewType('WorkflowTemplateId', str)


class WorkflowComplexity(Enum):
    """Workflow complexity levels."""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class OptimizationGoal(Enum):
    """Workflow optimization goals."""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"
    COST = "cost"
    SIMPLICITY = "simplicity"
    MAINTAINABILITY = "maintainability"


class IntelligenceLevel(Enum):
    """AI intelligence analysis levels."""
    BASIC = "basic"
    STANDARD = "standard"
    SMART = "smart"
    AI_POWERED = "ai_powered"
    ML_ENHANCED = "ml_enhanced"


class PatternType(Enum):
    """Types of workflow patterns."""
    EFFICIENCY = "efficiency"
    REUSABILITY = "reusability"
    COMPLEXITY = "complexity"
    INNOVATION = "innovation"
    ANTI_PATTERN = "anti_pattern"
    BEST_PRACTICE = "best_practice"


class WorkflowIntent(Enum):
    """Recognized workflow intents from NLP."""
    AUTOMATION = "automation"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    FILE_MANAGEMENT = "file_management"
    SYSTEM_CONTROL = "system_control"
    MONITORING = "monitoring"
    INTEGRATION = "integration"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"


class OptimizationImpact(Enum):
    """Impact levels for optimizations."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class WorkflowComponent:
    """Individual workflow component definition."""
    component_id: str
    component_type: str  # action|condition|trigger|group
    name: str
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    estimated_execution_time: timedelta
    reliability_score: float  # 0.0 to 1.0
    complexity_score: float   # 0.0 to 1.0
    
    def __post_init__(self):
        if not (0.0 <= self.reliability_score <= 1.0):
            raise ValidationError("reliability_score", self.reliability_score, "must be between 0.0 and 1.0")
        
        if not (0.0 <= self.complexity_score <= 1.0):
            raise ValidationError("complexity_score", self.complexity_score, "must be between 0.0 and 1.0")


@dataclass(frozen=True)
class WorkflowPattern:
    """Identified workflow pattern."""
    pattern_id: PatternId
    pattern_type: PatternType
    name: str
    description: str
    components: List[WorkflowComponent]
    usage_frequency: float
    effectiveness_score: float  # 0.0 to 1.0
    complexity_reduction: float
    reusability_score: float   # 0.0 to 1.0
    detected_in_workflows: List[str]
    template_generated: bool
    confidence_score: float    # 0.0 to 1.0
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        for score_name, score_value in [
            ("effectiveness_score", self.effectiveness_score),
            ("reusability_score", self.reusability_score),
            ("confidence_score", self.confidence_score)
        ]:
            if not (0.0 <= score_value <= 1.0):
                raise ValidationError(score_name, score_value, "must be between 0.0 and 1.0")


@dataclass(frozen=True)
class OptimizationRecommendation:
    """Workflow optimization recommendation."""
    recommendation_id: RecommendationId
    optimization_id: OptimizationId
    title: str
    description: str
    optimization_goals: List[OptimizationGoal]
    impact_level: OptimizationImpact
    implementation_effort: WorkflowComplexity
    expected_improvement: Dict[str, float]  # metric -> improvement percentage
    before_components: List[WorkflowComponent]
    after_components: List[WorkflowComponent]
    implementation_steps: List[str]
    risks_and_considerations: List[str]
    confidence_score: float  # 0.0 to 1.0
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValidationError("confidence_score", self.confidence_score, "must be between 0.0 and 1.0")


@dataclass(frozen=True)
class NLPProcessingResult:
    """Result of natural language processing."""
    processing_id: str
    original_text: str
    identified_intent: WorkflowIntent
    extracted_entities: Dict[str, List[str]]
    suggested_components: List[WorkflowComponent]
    suggested_tools: List[str]
    complexity_estimate: WorkflowComplexity
    confidence_score: float  # 0.0 to 1.0
    processing_time_ms: float
    language_detected: str
    processed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValidationError("confidence_score", self.confidence_score, "must be between 0.0 and 1.0")


@dataclass(frozen=True)
class WorkflowAnalysisResult:
    """Comprehensive workflow analysis result."""
    analysis_id: AnalysisSessionId
    workflow_id: str
    analysis_depth: IntelligenceLevel
    quality_score: float  # 0.0 to 1.0
    complexity_analysis: Dict[str, Any]
    performance_prediction: Dict[str, float]
    optimization_opportunities: List[OptimizationRecommendation]
    identified_patterns: List[WorkflowPattern]
    anti_patterns_detected: List[WorkflowPattern]
    cross_tool_dependencies: Dict[str, List[str]]
    resource_requirements: Dict[str, Any]
    reliability_assessment: Dict[str, float]
    maintainability_score: float  # 0.0 to 1.0
    improvement_suggestions: List[str]
    alternative_designs: List[Dict[str, Any]]
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        for score_name, score_value in [
            ("quality_score", self.quality_score),
            ("maintainability_score", self.maintainability_score)
        ]:
            if not (0.0 <= score_value <= 1.0):
                raise ValidationError(score_name, score_value, "must be between 0.0 and 1.0")


@dataclass(frozen=True)
class IntelligentWorkflowTemplate:
    """AI-generated workflow template."""
    template_id: WorkflowTemplateId
    name: str
    description: str
    category: str
    complexity: WorkflowComplexity
    components: List[WorkflowComponent]
    required_tools: List[str]
    estimated_setup_time: timedelta
    use_cases: List[str]
    customization_options: Dict[str, Any]
    success_rate: float  # 0.0 to 1.0
    adoption_count: int
    effectiveness_rating: float  # 0.0 to 5.0
    generated_from_pattern: Optional[PatternId]
    ai_confidence: float  # 0.0 to 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValidationError("success_rate", self.success_rate, "must be between 0.0 and 1.0")
        
        if not (0.0 <= self.ai_confidence <= 1.0):
            raise ValidationError("ai_confidence", self.ai_confidence, "must be between 0.0 and 1.0")
        
        if not (0.0 <= self.effectiveness_rating <= 5.0):
            raise ValidationError("effectiveness_rating", self.effectiveness_rating, "must be between 0.0 and 5.0")


@dataclass(frozen=True)
class WorkflowIntelligenceConfig:
    """Configuration for workflow intelligence engine."""
    enable_nlp_processing: bool = True
    enable_pattern_recognition: bool = True
    enable_optimization_analysis: bool = True
    enable_predictive_modeling: bool = True
    enable_cross_tool_analysis: bool = True
    minimum_confidence_threshold: float = 0.7
    maximum_analysis_time_ms: int = 5000
    nlp_language_models: List[str] = field(default_factory=lambda: ["en", "auto"])
    optimization_priorities: List[OptimizationGoal] = field(default_factory=lambda: [OptimizationGoal.EFFICIENCY])
    pattern_discovery_threshold: float = 0.8
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    def __post_init__(self):
        if not (0.0 <= self.minimum_confidence_threshold <= 1.0):
            raise ValidationError("minimum_confidence_threshold", self.minimum_confidence_threshold, "must be between 0.0 and 1.0")
        
        if not (0.0 <= self.pattern_discovery_threshold <= 1.0):
            raise ValidationError("pattern_discovery_threshold", self.pattern_discovery_threshold, "must be between 0.0 and 1.0")


# Required permissions for workflow intelligence operations
WORKFLOW_INTELLIGENCE_PERMISSIONS = {
    IntelligenceLevel.BASIC: [Permission.SYSTEM_CONTROL],
    IntelligenceLevel.STANDARD: [Permission.SYSTEM_CONTROL, Permission.FILE_ACCESS],
    IntelligenceLevel.AI_POWERED: [Permission.SYSTEM_CONTROL, Permission.FILE_ACCESS, Permission.NETWORK_ACCESS],
    IntelligenceLevel.ML_ENHANCED: [Permission.SYSTEM_CONTROL, Permission.FILE_ACCESS, Permission.NETWORK_ACCESS, Permission.APPLICATION_CONTROL],
}


# Workflow intelligence utility functions
def create_workflow_intelligence_id(workflow_name: str) -> WorkflowIntelligenceId:
    """Create a unique workflow intelligence identifier."""
    return WorkflowIntelligenceId(f"wi_{workflow_name}_{uuid.uuid4().hex[:8]}")


def create_pattern_id(pattern_type: PatternType) -> PatternId:
    """Create a unique pattern identifier."""
    return PatternId(f"pattern_{pattern_type.value}_{uuid.uuid4().hex[:8]}")


def create_optimization_id(goal: OptimizationGoal) -> OptimizationId:
    """Create a unique optimization identifier."""
    return OptimizationId(f"opt_{goal.value}_{uuid.uuid4().hex[:8]}")


def create_recommendation_id(optimization_id: OptimizationId) -> RecommendationId:
    """Create a unique recommendation identifier."""
    return RecommendationId(f"rec_{optimization_id}_{uuid.uuid4().hex[:8]}")


def create_analysis_session_id() -> AnalysisSessionId:
    """Create a unique analysis session identifier."""
    return AnalysisSessionId(f"analysis_{uuid.uuid4().hex[:8]}")


def create_template_id(category: str) -> WorkflowTemplateId:
    """Create a unique workflow template identifier."""
    return WorkflowTemplateId(f"template_{category}_{uuid.uuid4().hex[:8]}")


def validate_workflow_component(component: WorkflowComponent) -> bool:
    """Validate workflow component structure and constraints."""
    if not component.component_id or len(component.component_id) == 0:
        return False
    
    if not component.name or len(component.name.strip()) == 0:
        return False
    
    if component.component_type not in ["action", "condition", "trigger", "group"]:
        return False
    
    return True


def calculate_workflow_complexity_score(components: List[WorkflowComponent]) -> float:
    """Calculate overall workflow complexity score."""
    if not components:
        return 0.0
    
    # Complexity factors
    component_count_factor = min(1.0, len(components) / 20.0)  # Normalize by expected max
    avg_complexity = sum(c.complexity_score for c in components) / len(components)
    dependency_factor = sum(len(c.dependencies) for c in components) / len(components) / 5.0  # Normalize
    
    # Weighted combination
    overall_complexity = (component_count_factor * 0.3 + 
                         avg_complexity * 0.5 + 
                         min(1.0, dependency_factor) * 0.2)
    
    return min(1.0, max(0.0, overall_complexity))


def estimate_workflow_execution_time(components: List[WorkflowComponent]) -> timedelta:
    """Estimate total workflow execution time."""
    if not components:
        return timedelta(seconds=0)
    
    # Simple sequential execution time estimation
    total_seconds = sum(c.estimated_execution_time.total_seconds() for c in components)
    
    # Add overhead for workflow orchestration (10% overhead)
    total_seconds *= 1.1
    
    return timedelta(seconds=total_seconds)


class WorkflowIntelligenceError(Exception):
    """Base exception for workflow intelligence-related errors."""
    pass


class NLPProcessingError(WorkflowIntelligenceError):
    """Exception for natural language processing errors."""
    pass


class PatternRecognitionError(WorkflowIntelligenceError):
    """Exception for pattern recognition errors."""
    pass


class OptimizationError(WorkflowIntelligenceError):
    """Exception for workflow optimization errors."""
    pass