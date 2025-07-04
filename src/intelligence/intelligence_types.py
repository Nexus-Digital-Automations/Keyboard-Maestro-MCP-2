"""
Shared type definitions for the intelligence module.

This module contains common enums and types used across the intelligence system
to avoid circular imports between modules.
"""

from enum import Enum


class IntelligenceOperation(Enum):
    """Intelligence system operation types with comprehensive capabilities."""
    ANALYZE = "analyze"              # Analyze behavior patterns and usage
    LEARN = "learn"                  # Learn from user actions and feedback
    SUGGEST = "suggest"              # Generate intelligent automation suggestions
    OPTIMIZE = "optimize"            # Optimize existing automations for performance
    PREDICT = "predict"              # Predict user intent and future needs
    INSIGHTS = "insights"            # Generate insights about usage and effectiveness


class AnalysisScope(Enum):
    """Scope of behavioral analysis with privacy considerations."""
    USER_BEHAVIOR = "user_behavior"         # Individual user behavior patterns
    AUTOMATION_PATTERNS = "automation_patterns"  # Automation usage patterns across tools
    PERFORMANCE = "performance"             # System performance and efficiency analysis
    USAGE = "usage"                         # Tool usage analytics and trends
    WORKFLOW = "workflow"                   # Workflow efficiency and optimization
    ERROR_PATTERNS = "error_patterns"       # Error occurrence and prevention analysis


class LearningMode(Enum):
    """Machine learning approach modes for adaptive intelligence."""
    ADAPTIVE = "adaptive"                   # Adaptive online learning with real-time updates
    SUPERVISED = "supervised"               # Supervised learning with labeled feedback
    UNSUPERVISED = "unsupervised"          # Unsupervised pattern discovery and clustering
    REINFORCEMENT = "reinforcement"         # Reinforcement learning from user outcomes