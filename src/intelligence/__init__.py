"""Workflow intelligence package for AI-powered workflow analysis and optimization.

This package provides comprehensive workflow intelligence capabilities including:
- Natural language workflow creation and parsing
- AI-powered workflow analysis and optimization
- Pattern recognition and template matching
- Cross-tool optimization and performance analysis
- Visual workflow intelligence and suggestions
"""

from .nlp_processor import NLPProcessor
from .workflow_analyzer import WorkflowAnalyzer

# Import existing intelligence components if available
try:
    from .optimization_engine import (
        OptimizationEngine,  # noqa: F401 # Used in __all__ if available
        OptimizationRecommendation,  # noqa: F401 # Used in __all__ if available
    )
    from .pattern_recognizer import (  # noqa: F401 # Used in __all__ if available
        PatternRecognizer,
        WorkflowPattern,
    )

    _EXTENDED_COMPONENTS = True
except ImportError:
    _EXTENDED_COMPONENTS = False

__all__ = [
    "NLPProcessor",
    # Core classes (always available)
    "WorkflowAnalyzer",
]

# Add extended components if available
if _EXTENDED_COMPONENTS:
    __all__.extend(
        [
            "OptimizationEngine",
            "OptimizationRecommendation",
            "PatternRecognizer",
            "WorkflowPattern",
        ],
    )
