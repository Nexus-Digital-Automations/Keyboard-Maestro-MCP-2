"""
Workflow intelligence package for AI-powered workflow analysis and optimization.

This package provides comprehensive workflow intelligence capabilities including:
- Natural language workflow creation and parsing
- AI-powered workflow analysis and optimization  
- Pattern recognition and template matching
- Cross-tool optimization and performance analysis
- Visual workflow intelligence and suggestions
"""

from .workflow_analyzer import WorkflowAnalyzer
from .nlp_processor import NLPProcessor

# Import existing intelligence components if available
try:
    from .pattern_recognizer import PatternRecognizer, WorkflowPattern
    from .optimization_engine import OptimizationEngine, OptimizationRecommendation
    _EXTENDED_COMPONENTS = True
except ImportError:
    _EXTENDED_COMPONENTS = False

__all__ = [
    # Core classes (always available)
    "WorkflowAnalyzer", "NLPProcessor"
]

# Add extended components if available
if _EXTENDED_COMPONENTS:
    __all__.extend([
        "PatternRecognizer", "OptimizationEngine",
        "WorkflowPattern", "OptimizationRecommendation"
    ])