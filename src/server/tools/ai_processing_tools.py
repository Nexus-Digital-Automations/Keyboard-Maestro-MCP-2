"""
AI Processing Tools - Main module for AI processing capabilities.

This is the main entry point for AI processing tools, providing a unified interface
to the modular AI processing system. This module delegates to specialized sub-modules
for better maintainability and adherence to ADDER+ architectural principles.

Modules:
- ai_core_tools: Core AI processing manager and basic operations
- ai_intelligence_tools: Advanced intelligence operations and batch processing
- ai_model_management: Model listing, caching, and cost optimization

Security: All AI operations include comprehensive validation and threat detection.
Performance: Optimized modular architecture with intelligent resource management.
Type Safety: Complete integration with AI processing architecture.
"""

# Import all functions from specialized modules to maintain API compatibility
from .ai_core_tools import (
    AIProcessingManager,
    ai_manager,
    km_ai_processing,
    km_ai_status,
)
from .ai_intelligence_tools import km_ai_batch, km_ai_intelligence
from .ai_model_management import km_ai_cache, km_ai_cost_optimization, km_ai_models

# Re-export all functions to maintain backward compatibility
__all__ = [
    # Core AI processing
    "km_ai_processing",
    "km_ai_status",
    "ai_manager",
    "AIProcessingManager",
    # AI intelligence and batch processing
    "km_ai_intelligence",
    "km_ai_batch",
    # Model management and optimization
    "km_ai_cache",
    "km_ai_cost_optimization",
    "km_ai_models",
]
