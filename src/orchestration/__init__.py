"""
Ecosystem orchestration package for coordinating all 46+ automation tools.

This package provides comprehensive orchestration capabilities including:
- Tool registry and capability mapping
- Workflow orchestration and coordination
- Performance monitoring and optimization
- Strategic automation planning
- Enterprise-grade security orchestration
"""

from .ecosystem_architecture import (
    ToolCategory, ExecutionMode, OptimizationTarget,
    ToolDescriptor, WorkflowStep, EcosystemWorkflow,
    SystemPerformanceMetrics, OrchestrationError
)

from .tool_registry import ComprehensiveToolRegistry as ToolRegistry
from .ecosystem_orchestrator import EcosystemOrchestrator

__all__ = [
    # Core types
    "ToolCategory", "ExecutionMode", "OptimizationTarget", 
    "OrchestrationError",
    
    # Data structures
    "ToolDescriptor", "WorkflowStep", "EcosystemWorkflow", 
    "SystemPerformanceMetrics",
    
    # Main classes
    "ToolRegistry", "EcosystemOrchestrator"
]