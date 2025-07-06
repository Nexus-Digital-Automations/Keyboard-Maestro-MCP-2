"""
Autonomous agent management system for intelligent automation.

This package provides comprehensive agent lifecycle management, decision-making,
learning systems, and resource optimization for self-managing automation agents.
"""

from .agent_manager import AgentManager, AgentMetrics, AgentState, AutonomousAgent
from .communication_hub import CommunicationHub, Message, MessageType
from .decision_engine import ActionPlan, DecisionContext, DecisionEngine
from .goal_manager import GoalDecomposition, GoalManager, GoalMetrics
from .learning_system import LearningModel, LearningSystem, Pattern
from .resource_optimizer import (
    ResourceAllocation,
    ResourceOptimizer,
    ResourcePrediction,
)

__all__ = [
    # Agent Management
    "AgentManager",
    "AutonomousAgent",
    "AgentState",
    "AgentMetrics",
    # Decision Making
    "DecisionEngine",
    "DecisionContext",
    "ActionPlan",
    # Learning System
    "LearningSystem",
    "Pattern",
    "LearningModel",
    # Goal Management
    "GoalManager",
    "GoalDecomposition",
    "GoalMetrics",
    # Resource Optimization
    "ResourceOptimizer",
    "ResourceAllocation",
    "ResourcePrediction",
    # Communication
    "CommunicationHub",
    "Message",
    "MessageType",
]
