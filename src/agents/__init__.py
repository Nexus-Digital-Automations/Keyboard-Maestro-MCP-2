"""Autonomous agent management system for intelligent automation.

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
    "ActionPlan",
    # Agent Management
    "AgentManager",
    "AgentMetrics",
    "AgentState",
    "AutonomousAgent",
    # Communication
    "CommunicationHub",
    "DecisionContext",
    # Decision Making
    "DecisionEngine",
    "GoalDecomposition",
    # Goal Management
    "GoalManager",
    "GoalMetrics",
    "LearningModel",
    # Learning System
    "LearningSystem",
    "Message",
    "MessageType",
    "Pattern",
    "ResourceAllocation",
    # Resource Optimization
    "ResourceOptimizer",
    "ResourcePrediction",
]
