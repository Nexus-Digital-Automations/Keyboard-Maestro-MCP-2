"""
Autonomous agent management system for intelligent automation.

This package provides comprehensive agent lifecycle management, decision-making,
learning systems, and resource optimization for self-managing automation agents.
"""

from .agent_manager import AgentManager, AutonomousAgent, AgentState, AgentMetrics
from .decision_engine import DecisionEngine, DecisionContext, ActionPlan
from .learning_system import LearningSystem, Pattern, LearningModel
from .goal_manager import GoalManager, GoalDecomposition, GoalMetrics
from .resource_optimizer import ResourceOptimizer, ResourceAllocation, ResourcePrediction
from .communication_hub import CommunicationHub, Message, MessageType

__all__ = [
    # Agent Management
    'AgentManager',
    'AutonomousAgent', 
    'AgentState',
    'AgentMetrics',
    
    # Decision Making
    'DecisionEngine',
    'DecisionContext',
    'ActionPlan',
    
    # Learning System
    'LearningSystem',
    'Pattern',
    'LearningModel',
    
    # Goal Management
    'GoalManager',
    'GoalDecomposition',
    'GoalMetrics',
    
    # Resource Optimization
    'ResourceOptimizer',
    'ResourceAllocation',
    'ResourcePrediction',
    
    # Communication
    'CommunicationHub',
    'Message',
    'MessageType'
]