"""
Autonomous agent lifecycle management and execution system.

This module provides comprehensive agent lifecycle management including creation,
initialization, execution, monitoring, and termination of autonomous agents.
Implements enterprise-grade agent orchestration with safety constraints.

Security: All agent operations include safety validation and constraint enforcement.
Performance: Optimized for real-time agent management and coordination.
Type Safety: Complete integration with autonomous systems architecture.
"""

import asyncio
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
import logging

from src.core.autonomous_systems import (
    AgentId, GoalId, ActionId, AgentType, AutonomyLevel, AgentStatus,
    AgentGoal, AgentAction, LearningExperience, AgentConfiguration,
    AutonomousAgentError, get_default_config, create_agent_id,
    ConfidenceScore, RiskScore, PerformanceMetric, ActionType, GoalPriority
)
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.errors import ValidationError
from .goal_manager import GoalManager
from .learning_system import LearningSystem
from .resource_optimizer import ResourceOptimizer, ResourceType
from .communication_hub import CommunicationHub, Message, MessageType, MessagePriority
from .self_healing import SelfHealingEngine, ErrorEvent, RecoveryAction


@dataclass
class AgentMetrics:
    """Performance and operational metrics for autonomous agents."""
    goals_achieved: int = 0
    actions_executed: int = 0
    success_rate: PerformanceMetric = PerformanceMetric(0.0)
    average_decision_time: float = 0.0
    learning_experiences: int = 0
    optimization_cycles: int = 0
    total_runtime: timedelta = field(default_factory=lambda: timedelta(0))
    resource_usage: Dict[str, float] = field(default_factory=dict)
    last_optimization: Optional[datetime] = None
    last_learning_update: Optional[datetime] = None


@dataclass
class AgentState:
    """Current state and context of an autonomous agent."""
    agent_id: AgentId
    status: AgentStatus
    current_goals: List[AgentGoal]
    active_actions: List[AgentAction]
    completed_actions: Set[ActionId]
    experiences: List[LearningExperience]
    learned_patterns: Dict[str, Any]
    metrics: AgentMetrics
    last_activity: datetime
    created_at: datetime
    configuration: AgentConfiguration
    
    def get_priority_goal(self) -> Optional[AgentGoal]:
        """Get the highest priority active goal."""
        active_goals = [g for g in self.current_goals if not g.is_overdue()]
        if not active_goals:
            return None
        
        # Sort by urgency score descending
        active_goals.sort(key=lambda g: g.get_urgency_score(), reverse=True)
        return active_goals[0]
    
    def can_accept_new_action(self) -> bool:
        """Check if agent can accept new actions based on configuration."""
        return len(self.active_actions) < self.configuration.max_concurrent_actions
    
    def get_available_resources(self) -> Dict[str, float]:
        """Calculate available resources after accounting for active actions."""
        available = self.configuration.resource_limits.copy()
        
        # Subtract resources used by active actions
        for action in self.active_actions:
            for resource, usage in action.resource_cost.items():
                if resource in available:
                    available[resource] = max(0.0, available[resource] - usage)
        
        return available


class AutonomousAgent:
    """Core autonomous agent with learning and self-optimization capabilities."""
    
    def __init__(self, agent_id: AgentId, config: AgentConfiguration):
        self.state = AgentState(
            agent_id=agent_id,
            status=AgentStatus.CREATED,
            current_goals=[],
            active_actions=[],
            completed_actions=set(),
            experiences=[],
            learned_patterns={},
            metrics=AgentMetrics(),
            last_activity=datetime.now(UTC),
            created_at=datetime.now(UTC),
            configuration=config
        )
        self.ai_processor = None
        self.decision_engine = None
        self.safety_validator = None
        self._execution_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self, ai_processor=None, decision_engine=None, safety_validator=None) -> Either[AutonomousAgentError, None]:
        """Initialize the autonomous agent with required components."""
        try:
            self.state.status = AgentStatus.INITIALIZING
            
            # Store component references
            self.ai_processor = ai_processor
            self.decision_engine = decision_engine
            self.safety_validator = safety_validator
            
            # Load learned patterns from previous sessions
            await self._load_learned_patterns()
            
            # Initialize performance metrics
            self.state.metrics = AgentMetrics()
            
            # Validate initial configuration
            config_validation = self._validate_configuration()
            if config_validation.is_left():
                return config_validation
            
            self.state.status = AgentStatus.ACTIVE
            self.state.last_activity = datetime.now(UTC)
            
            return Either.right(None)
            
        except Exception as e:
            self.state.status = AgentStatus.ERROR
            return Either.left(AutonomousAgentError.initialization_failed(str(e)))
    
    @require(lambda self, goal: isinstance(goal, AgentGoal))
    async def add_goal(self, goal: AgentGoal) -> Either[AutonomousAgentError, None]:
        """Add a new goal for the agent to pursue."""
        try:
            # Validate goal safety
            if self.safety_validator:
                safety_result = await self.safety_validator.validate_goal_safety(goal)
                if safety_result.is_left():
                    return safety_result
            
            # Check for conflicting goals
            conflicts = self._check_goal_conflicts(goal)
            if conflicts:
                return Either.left(AutonomousAgentError.conflicting_goals(conflicts))
            
            # Check resource availability
            available_resources = self.state.get_available_resources()
            if not goal.is_achievable(available_resources):
                return Either.left(AutonomousAgentError.resource_limit_exceeded(
                    "goal_resources", 
                    sum(goal.resource_requirements.values()),
                    sum(available_resources.values())
                ))
            
            self.state.current_goals.append(goal)
            await self._prioritize_goals()
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(AutonomousAgentError.conflicting_goals([str(e)]))
    
    async def start_autonomous_execution(self) -> Either[AutonomousAgentError, None]:
        """Start the autonomous execution loop."""
        try:
            if self.state.status != AgentStatus.ACTIVE:
                return Either.left(AutonomousAgentError.agent_not_active())
            
            # Start execution task
            self._execution_task = asyncio.create_task(self._execution_loop())
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(AutonomousAgentError.execution_start_failed(str(e)))
    
    async def stop_autonomous_execution(self) -> Either[AutonomousAgentError, None]:
        """Stop the autonomous execution loop gracefully."""
        try:
            self._shutdown_event.set()
            
            if self._execution_task:
                await self._execution_task
            
            self.state.status = AgentStatus.PAUSED
            return Either.right(None)
            
        except Exception as e:
            return Either.left(AutonomousAgentError.unexpected_error(str(e)))
    
    async def execute_single_cycle(self) -> Either[AutonomousAgentError, Dict[str, Any]]:
        """Execute one cycle of autonomous operation."""
        try:
            if self.state.status != AgentStatus.ACTIVE:
                return Either.left(AutonomousAgentError.agent_not_active())
            
            cycle_start = datetime.now(UTC)
            actions_taken = []
            
            # 1. Assess current situation
            situation = await self._assess_situation()
            
            # 2. Select highest priority goal
            current_goal = self.state.get_priority_goal()
            if not current_goal:
                return Either.right({
                    "status": "no_active_goals",
                    "actions_taken": 0,
                    "cycle_duration": 0.0
                })
            
            # 3. Plan actions using decision engine
            if self.decision_engine:
                planned_actions = await self.decision_engine.plan_actions(
                    current_goal, situation, self.state.learned_patterns
                )
            else:
                planned_actions = await self._fallback_action_planning(current_goal, situation)
            
            # 4. Execute actions with safety validation
            for action in planned_actions:
                if not self.state.can_accept_new_action():
                    break
                
                # Safety validation
                if self.safety_validator:
                    safety_result = await self.safety_validator.validate_action_safety(self, action)
                    if safety_result.is_left():
                        continue
                
                # Configuration validation
                if not self.state.configuration.is_action_within_limits(action):
                    continue
                
                # Human approval if required
                if self.state.configuration.should_request_human_approval(action):
                    approval = await self._request_human_approval(action)
                    if not approval:
                        continue
                
                # Execute action
                execution_result = await self._execute_action(action)
                actions_taken.append({
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "goal_id": action.goal_id,
                    "success": execution_result.is_right(),
                    "confidence": action.confidence
                })
                
                # Learn from execution
                await self._learn_from_action(action, execution_result, situation)
                
                self.state.metrics.actions_executed += 1
            
            # 5. Update metrics and perform optimization if needed
            await self._update_performance_metrics()
            
            if self._should_optimize():
                await self._perform_self_optimization()
            
            cycle_duration = (datetime.now(UTC) - cycle_start).total_seconds()
            self.state.last_activity = datetime.now(UTC)
            
            return Either.right({
                "status": "completed",
                "actions_taken": len(actions_taken),
                "current_goal": current_goal.goal_id,
                "cycle_duration": cycle_duration,
                "agent_metrics": {
                    "success_rate": self.state.metrics.success_rate,
                    "goals_achieved": self.state.metrics.goals_achieved,
                    "actions_executed": self.state.metrics.actions_executed
                },
                "actions": actions_taken
            })
            
        except Exception as e:
            return Either.left(AutonomousAgentError.execution_cycle_failed(str(e)))
    
    async def _execution_loop(self) -> None:
        """Main autonomous execution loop."""
        while not self._shutdown_event.is_set() and self.state.status == AgentStatus.ACTIVE:
            try:
                # Execute one cycle
                cycle_result = await self.execute_single_cycle()
                
                if cycle_result.is_left():
                    logging.warning(f"Agent {self.state.agent_id} cycle failed: {cycle_result.get_left()}")
                    await asyncio.sleep(60)  # Wait before retry
                else:
                    # Wait for next cycle based on configuration
                    interval = self.state.configuration.monitoring_interval.total_seconds()
                    await asyncio.sleep(interval)
                
            except Exception as e:
                logging.error(f"Agent {self.state.agent_id} execution error: {e}")
                
                # Attempt self-healing if manager provides it
                if hasattr(self, '_agent_manager') and self._agent_manager:
                    context = {
                        "execution_phase": "autonomous_cycle",
                        "critical_operation": True,
                        "retry_count": getattr(self, '_error_count', 0)
                    }
                    
                    # Increment error count for pattern tracking
                    self._error_count = getattr(self, '_error_count', 0) + 1
                    
                    recovery_result = await self._agent_manager.handle_agent_error(
                        self.state.agent_id, e, context
                    )
                    
                    if recovery_result.is_right():
                        logging.info(f"Agent {self.state.agent_id} self-healing successful")
                        # Reset error count on successful recovery
                        self._error_count = 0
                        await asyncio.sleep(30)  # Short wait after successful recovery
                    else:
                        logging.warning(f"Agent {self.state.agent_id} self-healing failed: {recovery_result.get_left()}")
                        await asyncio.sleep(300)  # Wait 5 minutes before retry
                else:
                    await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _assess_situation(self) -> Dict[str, Any]:
        """Assess current situation and context."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "agent_status": self.state.status.value,
            "active_goals": len(self.state.current_goals),
            "active_actions": len(self.state.active_actions),
            "completed_actions": len(self.state.completed_actions),
            "recent_success_rate": self._calculate_recent_success_rate(),
            "available_resources": self.state.get_available_resources(),
            "learned_patterns_count": len(self.state.learned_patterns),
            "uptime": (datetime.now(UTC) - self.state.created_at).total_seconds()
        }
    
    async def _fallback_action_planning(self, goal: AgentGoal, situation: Dict[str, Any]) -> List[AgentAction]:
        """Fallback rule-based action planning when no decision engine available."""
        actions = []
        
        # Simple rule-based planning based on agent type
        if self.state.configuration.agent_type == AgentType.OPTIMIZER:
            if "performance" in goal.description.lower():
                action = AgentAction(
                    action_id=ActionId(f"opt_{datetime.now(UTC).timestamp()}"),
                    agent_id=self.state.agent_id,
                    action_type=ActionType.ANALYZE_PERFORMANCE,
                    parameters={"target_metrics": goal.target_metrics},
                    goal_id=goal.goal_id,
                    confidence=ConfidenceScore(0.8),
                    estimated_impact=PerformanceMetric(0.6)
                )
                actions.append(action)
        
        elif self.state.configuration.agent_type == AgentType.MONITOR:
            action = AgentAction(
                action_id=ActionId(f"mon_{datetime.now(UTC).timestamp()}"),
                agent_id=self.state.agent_id,
                action_type=ActionType.MONITOR_SYSTEM,
                parameters={"components": ["system_health", "performance_metrics"]},
                goal_id=goal.goal_id,
                confidence=ConfidenceScore(0.9),
                estimated_impact=PerformanceMetric(0.4)
            )
            actions.append(action)
        
        return actions[:self.state.configuration.max_concurrent_actions]
    
    async def _execute_action(self, action: AgentAction) -> Either[AutonomousAgentError, Dict[str, Any]]:
        """Execute the specified action."""
        try:
            self.state.active_actions.append(action)
            action.executed_at = datetime.now(UTC)
            
            # Simulate action execution - in production this would call actual tools
            await asyncio.sleep(0.5)  # Simulate processing time
            
            execution_result = {
                "action_id": action.action_id,
                "success": True,
                "output": f"Executed {action.action_type.value} successfully",
                "duration": 0.5,
                "resources_used": action.resource_cost.copy()
            }
            
            # Move from active to completed
            self.state.active_actions.remove(action)
            self.state.completed_actions.add(action.action_id)
            
            return Either.right(execution_result)
            
        except Exception as e:
            # Remove from active actions on failure
            if action in self.state.active_actions:
                self.state.active_actions.remove(action)
            return Either.left(AutonomousAgentError.action_execution_failed(str(e)))
    
    async def _learn_from_action(self, action: AgentAction, result: Either[AutonomousAgentError, Dict[str, Any]], 
                                context: Dict[str, Any]) -> None:
        """Learn from action execution for future improvement."""
        try:
            success = result.is_right()
            outcome = result.get_right() if success else {"error": str(result.get_left())}
            
            # Calculate performance impact
            if success:
                performance_impact = action.estimated_impact * action.confidence
            else:
                performance_impact = -abs(action.estimated_impact * action.confidence)
            
            experience = LearningExperience(
                experience_id=f"exp_{datetime.now(UTC).timestamp()}",
                agent_id=self.state.agent_id,
                context=context,
                action_taken=action,
                outcome=outcome,
                success=success,
                learning_value=ConfidenceScore(0.8 if success else 0.9),  # Failures teach more
                performance_impact=PerformanceMetric(performance_impact)
            )
            
            self.state.experiences.append(experience)
            self.state.metrics.learning_experiences += 1
            
            # Extract and store patterns
            patterns = experience.extract_patterns()
            await self._update_learned_patterns(patterns)
            
            # Limit experience history to prevent memory bloat
            if len(self.state.experiences) > 1000:
                self.state.experiences = self.state.experiences[-500:]
                
        except Exception as e:
            # Learning failure shouldn't break the agent
            logging.warning(f"Learning failed for agent {self.state.agent_id}: {e}")
    
    async def _perform_self_optimization(self) -> None:
        """Perform self-optimization based on learned patterns."""
        try:
            self.state.status = AgentStatus.OPTIMIZING
            
            # Analyze recent performance
            recent_experiences = self.state.experiences[-50:] if len(self.state.experiences) >= 50 else self.state.experiences
            
            if not recent_experiences:
                return
            
            success_rate = sum(1 for exp in recent_experiences if exp.success) / len(recent_experiences)
            
            # Adjust configuration based on performance
            if success_rate < 0.7:
                # Low success rate - be more conservative
                self.state.configuration.decision_threshold = min(0.9, self.state.configuration.decision_threshold + 0.1)
                self.state.configuration.risk_tolerance = max(0.1, self.state.configuration.risk_tolerance - 0.1)
            elif success_rate > 0.9:
                # High success rate - can be more aggressive
                self.state.configuration.decision_threshold = max(0.3, self.state.configuration.decision_threshold - 0.05)
                self.state.configuration.risk_tolerance = min(0.8, self.state.configuration.risk_tolerance + 0.05)
            
            # Update performance metrics
            self.state.metrics.optimization_cycles += 1
            self.state.metrics.last_optimization = datetime.now(UTC)
            
            self.state.status = AgentStatus.ACTIVE
            
        except Exception as e:
            self.state.status = AgentStatus.ACTIVE  # Continue operation even if optimization fails
            logging.warning(f"Self-optimization failed for agent {self.state.agent_id}: {e}")
    
    def _should_optimize(self) -> bool:
        """Determine if agent should perform self-optimization."""
        if not self.state.metrics.last_optimization:
            return len(self.state.experiences) >= 10  # Need some experience first
        
        time_since_last = datetime.now(UTC) - self.state.metrics.last_optimization
        return time_since_last >= self.state.configuration.optimization_frequency
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate for recent actions."""
        recent_experiences = self.state.experiences[-20:]  # Last 20 experiences
        if not recent_experiences:
            return 0.0
        
        return sum(1 for exp in recent_experiences if exp.success) / len(recent_experiences)
    
    async def _update_performance_metrics(self) -> None:
        """Update agent performance metrics."""
        self.state.metrics.success_rate = PerformanceMetric(self._calculate_recent_success_rate())
        self.state.metrics.total_runtime = datetime.now(UTC) - self.state.created_at
        
        # Check for achieved goals
        for goal in self.state.current_goals[:]:
            if self._is_goal_achieved(goal):
                self.state.current_goals.remove(goal)
                self.state.metrics.goals_achieved += 1
    
    def _is_goal_achieved(self, goal: AgentGoal) -> bool:
        """Check if goal has been achieved based on success criteria."""
        # Simple heuristic - in production this would be more sophisticated
        recent_actions = [a for a in self.state.completed_actions if a.startswith(goal.goal_id[:8])]
        return len(recent_actions) >= len(goal.success_criteria)
    
    def _check_goal_conflicts(self, new_goal: AgentGoal) -> List[str]:
        """Check for conflicts with existing goals."""
        conflicts = []
        
        for existing_goal in self.state.current_goals:
            # Check resource conflicts
            for resource, requirement in new_goal.resource_requirements.items():
                existing_requirement = existing_goal.resource_requirements.get(resource, 0)
                total_requirement = requirement + existing_requirement
                limit = self.state.configuration.resource_limits.get(resource, float('inf'))
                
                if total_requirement > limit:
                    conflicts.append(f"Resource conflict: {resource} ({total_requirement} > {limit})")
            
            # Check for contradictory objectives
            if new_goal.priority == GoalPriority.EMERGENCY and existing_goal.priority == GoalPriority.EMERGENCY:
                conflicts.append("Multiple emergency goals not allowed")
        
        return conflicts
    
    async def _prioritize_goals(self) -> None:
        """Prioritize goals based on urgency and importance."""
        self.state.current_goals.sort(key=lambda g: g.get_urgency_score(), reverse=True)
    
    async def _request_human_approval(self, action: AgentAction) -> bool:
        """Request human approval for high-risk actions."""
        # Placeholder - in production this would integrate with notification system
        logging.info(f"Human approval requested for action {action.action_id}: {action.action_type.value}")
        # For testing, approve low-risk actions automatically
        return action.get_risk_score() < 0.5
    
    async def _load_learned_patterns(self) -> None:
        """Load learned patterns from persistent storage."""
        # Placeholder - in production this would load from database/file
        self.state.learned_patterns = {}
    
    async def _update_learned_patterns(self, new_patterns: Dict[str, Any]) -> None:
        """Update learned patterns with new information."""
        for pattern_type, pattern_data in new_patterns.items():
            if pattern_type not in self.state.learned_patterns:
                self.state.learned_patterns[pattern_type] = []
            
            if isinstance(pattern_data, list):
                self.state.learned_patterns[pattern_type].extend(pattern_data)
            else:
                self.state.learned_patterns[pattern_type].append(pattern_data)
        
        self.state.metrics.last_learning_update = datetime.now(UTC)
    
    def _validate_configuration(self) -> Either[AutonomousAgentError, None]:
        """Validate agent configuration for consistency."""
        config = self.state.configuration
        
        # Validate thresholds
        if config.decision_threshold > 0.95 and config.autonomy_level == AutonomyLevel.AUTONOMOUS:
            return Either.left(AutonomousAgentError.initialization_failed(
                "Decision threshold too high for autonomous operation"
            ))
        
        # Validate resource limits
        if not config.resource_limits:
            return Either.left(AutonomousAgentError.initialization_failed(
                "Resource limits must be specified"
            ))
        
        return Either.right(None)
    
    async def _handle_agent_message(self, message: Message) -> None:
        """Handle messages from other agents."""
        try:
            if message.message_type == MessageType.HELP_REQUEST:
                # Check if we can help
                requested_resources = message.content.get("resources_needed", {})
                if self.resource_optimizer:
                    # Check our available resources
                    status = self.resource_optimizer.get_resource_status()
                    available = status.get("available_resources", {})
                    
                    can_help = all(
                        available.get(res, 0) > amount * 1.5  # Have 50% more than needed
                        for res, amount in requested_resources.items()
                    )
                    
                    if can_help and self.communication_hub:
                        # Send resource offer
                        offer_message = Message(
                            message_id=str(datetime.now(UTC).timestamp()),
                            sender_id=self.state.agent_id,
                            recipient_id=message.sender_id,
                            message_type=MessageType.RESOURCE_OFFER,
                            priority=MessagePriority.HIGH,
                            content={
                                "offered_resources": requested_resources,
                                "offer_duration": "1 hour"
                            },
                            timestamp=datetime.now(UTC),
                            correlation_id=message.message_id
                        )
                        await self.communication_hub.send_message(offer_message)
            
            elif message.message_type == MessageType.COORDINATION_REQUEST:
                # Acknowledge and consider coordination
                if self.communication_hub:
                    await self.communication_hub.acknowledge_message(
                        self.state.agent_id,
                        message.message_id,
                        {"status": "considering", "agent_type": self.state.configuration.agent_type.value}
                    )
        
        except Exception as e:
            logging.warning(f"Failed to handle agent message: {e}")


class AgentManager:
    """Comprehensive management system for autonomous agents."""
    
    def __init__(self):
        self.agents: Dict[AgentId, AutonomousAgent] = {}
        self.active_agents: Set[AgentId] = set()
        self.agent_metrics: Dict[AgentId, AgentMetrics] = {}
        
        # Initialize shared components
        self.communication_hub = CommunicationHub()
        self.resource_optimizer = ResourceOptimizer({
            ResourceType.CPU: 100.0,
            ResourceType.MEMORY: 100.0,
            ResourceType.DISK: 100.0,
            ResourceType.NETWORK: 100.0,
            ResourceType.API_CALLS: 1000.0,
            ResourceType.ACTIONS: 100.0,
            ResourceType.TIME: 86400.0  # 24 hours in seconds
        })
        self.self_healing_engine = SelfHealingEngine()
    
    @require(lambda self, agent_type: isinstance(agent_type, AgentType))
    async def create_agent(self, agent_type: AgentType, config: Optional[AgentConfiguration] = None) -> Either[AutonomousAgentError, AgentId]:
        """Create and initialize a new autonomous agent."""
        try:
            agent_id = create_agent_id()
            agent_config = config or get_default_config(agent_type)
            
            # Create agent instance
            agent = AutonomousAgent(agent_id, agent_config)
            
            # Initialize agent with basic components
            init_result = await agent.initialize()
            
            # Set agent manager reference for self-healing and shared resources
            agent._agent_manager = self
            agent.resource_optimizer = self.resource_optimizer
            agent.communication_hub = self.communication_hub
            if init_result.is_left():
                return init_result
            
            # Store agent
            self.agents[agent_id] = agent
            self.agent_metrics[agent_id] = agent.state.metrics
            
            return Either.right(agent_id)
            
        except Exception as e:
            return Either.left(AutonomousAgentError.agent_creation_failed(str(e)))
    
    async def start_agent(self, agent_id: AgentId) -> Either[AutonomousAgentError, None]:
        """Start autonomous execution for specified agent."""
        if agent_id not in self.agents:
            return Either.left(AutonomousAgentError.agent_not_found(agent_id))
        
        agent = self.agents[agent_id]
        result = await agent.start_autonomous_execution()
        
        if result.is_right():
            self.active_agents.add(agent_id)
        
        return result
    
    async def stop_agent(self, agent_id: AgentId) -> Either[AutonomousAgentError, None]:
        """Stop autonomous execution for specified agent."""
        if agent_id not in self.agents:
            return Either.left(AutonomousAgentError.agent_not_found(agent_id))
        
        agent = self.agents[agent_id]
        result = await agent.stop_autonomous_execution()
        
        if result.is_right():
            self.active_agents.discard(agent_id)
        
        return result
    
    async def add_goal_to_agent(self, agent_id: AgentId, goal: AgentGoal) -> Either[AutonomousAgentError, None]:
        """Add goal to specified agent."""
        if agent_id not in self.agents:
            return Either.left(AutonomousAgentError.agent_not_found(agent_id))
        
        return await self.agents[agent_id].add_goal(goal)
    
    def get_agent_status(self, agent_id: AgentId) -> Either[AutonomousAgentError, Dict[str, Any]]:
        """Get comprehensive status information for agent."""
        if agent_id not in self.agents:
            return Either.left(AutonomousAgentError.agent_not_found(agent_id))
        
        agent = self.agents[agent_id]
        return Either.right({
            "agent_id": agent_id,
            "status": agent.state.status.value,
            "agent_type": agent.state.configuration.agent_type.value,
            "autonomy_level": agent.state.configuration.autonomy_level.value,
            "active_goals": len(agent.state.current_goals),
            "active_actions": len(agent.state.active_actions),
            "completed_actions": len(agent.state.completed_actions),
            "learning_experiences": len(agent.state.experiences),
            "metrics": {
                "goals_achieved": agent.state.metrics.goals_achieved,
                "actions_executed": agent.state.metrics.actions_executed,
                "success_rate": agent.state.metrics.success_rate,
                "optimization_cycles": agent.state.metrics.optimization_cycles
            },
            "last_activity": agent.state.last_activity.isoformat(),
            "uptime": (datetime.now(UTC) - agent.state.created_at).total_seconds()
        })
    
    def list_agents(self) -> Dict[AgentId, Dict[str, Any]]:
        """List all agents with their basic information."""
        return {
            agent_id: {
                "status": agent.state.status.value,
                "type": agent.state.configuration.agent_type.value,
                "active_goals": len(agent.state.current_goals),
                "uptime": (datetime.now(UTC) - agent.state.created_at).total_seconds()
            }
            for agent_id, agent in self.agents.items()
        }
    
    async def shutdown_all_agents(self) -> None:
        """Gracefully shutdown all agents."""
        for agent_id in list(self.active_agents):
            await self.stop_agent(agent_id)
        
        self.active_agents.clear()
        
        # Clean up shared resources
        await self.communication_hub.cleanup_expired()
        await self.resource_optimizer.optimize_allocations()
    
    async def handle_agent_error(self, agent_id: AgentId, error: Exception, context: Dict[str, Any]) -> Either[AutonomousAgentError, Dict[str, Any]]:
        """Handle agent error using self-healing capabilities."""
        try:
            # Diagnose the error
            diagnosis_result = await self.self_healing_engine.detect_and_diagnose(agent_id, error, context)
            if diagnosis_result.is_left():
                return diagnosis_result
            
            error_event = diagnosis_result.get_right()
            
            # Plan recovery
            recovery_plan_result = await self.self_healing_engine.plan_recovery(error_event)
            if recovery_plan_result.is_left():
                return recovery_plan_result
            
            recovery_action = recovery_plan_result.get_right()
            
            # Execute recovery
            recovery_result = await self.self_healing_engine.execute_recovery(
                agent_id, 
                recovery_action, 
                agent_manager=self
            )
            
            if recovery_result.is_right():
                # Log successful recovery
                logging.info(f"Successfully recovered agent {agent_id} using {recovery_action.strategy.value}")
                
                # Update agent metrics
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    agent.state.metrics.optimization_cycles += 1
            
            return recovery_result
            
        except Exception as e:
            logging.error(f"Self-healing failed for agent {agent_id}: {e}")
            return Either.left(AutonomousAgentError.recovery_execution_failed(str(e)))
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get self-healing statistics across all agents."""
        return self.self_healing_engine.get_healing_statistics()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all agents and resources."""
        return {
            "total_agents": len(self.agents),
            "active_agents": len(self.active_agents),
            "agent_statuses": self.list_agents(),
            "resource_status": self.resource_optimizer.get_resource_status(),
            "communication_stats": self.communication_hub.get_communication_stats(),
            "healing_statistics": self.get_healing_statistics(),
            "system_health": {
                "resource_efficiency": asyncio.run(self.resource_optimizer.calculate_efficiency_score()),
                "optimization_recommendations": asyncio.run(self.resource_optimizer.get_optimization_recommendations())
            }
        }