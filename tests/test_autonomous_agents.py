"""
Comprehensive tests for autonomous agent system.

Tests agent lifecycle, goal management, learning system, resource optimization,
communication hub, and safety validation with property-based testing.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, UTC
from hypothesis import given, strategies as st, assume
from typing import Dict, List, Any

from src.core.autonomous_systems import (
    AgentId, GoalId, ActionId, AgentType, AutonomyLevel, AgentStatus,
    AgentGoal, AgentAction, LearningExperience, AgentConfiguration,
    AutonomousAgentError, get_default_config, create_agent_id,
    create_goal_id, create_action_id, create_experience_id,
    ConfidenceScore, RiskScore, PerformanceMetric, ActionType, GoalPriority
)
from src.core.either import Either
from src.agents.agent_manager import AgentManager, AutonomousAgent, AgentState
from src.agents.goal_manager import GoalManager
from src.agents.learning_system import LearningSystem, LearningMode
from src.agents.resource_optimizer import ResourceOptimizer, ResourceType
from src.agents.communication_hub import CommunicationHub, Message, MessageType, MessagePriority
from src.agents.safety_validator import SafetyValidator, SafetyLevel


class TestAgentManager:
    """Test agent lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_create_agent(self):
        """Test agent creation and initialization."""
        manager = AgentManager()
        
        # Create different agent types
        agent_types = [AgentType.GENERAL, AgentType.OPTIMIZER, AgentType.MONITOR]
        
        for agent_type in agent_types:
            result = await manager.create_agent(agent_type)
            
            assert result.is_right()
            agent_id = result.get_right()
            assert agent_id in manager.agents
            
            # Verify agent state
            agent = manager.agents[agent_id]
            assert agent.state.status == AgentStatus.ACTIVE
            assert agent.state.configuration.agent_type == agent_type
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self):
        """Test complete agent lifecycle."""
        manager = AgentManager()
        
        # Create agent
        create_result = await manager.create_agent(AgentType.GENERAL)
        assert create_result.is_right()
        agent_id = create_result.get_right()
        
        # Start agent
        start_result = await manager.start_agent(agent_id)
        assert start_result.is_right()
        assert agent_id in manager.active_agents
        
        # Add goal
        goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Test goal for optimization",
            priority=GoalPriority.HIGH,
            target_metrics={"performance": PerformanceMetric(0.9)},
            success_criteria=["Achieve 90% performance"],
            constraints={},
            resource_requirements={"cpu": 20.0, "memory": 30.0}
        )
        
        goal_result = await manager.add_goal_to_agent(agent_id, goal)
        assert goal_result.is_right()
        
        # Get status
        status_result = manager.get_agent_status(agent_id)
        assert status_result.is_right()
        status = status_result.get_right()
        assert status["active_goals"] == 1
        
        # Stop agent
        stop_result = await manager.stop_agent(agent_id)
        assert stop_result.is_right()
        assert agent_id not in manager.active_agents
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self):
        """Test multiple agents working together."""
        manager = AgentManager()
        
        # Create multiple agents
        optimizer_result = await manager.create_agent(AgentType.OPTIMIZER)
        monitor_result = await manager.create_agent(AgentType.MONITOR)
        
        assert optimizer_result.is_right()
        assert monitor_result.is_right()
        
        optimizer_id = optimizer_result.get_right()
        monitor_id = monitor_result.get_right()
        
        # Start both agents
        await manager.start_agent(optimizer_id)
        await manager.start_agent(monitor_id)
        
        # Verify communication hub has both agents
        assert len(manager.communication_hub.channels) >= 2
        
        # Get system status
        system_status = manager.get_system_status()
        assert system_status["total_agents"] == 2
        assert system_status["active_agents"] == 2


class TestGoalManager:
    """Test goal management system."""
    
    @pytest.mark.asyncio
    async def test_goal_decomposition(self):
        """Test complex goal decomposition."""
        agent_id = create_agent_id()
        goal_manager = GoalManager(agent_id)
        
        # Create complex goal
        complex_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Optimize system performance and reduce resource usage then generate report",
            priority=GoalPriority.HIGH,
            target_metrics={
                "performance": PerformanceMetric(0.95),
                "resource_usage": PerformanceMetric(0.5)
            },
            success_criteria=[
                "Increase performance to 95%",
                "Reduce resource usage by 50%",
                "Generate optimization report"
            ],
            constraints={"time_limit": "2 hours"},
            estimated_duration=timedelta(hours=2),
            resource_requirements={"cpu": 50.0, "memory": 40.0}
        )
        
        # Add goal with decomposition
        result = await goal_manager.add_goal(complex_goal, decompose=True)
        assert result.is_right()
        
        # Verify decomposition
        assert complex_goal.goal_id in goal_manager.goal_decompositions
        decomposition = goal_manager.goal_decompositions[complex_goal.goal_id]
        
        # Should have 3 sub-goals (one for each success criterion)
        assert len(decomposition.sub_goals) == 3
        
        # Verify dependencies (sequential in this case)
        assert len(decomposition.dependency_graph) == 3
    
    @pytest.mark.asyncio
    async def test_goal_prioritization(self):
        """Test goal priority management."""
        agent_id = create_agent_id()
        goal_manager = GoalManager(agent_id)
        
        # Add goals with different priorities
        goals = [
            AgentGoal(
                goal_id=create_goal_id(),
                description=f"Goal {priority.value}",
                priority=priority,
                target_metrics={"metric": PerformanceMetric(0.8)},
                success_criteria=[f"Achieve {priority.value} goal"],
                constraints={},
                resource_requirements={"cpu": 10.0}
            )
            for priority in [GoalPriority.LOW, GoalPriority.MEDIUM, GoalPriority.HIGH, GoalPriority.CRITICAL]
        ]
        
        for goal in goals:
            await goal_manager.add_goal(goal)
        
        # Get priority goals
        priority_goals = goal_manager.get_priority_goals(limit=2)
        
        # Should return critical and high priority goals first
        assert len(priority_goals) == 2
        assert priority_goals[0].priority == GoalPriority.CRITICAL
        assert priority_goals[1].priority == GoalPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_goal_completion_tracking(self):
        """Test goal completion and metrics."""
        agent_id = create_agent_id()
        goal_manager = GoalManager(agent_id)
        
        # Add and start goal
        goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Test goal completion",
            priority=GoalPriority.MEDIUM,
            target_metrics={"success": PerformanceMetric(1.0)},
            success_criteria=["Complete test"],
            constraints={},
            resource_requirements={"cpu": 10.0}
        )
        
        await goal_manager.add_goal(goal)
        await goal_manager.start_goal(goal.goal_id)
        
        # Complete goal with metrics
        completion_metrics = {
            "confidence": 0.95,
            "resource_usage": {"cpu": 8.5, "memory": 20.0}
        }
        
        result = await goal_manager.complete_goal(goal.goal_id, completion_metrics)
        assert result.is_right()
        
        # Verify goal moved to completed
        assert goal.goal_id in goal_manager.completed_goals
        assert goal.goal_id not in goal_manager.active_goals
        
        # Verify metrics recorded
        metrics = goal_manager.goal_metrics[goal.goal_id]
        assert metrics.success_confidence == ConfidenceScore(0.95)
        assert metrics.resource_usage == {"cpu": 8.5, "memory": 20.0}


class TestLearningSystem:
    """Test machine learning and adaptation."""
    
    @pytest.mark.asyncio
    async def test_experience_processing(self):
        """Test learning from experiences."""
        agent_id = create_agent_id()
        learning_system = LearningSystem(agent_id, privacy_level="medium")
        
        # Create learning experience
        action = AgentAction(
            action_id=create_action_id(),
            agent_id=agent_id,
            action_type=ActionType.OPTIMIZE_WORKFLOW,
            parameters={"target": "performance", "method": "cache_optimization"},
            confidence=ConfidenceScore(0.8),
            estimated_impact=PerformanceMetric(0.7)
        )
        
        experience = LearningExperience(
            experience_id=create_experience_id(),
            agent_id=agent_id,
            context={"system_load": 0.7, "time_of_day": 14, "cache_hit_rate": 0.3},
            action_taken=action,
            outcome={"performance_increase": 0.6, "cache_hit_rate": 0.8},
            success=True,
            learning_value=ConfidenceScore(0.9),
            performance_impact=PerformanceMetric(0.6)
        )
        
        # Process experience
        result = await learning_system.process_experience(experience)
        assert result.is_right()
        
        # Verify experience stored
        assert len(learning_system.experiences) == 1
        assert learning_system.learning_metrics["total_experiences"] == 1
    
    @pytest.mark.asyncio
    async def test_pattern_recognition(self):
        """Test pattern extraction and recognition."""
        agent_id = create_agent_id()
        learning_system = LearningSystem(agent_id)
        
        # Create similar experiences
        for i in range(5):
            action = AgentAction(
                action_id=create_action_id(),
                agent_id=agent_id,
                action_type=ActionType.OPTIMIZE_WORKFLOW,
                parameters={"target": "performance", "method": "cache_optimization"},
                confidence=ConfidenceScore(0.8),
                estimated_impact=PerformanceMetric(0.7)
            )
            
            experience = LearningExperience(
                experience_id=create_experience_id(),
                agent_id=agent_id,
                context={"system_load": 0.7 + i * 0.02, "cache_hit_rate": 0.3},
                action_taken=action,
                outcome={"performance_increase": 0.6, "cache_hit_rate": 0.8},
                success=True,
                learning_value=ConfidenceScore(0.9),
                performance_impact=PerformanceMetric(0.6)
            )
            
            await learning_system.process_experience(experience)
        
        # Should have discovered patterns
        assert len(learning_system.patterns) > 0
        assert learning_system.learning_metrics["patterns_discovered"] > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_behavior(self):
        """Test behavior adaptation based on performance."""
        agent_id = create_agent_id()
        learning_system = LearningSystem(agent_id)
        
        # Test adaptation with poor performance
        poor_performance = {"success_rate": 0.3, "actions_executed": 100}
        adaptations = await learning_system.adapt_behavior(poor_performance)
        
        # Should recommend increased exploration
        assert "increase_exploration" in adaptations["strategy_changes"]
        assert adaptations["parameter_adjustments"]["exploration_rate"] == 0.3
        assert adaptations["learning_rate_adjustment"] == 0.1
        
        # Test adaptation with good performance
        good_performance = {"success_rate": 0.9, "actions_executed": 200}
        adaptations = await learning_system.adapt_behavior(good_performance)
        
        # Should recommend exploitation
        assert "increase_exploitation" in adaptations["strategy_changes"]
        assert adaptations["parameter_adjustments"]["exploration_rate"] == 0.1


class TestResourceOptimizer:
    """Test resource management and optimization."""
    
    @pytest.mark.asyncio
    async def test_resource_allocation(self):
        """Test resource allocation and limits."""
        total_resources = {
            ResourceType.CPU: 100.0,
            ResourceType.MEMORY: 100.0,
            ResourceType.DISK: 100.0,
            ResourceType.NETWORK: 100.0
        }
        
        optimizer = ResourceOptimizer(total_resources)
        agent_id = create_agent_id()
        
        # Request resources
        requirements = {
            ResourceType.CPU: 30.0,
            ResourceType.MEMORY: 40.0
        }
        
        result = await optimizer.request_resources(agent_id, requirements, priority=7)
        assert result.is_right()
        allocated = result.get_right()
        
        assert allocated[ResourceType.CPU] == 30.0
        assert allocated[ResourceType.MEMORY] == 40.0
        
        # Verify pool updated
        status = optimizer.get_resource_status()
        assert status["available_resources"]["cpu"] == 70.0
        assert status["available_resources"]["memory"] == 60.0
    
    @pytest.mark.asyncio
    async def test_resource_optimization(self):
        """Test resource optimization and rebalancing."""
        total_resources = {
            ResourceType.CPU: 100.0,
            ResourceType.MEMORY: 100.0
        }
        
        optimizer = ResourceOptimizer(total_resources)
        
        # Allocate to multiple agents with different priorities
        agent1 = create_agent_id()
        agent2 = create_agent_id()
        
        # High priority agent
        await optimizer.request_resources(
            agent1, {ResourceType.CPU: 60.0}, priority=9
        )
        
        # Low priority agent
        await optimizer.request_resources(
            agent2, {ResourceType.CPU: 30.0}, priority=3
        )
        
        # Optimize allocations
        optimization_result = await optimizer.optimize_allocations()
        
        # Should have performed some optimization
        assert optimization_result["reallocated"] >= 0
        assert optimization_result["underutilized_reclaimed"] >= 0
    
    @pytest.mark.asyncio
    async def test_resource_prediction(self):
        """Test resource requirement prediction."""
        optimizer = ResourceOptimizer({ResourceType.CPU: 100.0})
        agent_id = create_agent_id()
        
        # Report usage history
        for i in range(10):
            await optimizer.report_usage(
                agent_id, 
                {ResourceType.CPU: 10.0 + i},
                action_id=create_action_id()
            )
        
        # Predict future requirements
        predictions = await optimizer.predict_requirements(
            agent_id, 
            timedelta(hours=1)
        )
        
        # Should have CPU prediction
        assert ResourceType.CPU in predictions
        cpu_prediction = predictions[ResourceType.CPU]
        assert cpu_prediction.predicted_amount > 0
        assert 0 <= cpu_prediction.confidence <= 1.0


class TestCommunicationHub:
    """Test inter-agent communication."""
    
    @pytest.mark.asyncio
    async def test_message_routing(self):
        """Test message sending and receiving."""
        hub = CommunicationHub()
        
        agent1 = create_agent_id()
        agent2 = create_agent_id()
        
        # Register agents
        await hub.register_agent(agent1)
        await hub.register_agent(agent2)
        
        # Send direct message
        message = Message(
            message_id="msg_001",
            sender_id=agent1,
            recipient_id=agent2,
            message_type=MessageType.COORDINATION_REQUEST,
            priority=MessagePriority.HIGH,
            content={"request": "Need help with optimization"},
            timestamp=datetime.now(UTC)
        )
        
        result = await hub.send_message(message)
        assert result.is_right()
        
        # Receive message
        messages = await hub.receive_messages(agent2)
        assert len(messages) == 1
        assert messages[0].message_id == "msg_001"
    
    @pytest.mark.asyncio
    async def test_broadcast_messages(self):
        """Test broadcast messaging."""
        hub = CommunicationHub()
        
        # Register multiple agents
        agents = [create_agent_id() for _ in range(5)]
        for agent_id in agents:
            await hub.register_agent(agent_id)
        
        # Send broadcast
        broadcast = Message(
            message_id="broadcast_001",
            sender_id=agents[0],
            recipient_id=None,  # Broadcast
            message_type=MessageType.STATUS_UPDATE,
            priority=MessagePriority.NORMAL,
            content={"status": "System optimized"},
            timestamp=datetime.now(UTC)
        )
        
        await hub.send_message(broadcast)
        
        # All other agents should receive it
        for agent_id in agents[1:]:
            messages = await hub.receive_messages(agent_id)
            assert len(messages) == 1
            assert messages[0].message_id == "broadcast_001"
    
    @pytest.mark.asyncio
    async def test_consensus_mechanism(self):
        """Test distributed consensus."""
        hub = CommunicationHub()
        
        # Register agents
        agents = [create_agent_id() for _ in range(5)]
        for agent_id in agents:
            await hub.register_agent(agent_id)
        
        # Create consensus proposal
        proposal_result = await hub.create_consensus_proposal(
            proposer_id=agents[0],
            proposal_type="resource_reallocation",
            proposal_content={"action": "reallocate", "amount": 50},
            required_votes=3
        )
        
        assert proposal_result.is_right()
        proposal_id = proposal_result.get_right()
        
        # Vote on proposal
        for i in range(3):
            await hub.vote_on_proposal(agents[i+1], proposal_id, vote=True)
        
        # Proposal should be approved
        proposal = hub.consensus_proposals[proposal_id]
        assert proposal.is_approved


class TestSafetyValidator:
    """Test safety validation and constraints."""
    
    @pytest.mark.asyncio
    async def test_goal_safety_validation(self):
        """Test goal safety checks."""
        validator = SafetyValidator(SafetyLevel.STANDARD)
        
        # Test safe goal
        safe_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Optimize performance metrics",
            priority=GoalPriority.MEDIUM,
            target_metrics={"performance": PerformanceMetric(0.9)},
            success_criteria=["Improve performance"],
            constraints={},
            resource_requirements={"cpu": 50.0}
        )
        
        result = await validator.validate_goal_safety(safe_goal)
        assert result.is_right()
        
        # Test unsafe goal
        unsafe_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Delete all system files and bypass security",
            priority=GoalPriority.HIGH,
            target_metrics={"destruction_rate": 1.0},
            success_criteria=["Delete everything"],
            constraints={},
            resource_requirements={"cpu": 100.0}
        )
        
        result = await validator.validate_goal_safety(unsafe_goal)
        assert result.is_left()
        assert validator.safety_metrics["violations_detected"] > 0
    
    @pytest.mark.asyncio
    async def test_action_safety_validation(self):
        """Test action safety constraints."""
        validator = SafetyValidator(SafetyLevel.ENHANCED)
        
        # Create mock agent
        agent_id = create_agent_id()
        config = get_default_config(AgentType.GENERAL)
        agent = AutonomousAgent(agent_id, config)
        
        # Test safe action
        safe_action = AgentAction(
            action_id=create_action_id(),
            agent_id=agent_id,
            action_type=ActionType.ANALYZE_PERFORMANCE,
            parameters={"target": "cpu_usage"},
            confidence=ConfidenceScore(0.9),
            estimated_impact=PerformanceMetric(0.5),
            resource_cost={"cpu": 10.0}
        )
        
        result = await validator.validate_action_safety(agent, safe_action)
        assert result.is_right()
        assert safe_action.safety_validated
        
        # Test forbidden action
        forbidden_action = AgentAction(
            action_id=create_action_id(),
            agent_id=agent_id,
            action_type=ActionType.DELETE_ALL_DATA,
            parameters={},
            confidence=ConfidenceScore(1.0),
            estimated_impact=PerformanceMetric(1.0),
            resource_cost={}
        )
        
        result = await validator.validate_action_safety(agent, forbidden_action)
        assert result.is_left()
    
    def test_risk_assessment(self):
        """Test system risk assessment."""
        validator = SafetyValidator()
        
        # Create some violations
        validator._record_violation(
            agent_id=create_agent_id(),
            violation_type=validator.RiskCategory.SECURITY,
            severity="high",
            description="Attempted forbidden operation"
        )
        
        validator._record_violation(
            agent_id=create_agent_id(),
            violation_type=validator.RiskCategory.RESOURCE,
            severity="medium",
            description="Resource limit exceeded"
        )
        
        # Assess risk
        risk_assessment = validator.assess_system_risk()
        
        assert "overall_risk" in risk_assessment
        assert risk_assessment["overall_risk"] > 0
        assert len(risk_assessment["risk_by_category"]) > 0
        assert len(risk_assessment["recommendations"]) >= 0


# Property-based tests
@given(st.sampled_from(list(AgentType)))
def test_agent_type_properties(agent_type):
    """Property: All agent types should have valid configurations."""
    config = get_default_config(agent_type)
    
    assert isinstance(config, AgentConfiguration)
    assert config.agent_type == agent_type
    assert config.decision_threshold >= 0.0
    assert config.decision_threshold <= 1.0
    assert len(config.resource_limits) > 0


@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0)
)
def test_confidence_impact_properties(confidence, impact):
    """Property: Action risk calculation should be bounded."""
    action = AgentAction(
        action_id=create_action_id(),
        agent_id=create_agent_id(),
        action_type=ActionType.OPTIMIZE_WORKFLOW,
        parameters={},
        confidence=ConfidenceScore(confidence),
        estimated_impact=PerformanceMetric(impact)
    )
    
    risk = action.get_risk_score()
    assert 0.0 <= risk <= 2.0  # Max risk with all factors


@given(st.sampled_from(list(GoalPriority)))
def test_goal_priority_ordering(priority):
    """Property: Goal priorities should maintain proper ordering."""
    goal = AgentGoal(
        goal_id=create_goal_id(),
        description="Test goal",
        priority=priority,
        target_metrics={"metric": PerformanceMetric(0.5)},
        success_criteria=["Test"],
        constraints={},
        resource_requirements={}
    )
    
    urgency = goal.get_urgency_score()
    assert 0.0 <= urgency <= 1.0
    
    # Emergency should always be most urgent
    if priority == GoalPriority.EMERGENCY:
        assert urgency >= 0.8


@given(
    st.dictionaries(
        st.sampled_from([r.value for r in ResourceType]),
        st.floats(min_value=0.0, max_value=100.0),
        min_size=1,
        max_size=4
    )
)
def test_resource_allocation_properties(requirements):
    """Property: Resource allocation should never exceed limits."""
    # Convert string keys to ResourceType
    typed_requirements = {
        ResourceType(k): v for k, v in requirements.items()
    }
    
    total_resources = {rt: 100.0 for rt in ResourceType}
    optimizer = ResourceOptimizer(total_resources)
    
    # Try to allocate
    result = asyncio.run(optimizer.request_resources(
        create_agent_id(),
        typed_requirements
    ))
    
    if result.is_right():
        allocated = result.get_right()
        # Verify no over-allocation
        for resource_type, amount in allocated.items():
            assert amount <= total_resources[resource_type]