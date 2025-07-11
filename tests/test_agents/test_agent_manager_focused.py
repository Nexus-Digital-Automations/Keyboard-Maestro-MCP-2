"""Comprehensive tests for src/agents/agent_manager.py.

This module provides targeted tests for the autonomous agent management system to achieve
significant coverage improvement toward the mandatory 95% threshold by covering
all classes, protocols, methods, and error handling scenarios.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.agents.agent_manager import (
    AgentManager,
    AgentMetrics,
    AgentState,
    AIProcessorProtocol,
    AutonomousAgent,
    DecisionEngineProtocol,
    SafetyValidatorProtocol,
)
from src.core.autonomous_systems import (
    ActionId,
    ActionType,
    AgentAction,
    AgentGoal,
    AgentId,
    AgentStatus,
    AgentType,
    AutonomousAgentError,
    AutonomyLevel,
    ConfidenceScore,
    GoalPriority,
    LearningExperience,
    PerformanceMetric,
    create_agent_id,
    get_default_config,
)
from src.core.either import Either


class TestAgentMetrics:
    """Test AgentMetrics dataclass functionality."""

    def test_agent_metrics_creation_default(self):
        """Test AgentMetrics creation with default values."""
        metrics = AgentMetrics()

        assert metrics.goals_achieved == 0
        assert metrics.actions_executed == 0
        assert metrics.success_rate == PerformanceMetric(0.0)
        assert metrics.average_decision_time == 0.0
        assert metrics.learning_experiences == 0
        assert metrics.optimization_cycles == 0
        assert metrics.total_runtime == timedelta(0)
        assert metrics.resource_usage == {}
        assert metrics.last_optimization is None
        assert metrics.last_learning_update is None

    def test_agent_metrics_creation_with_values(self):
        """Test AgentMetrics creation with specific values."""
        runtime = timedelta(hours=2, minutes=30)
        last_opt = datetime.now(UTC)
        last_learning = datetime.now(UTC)

        metrics = AgentMetrics(
            goals_achieved=5,
            actions_executed=25,
            success_rate=PerformanceMetric(0.85),
            average_decision_time=1.2,
            learning_experiences=10,
            optimization_cycles=3,
            total_runtime=runtime,
            resource_usage={"cpu": 75.5, "memory": 60.2},
            last_optimization=last_opt,
            last_learning_update=last_learning,
        )

        assert metrics.goals_achieved == 5
        assert metrics.actions_executed == 25
        assert metrics.success_rate == PerformanceMetric(0.85)
        assert metrics.average_decision_time == 1.2
        assert metrics.learning_experiences == 10
        assert metrics.optimization_cycles == 3
        assert metrics.total_runtime == runtime
        assert metrics.resource_usage == {"cpu": 75.5, "memory": 60.2}
        assert metrics.last_optimization == last_opt
        assert metrics.last_learning_update == last_learning


class TestAgentState:
    """Test AgentState dataclass functionality comprehensively."""

    @pytest.fixture
    def sample_agent_state(self):
        """Create sample AgentState for testing."""
        agent_id = AgentId("test-agent-001")
        config = get_default_config(AgentType.GENERAL)

        goal1 = AgentGoal(
            goal_id="goal-1",
            description="Test goal 1",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1", "criterion2"],
            constraints={"max_runtime": 3600},
        )

        action1 = AgentAction(
            action_id=ActionId("action-1"),
            agent_id=agent_id,
            action_type=ActionType.ANALYZE_PERFORMANCE,
            parameters={"target": "system"},
            goal_id="goal-1",
            confidence=ConfidenceScore(0.8),
            estimated_impact=PerformanceMetric(0.6),
            resource_cost={"cpu": 5.0},
        )

        experience1 = LearningExperience(
            experience_id="exp-1",
            agent_id=agent_id,
            context={"situation": "test"},
            action_taken=action1,
            outcome={"success": True},
            success=True,
            learning_value=ConfidenceScore(0.9),
            performance_impact=PerformanceMetric(0.7),
        )

        return AgentState(
            agent_id=agent_id,
            status=AgentStatus.ACTIVE,
            current_goals=[goal1],
            active_actions=[action1],
            completed_actions={ActionId("completed-1"), ActionId("completed-2")},
            experiences=[experience1],
            learned_patterns={"pattern1": ["data1", "data2"]},
            metrics=AgentMetrics(goals_achieved=2, actions_executed=10),
            last_activity=datetime.now(UTC),
            created_at=datetime.now(UTC) - timedelta(hours=1),
            configuration=config,
        )

    def test_agent_state_get_priority_goal_with_active_goals(self, sample_agent_state):
        """Test get_priority_goal with active goals."""
        # Add a higher priority goal
        high_priority_goal = AgentGoal(
            goal_id="goal-2",
            description="High priority goal",
            priority=GoalPriority.HIGH,
            target_metrics={"accuracy": 0.95},
            resource_requirements={"cpu": 15.0},
            deadline=datetime.now(UTC) + timedelta(hours=2),
            success_criteria=["high_criterion"],
            constraints={"max_runtime": 7200},
        )
        sample_agent_state.current_goals.append(high_priority_goal)

        priority_goal = sample_agent_state.get_priority_goal()
        assert priority_goal is not None
        assert priority_goal.goal_id == "goal-2"
        assert priority_goal.priority == GoalPriority.HIGH

    def test_agent_state_get_priority_goal_no_active_goals(self, sample_agent_state):
        """Test get_priority_goal with no active goals."""
        sample_agent_state.current_goals.clear()

        priority_goal = sample_agent_state.get_priority_goal()
        assert priority_goal is None

    def test_agent_state_get_priority_goal_overdue_goals(self, sample_agent_state):
        """Test get_priority_goal filtering out overdue goals."""
        # Make the existing goal overdue
        overdue_goal = sample_agent_state.current_goals[0]
        overdue_goal.deadline = datetime.now(UTC) - timedelta(hours=1)  # Past deadline

        priority_goal = sample_agent_state.get_priority_goal()
        # Should return None since the only goal is overdue
        assert priority_goal is None

    def test_agent_state_can_accept_new_action_under_limit(self, sample_agent_state):
        """Test can_accept_new_action when under limit."""
        # Default config has max_concurrent_actions = 5
        sample_agent_state.active_actions = [
            sample_agent_state.active_actions[0]
        ]  # Only 1 action

        can_accept = sample_agent_state.can_accept_new_action()
        assert can_accept is True

    def test_agent_state_can_accept_new_action_at_limit(self, sample_agent_state):
        """Test can_accept_new_action when at limit."""
        # Fill up to max_concurrent_actions limit
        max_actions = sample_agent_state.configuration.max_concurrent_actions
        sample_agent_state.active_actions = [
            sample_agent_state.active_actions[0]
        ] * max_actions

        can_accept = sample_agent_state.can_accept_new_action()
        assert can_accept is False

    def test_agent_state_get_available_resources(self, sample_agent_state):
        """Test get_available_resources calculation."""
        # Set up configuration resource limits
        sample_agent_state.configuration.resource_limits = {
            "cpu": 100.0,
            "memory": 80.0,
        }

        # Active action uses some resources
        sample_agent_state.active_actions[0].resource_cost = {
            "cpu": 30.0,
            "memory": 20.0,
        }

        available = sample_agent_state.get_available_resources()

        assert available["cpu"] == 70.0  # 100 - 30
        assert available["memory"] == 60.0  # 80 - 20

    def test_agent_state_get_available_resources_over_limit(self, sample_agent_state):
        """Test get_available_resources when usage exceeds limits."""
        # Set up configuration resource limits
        sample_agent_state.configuration.resource_limits = {"cpu": 50.0}

        # Active action uses more than available
        sample_agent_state.active_actions[0].resource_cost = {"cpu": 60.0}

        available = sample_agent_state.get_available_resources()

        assert available["cpu"] == 0.0  # Should not go negative


class TestAutonomousAgent:
    """Test AutonomousAgent class functionality comprehensively."""

    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return get_default_config(AgentType.GENERAL)

    @pytest.fixture
    def autonomous_agent(self, agent_config):
        """Create test autonomous agent."""
        agent_id = create_agent_id()
        return AutonomousAgent(agent_id, agent_config)

    @pytest.fixture
    def mock_ai_processor(self):
        """Create mock AI processor."""
        processor = MagicMock(spec=AIProcessorProtocol)
        processor.process = AsyncMock(return_value={"result": "processed"})
        return processor

    @pytest.fixture
    def mock_decision_engine(self):
        """Create mock decision engine."""
        engine = MagicMock(spec=DecisionEngineProtocol)
        engine.make_decision = AsyncMock(return_value={"decision": "proceed"})
        engine.plan_actions = AsyncMock(return_value=[])
        return engine

    @pytest.fixture
    def mock_safety_validator(self):
        """Create mock safety validator."""
        validator = MagicMock(spec=SafetyValidatorProtocol)
        validator.validate = AsyncMock(return_value=True)
        validator.validate_goal_safety = AsyncMock(return_value=Either.right(None))
        validator.validate_action_safety = AsyncMock(return_value=Either.right(None))
        return validator

    def test_autonomous_agent_creation(self, autonomous_agent):
        """Test AutonomousAgent creation and initial state."""
        assert autonomous_agent.state.status == AgentStatus.CREATED
        assert autonomous_agent.state.current_goals == []
        assert autonomous_agent.state.active_actions == []
        assert autonomous_agent.state.completed_actions == set()
        assert autonomous_agent.state.experiences == []
        assert autonomous_agent.state.learned_patterns == {}
        assert isinstance(autonomous_agent.state.metrics, AgentMetrics)
        assert autonomous_agent.ai_processor is None
        assert autonomous_agent.decision_engine is None
        assert autonomous_agent.safety_validator is None

    @pytest.mark.asyncio
    async def test_autonomous_agent_initialize_success(
        self,
        autonomous_agent,
        mock_ai_processor,
        mock_decision_engine,
        mock_safety_validator,
    ):
        """Test successful agent initialization."""
        result = await autonomous_agent.initialize(
            ai_processor=mock_ai_processor,
            decision_engine=mock_decision_engine,
            safety_validator=mock_safety_validator,
        )

        assert result.is_right()
        assert autonomous_agent.state.status == AgentStatus.ACTIVE
        assert autonomous_agent.ai_processor == mock_ai_processor
        assert autonomous_agent.decision_engine == mock_decision_engine
        assert autonomous_agent.safety_validator == mock_safety_validator

    @pytest.mark.asyncio
    async def test_autonomous_agent_initialize_with_none_components(
        self, autonomous_agent
    ):
        """Test agent initialization with None components."""
        result = await autonomous_agent.initialize(
            ai_processor=None,
            decision_engine=None,
            safety_validator=None,
        )

        assert result.is_right()
        assert autonomous_agent.state.status == AgentStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_autonomous_agent_initialize_validation_failure(
        self, autonomous_agent
    ):
        """Test agent initialization with configuration validation failure."""
        # Create config with invalid settings
        autonomous_agent.state.configuration.decision_threshold = (
            0.99  # Too high for autonomous
        )
        autonomous_agent.state.configuration.autonomy_level = AutonomyLevel.AUTONOMOUS

        result = await autonomous_agent.initialize()

        assert result.is_left()
        assert autonomous_agent.state.status == AgentStatus.ERROR

    @pytest.mark.asyncio
    async def test_autonomous_agent_initialize_exception_handling(
        self, autonomous_agent
    ):
        """Test agent initialization exception handling."""
        # Mock _load_learned_patterns to raise exception
        with patch.object(
            autonomous_agent,
            "_load_learned_patterns",
            side_effect=Exception("Load failed"),
        ):
            result = await autonomous_agent.initialize()

            assert result.is_left()
            assert autonomous_agent.state.status == AgentStatus.ERROR

    @pytest.mark.asyncio
    async def test_autonomous_agent_add_goal_success(
        self, autonomous_agent, mock_safety_validator
    ):
        """Test successful goal addition."""
        await autonomous_agent.initialize(safety_validator=mock_safety_validator)

        goal = AgentGoal(
            goal_id="test-goal",
            description="Test goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        result = await autonomous_agent.add_goal(goal)

        assert result.is_right()
        assert len(autonomous_agent.state.current_goals) == 1
        assert autonomous_agent.state.current_goals[0] == goal

    @pytest.mark.asyncio
    async def test_autonomous_agent_add_goal_safety_validation_failure(
        self,
        autonomous_agent,
        mock_safety_validator,
    ):
        """Test goal addition with safety validation failure."""
        mock_safety_validator.validate_goal_safety.return_value = Either.left(
            AutonomousAgentError.safety_violation("Unsafe goal")
        )

        await autonomous_agent.initialize(safety_validator=mock_safety_validator)

        goal = AgentGoal(
            goal_id="unsafe-goal",
            description="Unsafe goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        result = await autonomous_agent.add_goal(goal)

        assert result.is_left()
        assert len(autonomous_agent.state.current_goals) == 0

    @pytest.mark.asyncio
    async def test_autonomous_agent_add_goal_resource_limit_exceeded(
        self, autonomous_agent
    ):
        """Test goal addition with resource limit exceeded."""
        await autonomous_agent.initialize()

        # Create goal that requires more resources than available
        goal = AgentGoal(
            goal_id="resource-heavy-goal",
            description="Resource heavy goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 1000.0},  # Exceeds typical limits
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        result = await autonomous_agent.add_goal(goal)

        assert result.is_left()
        assert len(autonomous_agent.state.current_goals) == 0

    @pytest.mark.asyncio
    async def test_autonomous_agent_add_goal_conflicts_with_existing(
        self, autonomous_agent
    ):
        """Test goal addition with conflicts."""
        await autonomous_agent.initialize()

        # Add first emergency goal
        goal1 = AgentGoal(
            goal_id="emergency-goal-1",
            description="Emergency goal 1",
            priority=GoalPriority.EMERGENCY,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        await autonomous_agent.add_goal(goal1)

        # Try to add second emergency goal (should conflict)
        goal2 = AgentGoal(
            goal_id="emergency-goal-2",
            description="Emergency goal 2",
            priority=GoalPriority.EMERGENCY,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        result = await autonomous_agent.add_goal(goal2)

        assert result.is_left()
        assert len(autonomous_agent.state.current_goals) == 1  # Only first goal added

    @pytest.mark.asyncio
    async def test_autonomous_agent_start_autonomous_execution_success(
        self, autonomous_agent
    ):
        """Test starting autonomous execution."""
        await autonomous_agent.initialize()

        result = await autonomous_agent.start_autonomous_execution()

        assert result.is_right()
        assert autonomous_agent._execution_task is not None

    @pytest.mark.asyncio
    async def test_autonomous_agent_start_autonomous_execution_not_active(
        self, autonomous_agent
    ):
        """Test starting autonomous execution when agent not active."""
        # Agent is in CREATED state, not ACTIVE
        result = await autonomous_agent.start_autonomous_execution()

        assert result.is_left()

    @pytest.mark.asyncio
    async def test_autonomous_agent_stop_autonomous_execution(self, autonomous_agent):
        """Test stopping autonomous execution."""
        await autonomous_agent.initialize()
        await autonomous_agent.start_autonomous_execution()

        result = await autonomous_agent.stop_autonomous_execution()

        assert result.is_right()
        assert autonomous_agent.state.status == AgentStatus.PAUSED

    @pytest.mark.asyncio
    async def test_autonomous_agent_execute_single_cycle_no_goals(
        self, autonomous_agent
    ):
        """Test single cycle execution with no goals."""
        await autonomous_agent.initialize()

        result = await autonomous_agent.execute_single_cycle()

        assert result.is_right()
        cycle_data = result.get_right()
        assert cycle_data["status"] == "no_active_goals"
        assert cycle_data["actions_taken"] == 0

    @pytest.mark.asyncio
    async def test_autonomous_agent_execute_single_cycle_not_active(
        self, autonomous_agent
    ):
        """Test single cycle execution when agent not active."""
        # Agent is in CREATED state
        result = await autonomous_agent.execute_single_cycle()

        assert result.is_left()

    @pytest.mark.asyncio
    async def test_autonomous_agent_execute_single_cycle_with_goal_and_decision_engine(
        self,
        autonomous_agent,
        mock_decision_engine,
    ):
        """Test single cycle execution with goal and decision engine."""
        # Create an action that the decision engine will return
        test_action = AgentAction(
            action_id=ActionId("test-action"),
            agent_id=autonomous_agent.state.agent_id,
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={"components": ["test"]},
            goal_id="test-goal",
            confidence=ConfidenceScore(0.8),
            estimated_impact=PerformanceMetric(0.6),
            resource_cost={"cpu": 5.0},
        )

        mock_decision_engine.plan_actions.return_value = [test_action]

        await autonomous_agent.initialize(decision_engine=mock_decision_engine)

        # Add a goal
        goal = AgentGoal(
            goal_id="test-goal",
            description="Test goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        await autonomous_agent.add_goal(goal)

        result = await autonomous_agent.execute_single_cycle()

        assert result.is_right()
        cycle_data = result.get_right()
        assert cycle_data["status"] == "completed"
        assert cycle_data["actions_taken"] >= 0

    @pytest.mark.asyncio
    async def test_autonomous_agent_execute_single_cycle_fallback_planning(
        self, autonomous_agent
    ):
        """Test single cycle execution with fallback action planning."""
        await autonomous_agent.initialize()  # No decision engine

        # Add a goal
        goal = AgentGoal(
            goal_id="test-goal",
            description="Test performance goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        await autonomous_agent.add_goal(goal)

        result = await autonomous_agent.execute_single_cycle()

        assert result.is_right()
        cycle_data = result.get_right()
        assert cycle_data["status"] == "completed"

    def test_autonomous_agent_assess_situation(self, autonomous_agent):
        """Test _assess_situation method."""
        situation = asyncio.run(autonomous_agent._assess_situation())

        assert "timestamp" in situation
        assert "agent_status" in situation
        assert "active_goals" in situation
        assert "active_actions" in situation
        assert "completed_actions" in situation
        assert "recent_success_rate" in situation
        assert "available_resources" in situation
        assert "learned_patterns_count" in situation
        assert "uptime" in situation

    @pytest.mark.asyncio
    async def test_autonomous_agent_fallback_action_planning_optimizer(
        self, autonomous_agent
    ):
        """Test fallback action planning for optimizer agent."""
        # Set agent type to optimizer
        autonomous_agent.state.configuration.agent_type = AgentType.OPTIMIZER

        goal = AgentGoal(
            goal_id="performance-goal",
            description="Improve performance metrics",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        actions = await autonomous_agent._fallback_action_planning(goal, {})

        assert len(actions) > 0
        assert actions[0].action_type == ActionType.ANALYZE_PERFORMANCE

    @pytest.mark.asyncio
    async def test_autonomous_agent_fallback_action_planning_monitor(
        self, autonomous_agent
    ):
        """Test fallback action planning for monitor agent."""
        # Set agent type to monitor
        autonomous_agent.state.configuration.agent_type = AgentType.MONITOR

        goal = AgentGoal(
            goal_id="monitor-goal",
            description="Monitor system",
            priority=GoalPriority.MEDIUM,
            target_metrics={"uptime": 0.99},
            resource_requirements={"cpu": 5.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        actions = await autonomous_agent._fallback_action_planning(goal, {})

        assert len(actions) > 0
        assert actions[0].action_type == ActionType.MONITOR_SYSTEM

    @pytest.mark.asyncio
    async def test_autonomous_agent_execute_action_success(self, autonomous_agent):
        """Test successful action execution."""
        action = AgentAction(
            action_id=ActionId("test-action"),
            agent_id=autonomous_agent.state.agent_id,
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={"components": ["test"]},
            goal_id="test-goal",
            confidence=ConfidenceScore(0.8),
            estimated_impact=PerformanceMetric(0.6),
            resource_cost={"cpu": 5.0},
        )

        result = await autonomous_agent._execute_action(action)

        assert result.is_right()
        execution_data = result.get_right()
        assert execution_data["success"] is True
        assert execution_data["action_id"] == action.action_id
        assert action.action_id in autonomous_agent.state.completed_actions

    @pytest.mark.asyncio
    async def test_autonomous_agent_execute_action_exception(self, autonomous_agent):
        """Test action execution with exception."""
        action = AgentAction(
            action_id=ActionId("failing-action"),
            agent_id=autonomous_agent.state.agent_id,
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={"components": ["test"]},
            goal_id="test-goal",
            confidence=ConfidenceScore(0.8),
            estimated_impact=PerformanceMetric(0.6),
            resource_cost={"cpu": 5.0},
        )

        # Mock sleep to raise exception
        with patch("asyncio.sleep", side_effect=Exception("Execution failed")):
            result = await autonomous_agent._execute_action(action)

            assert result.is_left()
            assert action not in autonomous_agent.state.active_actions

    @pytest.mark.asyncio
    async def test_autonomous_agent_learn_from_action_success(self, autonomous_agent):
        """Test learning from successful action."""
        action = AgentAction(
            action_id=ActionId("test-action"),
            agent_id=autonomous_agent.state.agent_id,
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={"components": ["test"]},
            goal_id="test-goal",
            confidence=ConfidenceScore(0.8),
            estimated_impact=PerformanceMetric(0.6),
            resource_cost={"cpu": 5.0},
        )

        result = Either.right({"success": True, "output": "completed"})
        context = {"situation": "test"}

        await autonomous_agent._learn_from_action(action, result, context)

        assert len(autonomous_agent.state.experiences) == 1
        assert autonomous_agent.state.experiences[0].success is True
        assert autonomous_agent.state.metrics.learning_experiences == 1

    @pytest.mark.asyncio
    async def test_autonomous_agent_learn_from_action_failure(self, autonomous_agent):
        """Test learning from failed action."""
        action = AgentAction(
            action_id=ActionId("test-action"),
            agent_id=autonomous_agent.state.agent_id,
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={"components": ["test"]},
            goal_id="test-goal",
            confidence=ConfidenceScore(0.8),
            estimated_impact=PerformanceMetric(0.6),
            resource_cost={"cpu": 5.0},
        )

        result = Either.left(AutonomousAgentError.action_execution_failed("Failed"))
        context = {"situation": "test"}

        await autonomous_agent._learn_from_action(action, result, context)

        assert len(autonomous_agent.state.experiences) == 1
        assert autonomous_agent.state.experiences[0].success is False
        assert autonomous_agent.state.metrics.learning_experiences == 1

    @pytest.mark.asyncio
    async def test_autonomous_agent_learn_from_action_experience_limit(
        self, autonomous_agent
    ):
        """Test experience history limiting."""
        # Fill up experiences beyond limit
        for i in range(1050):  # More than 1000 limit
            action = AgentAction(
                action_id=ActionId(f"action-{i}"),
                agent_id=autonomous_agent.state.agent_id,
                action_type=ActionType.MONITOR_SYSTEM,
                parameters={"components": ["test"]},
                goal_id="test-goal",
                confidence=ConfidenceScore(0.8),
                estimated_impact=PerformanceMetric(0.6),
                resource_cost={"cpu": 5.0},
            )

            result = Either.right({"success": True})

            await autonomous_agent._learn_from_action(action, result, {})

        # Should be limited to 500 (from last 1000, keep last 500)
        assert len(autonomous_agent.state.experiences) == 500

    @pytest.mark.asyncio
    async def test_autonomous_agent_perform_self_optimization_low_success(
        self, autonomous_agent
    ):
        """Test self-optimization with low success rate."""
        # Add experiences with low success rate
        for i in range(10):
            experience = LearningExperience(
                experience_id=f"exp-{i}",
                agent_id=autonomous_agent.state.agent_id,
                context={"test": True},
                action_taken=MagicMock(),
                outcome={"success": i < 3},  # 30% success rate
                success=i < 3,
                learning_value=ConfidenceScore(0.8),
                performance_impact=PerformanceMetric(0.6),
            )
            autonomous_agent.state.experiences.append(experience)

        initial_threshold = autonomous_agent.state.configuration.decision_threshold
        initial_risk_tolerance = autonomous_agent.state.configuration.risk_tolerance

        await autonomous_agent._perform_self_optimization()

        # Should become more conservative
        assert (
            autonomous_agent.state.configuration.decision_threshold > initial_threshold
        )
        assert (
            autonomous_agent.state.configuration.risk_tolerance < initial_risk_tolerance
        )
        assert autonomous_agent.state.metrics.optimization_cycles == 1

    @pytest.mark.asyncio
    async def test_autonomous_agent_perform_self_optimization_high_success(
        self, autonomous_agent
    ):
        """Test self-optimization with high success rate."""
        # Add experiences with high success rate
        for i in range(10):
            experience = LearningExperience(
                experience_id=f"exp-{i}",
                agent_id=autonomous_agent.state.agent_id,
                context={"test": True},
                action_taken=MagicMock(),
                outcome={"success": i < 9},  # 90% success rate
                success=i < 9,
                learning_value=ConfidenceScore(0.8),
                performance_impact=PerformanceMetric(0.6),
            )
            autonomous_agent.state.experiences.append(experience)

        initial_threshold = autonomous_agent.state.configuration.decision_threshold
        initial_risk_tolerance = autonomous_agent.state.configuration.risk_tolerance

        await autonomous_agent._perform_self_optimization()

        # Should become more aggressive
        assert (
            autonomous_agent.state.configuration.decision_threshold < initial_threshold
        )
        assert (
            autonomous_agent.state.configuration.risk_tolerance > initial_risk_tolerance
        )

    @pytest.mark.asyncio
    async def test_autonomous_agent_perform_self_optimization_no_experiences(
        self, autonomous_agent
    ):
        """Test self-optimization with no experiences."""
        await autonomous_agent._perform_self_optimization()

        # Should complete without changes
        assert autonomous_agent.state.status == AgentStatus.ACTIVE

    def test_autonomous_agent_should_optimize_no_previous_optimization(
        self, autonomous_agent
    ):
        """Test should_optimize with no previous optimization."""
        # Add some experiences
        for i in range(15):
            experience = LearningExperience(
                experience_id=f"exp-{i}",
                agent_id=autonomous_agent.state.agent_id,
                context={"test": True},
                action_taken=MagicMock(),
                outcome={"success": True},
                success=True,
                learning_value=ConfidenceScore(0.8),
                performance_impact=PerformanceMetric(0.6),
            )
            autonomous_agent.state.experiences.append(experience)

        should_optimize = autonomous_agent._should_optimize()
        assert should_optimize is True

    def test_autonomous_agent_should_optimize_insufficient_experience(
        self, autonomous_agent
    ):
        """Test should_optimize with insufficient experience."""
        # Add only a few experiences
        for i in range(5):
            experience = LearningExperience(
                experience_id=f"exp-{i}",
                agent_id=autonomous_agent.state.agent_id,
                context={"test": True},
                action_taken=MagicMock(),
                outcome={"success": True},
                success=True,
                learning_value=ConfidenceScore(0.8),
                performance_impact=PerformanceMetric(0.6),
            )
            autonomous_agent.state.experiences.append(experience)

        should_optimize = autonomous_agent._should_optimize()
        assert should_optimize is False

    def test_autonomous_agent_should_optimize_recent_optimization(
        self, autonomous_agent
    ):
        """Test should_optimize with recent optimization."""
        # Set recent optimization time
        autonomous_agent.state.metrics.last_optimization = datetime.now(
            UTC
        ) - timedelta(minutes=30)

        # Optimization frequency is typically 1 hour, so should be False
        should_optimize = autonomous_agent._should_optimize()
        assert should_optimize is False

    def test_autonomous_agent_calculate_recent_success_rate(self, autonomous_agent):
        """Test calculation of recent success rate."""
        # Add mixed success experiences
        for i in range(30):
            experience = LearningExperience(
                experience_id=f"exp-{i}",
                agent_id=autonomous_agent.state.agent_id,
                context={"test": True},
                action_taken=MagicMock(),
                outcome={"success": i % 2 == 0},  # 50% success rate
                success=i % 2 == 0,
                learning_value=ConfidenceScore(0.8),
                performance_impact=PerformanceMetric(0.6),
            )
            autonomous_agent.state.experiences.append(experience)

        success_rate = autonomous_agent._calculate_recent_success_rate()
        # Should calculate from last 20 experiences
        assert 0.4 <= success_rate <= 0.6  # Approximately 50%

    def test_autonomous_agent_calculate_recent_success_rate_no_experiences(
        self, autonomous_agent
    ):
        """Test success rate calculation with no experiences."""
        success_rate = autonomous_agent._calculate_recent_success_rate()
        assert success_rate == 0.0

    @pytest.mark.asyncio
    async def test_autonomous_agent_update_performance_metrics(self, autonomous_agent):
        """Test performance metrics update."""
        # Add some experiences for success rate calculation
        for i in range(10):
            experience = LearningExperience(
                experience_id=f"exp-{i}",
                agent_id=autonomous_agent.state.agent_id,
                context={"test": True},
                action_taken=MagicMock(),
                outcome={"success": i < 8},  # 80% success rate
                success=i < 8,
                learning_value=ConfidenceScore(0.8),
                performance_impact=PerformanceMetric(0.6),
            )
            autonomous_agent.state.experiences.append(experience)

        # Add some goals that can be marked as achieved
        goal1 = AgentGoal(
            goal_id="goal-1",
            description="Test goal 1",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["crit1", "crit2"],
            constraints={"max_runtime": 3600},
        )

        autonomous_agent.state.current_goals.append(goal1)

        # Add completed actions that match goal
        autonomous_agent.state.completed_actions.add(ActionId("goal-1_action1"))
        autonomous_agent.state.completed_actions.add(ActionId("goal-1_action2"))

        initial_goals_achieved = autonomous_agent.state.metrics.goals_achieved

        await autonomous_agent._update_performance_metrics()

        assert autonomous_agent.state.metrics.success_rate == PerformanceMetric(0.8)
        assert autonomous_agent.state.metrics.total_runtime > timedelta(0)
        # Goal should be achieved and removed
        assert autonomous_agent.state.metrics.goals_achieved > initial_goals_achieved

    def test_autonomous_agent_is_goal_achieved_simple_heuristic(self, autonomous_agent):
        """Test simple goal achievement heuristic."""
        goal = AgentGoal(
            goal_id="test-goal",
            description="Test goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["crit1", "crit2"],
            constraints={"max_runtime": 3600},
        )

        # Add completed actions that match goal prefix
        autonomous_agent.state.completed_actions.add(ActionId("test-goa_action1"))
        autonomous_agent.state.completed_actions.add(ActionId("test-goa_action2"))

        is_achieved = autonomous_agent._is_goal_achieved(goal)
        assert is_achieved is True

    def test_autonomous_agent_is_goal_achieved_insufficient_actions(
        self, autonomous_agent
    ):
        """Test goal achievement with insufficient completed actions."""
        goal = AgentGoal(
            goal_id="test-goal",
            description="Test goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["crit1", "crit2"],
            constraints={"max_runtime": 3600},
        )

        # Add only one completed action (need 2 for criteria)
        autonomous_agent.state.completed_actions.add(ActionId("test-goa_action1"))

        is_achieved = autonomous_agent._is_goal_achieved(goal)
        assert is_achieved is False

    def test_autonomous_agent_check_goal_conflicts_resource_conflict(
        self, autonomous_agent
    ):
        """Test goal conflict detection for resource conflicts."""
        # Set up configuration with limited resources
        autonomous_agent.state.configuration.resource_limits = {"cpu": 50.0}

        # Add existing goal
        existing_goal = AgentGoal(
            goal_id="existing-goal",
            description="Existing goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 30.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["crit1"],
            constraints={"max_runtime": 3600},
        )
        autonomous_agent.state.current_goals.append(existing_goal)

        # Try to add new goal that would exceed limits
        new_goal = AgentGoal(
            goal_id="new-goal",
            description="New goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 25.0},  # 30 + 25 = 55 > 50
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["crit1"],
            constraints={"max_runtime": 3600},
        )

        conflicts = autonomous_agent._check_goal_conflicts(new_goal)
        assert len(conflicts) > 0
        assert "Resource conflict: cpu" in conflicts[0]

    def test_autonomous_agent_check_goal_conflicts_multiple_emergency_goals(
        self, autonomous_agent
    ):
        """Test goal conflict detection for multiple emergency goals."""
        # Add existing emergency goal
        existing_goal = AgentGoal(
            goal_id="emergency-1",
            description="Emergency goal 1",
            priority=GoalPriority.EMERGENCY,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["crit1"],
            constraints={"max_runtime": 3600},
        )
        autonomous_agent.state.current_goals.append(existing_goal)

        # Try to add another emergency goal
        new_goal = AgentGoal(
            goal_id="emergency-2",
            description="Emergency goal 2",
            priority=GoalPriority.EMERGENCY,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["crit1"],
            constraints={"max_runtime": 3600},
        )

        conflicts = autonomous_agent._check_goal_conflicts(new_goal)
        assert len(conflicts) > 0
        assert "Multiple emergency goals not allowed" in conflicts

    def test_autonomous_agent_check_goal_conflicts_no_conflicts(self, autonomous_agent):
        """Test goal conflict detection with no conflicts."""
        # Set up configuration with sufficient resources
        autonomous_agent.state.configuration.resource_limits = {"cpu": 100.0}

        # Add existing goal
        existing_goal = AgentGoal(
            goal_id="existing-goal",
            description="Existing goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 30.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["crit1"],
            constraints={"max_runtime": 3600},
        )
        autonomous_agent.state.current_goals.append(existing_goal)

        # Add new goal with no conflicts
        new_goal = AgentGoal(
            goal_id="new-goal",
            description="New goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 20.0},  # 30 + 20 = 50 < 100
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["crit1"],
            constraints={"max_runtime": 3600},
        )

        conflicts = autonomous_agent._check_goal_conflicts(new_goal)
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_autonomous_agent_prioritize_goals(self, autonomous_agent):
        """Test goal prioritization."""
        # Add goals with different priorities
        goal1 = AgentGoal(
            goal_id="low-priority",
            description="Low priority goal",
            priority=GoalPriority.LOW,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=2),
            success_criteria=["crit1"],
            constraints={"max_runtime": 3600},
        )

        goal2 = AgentGoal(
            goal_id="high-priority",
            description="High priority goal",
            priority=GoalPriority.HIGH,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["crit1"],
            constraints={"max_runtime": 3600},
        )

        goal3 = AgentGoal(
            goal_id="normal-priority",
            description="Normal priority goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=3),
            success_criteria=["crit1"],
            constraints={"max_runtime": 3600},
        )

        autonomous_agent.state.current_goals.extend([goal1, goal2, goal3])

        await autonomous_agent._prioritize_goals()

        # Goals should be sorted by urgency (high priority first)
        assert autonomous_agent.state.current_goals[0].priority == GoalPriority.HIGH

    @pytest.mark.asyncio
    async def test_autonomous_agent_request_human_approval_low_risk(
        self, autonomous_agent
    ):
        """Test human approval request for low-risk action."""
        action = AgentAction(
            action_id=ActionId("low-risk-action"),
            agent_id=autonomous_agent.state.agent_id,
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={"components": ["test"]},
            goal_id="test-goal",
            confidence=ConfidenceScore(0.9),  # High confidence = low risk
            estimated_impact=PerformanceMetric(0.3),  # Low impact
            resource_cost={"cpu": 5.0},
        )

        approval = await autonomous_agent._request_human_approval(action)
        assert approval is True  # Low risk actions auto-approved

    @pytest.mark.asyncio
    async def test_autonomous_agent_request_human_approval_high_risk(
        self, autonomous_agent
    ):
        """Test human approval request for high-risk action."""
        action = AgentAction(
            action_id=ActionId("high-risk-action"),
            agent_id=autonomous_agent.state.agent_id,
            action_type=ActionType.OPTIMIZE_PERFORMANCE,
            parameters={"system_changes": True},
            goal_id="test-goal",
            confidence=ConfidenceScore(0.3),  # Low confidence = high risk
            estimated_impact=PerformanceMetric(0.8),  # High impact
            resource_cost={"cpu": 50.0},
        )

        approval = await autonomous_agent._request_human_approval(action)
        assert approval is False  # High risk actions not auto-approved

    @pytest.mark.asyncio
    async def test_autonomous_agent_load_learned_patterns(self, autonomous_agent):
        """Test loading learned patterns (placeholder implementation)."""
        await autonomous_agent._load_learned_patterns()

        # Should initialize empty patterns
        assert autonomous_agent.state.learned_patterns == {}

    @pytest.mark.asyncio
    async def test_autonomous_agent_update_learned_patterns(self, autonomous_agent):
        """Test updating learned patterns."""
        new_patterns = {
            "pattern_type_1": ["data1", "data2"],
            "pattern_type_2": {"key": "value"},
        }

        await autonomous_agent._update_learned_patterns(new_patterns)

        assert "pattern_type_1" in autonomous_agent.state.learned_patterns
        assert "pattern_type_2" in autonomous_agent.state.learned_patterns
        assert autonomous_agent.state.learned_patterns["pattern_type_1"] == [
            "data1",
            "data2",
        ]
        assert autonomous_agent.state.learned_patterns["pattern_type_2"] == [
            {"key": "value"}
        ]
        assert autonomous_agent.state.metrics.last_learning_update is not None

    @pytest.mark.asyncio
    async def test_autonomous_agent_update_learned_patterns_extend_existing(
        self, autonomous_agent
    ):
        """Test updating learned patterns that extend existing patterns."""
        # Initialize with existing patterns
        autonomous_agent.state.learned_patterns = {"pattern_type_1": ["existing_data"]}

        new_patterns = {
            "pattern_type_1": ["new_data1", "new_data2"],
        }

        await autonomous_agent._update_learned_patterns(new_patterns)

        # Should extend, not replace
        assert len(autonomous_agent.state.learned_patterns["pattern_type_1"]) == 3
        assert (
            "existing_data" in autonomous_agent.state.learned_patterns["pattern_type_1"]
        )
        assert "new_data1" in autonomous_agent.state.learned_patterns["pattern_type_1"]

    def test_autonomous_agent_validate_configuration_success(self, autonomous_agent):
        """Test successful configuration validation."""
        # Set valid configuration
        autonomous_agent.state.configuration.decision_threshold = 0.7
        autonomous_agent.state.configuration.autonomy_level = AutonomyLevel.AUTONOMOUS
        autonomous_agent.state.configuration.resource_limits = {"cpu": 100.0}

        result = autonomous_agent._validate_configuration()

        assert result.is_right()

    def test_autonomous_agent_validate_configuration_high_threshold_autonomous(
        self, autonomous_agent
    ):
        """Test configuration validation with high threshold for autonomous operation."""
        # Set invalid configuration
        autonomous_agent.state.configuration.decision_threshold = 0.99
        autonomous_agent.state.configuration.autonomy_level = AutonomyLevel.AUTONOMOUS

        result = autonomous_agent._validate_configuration()

        assert result.is_left()

    def test_autonomous_agent_validate_configuration_no_resource_limits(
        self, autonomous_agent
    ):
        """Test configuration validation with no resource limits."""
        # Clear resource limits
        autonomous_agent.state.configuration.resource_limits = {}

        result = autonomous_agent._validate_configuration()

        assert result.is_left()


class TestAgentManager:
    """Test AgentManager class functionality comprehensively."""

    @pytest.fixture
    def agent_manager(self):
        """Create test agent manager."""
        return AgentManager()

    @pytest.mark.asyncio
    async def test_agent_manager_creation(self, agent_manager):
        """Test AgentManager creation and initialization."""
        assert len(agent_manager.agents) == 0
        assert len(agent_manager.active_agents) == 0
        assert len(agent_manager.agent_metrics) == 0
        assert agent_manager.communication_hub is not None
        assert agent_manager.resource_optimizer is not None
        assert agent_manager.self_healing_engine is not None

    @pytest.mark.asyncio
    async def test_agent_manager_create_agent_success(self, agent_manager):
        """Test successful agent creation."""
        result = await agent_manager.create_agent(AgentType.GENERAL)

        assert result.is_right()
        agent_id = result.get_right()
        assert agent_id in agent_manager.agents
        assert agent_id in agent_manager.agent_metrics
        assert len(agent_manager.agents) == 1

    @pytest.mark.asyncio
    async def test_agent_manager_create_agent_with_custom_config(self, agent_manager):
        """Test agent creation with custom configuration."""
        custom_config = get_default_config(AgentType.OPTIMIZER)
        custom_config.decision_threshold = 0.8

        result = await agent_manager.create_agent(AgentType.OPTIMIZER, custom_config)

        assert result.is_right()
        agent_id = result.get_right()
        agent = agent_manager.agents[agent_id]
        assert agent.state.configuration.decision_threshold == 0.8
        assert agent.state.configuration.agent_type == AgentType.OPTIMIZER

    @pytest.mark.asyncio
    async def test_agent_manager_create_agent_initialization_failure(
        self, agent_manager
    ):
        """Test agent creation with initialization failure."""
        # Create agent with invalid config that will fail validation
        invalid_config = get_default_config(AgentType.GENERAL)
        invalid_config.decision_threshold = 0.99
        invalid_config.autonomy_level = AutonomyLevel.AUTONOMOUS
        invalid_config.resource_limits = {}  # Empty resource limits

        result = await agent_manager.create_agent(AgentType.GENERAL, invalid_config)

        assert result.is_left()

    @pytest.mark.asyncio
    async def test_agent_manager_start_agent_success(self, agent_manager):
        """Test successful agent start."""
        # Create agent first
        create_result = await agent_manager.create_agent(AgentType.GENERAL)
        agent_id = create_result.get_right()

        result = await agent_manager.start_agent(agent_id)

        assert result.is_right()
        assert agent_id in agent_manager.active_agents

    @pytest.mark.asyncio
    async def test_agent_manager_start_agent_not_found(self, agent_manager):
        """Test starting non-existent agent."""
        non_existent_id = AgentId("non-existent")

        result = await agent_manager.start_agent(non_existent_id)

        assert result.is_left()

    @pytest.mark.asyncio
    async def test_agent_manager_stop_agent_success(self, agent_manager):
        """Test successful agent stop."""
        # Create and start agent first
        create_result = await agent_manager.create_agent(AgentType.GENERAL)
        agent_id = create_result.get_right()
        await agent_manager.start_agent(agent_id)

        result = await agent_manager.stop_agent(agent_id)

        assert result.is_right()
        assert agent_id not in agent_manager.active_agents

    @pytest.mark.asyncio
    async def test_agent_manager_stop_agent_not_found(self, agent_manager):
        """Test stopping non-existent agent."""
        non_existent_id = AgentId("non-existent")

        result = await agent_manager.stop_agent(non_existent_id)

        assert result.is_left()

    @pytest.mark.asyncio
    async def test_agent_manager_add_goal_to_agent_success(self, agent_manager):
        """Test successful goal addition to agent."""
        # Create agent
        create_result = await agent_manager.create_agent(AgentType.GENERAL)
        agent_id = create_result.get_right()

        goal = AgentGoal(
            goal_id="test-goal",
            description="Test goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        result = await agent_manager.add_goal_to_agent(agent_id, goal)

        assert result.is_right()
        agent = agent_manager.agents[agent_id]
        assert len(agent.state.current_goals) == 1

    @pytest.mark.asyncio
    async def test_agent_manager_add_goal_to_agent_not_found(self, agent_manager):
        """Test adding goal to non-existent agent."""
        non_existent_id = AgentId("non-existent")

        goal = AgentGoal(
            goal_id="test-goal",
            description="Test goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        result = await agent_manager.add_goal_to_agent(non_existent_id, goal)

        assert result.is_left()

    def test_agent_manager_get_agent_status_success(self, agent_manager):
        """Test successful agent status retrieval."""
        # Create agent
        create_result = asyncio.run(agent_manager.create_agent(AgentType.MONITOR))
        agent_id = create_result.get_right()

        result = agent_manager.get_agent_status(agent_id)

        assert result.is_right()
        status_data = result.get_right()
        assert status_data["agent_id"] == agent_id
        assert status_data["status"] == AgentStatus.ACTIVE.value
        assert status_data["agent_type"] == AgentType.MONITOR.value
        assert "metrics" in status_data
        assert "uptime" in status_data

    def test_agent_manager_get_agent_status_not_found(self, agent_manager):
        """Test agent status retrieval for non-existent agent."""
        non_existent_id = AgentId("non-existent")

        result = agent_manager.get_agent_status(non_existent_id)

        assert result.is_left()

    def test_agent_manager_list_agents_empty(self, agent_manager):
        """Test listing agents when none exist."""
        agents_list = agent_manager.list_agents()

        assert len(agents_list) == 0
        assert isinstance(agents_list, dict)

    def test_agent_manager_list_agents_with_agents(self, agent_manager):
        """Test listing agents when some exist."""
        # Create multiple agents
        asyncio.run(agent_manager.create_agent(AgentType.GENERAL))
        asyncio.run(agent_manager.create_agent(AgentType.OPTIMIZER))

        agents_list = agent_manager.list_agents()

        assert len(agents_list) == 2
        for _agent_id, agent_info in agents_list.items():
            assert "status" in agent_info
            assert "type" in agent_info
            assert "active_goals" in agent_info
            assert "uptime" in agent_info

    @pytest.mark.asyncio
    async def test_agent_manager_shutdown_all_agents(self, agent_manager):
        """Test shutting down all agents."""
        # Create and start multiple agents
        agent1_result = await agent_manager.create_agent(AgentType.GENERAL)
        agent2_result = await agent_manager.create_agent(AgentType.MONITOR)

        agent1_id = agent1_result.get_right()
        agent2_id = agent2_result.get_right()

        await agent_manager.start_agent(agent1_id)
        await agent_manager.start_agent(agent2_id)

        assert len(agent_manager.active_agents) == 2

        await agent_manager.shutdown_all_agents()

        assert len(agent_manager.active_agents) == 0

    @pytest.mark.asyncio
    async def test_agent_manager_handle_agent_error_success(self, agent_manager):
        """Test successful agent error handling."""
        # Create agent
        create_result = await agent_manager.create_agent(AgentType.GENERAL)
        agent_id = create_result.get_right()

        error = Exception("Test error")
        context = {"test": True}

        # Mock the self-healing engine methods
        mock_error_event = MagicMock()
        mock_recovery_action = MagicMock()
        mock_recovery_action.strategy.value = "restart"

        agent_manager.self_healing_engine.detect_and_diagnose = AsyncMock(
            return_value=Either.right(mock_error_event)
        )
        agent_manager.self_healing_engine.plan_recovery = AsyncMock(
            return_value=Either.right(mock_recovery_action)
        )
        agent_manager.self_healing_engine.execute_recovery = AsyncMock(
            return_value=Either.right({"recovery": "success"})
        )

        result = await agent_manager.handle_agent_error(agent_id, error, context)

        assert result.is_right()

        # Verify optimization cycles was incremented
        agent = agent_manager.agents[agent_id]
        assert agent.state.metrics.optimization_cycles > 0

    @pytest.mark.asyncio
    async def test_agent_manager_handle_agent_error_diagnosis_failure(
        self, agent_manager
    ):
        """Test agent error handling with diagnosis failure."""
        # Create agent
        create_result = await agent_manager.create_agent(AgentType.GENERAL)
        agent_id = create_result.get_right()

        error = Exception("Test error")
        context = {"test": True}

        # Mock diagnosis failure
        agent_manager.self_healing_engine.detect_and_diagnose = AsyncMock(
            return_value=Either.left(AutonomousAgentError.diagnosis_failed("Failed"))
        )

        result = await agent_manager.handle_agent_error(agent_id, error, context)

        assert result.is_left()

    @pytest.mark.asyncio
    async def test_agent_manager_handle_agent_error_recovery_planning_failure(
        self, agent_manager
    ):
        """Test agent error handling with recovery planning failure."""
        # Create agent
        create_result = await agent_manager.create_agent(AgentType.GENERAL)
        agent_id = create_result.get_right()

        error = Exception("Test error")
        context = {"test": True}

        # Mock successful diagnosis but failed recovery planning
        mock_error_event = MagicMock()
        agent_manager.self_healing_engine.detect_and_diagnose = AsyncMock(
            return_value=Either.right(mock_error_event)
        )
        agent_manager.self_healing_engine.plan_recovery = AsyncMock(
            return_value=Either.left(
                AutonomousAgentError.recovery_planning_failed("Failed")
            )
        )

        result = await agent_manager.handle_agent_error(agent_id, error, context)

        assert result.is_left()

    @pytest.mark.asyncio
    async def test_agent_manager_handle_agent_error_recovery_execution_failure(
        self, agent_manager
    ):
        """Test agent error handling with recovery execution failure."""
        # Create agent
        create_result = await agent_manager.create_agent(AgentType.GENERAL)
        agent_id = create_result.get_right()

        error = Exception("Test error")
        context = {"test": True}

        # Mock successful diagnosis and planning but failed execution
        mock_error_event = MagicMock()
        mock_recovery_action = MagicMock()

        agent_manager.self_healing_engine.detect_and_diagnose = AsyncMock(
            return_value=Either.right(mock_error_event)
        )
        agent_manager.self_healing_engine.plan_recovery = AsyncMock(
            return_value=Either.right(mock_recovery_action)
        )
        agent_manager.self_healing_engine.execute_recovery = AsyncMock(
            return_value=Either.left(
                AutonomousAgentError.recovery_execution_failed("Failed")
            )
        )

        result = await agent_manager.handle_agent_error(agent_id, error, context)

        assert result.is_left()

    @pytest.mark.asyncio
    async def test_agent_manager_handle_agent_error_exception_handling(
        self, agent_manager
    ):
        """Test agent error handling with exception during process."""
        # Create agent
        create_result = await agent_manager.create_agent(AgentType.GENERAL)
        agent_id = create_result.get_right()

        error = Exception("Test error")
        context = {"test": True}

        # Mock diagnosis to raise exception
        agent_manager.self_healing_engine.detect_and_diagnose = AsyncMock(
            side_effect=Exception("Diagnosis exception")
        )

        result = await agent_manager.handle_agent_error(agent_id, error, context)

        assert result.is_left()

    def test_agent_manager_get_healing_statistics(self, agent_manager):
        """Test getting healing statistics."""
        # Mock the self-healing engine method
        expected_stats = {"total_recoveries": 5, "success_rate": 0.8}
        agent_manager.self_healing_engine.get_healing_statistics = MagicMock(
            return_value=expected_stats
        )

        stats = agent_manager.get_healing_statistics()

        assert stats == expected_stats

    def test_agent_manager_get_system_status(self, agent_manager):
        """Test getting comprehensive system status."""
        # Create some agents
        asyncio.run(agent_manager.create_agent(AgentType.GENERAL))
        asyncio.run(agent_manager.create_agent(AgentType.MONITOR))

        # Mock the methods that return async values
        agent_manager.resource_optimizer.calculate_efficiency_score = AsyncMock(
            return_value=0.85
        )
        agent_manager.resource_optimizer.get_optimization_recommendations = AsyncMock(
            return_value=["recommendation1", "recommendation2"]
        )
        agent_manager.resource_optimizer.get_resource_status = MagicMock(
            return_value={"cpu": 75.0, "memory": 60.0}
        )
        agent_manager.communication_hub.get_communication_stats = MagicMock(
            return_value={"messages_sent": 10, "messages_received": 8}
        )
        agent_manager.self_healing_engine.get_healing_statistics = MagicMock(
            return_value={"total_recoveries": 3}
        )

        status = agent_manager.get_system_status()

        assert status["total_agents"] == 2
        assert status["active_agents"] == 0  # None started
        assert "agent_statuses" in status
        assert "resource_status" in status
        assert "communication_stats" in status
        assert "healing_statistics" in status
        assert "system_health" in status
        assert "resource_efficiency" in status["system_health"]
        assert "optimization_recommendations" in status["system_health"]


class TestProtocols:
    """Test protocol definitions and implementations."""

    def test_ai_processor_protocol(self):
        """Test AIProcessorProtocol structure."""

        # Create mock implementation
        class MockAIProcessor:
            async def process(self, data):
                return {"processed": data}

        processor = MockAIProcessor()

        # Should satisfy protocol
        assert isinstance(processor, AIProcessorProtocol)

    def test_decision_engine_protocol(self):
        """Test DecisionEngineProtocol structure."""

        # Create mock implementation
        class MockDecisionEngine:
            async def make_decision(self, context):
                return {"decision": "proceed"}

        engine = MockDecisionEngine()

        # Should satisfy protocol
        assert isinstance(engine, DecisionEngineProtocol)

    def test_safety_validator_protocol(self):
        """Test SafetyValidatorProtocol structure."""

        # Create mock implementation
        class MockSafetyValidator:
            async def validate(self, action):
                return True

        validator = MockSafetyValidator()

        # Should satisfy protocol
        assert isinstance(validator, SafetyValidatorProtocol)


class TestIntegrationScenarios:
    """Test integration scenarios between components."""

    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self):
        """Test complete agent lifecycle from creation to shutdown."""
        manager = AgentManager()

        # Create agent
        create_result = await manager.create_agent(AgentType.GENERAL)
        assert create_result.is_right()
        agent_id = create_result.get_right()

        # Add goal
        goal = AgentGoal(
            goal_id="lifecycle-goal",
            description="Test lifecycle goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"accuracy": 0.9},
            resource_requirements={"cpu": 10.0},
            deadline=datetime.now(UTC) + timedelta(hours=1),
            success_criteria=["criterion1"],
            constraints={"max_runtime": 3600},
        )

        goal_result = await manager.add_goal_to_agent(agent_id, goal)
        assert goal_result.is_right()

        # Start agent
        start_result = await manager.start_agent(agent_id)
        assert start_result.is_right()

        # Check status
        status_result = manager.get_agent_status(agent_id)
        assert status_result.is_right()

        # Stop agent
        stop_result = await manager.stop_agent(agent_id)
        assert stop_result.is_right()

        # Shutdown all
        await manager.shutdown_all_agents()

    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self):
        """Test coordination between multiple agents."""
        manager = AgentManager()

        # Create multiple agents
        general_result = await manager.create_agent(AgentType.GENERAL)
        monitor_result = await manager.create_agent(AgentType.MONITOR)
        optimizer_result = await manager.create_agent(AgentType.OPTIMIZER)

        assert all(
            result.is_right()
            for result in [general_result, monitor_result, optimizer_result]
        )

        # Get agent IDs
        general_id = general_result.get_right()
        monitor_id = monitor_result.get_right()
        optimizer_id = optimizer_result.get_right()

        # Start all agents
        for agent_id in [general_id, monitor_id, optimizer_id]:
            start_result = await manager.start_agent(agent_id)
            assert start_result.is_right()

        # Check system status
        system_status = manager.get_system_status()
        assert system_status["total_agents"] == 3
        assert system_status["active_agents"] == 3

        # Shutdown
        await manager.shutdown_all_agents()
        assert len(manager.active_agents) == 0

    @pytest.mark.asyncio
    async def test_agent_error_recovery_integration(self):
        """Test agent error recovery integration."""
        manager = AgentManager()

        # Create agent
        create_result = await manager.create_agent(AgentType.GENERAL)
        agent_id = create_result.get_right()

        # Mock successful error handling
        error = Exception("Integration test error")
        context = {"integration_test": True}

        # Mock the healing engine
        mock_error_event = MagicMock()
        mock_recovery_action = MagicMock()
        mock_recovery_action.strategy.value = "restart"

        manager.self_healing_engine.detect_and_diagnose = AsyncMock(
            return_value=Either.right(mock_error_event)
        )
        manager.self_healing_engine.plan_recovery = AsyncMock(
            return_value=Either.right(mock_recovery_action)
        )
        manager.self_healing_engine.execute_recovery = AsyncMock(
            return_value=Either.right({"recovery": "success"})
        )

        # Handle error
        recovery_result = await manager.handle_agent_error(agent_id, error, context)

        assert recovery_result.is_right()

        # Verify agent is still in manager
        assert agent_id in manager.agents

    @pytest.mark.asyncio
    async def test_agent_performance_optimization_cycle(self):
        """Test agent performance optimization cycle."""
        manager = AgentManager()

        # Create agent
        create_result = await manager.create_agent(AgentType.OPTIMIZER)
        agent_id = create_result.get_right()
        agent = manager.agents[agent_id]

        # Add experiences to trigger optimization
        for i in range(15):
            experience = LearningExperience(
                experience_id=f"opt-exp-{i}",
                agent_id=agent_id,
                context={"optimization_test": True},
                action_taken=MagicMock(),
                outcome={"success": i < 10},  # 66% success rate
                success=i < 10,
                learning_value=ConfidenceScore(0.8),
                performance_impact=PerformanceMetric(0.6),
            )
            agent.state.experiences.append(experience)

        initial_threshold = agent.state.configuration.decision_threshold

        # Trigger optimization
        await agent._perform_self_optimization()

        # Configuration should be adjusted
        assert agent.state.configuration.decision_threshold != initial_threshold
        assert agent.state.metrics.optimization_cycles > 0
