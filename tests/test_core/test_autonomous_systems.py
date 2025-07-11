"""Comprehensive test coverage for autonomous systems core module.

Tests the complete autonomous systems type system including branded types, enums,
dataclasses, and business logic following ADDER+ methodology for enterprise automation.
"""

from datetime import UTC, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st
from src.core.autonomous_systems import (
    ActionId,
    ActionType,
    AgentAction,
    AgentConfiguration,
    AgentGoal,
    AgentId,
    AgentStatus,
    AgentType,
    AutonomousAgentError,
    AutonomyLevel,
    ConfidenceScore,
    ExperienceId,
    GoalId,
    GoalPriority,
    LearningExperience,
    PerformanceMetric,
    RiskScore,
    create_action_id,
    create_agent_id,
    create_experience_id,
    create_goal_id,
    get_default_config,
)


class TestBrandedTypes:
    """Test branded types for autonomous systems."""

    def test_agent_id_creation(self):
        """Test AgentId branded type creation."""
        agent_id = AgentId("agent_123")
        assert isinstance(agent_id, str)
        assert agent_id == "agent_123"

    def test_goal_id_creation(self):
        """Test GoalId branded type creation."""
        goal_id = GoalId("goal_abc")
        assert isinstance(goal_id, str)
        assert goal_id == "goal_abc"

    def test_action_id_creation(self):
        """Test ActionId branded type creation."""
        action_id = ActionId("action_xyz")
        assert isinstance(action_id, str)
        assert action_id == "action_xyz"

    def test_experience_id_creation(self):
        """Test ExperienceId branded type creation."""
        exp_id = ExperienceId("exp_def")
        assert isinstance(exp_id, str)
        assert exp_id == "exp_def"

    def test_confidence_score_creation(self):
        """Test ConfidenceScore branded type creation."""
        confidence = ConfidenceScore(0.85)
        assert isinstance(confidence, float)
        assert confidence == 0.85

    def test_risk_score_creation(self):
        """Test RiskScore branded type creation."""
        risk = RiskScore(0.3)
        assert isinstance(risk, float)
        assert risk == 0.3

    def test_performance_metric_creation(self):
        """Test PerformanceMetric branded type creation."""
        metric = PerformanceMetric(95.5)
        assert isinstance(metric, float)
        assert metric == 95.5


class TestAgentTypeEnum:
    """Test AgentType enum values and behavior."""

    def test_agent_type_values(self):
        """Test all AgentType enum values."""
        assert AgentType.GENERAL.value == "general"
        assert AgentType.OPTIMIZER.value == "optimizer"
        assert AgentType.MONITOR.value == "monitor"
        assert AgentType.LEARNER.value == "learner"
        assert AgentType.COORDINATOR.value == "coordinator"
        assert AgentType.HEALER.value == "healer"
        assert AgentType.PLANNER.value == "planner"
        assert AgentType.RESOURCE_MANAGER.value == "resource_manager"

    def test_agent_type_enum_complete(self):
        """Test AgentType enum completeness."""
        expected_types = {
            "general",
            "optimizer",
            "monitor",
            "learner",
            "coordinator",
            "healer",
            "planner",
            "resource_manager",
        }
        actual_types = {at.value for at in AgentType}
        assert actual_types == expected_types


class TestAutonomyLevelEnum:
    """Test AutonomyLevel enum values and behavior."""

    def test_autonomy_level_values(self):
        """Test all AutonomyLevel enum values."""
        assert AutonomyLevel.MANUAL.value == "manual"
        assert AutonomyLevel.SUPERVISED.value == "supervised"
        assert AutonomyLevel.AUTONOMOUS.value == "autonomous"
        assert AutonomyLevel.FULL.value == "full"


class TestGoalPriorityEnum:
    """Test GoalPriority enum values and behavior."""

    def test_goal_priority_values(self):
        """Test all GoalPriority enum values."""
        assert GoalPriority.LOW.value == "low"
        assert GoalPriority.MEDIUM.value == "medium"
        assert GoalPriority.HIGH.value == "high"
        assert GoalPriority.CRITICAL.value == "critical"
        assert GoalPriority.EMERGENCY.value == "emergency"


class TestAgentStatusEnum:
    """Test AgentStatus enum values and behavior."""

    def test_agent_status_values(self):
        """Test all AgentStatus enum values."""
        assert AgentStatus.CREATED.value == "created"
        assert AgentStatus.INITIALIZING.value == "initializing"
        assert AgentStatus.ACTIVE.value == "active"
        assert AgentStatus.LEARNING.value == "learning"
        assert AgentStatus.OPTIMIZING.value == "optimizing"
        assert AgentStatus.PAUSED.value == "paused"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.TERMINATED.value == "terminated"


class TestActionTypeEnum:
    """Test ActionType enum values and behavior."""

    def test_action_type_values(self):
        """Test critical ActionType enum values."""
        assert ActionType.ANALYZE_PERFORMANCE.value == "analyze_performance"
        assert ActionType.OPTIMIZE_WORKFLOW.value == "optimize_workflow"
        assert ActionType.MONITOR_SYSTEM.value == "monitor_system"
        assert ActionType.EXECUTE_AUTOMATION.value == "execute_automation"
        assert ActionType.LEARN_PATTERN.value == "learn_pattern"

    def test_dangerous_action_types_exist(self):
        """Test that dangerous action types are properly defined for safety."""
        dangerous_actions = {
            ActionType.DELETE_ALL_DATA,
            ActionType.DISABLE_SECURITY,
            ActionType.MODIFY_SYSTEM_CONFIG,
            ActionType.EXECUTE_SYSTEM_COMMAND,
            ActionType.MODIFY_CRITICAL_CONFIG,
            ActionType.ACCESS_SENSITIVE_DATA,
        }
        assert len(dangerous_actions) == 6
        assert ActionType.DELETE_ALL_DATA.value == "delete_all_data"
        assert ActionType.DISABLE_SECURITY.value == "disable_security"


class TestIDCreationFunctions:
    """Test ID creation utility functions."""

    def test_create_agent_id(self):
        """Test agent ID creation function."""
        agent_id = create_agent_id()
        assert isinstance(agent_id, AgentId)
        assert agent_id.startswith("agent_")
        assert len(agent_id) > 20  # Should contain timestamp and uuid

    def test_create_goal_id(self):
        """Test goal ID creation function."""
        goal_id = create_goal_id()
        assert isinstance(goal_id, GoalId)
        assert goal_id.startswith("goal_")
        assert len(goal_id) > 20  # Should contain uuid

    def test_create_action_id(self):
        """Test action ID creation function."""
        action_id = create_action_id()
        assert isinstance(action_id, ActionId)
        assert action_id.startswith("action_")
        assert len(action_id) > 20  # Should contain uuid

    def test_create_experience_id(self):
        """Test experience ID creation function."""
        exp_id = create_experience_id()
        assert isinstance(exp_id, ExperienceId)
        assert exp_id.startswith("exp_")
        assert len(exp_id) > 20  # Should contain uuid

    def test_id_uniqueness(self):
        """Test that created IDs are unique."""
        ids = [create_agent_id() for _ in range(10)]
        assert len(set(ids)) == 10  # All should be unique


class TestAgentGoal:
    """Test AgentGoal dataclass functionality."""

    def test_agent_goal_creation(self):
        """Test AgentGoal creation with valid parameters."""
        goal = AgentGoal(
            goal_id=GoalId("test_goal"),
            description="Optimize system performance",
            priority=GoalPriority.HIGH,
            target_metrics={"efficiency": PerformanceMetric(0.9)},
            success_criteria=["Achieve 90% efficiency", "Reduce latency by 20%"],
            constraints={"max_cpu": 80.0, "safety_mode": True},
        )

        assert goal.goal_id == GoalId("test_goal")
        assert goal.description == "Optimize system performance"
        assert goal.priority == GoalPriority.HIGH
        assert goal.target_metrics["efficiency"] == PerformanceMetric(0.9)
        assert len(goal.success_criteria) == 2
        assert goal.constraints["max_cpu"] == 80.0

    def test_agent_goal_with_deadline(self):
        """Test AgentGoal with deadline functionality."""
        deadline = datetime.now(UTC) + timedelta(hours=24)
        goal = AgentGoal(
            goal_id=GoalId("deadline_goal"),
            description="Time-sensitive optimization",
            priority=GoalPriority.CRITICAL,
            target_metrics={"speed": PerformanceMetric(0.8)},
            success_criteria=["Complete within deadline"],
            constraints={},
            deadline=deadline,
        )

        assert goal.deadline == deadline
        assert not goal.is_overdue()

    def test_goal_overdue_check(self):
        """Test goal overdue detection."""
        past_deadline = datetime.now(UTC) - timedelta(hours=1)
        goal = AgentGoal(
            goal_id=GoalId("overdue_goal"),
            description="Overdue task",
            priority=GoalPriority.MEDIUM,
            target_metrics={"completion": PerformanceMetric(1.0)},
            success_criteria=["Complete task"],
            constraints={},
            deadline=past_deadline,
        )

        assert goal.is_overdue()

    def test_goal_urgency_score_calculation(self):
        """Test goal urgency score calculation."""
        # Test emergency priority
        emergency_goal = AgentGoal(
            goal_id=GoalId("emergency"),
            description="Emergency task",
            priority=GoalPriority.EMERGENCY,
            target_metrics={"response": PerformanceMetric(1.0)},
            success_criteria=["Immediate response"],
            constraints={},
        )
        assert emergency_goal.get_urgency_score() == ConfidenceScore(1.0)

        # Test low priority
        low_goal = AgentGoal(
            goal_id=GoalId("low"),
            description="Low priority task",
            priority=GoalPriority.LOW,
            target_metrics={"maintenance": PerformanceMetric(0.5)},
            success_criteria=["Routine maintenance"],
            constraints={},
        )
        assert low_goal.get_urgency_score() == ConfidenceScore(0.2)

    def test_goal_resource_achievability(self):
        """Test goal achievability with resources."""
        goal = AgentGoal(
            goal_id=GoalId("resource_goal"),
            description="Resource-intensive task",
            priority=GoalPriority.MEDIUM,
            target_metrics={"efficiency": PerformanceMetric(0.8)},
            success_criteria=["Use resources efficiently"],
            constraints={},
            resource_requirements={"cpu": 50.0, "memory": 1024.0},
        )

        # Test with sufficient resources
        sufficient_resources = {"cpu": 60.0, "memory": 2048.0}
        assert goal.is_achievable(sufficient_resources)

        # Test with insufficient resources
        insufficient_resources = {"cpu": 30.0, "memory": 512.0}
        assert not goal.is_achievable(insufficient_resources)


class TestAgentAction:
    """Test AgentAction dataclass functionality."""

    def test_agent_action_creation(self):
        """Test AgentAction creation with valid parameters."""
        action = AgentAction(
            action_id=ActionId("test_action"),
            agent_id=AgentId("test_agent"),
            action_type=ActionType.OPTIMIZE_WORKFLOW,
            parameters={"target": "performance", "threshold": 0.8},
            confidence=ConfidenceScore(0.9),
            estimated_impact=PerformanceMetric(0.7),
        )

        assert action.action_id == ActionId("test_action")
        assert action.agent_id == AgentId("test_agent")
        assert action.action_type == ActionType.OPTIMIZE_WORKFLOW
        assert action.parameters["target"] == "performance"
        assert action.confidence == ConfidenceScore(0.9)
        assert action.estimated_impact == PerformanceMetric(0.7)

    def test_action_confidence_check(self):
        """Test action confidence evaluation."""
        high_confidence_action = AgentAction(
            action_id=ActionId("high_conf"),
            agent_id=AgentId("agent"),
            action_type=ActionType.ANALYZE_PERFORMANCE,
            parameters={},
            confidence=ConfidenceScore(0.95),
            estimated_impact=PerformanceMetric(0.5),
        )

        low_confidence_action = AgentAction(
            action_id=ActionId("low_conf"),
            agent_id=AgentId("agent"),
            action_type=ActionType.ANALYZE_PERFORMANCE,
            parameters={},
            confidence=ConfidenceScore(0.6),
            estimated_impact=PerformanceMetric(0.5),
        )

        assert high_confidence_action.is_high_confidence()
        assert not low_confidence_action.is_high_confidence()
        assert not low_confidence_action.is_high_confidence(ConfidenceScore(0.8))

    def test_action_risk_score_calculation(self):
        """Test action risk score calculation."""
        # High confidence, low impact, safety validated
        safe_action = AgentAction(
            action_id=ActionId("safe"),
            agent_id=AgentId("agent"),
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={},
            confidence=ConfidenceScore(0.9),
            estimated_impact=PerformanceMetric(0.1),
            safety_validated=True,
        )

        # Low confidence, high impact, not validated
        risky_action = AgentAction(
            action_id=ActionId("risky"),
            agent_id=AgentId("agent"),
            action_type=ActionType.MODIFY_SYSTEM_CONFIG,
            parameters={},
            confidence=ConfidenceScore(0.3),
            estimated_impact=PerformanceMetric(0.9),
            safety_validated=False,
        )

        safe_risk = safe_action.get_risk_score()
        risky_risk = risky_action.get_risk_score()

        assert safe_risk < risky_risk
        assert safe_risk < RiskScore(0.3)
        assert risky_risk > RiskScore(0.5)

    def test_action_prerequisites_check(self):
        """Test action prerequisite validation."""
        action = AgentAction(
            action_id=ActionId("dependent"),
            agent_id=AgentId("agent"),
            action_type=ActionType.OPTIMIZE_WORKFLOW,
            parameters={},
            prerequisites=[ActionId("prereq1"), ActionId("prereq2")],
        )

        # Test with incomplete prerequisites
        completed_partial = {ActionId("prereq1")}
        assert not action.can_execute_now(completed_partial)

        # Test with all prerequisites completed
        completed_all = {ActionId("prereq1"), ActionId("prereq2")}
        assert action.can_execute_now(completed_all)

    def test_action_cost_calculation(self):
        """Test action cost calculation."""
        action = AgentAction(
            action_id=ActionId("costly"),
            agent_id=AgentId("agent"),
            action_type=ActionType.EXECUTE_AUTOMATION,
            parameters={},
            resource_cost={"cpu": 25.0, "memory": 512.0, "network": 10.0},
        )

        total_cost = action.estimate_total_cost()
        assert total_cost == 547.0  # 25.0 + 512.0 + 10.0


class TestLearningExperience:
    """Test LearningExperience dataclass functionality."""

    def test_learning_experience_creation(self):
        """Test LearningExperience creation with valid parameters."""
        action = AgentAction(
            action_id=ActionId("learned_action"),
            agent_id=AgentId("learner_agent"),
            action_type=ActionType.LEARN_PATTERN,
            parameters={"pattern": "performance_optimization"},
        )

        experience = LearningExperience(
            experience_id=ExperienceId("exp_123"),
            agent_id=AgentId("learner_agent"),
            context={"system_load": 0.7, "time_of_day": "morning"},
            action_taken=action,
            outcome={"improvement": 0.15, "success_rate": 0.92},
            success=True,
            learning_value=ConfidenceScore(0.8),
            performance_impact=PerformanceMetric(0.15),
        )

        assert experience.experience_id == ExperienceId("exp_123")
        assert experience.success
        assert experience.learning_value == ConfidenceScore(0.8)
        assert experience.performance_impact == PerformanceMetric(0.15)

    def test_experience_pattern_extraction(self):
        """Test learning pattern extraction from experience."""
        action = AgentAction(
            action_id=ActionId("test_action"),
            agent_id=AgentId("agent"),
            action_type=ActionType.OPTIMIZE_WORKFLOW,
            parameters={"optimization_level": 0.8},
        )

        success_experience = LearningExperience(
            experience_id=ExperienceId("success_exp"),
            agent_id=AgentId("agent"),
            context={"load": 0.5, "users": 100, "active": True},
            action_taken=action,
            outcome={"performance_gain": 0.2},
            success=True,
            learning_value=ConfidenceScore(0.9),
            performance_impact=PerformanceMetric(0.2),
        )

        patterns = success_experience.extract_patterns()

        assert "success_indicators" in patterns
        assert "failure_indicators" in patterns
        assert "optimal_parameters" in patterns
        assert "performance_correlations" in patterns
        assert len(patterns["success_indicators"]) > 0
        assert "optimization_level" in patterns["optimal_parameters"]

    def test_experience_learning_weight(self):
        """Test learning weight calculation."""
        # Normal experience
        normal_exp = LearningExperience(
            experience_id=ExperienceId("normal"),
            agent_id=AgentId("agent"),
            context={},
            action_taken=AgentAction(
                action_id=ActionId("action"),
                agent_id=AgentId("agent"),
                action_type=ActionType.MONITOR_SYSTEM,
                parameters={},
            ),
            outcome={},
            success=True,
            learning_value=ConfidenceScore(0.5),
            performance_impact=PerformanceMetric(0.1),
        )

        # High-impact failure with unexpected results
        impactful_exp = LearningExperience(
            experience_id=ExperienceId("impactful"),
            agent_id=AgentId("agent"),
            context={},
            action_taken=AgentAction(
                action_id=ActionId("action"),
                agent_id=AgentId("agent"),
                action_type=ActionType.OPTIMIZE_WORKFLOW,
                parameters={},
            ),
            outcome={},
            success=False,
            learning_value=ConfidenceScore(0.8),
            performance_impact=PerformanceMetric(0.7),
            unexpected_results=["system_slowdown", "memory_leak"],
        )

        normal_weight = normal_exp.get_learning_weight()
        impactful_weight = impactful_exp.get_learning_weight()

        assert impactful_weight > normal_weight
        assert impactful_weight <= 1.0  # Should be capped at 1.0


class TestAgentConfiguration:
    """Test AgentConfiguration dataclass functionality."""

    def test_agent_configuration_creation(self):
        """Test AgentConfiguration creation with valid parameters."""
        config = AgentConfiguration(
            agent_type=AgentType.OPTIMIZER,
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            max_concurrent_actions=5,
            decision_threshold=ConfidenceScore(0.8),
            risk_tolerance=RiskScore(0.4),
            learning_rate=0.15,
        )

        assert config.agent_type == AgentType.OPTIMIZER
        assert config.autonomy_level == AutonomyLevel.AUTONOMOUS
        assert config.max_concurrent_actions == 5
        assert config.decision_threshold == ConfidenceScore(0.8)
        assert config.risk_tolerance == RiskScore(0.4)
        assert config.learning_rate == 0.15

    def test_configuration_action_limits_check(self):
        """Test configuration action limits validation."""
        config = AgentConfiguration(
            agent_type=AgentType.MONITOR,
            autonomy_level=AutonomyLevel.SUPERVISED,
            decision_threshold=ConfidenceScore(0.7),
            risk_tolerance=RiskScore(0.3),
            resource_limits={"cpu": 50.0, "memory": 1024.0},
        )

        # Action within limits
        safe_action = AgentAction(
            action_id=ActionId("safe"),
            agent_id=AgentId("agent"),
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={},
            confidence=ConfidenceScore(0.8),
            estimated_impact=PerformanceMetric(0.2),
            resource_cost={"cpu": 30.0, "memory": 512.0},
            safety_validated=True,
        )

        # Action exceeding limits
        risky_action = AgentAction(
            action_id=ActionId("risky"),
            agent_id=AgentId("agent"),
            action_type=ActionType.MODIFY_SYSTEM_CONFIG,
            parameters={},
            confidence=ConfidenceScore(0.5),  # Below threshold
            estimated_impact=PerformanceMetric(0.9),
            resource_cost={"cpu": 70.0},  # Exceeds limit
        )

        assert config.is_action_within_limits(safe_action)
        assert not config.is_action_within_limits(risky_action)

    def test_configuration_human_approval_logic(self):
        """Test human approval requirement logic."""
        # Manual mode configuration
        manual_config = AgentConfiguration(
            agent_type=AgentType.GENERAL,
            autonomy_level=AutonomyLevel.MANUAL,
            human_approval_required=False,  # Still requires approval due to manual mode
        )

        # Autonomous configuration
        auto_config = AgentConfiguration(
            agent_type=AgentType.OPTIMIZER,
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            human_approval_required=False,
        )

        normal_action = AgentAction(
            action_id=ActionId("normal"),
            agent_id=AgentId("agent"),
            action_type=ActionType.OPTIMIZE_WORKFLOW,
            parameters={},
            confidence=ConfidenceScore(0.8),
            estimated_impact=PerformanceMetric(0.5),
        )

        high_risk_action = AgentAction(
            action_id=ActionId("high_risk"),
            agent_id=AgentId("agent"),
            action_type=ActionType.MODIFY_CRITICAL_CONFIG,
            parameters={},
            confidence=ConfidenceScore(0.3),
            estimated_impact=PerformanceMetric(0.95),
        )

        # Manual mode always requires approval
        assert manual_config.should_request_human_approval(normal_action)

        # Auto mode doesn't require approval for normal actions
        assert not auto_config.should_request_human_approval(normal_action)

        # High-risk actions require approval regardless
        assert auto_config.should_request_human_approval(high_risk_action)


class TestAutonomousAgentError:
    """Test AutonomousAgentError class methods."""

    def test_agent_not_found_error(self):
        """Test agent not found error creation."""
        error = AutonomousAgentError.agent_not_found(AgentId("missing_agent"))
        assert error.field_name == "agent_id"
        assert error.value == AgentId("missing_agent")
        assert "missing_agent" in error.message

    def test_action_too_risky_error(self):
        """Test action too risky error creation."""
        error = AutonomousAgentError.action_too_risky(RiskScore(0.9), RiskScore(0.5))
        assert error.field_name == "risk_score"
        assert error.value == RiskScore(0.9)
        assert "0.9" in error.message
        assert "0.5" in error.message

    def test_resource_limit_exceeded_error(self):
        """Test resource limit exceeded error creation."""
        error = AutonomousAgentError.resource_limit_exceeded("cpu", 75.0, 50.0)
        assert error.field_name == "resource_usage"
        assert error.value == 75.0
        assert "cpu" in error.message
        assert "75.0" in error.message
        assert "50.0" in error.message


class TestDefaultConfigurations:
    """Test default agent configurations."""

    def test_default_config_exists_for_all_types(self):
        """Test that default configurations exist for all agent types."""
        core_types = [
            AgentType.GENERAL,
            AgentType.OPTIMIZER,
            AgentType.MONITOR,
            AgentType.LEARNER,
            AgentType.HEALER,
        ]

        for agent_type in core_types:
            config = get_default_config(agent_type)
            assert isinstance(config, AgentConfiguration)
            assert config.agent_type == agent_type

    def test_default_config_properties(self):
        """Test default configuration properties."""
        optimizer_config = get_default_config(AgentType.OPTIMIZER)
        assert optimizer_config.autonomy_level == AutonomyLevel.AUTONOMOUS
        assert optimizer_config.decision_threshold == ConfidenceScore(0.8)
        assert optimizer_config.risk_tolerance == RiskScore(0.4)

        monitor_config = get_default_config(AgentType.MONITOR)
        assert monitor_config.decision_threshold == ConfidenceScore(0.9)
        assert monitor_config.risk_tolerance == RiskScore(0.2)

    def test_fallback_to_general_config(self):
        """Test fallback to general configuration for unknown types."""
        # Create a mock agent type that doesn't exist in defaults
        general_config = get_default_config(AgentType.GENERAL)
        coordinator_config = get_default_config(AgentType.COORDINATOR)

        # Should fallback to general config
        assert coordinator_config.agent_type == general_config.agent_type
        assert coordinator_config.autonomy_level == general_config.autonomy_level


class TestPropertyBasedValidation:
    """Property-based tests for autonomous systems."""

    @given(st.text(min_size=1, max_size=50))
    def test_agent_id_properties(self, agent_name):
        """Property test for agent ID creation."""
        agent_id = AgentId(agent_name)
        assert isinstance(agent_id, str)
        assert agent_id == agent_name

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_confidence_score_properties(self, confidence):
        """Property test for confidence score validation."""
        if 0.0 <= confidence <= 1.0:
            conf_score = ConfidenceScore(confidence)
            assert conf_score == confidence
            assert isinstance(conf_score, float)

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_risk_score_properties(self, risk):
        """Property test for risk score validation."""
        if 0.0 <= risk <= 1.0:
            risk_score = RiskScore(risk)
            assert risk_score == risk
            assert isinstance(risk_score, float)

    @given(st.integers(min_value=1, max_value=10))
    def test_max_concurrent_actions_properties(self, max_actions):
        """Property test for max concurrent actions validation."""
        config = AgentConfiguration(
            agent_type=AgentType.GENERAL,
            autonomy_level=AutonomyLevel.SUPERVISED,
            max_concurrent_actions=max_actions,
        )
        assert config.max_concurrent_actions == max_actions
        assert 1 <= config.max_concurrent_actions <= 10


class TestIntegrationScenarios:
    """Integration test scenarios for autonomous systems."""

    def test_complete_agent_lifecycle(self):
        """Test complete autonomous agent lifecycle."""
        # Create agent configuration
        config = AgentConfiguration(
            agent_type=AgentType.OPTIMIZER,
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            decision_threshold=ConfidenceScore(0.8),
            risk_tolerance=RiskScore(0.4),
        )

        # Create agent goal
        goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Optimize system performance",
            priority=GoalPriority.HIGH,
            target_metrics={"efficiency": PerformanceMetric(0.9)},
            success_criteria=["Achieve 90% efficiency"],
            constraints={"max_resource_usage": 0.8},
        )

        # Create agent action
        action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.OPTIMIZE_WORKFLOW,
            parameters={"optimization_target": "efficiency"},
            goal_id=goal.goal_id,
            confidence=ConfidenceScore(0.85),
            estimated_impact=PerformanceMetric(0.15),
        )

        # Validate action within configuration limits
        assert config.is_action_within_limits(action)
        assert not config.should_request_human_approval(action)

        # Create learning experience
        experience = LearningExperience(
            experience_id=create_experience_id(),
            agent_id=action.agent_id,
            context={"system_state": "normal", "load": 0.6},
            action_taken=action,
            outcome={"efficiency_gain": 0.12},
            success=True,
            learning_value=ConfidenceScore(0.9),
            performance_impact=PerformanceMetric(0.12),
        )

        # Validate complete workflow
        assert goal.priority == GoalPriority.HIGH
        assert action.goal_id == goal.goal_id
        assert experience.action_taken == action
        assert experience.success

    def test_multi_agent_coordination_scenario(self):
        """Test multi-agent coordination scenario."""
        # Create coordinator agent (configuration for reference)
        AgentConfiguration(
            agent_type=AgentType.COORDINATOR,
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            max_concurrent_actions=5,
        )

        # Create shared goal
        shared_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Coordinate system optimization",
            priority=GoalPriority.CRITICAL,
            target_metrics={"system_efficiency": PerformanceMetric(0.95)},
            success_criteria=["All subsystems optimized", "No conflicts detected"],
            constraints={"coordination_required": True},
        )

        # Create actions for different agents
        agents = [create_agent_id() for _ in range(3)]
        actions = []

        for i, agent_id in enumerate(agents):
            action = AgentAction(
                action_id=create_action_id(),
                agent_id=agent_id,
                action_type=ActionType.COORDINATE_AGENTS,
                parameters={"subsystem": f"subsystem_{i}"},
                goal_id=shared_goal.goal_id,
                confidence=ConfidenceScore(0.8),
                estimated_impact=PerformanceMetric(0.3),
            )
            actions.append(action)

        # Validate coordination scenario
        assert len(actions) == 3
        assert all(action.goal_id == shared_goal.goal_id for action in actions)
        assert shared_goal.priority == GoalPriority.CRITICAL

    def test_learning_and_adaptation_scenario(self):
        """Test learning and adaptation scenario."""
        # Create learner configuration (for reference)
        get_default_config(AgentType.LEARNER)
        agent_id = create_agent_id()

        # Create series of learning experiences
        experiences = []
        for i in range(5):
            action = AgentAction(
                action_id=create_action_id(),
                agent_id=agent_id,
                action_type=ActionType.LEARN_PATTERN,
                parameters={"pattern_type": f"pattern_{i}"},
                confidence=ConfidenceScore(0.6 + i * 0.05),  # Increasing confidence
            )

            experience = LearningExperience(
                experience_id=create_experience_id(),
                agent_id=agent_id,
                context={"iteration": i, "complexity": i * 0.2},
                action_taken=action,
                outcome={"pattern_recognition": 0.7 + i * 0.05},
                success=i >= 2,  # Fails first 2, succeeds later
                learning_value=ConfidenceScore(0.8),
                performance_impact=PerformanceMetric(i * 0.1),
            )
            experiences.append(experience)

        # Analyze learning progression
        success_rate = sum(1 for exp in experiences if exp.success) / len(experiences)
        avg_confidence = sum(exp.action_taken.confidence for exp in experiences) / len(
            experiences
        )

        assert success_rate == 0.6  # 3 out of 5 succeeded
        assert avg_confidence > ConfidenceScore(0.6)  # Confidence improved over time
        assert (
            experiences[-1].action_taken.confidence
            > experiences[0].action_taken.confidence
        )

    def test_risk_management_scenario(self):
        """Test comprehensive risk management scenario."""
        # Create conservative configuration
        conservative_config = AgentConfiguration(
            agent_type=AgentType.MONITOR,
            autonomy_level=AutonomyLevel.SUPERVISED,
            decision_threshold=ConfidenceScore(0.9),
            risk_tolerance=RiskScore(0.2),
            human_approval_required=True,
        )

        # Create potentially dangerous actions
        dangerous_actions = [
            AgentAction(
                action_id=create_action_id(),
                agent_id=create_agent_id(),
                action_type=ActionType.DELETE_ALL_DATA,
                parameters={},
                confidence=ConfidenceScore(0.6),  # Lower confidence for realistic risk
                estimated_impact=PerformanceMetric(1.0),
                safety_validated=False,  # Explicitly not validated
            ),
            AgentAction(
                action_id=create_action_id(),
                agent_id=create_agent_id(),
                action_type=ActionType.MODIFY_CRITICAL_CONFIG,
                parameters={},
                confidence=ConfidenceScore(0.7),  # Lower confidence for realistic risk
                estimated_impact=PerformanceMetric(0.9),
                safety_validated=False,  # Explicitly not validated
            ),
        ]

        # All dangerous actions should require human approval
        for action in dangerous_actions:
            assert conservative_config.should_request_human_approval(action)
            # For dangerous actions with high impact and no safety validation,
            # risk should be elevated regardless of specific threshold
            risk_score = action.get_risk_score()
            assert risk_score >= RiskScore(0.3)  # Should be significantly risky
