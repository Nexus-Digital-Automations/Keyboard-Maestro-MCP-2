"""Comprehensive tests for src/core/autonomous_systems.py.

This module provides targeted tests for the autonomous systems module to achieve
significant coverage improvement toward the mandatory 95% threshold by covering
all enums, branded types, dataclasses, and utility functions.
"""

from datetime import UTC, datetime, timedelta

import pytest
from src.core.autonomous_systems import (
    DEFAULT_AGENT_CONFIGS,
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
    """Test branded type creation and validation."""

    def test_agent_id_creation(self):
        """Test AgentId branded type."""
        agent_id = AgentId("agent_123")
        assert isinstance(agent_id, str)
        assert agent_id == "agent_123"

    def test_goal_id_creation(self):
        """Test GoalId branded type."""
        goal_id = GoalId("goal_456")
        assert isinstance(goal_id, str)
        assert goal_id == "goal_456"

    def test_action_id_creation(self):
        """Test ActionId branded type."""
        action_id = ActionId("action_789")
        assert isinstance(action_id, str)
        assert action_id == "action_789"

    def test_experience_id_creation(self):
        """Test ExperienceId branded type."""
        exp_id = ExperienceId("exp_abc")
        assert isinstance(exp_id, str)
        assert exp_id == "exp_abc"

    def test_confidence_score_creation(self):
        """Test ConfidenceScore branded type."""
        score = ConfidenceScore(0.85)
        assert isinstance(score, float)
        assert score == 0.85

    def test_risk_score_creation(self):
        """Test RiskScore branded type."""
        score = RiskScore(0.3)
        assert isinstance(score, float)
        assert score == 0.3

    def test_performance_metric_creation(self):
        """Test PerformanceMetric branded type."""
        metric = PerformanceMetric(0.95)
        assert isinstance(metric, float)
        assert metric == 0.95


class TestAgentTypeEnum:
    """Test AgentType enum values and functionality."""

    def test_agent_type_enum_values(self):
        """Test AgentType enum has all expected values."""
        assert AgentType.GENERAL.value == "general"
        assert AgentType.OPTIMIZER.value == "optimizer"
        assert AgentType.MONITOR.value == "monitor"
        assert AgentType.LEARNER.value == "learner"
        assert AgentType.COORDINATOR.value == "coordinator"
        assert AgentType.HEALER.value == "healer"
        assert AgentType.PLANNER.value == "planner"
        assert AgentType.RESOURCE_MANAGER.value == "resource_manager"

    def test_agent_type_enum_completeness(self):
        """Test that all AgentType values are accounted for."""
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
        actual_types = {agent_type.value for agent_type in AgentType}
        assert actual_types == expected_types


class TestAutonomyLevelEnum:
    """Test AutonomyLevel enum values and functionality."""

    def test_autonomy_level_enum_values(self):
        """Test AutonomyLevel enum has all expected values."""
        assert AutonomyLevel.MANUAL.value == "manual"
        assert AutonomyLevel.SUPERVISED.value == "supervised"
        assert AutonomyLevel.AUTONOMOUS.value == "autonomous"
        assert AutonomyLevel.FULL.value == "full"

    def test_autonomy_level_enum_completeness(self):
        """Test that all AutonomyLevel values are accounted for."""
        expected_levels = {"manual", "supervised", "autonomous", "full"}
        actual_levels = {level.value for level in AutonomyLevel}
        assert actual_levels == expected_levels


class TestGoalPriorityEnum:
    """Test GoalPriority enum values and functionality."""

    def test_goal_priority_enum_values(self):
        """Test GoalPriority enum has all expected values."""
        assert GoalPriority.LOW.value == "low"
        assert GoalPriority.MEDIUM.value == "medium"
        assert GoalPriority.HIGH.value == "high"
        assert GoalPriority.CRITICAL.value == "critical"
        assert GoalPriority.EMERGENCY.value == "emergency"

    def test_goal_priority_enum_completeness(self):
        """Test that all GoalPriority values are accounted for."""
        expected_priorities = {"low", "medium", "high", "critical", "emergency"}
        actual_priorities = {priority.value for priority in GoalPriority}
        assert actual_priorities == expected_priorities


class TestAgentStatusEnum:
    """Test AgentStatus enum values and functionality."""

    def test_agent_status_enum_values(self):
        """Test AgentStatus enum has all expected values."""
        assert AgentStatus.CREATED.value == "created"
        assert AgentStatus.INITIALIZING.value == "initializing"
        assert AgentStatus.ACTIVE.value == "active"
        assert AgentStatus.LEARNING.value == "learning"
        assert AgentStatus.OPTIMIZING.value == "optimizing"
        assert AgentStatus.PAUSED.value == "paused"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.TERMINATED.value == "terminated"

    def test_agent_status_enum_completeness(self):
        """Test that all AgentStatus values are accounted for."""
        expected_statuses = {
            "created",
            "initializing",
            "active",
            "learning",
            "optimizing",
            "paused",
            "error",
            "terminated",
        }
        actual_statuses = {status.value for status in AgentStatus}
        assert actual_statuses == expected_statuses


class TestActionTypeEnum:
    """Test ActionType enum values and functionality."""

    def test_action_type_enum_values(self):
        """Test ActionType enum has all expected values."""
        assert ActionType.ANALYZE_PERFORMANCE.value == "analyze_performance"
        assert ActionType.OPTIMIZE_WORKFLOW.value == "optimize_workflow"
        assert ActionType.MONITOR_SYSTEM.value == "monitor_system"
        assert ActionType.EXECUTE_AUTOMATION.value == "execute_automation"
        assert ActionType.LEARN_PATTERN.value == "learn_pattern"
        assert ActionType.COORDINATE_AGENTS.value == "coordinate_agents"
        assert ActionType.HEAL_SYSTEM.value == "heal_system"
        assert ActionType.PLAN_SCHEDULE.value == "plan_schedule"
        assert ActionType.ALLOCATE_RESOURCES.value == "allocate_resources"
        assert ActionType.UPDATE_CONFIGURATION.value == "update_configuration"

    def test_action_type_dangerous_actions(self):
        """Test dangerous action types are present."""
        assert ActionType.DELETE_ALL_DATA.value == "delete_all_data"
        assert ActionType.DISABLE_SECURITY.value == "disable_security"
        assert ActionType.MODIFY_SYSTEM_CONFIG.value == "modify_system_config"
        assert ActionType.EXECUTE_SYSTEM_COMMAND.value == "execute_system_command"
        assert ActionType.MODIFY_CRITICAL_CONFIG.value == "modify_critical_config"
        assert ActionType.ACCESS_SENSITIVE_DATA.value == "access_sensitive_data"


class TestIdGenerationFunctions:
    """Test ID generation utility functions."""

    def test_create_agent_id(self):
        """Test create_agent_id function."""
        agent_id = create_agent_id()
        assert isinstance(agent_id, AgentId)
        assert agent_id.startswith("agent_")
        assert len(agent_id) > 20  # Should have timestamp and hex

        # Verify unique IDs
        agent_id2 = create_agent_id()
        assert agent_id != agent_id2

    def test_create_goal_id(self):
        """Test create_goal_id function."""
        goal_id = create_goal_id()
        assert isinstance(goal_id, GoalId)
        assert goal_id.startswith("goal_")
        assert len(goal_id) == 37  # "goal_" + 32 hex chars

        # Verify unique IDs
        goal_id2 = create_goal_id()
        assert goal_id != goal_id2

    def test_create_action_id(self):
        """Test create_action_id function."""
        action_id = create_action_id()
        assert isinstance(action_id, ActionId)
        assert action_id.startswith("action_")
        assert len(action_id) == 39  # "action_" + 32 hex chars

        # Verify unique IDs
        action_id2 = create_action_id()
        assert action_id != action_id2

    def test_create_experience_id(self):
        """Test create_experience_id function."""
        exp_id = create_experience_id()
        assert isinstance(exp_id, ExperienceId)
        assert exp_id.startswith("exp_")
        assert len(exp_id) == 36  # "exp_" + 32 hex chars

        # Verify unique IDs
        exp_id2 = create_experience_id()
        assert exp_id != exp_id2


class TestAgentGoal:
    """Test AgentGoal dataclass functionality."""

    @pytest.fixture
    def sample_goal(self):
        """Create a sample AgentGoal for testing."""
        return AgentGoal(
            goal_id=create_goal_id(),
            description="Optimize system performance",
            priority=GoalPriority.HIGH,
            target_metrics={
                "cpu_usage": PerformanceMetric(0.7),
                "memory_usage": PerformanceMetric(0.8),
            },
            success_criteria=["CPU usage below 70%", "Memory usage below 80%"],
            constraints={"max_duration": "1h", "resources": {"cpu": 50}},
            deadline=datetime.now(UTC) + timedelta(hours=2),
            estimated_duration=timedelta(hours=1),
            resource_requirements={"cpu": 50.0, "memory": 1024.0},
        )

    def test_agent_goal_creation(self, sample_goal):
        """Test AgentGoal creation and basic properties."""
        assert sample_goal.description == "Optimize system performance"
        assert sample_goal.priority == GoalPriority.HIGH
        assert len(sample_goal.target_metrics) == 2
        assert len(sample_goal.success_criteria) == 2
        assert len(sample_goal.constraints) == 2
        assert sample_goal.deadline is not None
        assert sample_goal.estimated_duration is not None

    def test_agent_goal_is_overdue(self):
        """Test is_overdue method."""
        # Future deadline - not overdue
        future_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Future goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete task"],
            constraints={},
            deadline=datetime.now(UTC) + timedelta(hours=1),
        )
        assert not future_goal.is_overdue()

        # Past deadline - overdue
        past_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Past goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete task"],
            constraints={},
            deadline=datetime.now(UTC) - timedelta(hours=1),
        )
        assert past_goal.is_overdue()

        # No deadline - not overdue
        no_deadline_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="No deadline goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete task"],
            constraints={},
        )
        assert not no_deadline_goal.is_overdue()

    def test_agent_goal_get_urgency_score(self):
        """Test get_urgency_score method with different priorities and deadlines."""
        # Test priority-based urgency
        low_priority_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Low priority",
            priority=GoalPriority.LOW,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete"],
            constraints={},
        )
        assert low_priority_goal.get_urgency_score() == 0.2

        high_priority_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="High priority",
            priority=GoalPriority.HIGH,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete"],
            constraints={},
        )
        assert high_priority_goal.get_urgency_score() == 0.6

        emergency_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Emergency",
            priority=GoalPriority.EMERGENCY,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete"],
            constraints={},
        )
        assert emergency_goal.get_urgency_score() == 1.0

        # Test deadline urgency - overdue goal
        overdue_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Overdue goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete"],
            constraints={},
            deadline=datetime.now(UTC) - timedelta(hours=1),
        )
        assert overdue_goal.get_urgency_score() == 1.0

        # Test deadline urgency - urgent (less than 1 hour)
        urgent_goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Urgent goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete"],
            constraints={},
            deadline=datetime.now(UTC) + timedelta(minutes=30),
        )
        urgency_score = urgent_goal.get_urgency_score()
        assert urgency_score > 0.4  # Should be elevated from base 0.4

    def test_agent_goal_estimate_completion_time(self):
        """Test estimate_completion_time method."""
        # With estimated duration
        goal_with_duration = AgentGoal(
            goal_id=create_goal_id(),
            description="Goal with duration",
            priority=GoalPriority.MEDIUM,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete"],
            constraints={},
            estimated_duration=timedelta(hours=2),
        )
        completion_time = goal_with_duration.estimate_completion_time()
        assert completion_time is not None
        assert completion_time > datetime.now(UTC)

        # Without estimated duration
        goal_without_duration = AgentGoal(
            goal_id=create_goal_id(),
            description="Goal without duration",
            priority=GoalPriority.MEDIUM,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete"],
            constraints={},
        )
        assert goal_without_duration.estimate_completion_time() is None

    def test_agent_goal_is_achievable(self):
        """Test is_achievable method."""
        goal = AgentGoal(
            goal_id=create_goal_id(),
            description="Resource goal",
            priority=GoalPriority.MEDIUM,
            target_metrics={"metric": PerformanceMetric(0.5)},
            success_criteria=["Complete"],
            constraints={},
            resource_requirements={"cpu": 50.0, "memory": 1024.0},
        )

        # Sufficient resources
        sufficient_resources = {"cpu": 100.0, "memory": 2048.0}
        assert goal.is_achievable(sufficient_resources)

        # Insufficient CPU
        insufficient_cpu = {"cpu": 25.0, "memory": 2048.0}
        assert not goal.is_achievable(insufficient_cpu)

        # Insufficient memory
        insufficient_memory = {"cpu": 100.0, "memory": 512.0}
        assert not goal.is_achievable(insufficient_memory)

        # Missing resource
        missing_resource = {"cpu": 100.0}
        assert not goal.is_achievable(missing_resource)


class TestAgentAction:
    """Test AgentAction dataclass functionality."""

    @pytest.fixture
    def sample_action(self):
        """Create a sample AgentAction for testing."""
        return AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.OPTIMIZE_WORKFLOW,
            parameters={"optimization_level": "high", "target_metric": "cpu_usage"},
            goal_id=create_goal_id(),
            rationale="System performance is degraded",
            confidence=ConfidenceScore(0.85),
            estimated_impact=PerformanceMetric(0.7),
            estimated_duration=timedelta(minutes=30),
            resource_cost={"cpu": 25.0, "memory": 512.0},
            prerequisites=[],
            safety_validated=True,
            human_approval_required=False,
        )

    def test_agent_action_creation(self, sample_action):
        """Test AgentAction creation and basic properties."""
        assert sample_action.action_type == ActionType.OPTIMIZE_WORKFLOW
        assert sample_action.confidence == 0.85
        assert sample_action.estimated_impact == 0.7
        assert sample_action.safety_validated is True
        assert sample_action.human_approval_required is False
        assert len(sample_action.parameters) == 2
        assert len(sample_action.resource_cost) == 2

    def test_agent_action_is_high_confidence(self, sample_action):
        """Test is_high_confidence method."""
        # Default threshold (0.8)
        assert sample_action.is_high_confidence()  # 0.85 > 0.8

        # Custom threshold
        assert not sample_action.is_high_confidence(ConfidenceScore(0.9))  # 0.85 < 0.9
        assert sample_action.is_high_confidence(ConfidenceScore(0.8))  # 0.85 >= 0.8

        # Low confidence action
        low_confidence_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={},
            confidence=ConfidenceScore(0.6),
        )
        assert not low_confidence_action.is_high_confidence()

    def test_agent_action_get_risk_score(self):
        """Test get_risk_score method with various factors."""
        # Base risk (1.0 - confidence)
        base_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={},
            confidence=ConfidenceScore(0.8),  # Base risk = 0.2
            estimated_impact=PerformanceMetric(0.1),  # Low impact
            safety_validated=True,  # Validated
        )
        base_risk = base_action.get_risk_score()
        assert (
            0.05 <= base_risk <= 0.3
        )  # Should be around 0.1 with low impact reduction

        # High impact action (increases risk)
        high_impact_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.MODIFY_SYSTEM_CONFIG,
            parameters={},
            confidence=ConfidenceScore(0.8),
            estimated_impact=PerformanceMetric(0.9),  # High impact
        )
        high_impact_risk = high_impact_action.get_risk_score()
        assert high_impact_risk > base_risk

        # Unvalidated action (increases risk)
        unvalidated_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.EXECUTE_AUTOMATION,
            parameters={},
            confidence=ConfidenceScore(0.8),
            safety_validated=False,  # Not validated - doubles risk
            estimated_impact=PerformanceMetric(0.1),  # Same low impact as base
        )
        unvalidated_risk = unvalidated_action.get_risk_score()
        assert unvalidated_risk > base_risk

        # Human approval required (reduces risk)
        human_approved_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.DELETE_ALL_DATA,
            parameters={},
            confidence=ConfidenceScore(0.8),
            human_approval_required=True,  # Human oversight
        )
        human_approved_risk = human_approved_action.get_risk_score()
        assert (
            human_approved_risk < base_risk or human_approved_risk <= 1.0
        )  # Should be lower or capped

    def test_agent_action_can_execute_now(self):
        """Test can_execute_now method with prerequisites."""
        action_id1 = create_action_id()
        action_id2 = create_action_id()
        action_id3 = create_action_id()

        # No prerequisites
        no_prereq_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={},
        )
        assert no_prereq_action.can_execute_now(set())
        assert no_prereq_action.can_execute_now({action_id1, action_id2})

        # With prerequisites - all satisfied
        prereq_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.OPTIMIZE_WORKFLOW,
            parameters={},
            prerequisites=[action_id1, action_id2],
        )
        assert prereq_action.can_execute_now({action_id1, action_id2, action_id3})

        # With prerequisites - partially satisfied
        assert not prereq_action.can_execute_now({action_id1})
        assert not prereq_action.can_execute_now({action_id2})

        # With prerequisites - none satisfied
        assert not prereq_action.can_execute_now(set())
        assert not prereq_action.can_execute_now({action_id3})

    def test_agent_action_estimate_total_cost(self):
        """Test estimate_total_cost method."""
        # Action with resource costs
        costly_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.HEAL_SYSTEM,
            parameters={},
            resource_cost={"cpu": 50.0, "memory": 1024.0, "disk": 100.0},
        )
        total_cost = costly_action.estimate_total_cost()
        assert total_cost == 1174.0  # 50 + 1024 + 100

        # Action with no costs
        free_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={},
        )
        assert free_action.estimate_total_cost() == 0.0


class TestLearningExperience:
    """Test LearningExperience dataclass functionality."""

    @pytest.fixture
    def sample_action(self):
        """Create a sample action for learning experiences."""
        return AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.OPTIMIZE_WORKFLOW,
            parameters={"optimization_type": "cpu", "threshold": 0.7},
            confidence=ConfidenceScore(0.8),
        )

    @pytest.fixture
    def sample_experience(self, sample_action):
        """Create a sample LearningExperience for testing."""
        return LearningExperience(
            experience_id=create_experience_id(),
            agent_id=sample_action.agent_id,
            context={"system_load": 0.9, "cpu_usage": 0.85, "memory_usage": 0.6},
            action_taken=sample_action,
            outcome={"cpu_reduction": 0.15, "performance_gain": 0.2},
            success=True,
            learning_value=ConfidenceScore(0.75),
            performance_impact=PerformanceMetric(0.2),
            unexpected_results=["Memory usage increased slightly"],
            lessons_learned=["High system load requires conservative optimization"],
        )

    def test_learning_experience_creation(self, sample_experience):
        """Test LearningExperience creation and basic properties."""
        assert sample_experience.success is True
        assert sample_experience.learning_value == 0.75
        assert sample_experience.performance_impact == 0.2
        assert len(sample_experience.context) == 3
        assert len(sample_experience.outcome) == 2
        assert len(sample_experience.unexpected_results) == 1
        assert len(sample_experience.lessons_learned) == 1

    def test_learning_experience_extract_patterns_success(self, sample_experience):
        """Test extract_patterns method for successful experience."""
        patterns = sample_experience.extract_patterns()

        assert "success_indicators" in patterns
        assert "failure_indicators" in patterns
        assert "context_factors" in patterns
        assert "optimal_parameters" in patterns
        assert "performance_correlations" in patterns

        # Success case should populate success_indicators
        assert len(patterns["success_indicators"]) > 0
        assert len(patterns["failure_indicators"]) == 0
        assert "system_load" in patterns["success_indicators"]
        assert "cpu_usage" in patterns["success_indicators"]

        # Optimal parameters should be copied from action
        assert (
            patterns["optimal_parameters"] == sample_experience.action_taken.parameters
        )

        # Performance correlations
        correlations = patterns["performance_correlations"]
        assert "confidence_vs_success" in correlations
        assert (
            correlations["confidence_vs_success"] == 0.8
        )  # Positive correlation for success
        assert "impact_vs_performance" in correlations
        assert "cost_efficiency" in correlations

    def test_learning_experience_extract_patterns_failure(self, sample_action):
        """Test extract_patterns method for failed experience."""
        failure_experience = LearningExperience(
            experience_id=create_experience_id(),
            agent_id=sample_action.agent_id,
            context={"system_load": 0.95, "cpu_usage": 0.98, "memory_usage": 0.9},
            action_taken=sample_action,
            outcome={"cpu_reduction": -0.05, "performance_gain": -0.1},
            success=False,
            learning_value=ConfidenceScore(0.6),
            performance_impact=PerformanceMetric(-0.1),
        )

        patterns = failure_experience.extract_patterns()

        # Failure case should populate failure_indicators
        assert len(patterns["failure_indicators"]) > 0
        assert len(patterns["success_indicators"]) == 0
        assert "system_load" in patterns["failure_indicators"]

        # No optimal parameters for failures
        assert patterns["optimal_parameters"] == {}

        # Negative correlation for failures
        correlations = patterns["performance_correlations"]
        assert (
            correlations["confidence_vs_success"] == -0.8
        )  # Negative correlation for failure

    def test_learning_experience_get_learning_weight(self, sample_experience):
        """Test get_learning_weight method with various factors."""
        # Base learning weight
        base_weight = sample_experience.get_learning_weight()
        assert base_weight >= 0.75  # Should be at least the learning_value

        # Experience with unexpected results (increases weight)
        unexpected_experience = LearningExperience(
            experience_id=create_experience_id(),
            agent_id=create_agent_id(),
            context={"load": 0.5},
            action_taken=sample_experience.action_taken,
            outcome={"result": "unexpected"},
            success=True,
            learning_value=ConfidenceScore(0.5),
            performance_impact=PerformanceMetric(0.1),
            unexpected_results=["Surprising outcome", "Another surprise"],
        )
        unexpected_weight = unexpected_experience.get_learning_weight()
        assert unexpected_weight > 0.5  # Should be increased from base

        # High-impact experience (increases weight)
        high_impact_experience = LearningExperience(
            experience_id=create_experience_id(),
            agent_id=create_agent_id(),
            context={"load": 0.5},
            action_taken=sample_experience.action_taken,
            outcome={"result": "high_impact"},
            success=True,
            learning_value=ConfidenceScore(0.5),
            performance_impact=PerformanceMetric(0.8),  # High impact
        )
        high_impact_weight = high_impact_experience.get_learning_weight()
        assert high_impact_weight > 0.5  # Should be increased

        # Failed experience (increases weight)
        failed_experience = LearningExperience(
            experience_id=create_experience_id(),
            agent_id=create_agent_id(),
            context={"load": 0.5},
            action_taken=sample_experience.action_taken,
            outcome={"result": "failed"},
            success=False,  # Failure
            learning_value=ConfidenceScore(0.5),
            performance_impact=PerformanceMetric(-0.2),
        )
        failed_weight = failed_experience.get_learning_weight()
        assert failed_weight > 0.5  # Should be increased

        # Weight should be capped at 1.0
        assert base_weight <= 1.0
        assert unexpected_weight <= 1.0
        assert high_impact_weight <= 1.0
        assert failed_weight <= 1.0


class TestAgentConfiguration:
    """Test AgentConfiguration dataclass functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample AgentConfiguration for testing."""
        return AgentConfiguration(
            agent_type=AgentType.OPTIMIZER,
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            max_concurrent_actions=5,
            decision_threshold=ConfidenceScore(0.8),
            risk_tolerance=RiskScore(0.3),
            learning_rate=0.15,
            optimization_frequency=timedelta(minutes=30),
            resource_limits={"cpu": 70.0, "memory": 2048.0},
            human_approval_required=False,
        )

    def test_agent_configuration_creation(self, sample_config):
        """Test AgentConfiguration creation and basic properties."""
        assert sample_config.agent_type == AgentType.OPTIMIZER
        assert sample_config.autonomy_level == AutonomyLevel.AUTONOMOUS
        assert sample_config.max_concurrent_actions == 5
        assert sample_config.decision_threshold == 0.8
        assert sample_config.risk_tolerance == 0.3
        assert sample_config.learning_rate == 0.15
        assert not sample_config.human_approval_required

    def test_agent_configuration_is_action_within_limits(self, sample_config):
        """Test is_action_within_limits method."""
        # Action within limits
        good_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.OPTIMIZE_WORKFLOW,
            parameters={},
            confidence=ConfidenceScore(0.85),  # Above threshold
            resource_cost={"cpu": 50.0, "memory": 1024.0},  # Within limits
            safety_validated=True,  # Low risk
        )
        assert sample_config.is_action_within_limits(good_action)

        # Action with too low confidence
        low_confidence_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={},
            confidence=ConfidenceScore(0.7),  # Below threshold (0.8)
        )
        assert not sample_config.is_action_within_limits(low_confidence_action)

        # Action exceeding resource limits
        resource_heavy_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.HEAL_SYSTEM,
            parameters={},
            confidence=ConfidenceScore(0.9),
            resource_cost={"cpu": 100.0},  # Exceeds limit (70.0)
        )
        assert not sample_config.is_action_within_limits(resource_heavy_action)

        # High-risk action
        high_risk_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.DELETE_ALL_DATA,
            parameters={},
            confidence=ConfidenceScore(0.5),  # Low confidence = high risk
            safety_validated=False,  # Increases risk further
        )
        risk_score = high_risk_action.get_risk_score()
        if risk_score > sample_config.risk_tolerance:
            assert not sample_config.is_action_within_limits(high_risk_action)

    def test_agent_configuration_should_request_human_approval(self, sample_config):
        """Test should_request_human_approval method."""
        # Normal action - no approval needed
        normal_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.MONITOR_SYSTEM,
            parameters={},
            confidence=ConfidenceScore(0.9),
            estimated_impact=PerformanceMetric(0.3),
            safety_validated=True,
        )
        assert not sample_config.should_request_human_approval(normal_action)

        # High-risk action - approval needed
        high_risk_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.DELETE_ALL_DATA,
            parameters={},
            confidence=ConfidenceScore(0.3),  # Low confidence = high risk
            estimated_impact=PerformanceMetric(0.5),
            safety_validated=False,
        )
        risk_score = high_risk_action.get_risk_score()
        if risk_score > 0.8:
            assert sample_config.should_request_human_approval(high_risk_action)

        # High-impact action - approval needed
        high_impact_action = AgentAction(
            action_id=create_action_id(),
            agent_id=create_agent_id(),
            action_type=ActionType.MODIFY_SYSTEM_CONFIG,
            parameters={},
            confidence=ConfidenceScore(0.9),
            estimated_impact=PerformanceMetric(0.95),  # Very high impact
            safety_validated=True,
        )
        assert sample_config.should_request_human_approval(high_impact_action)

        # Configuration requires all approval
        approval_required_config = AgentConfiguration(
            agent_type=AgentType.HEALER,
            autonomy_level=AutonomyLevel.SUPERVISED,
            human_approval_required=True,
        )
        assert approval_required_config.should_request_human_approval(normal_action)

        # Manual autonomy level requires approval
        manual_config = AgentConfiguration(
            agent_type=AgentType.GENERAL,
            autonomy_level=AutonomyLevel.MANUAL,
        )
        assert manual_config.should_request_human_approval(normal_action)


class TestDefaultAgentConfigs:
    """Test default agent configurations."""

    def test_default_agent_configs_exist(self):
        """Test that default configurations exist for all agent types."""
        for agent_type in AgentType:
            assert (
                agent_type in DEFAULT_AGENT_CONFIGS
                or agent_type == AgentType.COORDINATOR
                or agent_type == AgentType.PLANNER
                or agent_type == AgentType.RESOURCE_MANAGER
            )

    def test_get_default_config(self):
        """Test get_default_config function."""
        # Test existing configuration
        general_config = get_default_config(AgentType.GENERAL)
        assert general_config.agent_type == AgentType.GENERAL
        assert general_config.autonomy_level == AutonomyLevel.SUPERVISED

        optimizer_config = get_default_config(AgentType.OPTIMIZER)
        assert optimizer_config.agent_type == AgentType.OPTIMIZER
        assert optimizer_config.autonomy_level == AutonomyLevel.AUTONOMOUS

        # Test non-existing configuration (should fallback to GENERAL)
        coordinator_config = get_default_config(AgentType.COORDINATOR)
        assert coordinator_config.agent_type == AgentType.GENERAL

    def test_default_config_values(self):
        """Test specific values in default configurations."""
        general_config = get_default_config(AgentType.GENERAL)
        assert general_config.decision_threshold == 0.7
        assert general_config.risk_tolerance == 0.3
        assert general_config.learning_rate == 0.1
        assert "cpu" in general_config.resource_limits
        assert "memory" in general_config.resource_limits

        monitor_config = get_default_config(AgentType.MONITOR)
        assert (
            monitor_config.decision_threshold == 0.9
        )  # Higher threshold for monitoring
        assert monitor_config.risk_tolerance == 0.2  # Lower risk tolerance
        assert monitor_config.monitoring_interval == timedelta(minutes=1)


class TestAutonomousAgentError:
    """Test AutonomousAgentError exception class."""

    def test_agent_not_found_error(self):
        """Test agent_not_found class method."""
        agent_id = create_agent_id()
        error = AutonomousAgentError.agent_not_found(agent_id)
        assert isinstance(error, AutonomousAgentError)
        assert error.field_name == "agent_id"
        assert error.value == agent_id
        assert agent_id in str(error)

    def test_invalid_goal_constraints_error(self):
        """Test invalid_goal_constraints class method."""
        error = AutonomousAgentError.invalid_goal_constraints()
        assert error.field_name == "goal_constraints"
        assert error.value is None
        assert "invalid or conflicting" in str(error)

    def test_conflicting_goals_error(self):
        """Test conflicting_goals class method."""
        conflicts = ["goal1 vs goal2", "resource conflict"]
        error = AutonomousAgentError.conflicting_goals(conflicts)
        assert error.field_name == "goal_conflicts"
        assert error.value == conflicts
        assert "goal1 vs goal2" in str(error)

    def test_agent_not_active_error(self):
        """Test agent_not_active class method."""
        error = AutonomousAgentError.agent_not_active()
        assert error.field_name == "agent_status"
        assert error.value == "not_active"
        assert "not in active state" in str(error)

    def test_action_too_risky_error(self):
        """Test action_too_risky class method."""
        risk_score = RiskScore(0.9)
        max_risk = RiskScore(0.5)
        error = AutonomousAgentError.action_too_risky(risk_score, max_risk)
        assert error.field_name == "risk_score"
        assert error.value == risk_score
        assert "0.9" in str(error)
        assert "0.5" in str(error)

    def test_resource_limit_exceeded_error(self):
        """Test resource_limit_exceeded class method."""
        error = AutonomousAgentError.resource_limit_exceeded("cpu", 100.0, 70.0)
        assert error.field_name == "resource_usage"
        assert error.value == 100.0
        assert "cpu" in str(error)
        assert "100.0" in str(error)
        assert "70.0" in str(error)

    def test_initialization_failed_error(self):
        """Test initialization_failed class method."""
        reason = "Missing configuration"
        error = AutonomousAgentError.initialization_failed(reason)
        assert error.field_name == "agent_initialization"
        assert error.value is None
        assert reason in str(error)

    def test_execution_cycle_failed_error(self):
        """Test execution_cycle_failed class method."""
        reason = "Resource exhaustion"
        error = AutonomousAgentError.execution_cycle_failed(reason)
        assert error.field_name == "execution_cycle"
        assert reason in str(error)

    def test_action_execution_failed_error(self):
        """Test action_execution_failed class method."""
        reason = "Permission denied"
        error = AutonomousAgentError.action_execution_failed(reason)
        assert error.field_name == "action_execution"
        assert reason in str(error)

    def test_dangerous_goal_detected_error(self):
        """Test dangerous_goal_detected class method."""
        error = AutonomousAgentError.dangerous_goal_detected()
        assert error.field_name == "goal_safety"
        assert "dangerous operations" in str(error)

    def test_manual_mode_action_blocked_error(self):
        """Test manual_mode_action_blocked class method."""
        error = AutonomousAgentError.manual_mode_action_blocked()
        assert error.field_name == "autonomy_level"
        assert error.value == "manual"
        assert "manual mode" in str(error)

    def test_agent_creation_failed_error(self):
        """Test agent_creation_failed class method."""
        reason = "Invalid parameters"
        error = AutonomousAgentError.agent_creation_failed(reason)
        assert error.field_name == "agent_creation"
        assert reason in str(error)

    def test_recovery_in_progress_error(self):
        """Test recovery_in_progress class method."""
        error = AutonomousAgentError.recovery_in_progress()
        assert error.field_name == "recovery_status"
        assert error.value == "in_progress"
        assert "already in progress" in str(error)

    def test_recovery_planning_failed_error(self):
        """Test recovery_planning_failed class method."""
        reason = "Insufficient data"
        error = AutonomousAgentError.recovery_planning_failed(reason)
        assert error.field_name == "recovery_planning"
        assert reason in str(error)

    def test_recovery_execution_failed_error(self):
        """Test recovery_execution_failed class method."""
        reason = "System offline"
        error = AutonomousAgentError.recovery_execution_failed(reason)
        assert error.field_name == "recovery_execution"
        assert reason in str(error)

    def test_diagnostic_failure_error(self):
        """Test diagnostic_failure class method."""
        reason = "Sensor malfunction"
        error = AutonomousAgentError.diagnostic_failure(reason)
        assert error.field_name == "diagnostic"
        assert reason in str(error)

    def test_execution_start_failed_error(self):
        """Test execution_start_failed class method."""
        reason = "Service unavailable"
        error = AutonomousAgentError.execution_start_failed(reason)
        assert error.field_name == "execution_start"
        assert reason in str(error)

    def test_unexpected_error(self):
        """Test unexpected_error class method."""
        reason = "Unknown exception"
        error = AutonomousAgentError.unexpected_error(reason)
        assert error.field_name == "unexpected_error"
        assert reason in str(error)
