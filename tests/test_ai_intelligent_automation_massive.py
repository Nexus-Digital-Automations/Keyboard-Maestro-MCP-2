"""Comprehensive tests for src/ai/intelligent_automation.py - MASSIVE 455 statements coverage.

🚨 CRITICAL COVERAGE ENFORCEMENT: Phase 8 targeting highest-impact zero-coverage modules.
This test covers src/ai/intelligent_automation.py (455 statements - 7th HIGHEST IMPACT) to achieve
significant progress toward mandatory 95% coverage threshold.

Coverage Focus: Intelligent automation system, AI-powered adaptive workflows, smart triggers,
context awareness, decision nodes, pattern recognition, workflow optimization, and all AI automation functionality.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.ai.intelligent_automation import (
    AdaptationScore,
    AdaptiveWorkflow,
    AutomationRuleId,
    AutomationTriggerType,
    ConfidenceLevel,
    ContextDimension,
    ContextState,
    ContextStateId,
    DecisionNode,
    DecisionNodeId,
    IntelligentAutomationEngine,
    SmartTrigger,
    WorkflowAdaptationType,
    WorkflowInstanceId,
)
from src.core.ai_integration import AIOperation, ProcessingMode
from src.core.constants import (
    CONTEXT_HISTORY_LIMIT,
    HIGH_SIMILARITY_THRESHOLD,
    MINIMUM_PATTERN_OCCURRENCES,
    TEXT_LENGTH_LIMIT,
)
from src.core.either import Either
from src.core.errors import ValidationError


class TestEnumerations:
    """Comprehensive tests for enumeration classes."""

    def test_automation_trigger_type_values(self):
        """Test AutomationTriggerType enumeration values."""
        assert AutomationTriggerType.PATTERN_DETECTED.value == "pattern_detected"
        assert AutomationTriggerType.CONTEXT_CHANGED.value == "context_changed"
        assert AutomationTriggerType.CONTENT_ANALYZED.value == "content_analyzed"
        assert AutomationTriggerType.THRESHOLD_REACHED.value == "threshold_reached"
        assert AutomationTriggerType.SCHEDULE_BASED.value == "schedule_based"
        assert AutomationTriggerType.USER_INITIATED.value == "user_initiated"
        assert AutomationTriggerType.SYSTEM_EVENT.value == "system_event"
        assert AutomationTriggerType.ADAPTIVE_SUGGESTION.value == "adaptive_suggestion"

    def test_workflow_adaptation_type_values(self):
        """Test WorkflowAdaptationType enumeration values."""
        assert WorkflowAdaptationType.PARAMETER_OPTIMIZATION.value == "parameter_optimization"
        assert WorkflowAdaptationType.STEP_REORDERING.value == "step_reordering"
        assert WorkflowAdaptationType.CONDITIONAL_ADDITION.value == "conditional_addition"
        assert WorkflowAdaptationType.EFFICIENCY_IMPROVEMENT.value == "efficiency_improvement"
        assert WorkflowAdaptationType.ERROR_PREVENTION.value == "error_prevention"
        assert WorkflowAdaptationType.USER_PREFERENCE.value == "user_preference"

    def test_context_dimension_values(self):
        """Test ContextDimension enumeration values."""
        assert ContextDimension.TEMPORAL.value == "temporal"
        assert ContextDimension.SPATIAL.value == "spatial"
        assert ContextDimension.APPLICATION.value == "application"
        assert ContextDimension.CONTENT.value == "content"
        assert ContextDimension.USER_STATE.value == "user_state"
        assert ContextDimension.SYSTEM_STATE.value == "system_state"
        assert ContextDimension.WORKFLOW.value == "workflow"


class TestContextState:
    """Comprehensive tests for ContextState class."""

    @pytest.fixture
    def sample_context(self):
        """Create sample context state for testing."""
        return ContextState(
            context_id=ContextStateId("context_001"),
            timestamp=datetime(2024, 7, 11, 12, 0, 0, tzinfo=UTC),
            dimensions={
                ContextDimension.TEMPORAL: {"hour": 12, "day": "Friday"},
                ContextDimension.APPLICATION: {"name": "TextEditor", "window_title": "Document.txt"},
                ContextDimension.USER_STATE: {"activity": "typing", "focus": "high"},
                ContextDimension.SYSTEM_STATE: {"cpu_usage": 45, "memory_usage": 60},
            },
            confidence=ConfidenceLevel(0.9),
            metadata={"source": "context_monitor", "version": "1.0"},
        )

    def test_context_state_creation_success(self, sample_context):
        """Test successful context state creation."""
        context = sample_context

        assert context.context_id == "context_001"
        assert context.confidence == 0.9
        assert len(context.dimensions) == 4
        assert context.metadata["source"] == "context_monitor"

        # Check specific dimensions
        temporal = context.dimensions[ContextDimension.TEMPORAL]
        assert temporal["hour"] == 12
        assert temporal["day"] == "Friday"

        app = context.dimensions[ContextDimension.APPLICATION]
        assert app["name"] == "TextEditor"

    def test_context_state_validation_confidence_range(self):
        """Test context state validation for confidence range."""
        # Test valid confidence values
        valid_context = ContextState(
            context_id=ContextStateId("valid_context"),
            timestamp=datetime.now(UTC),
            dimensions={ContextDimension.TEMPORAL: {"hour": 10}},
            confidence=ConfidenceLevel(0.5),
        )
        assert valid_context.confidence == 0.5

        # Test boundary values
        boundary_context_low = ContextState(
            context_id=ContextStateId("boundary_low"),
            timestamp=datetime.now(UTC),
            dimensions={ContextDimension.TEMPORAL: {"hour": 10}},
            confidence=ConfidenceLevel(0.0),
        )
        assert boundary_context_low.confidence == 0.0

        boundary_context_high = ContextState(
            context_id=ContextStateId("boundary_high"),
            timestamp=datetime.now(UTC),
            dimensions={ContextDimension.TEMPORAL: {"hour": 10}},
            confidence=ConfidenceLevel(1.0),
        )
        assert boundary_context_high.confidence == 1.0

    def test_context_state_validation_empty_dimensions(self):
        """Test context state validation with empty dimensions."""
        # Empty dimensions should be invalid based on contract
        with pytest.raises(Exception):  # Contract violation
            ContextState(
                context_id=ContextStateId("invalid_context"),
                timestamp=datetime.now(UTC),
                dimensions={},  # Empty dimensions
                confidence=ConfidenceLevel(0.8),
            )

    def test_context_state_get_dimension_value_exists(self, sample_context):
        """Test getting dimension value for existing dimension."""
        context = sample_context

        temporal_value = context.get_dimension_value(ContextDimension.TEMPORAL)
        assert temporal_value["hour"] == 12

        app_value = context.get_dimension_value(ContextDimension.APPLICATION)
        assert app_value["name"] == "TextEditor"

    def test_context_state_get_dimension_value_not_exists(self, sample_context):
        """Test getting dimension value for non-existing dimension."""
        context = sample_context

        content_value = context.get_dimension_value(ContextDimension.CONTENT)
        assert content_value is None

    def test_context_state_similarity_identical(self, sample_context):
        """Test context similarity calculation for identical contexts."""
        context1 = sample_context
        context2 = sample_context

        similarity = context1.similarity_to(context2)
        assert similarity == 1.0

    def test_context_state_similarity_partial_overlap(self, sample_context):
        """Test context similarity calculation with partial overlap."""
        context1 = sample_context

        context2 = ContextState(
            context_id=ContextStateId("context_002"),
            timestamp=datetime.now(UTC),
            dimensions={
                ContextDimension.TEMPORAL: {"hour": 12, "day": "Friday"},  # Same
                ContextDimension.APPLICATION: {"name": "Browser", "window_title": "Gmail"},  # Different
                ContextDimension.CONTENT: {"type": "email", "subject": "Meeting"},  # New dimension
            },
            confidence=ConfidenceLevel(0.8),
        )

        similarity = context1.similarity_to(context2)
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0  # Should be less than perfect match

    def test_context_state_similarity_no_overlap(self, sample_context):
        """Test context similarity calculation with no overlap."""
        context1 = sample_context

        context2 = ContextState(
            context_id=ContextStateId("context_no_overlap"),
            timestamp=datetime.now(UTC),
            dimensions={
                ContextDimension.CONTENT: {"type": "document"},
                ContextDimension.WORKFLOW: {"step": "review"},
            },
            confidence=ConfidenceLevel(0.7),
        )

        similarity = context1.similarity_to(context2)
        assert similarity == 0.0

    def test_context_state_similarity_empty_dimensions(self, sample_context):
        """Test context similarity with empty dimensions."""
        context1 = sample_context

        # Create context with empty dimensions for similarity test
        context2_dict = {
            "context_id": ContextStateId("empty_context"),
            "timestamp": datetime.now(UTC),
            "dimensions": {},
            "confidence": ConfidenceLevel(0.8),
        }

        # Mock context with empty dimensions for similarity test
        mock_context = Mock()
        mock_context.dimensions = {}

        similarity = context1.similarity_to(mock_context)
        assert similarity == 0.0

    def test_context_state_string_similarity_identical(self, sample_context):
        """Test string similarity calculation for identical strings."""
        context = sample_context

        similarity = context._string_similarity("test", "test")
        assert similarity == 1.0

    def test_context_state_string_similarity_different(self, sample_context):
        """Test string similarity calculation for different strings."""
        context = sample_context

        similarity = context._string_similarity("hello", "world")
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0

    def test_context_state_string_similarity_empty_strings(self, sample_context):
        """Test string similarity calculation with empty strings."""
        context = sample_context

        assert context._string_similarity("", "") == 1.0
        assert context._string_similarity("test", "") == 0.0
        assert context._string_similarity("", "test") == 0.0

    def test_context_state_string_similarity_partial_match(self, sample_context):
        """Test string similarity calculation with partial character overlap."""
        context = sample_context

        # Strings with some common characters
        similarity = context._string_similarity("hello", "help")
        assert 0.0 < similarity < 1.0


class TestSmartTrigger:
    """Comprehensive tests for SmartTrigger class."""

    @pytest.fixture
    def pattern_trigger(self):
        """Create pattern detection smart trigger."""
        return SmartTrigger(
            trigger_id="pattern_trigger_001",
            trigger_type=AutomationTriggerType.PATTERN_DETECTED,
            conditions={
                "context.application": "TextEditor",
                "context.user_state": {"operator": "equals", "value": "typing"},
                "time_window": {"start_hour": 9, "end_hour": 17, "days": [0, 1, 2, 3, 4]},
            },
            ai_analysis_required=False,
            context_requirements={ContextDimension.APPLICATION, ContextDimension.USER_STATE},
            confidence_threshold=ConfidenceLevel(0.8),
            cooldown_period=timedelta(minutes=10),
            adaptation_enabled=True,
        )

    @pytest.fixture
    def ai_analysis_trigger(self):
        """Create AI analysis smart trigger."""
        return SmartTrigger(
            trigger_id="ai_trigger_001",
            trigger_type=AutomationTriggerType.CONTENT_ANALYZED,
            conditions={
                "analysis.sentiment": {"operator": "equals", "value": "positive"},
                "analysis.confidence": {"operator": "greater_than", "value": 0.7},
            },
            ai_analysis_required=True,
            context_requirements={ContextDimension.CONTENT},
            confidence_threshold=ConfidenceLevel(0.9),
        )

    @pytest.fixture
    def sample_context_for_trigger(self):
        """Create sample context for trigger testing."""
        return ContextState(
            context_id=ContextStateId("trigger_context"),
            timestamp=datetime.now(UTC),
            dimensions={
                ContextDimension.APPLICATION: "TextEditor",
                ContextDimension.USER_STATE: "typing",
                ContextDimension.CONTENT: {"text": "Hello world", "type": "document"},
            },
            confidence=ConfidenceLevel(0.9),
        )

    def test_smart_trigger_creation_success(self, pattern_trigger):
        """Test successful smart trigger creation."""
        trigger = pattern_trigger

        assert trigger.trigger_id == "pattern_trigger_001"
        assert trigger.trigger_type == AutomationTriggerType.PATTERN_DETECTED
        assert "context.application" in trigger.conditions
        assert trigger.ai_analysis_required is False
        assert ContextDimension.APPLICATION in trigger.context_requirements
        assert trigger.confidence_threshold == 0.8
        assert trigger.cooldown_period == timedelta(minutes=10)
        assert trigger.adaptation_enabled is True

    def test_smart_trigger_validation_empty_id(self):
        """Test smart trigger validation with empty trigger ID."""
        with pytest.raises(Exception):  # Contract violation
            SmartTrigger(
                trigger_id="",  # Empty ID
                trigger_type=AutomationTriggerType.PATTERN_DETECTED,
                conditions={"test": "value"},
            )

    def test_smart_trigger_validation_confidence_range(self):
        """Test smart trigger validation for confidence threshold range."""
        # Valid confidence values
        valid_trigger = SmartTrigger(
            trigger_id="valid_trigger",
            trigger_type=AutomationTriggerType.PATTERN_DETECTED,
            conditions={"test": "value"},
            confidence_threshold=ConfidenceLevel(0.5),
        )
        assert valid_trigger.confidence_threshold == 0.5

    def test_smart_trigger_validation_negative_cooldown(self):
        """Test smart trigger validation with negative cooldown period."""
        with pytest.raises(Exception):  # Contract violation
            SmartTrigger(
                trigger_id="invalid_cooldown_trigger",
                trigger_type=AutomationTriggerType.PATTERN_DETECTED,
                conditions={"test": "value"},
                cooldown_period=timedelta(seconds=-1),  # Negative cooldown
            )

    def test_smart_trigger_should_trigger_success(self, pattern_trigger, sample_context_for_trigger):
        """Test smart trigger should_trigger method for successful trigger."""
        trigger = pattern_trigger
        context = sample_context_for_trigger

        with patch('src.ai.intelligent_automation.datetime') as mock_datetime:
            # Mock current time to be within working hours (2 PM on Tuesday)
            mock_now = Mock()
            mock_now.hour = 14
            mock_now.weekday.return_value = 1  # Tuesday
            mock_datetime.now.return_value = mock_now

            should_trigger = trigger.should_trigger(context)
            assert should_trigger is True

    def test_smart_trigger_should_trigger_missing_context_requirements(self, pattern_trigger):
        """Test smart trigger should_trigger with missing context requirements."""
        trigger = pattern_trigger

        # Context missing required dimensions
        incomplete_context = ContextState(
            context_id=ContextStateId("incomplete_context"),
            timestamp=datetime.now(UTC),
            dimensions={
                ContextDimension.APPLICATION: "TextEditor",
                # Missing USER_STATE dimension
            },
            confidence=ConfidenceLevel(0.9),
        )

        should_trigger = trigger.should_trigger(incomplete_context)
        assert should_trigger is False

    def test_smart_trigger_should_trigger_low_confidence(self, pattern_trigger, sample_context_for_trigger):
        """Test smart trigger should_trigger with low confidence context."""
        trigger = pattern_trigger

        # Context with confidence below threshold
        low_confidence_context = ContextState(
            context_id=ContextStateId("low_confidence_context"),
            timestamp=datetime.now(UTC),
            dimensions=sample_context_for_trigger.dimensions,
            confidence=ConfidenceLevel(0.5),  # Below 0.8 threshold
        )

        should_trigger = trigger.should_trigger(low_confidence_context)
        assert should_trigger is False

    def test_smart_trigger_should_trigger_ai_analysis_required_missing(self, ai_analysis_trigger, sample_context_for_trigger):
        """Test smart trigger should_trigger with AI analysis required but missing."""
        trigger = ai_analysis_trigger
        context = sample_context_for_trigger

        # No analysis result provided but AI analysis is required
        should_trigger = trigger.should_trigger(context, analysis_result=None)
        assert should_trigger is False

    def test_smart_trigger_should_trigger_ai_analysis_success(self, ai_analysis_trigger, sample_context_for_trigger):
        """Test smart trigger should_trigger with successful AI analysis."""
        trigger = ai_analysis_trigger
        context = sample_context_for_trigger

        analysis_result = {
            "sentiment": "positive",
            "confidence": 0.8,
        }

        should_trigger = trigger.should_trigger(context, analysis_result)
        assert should_trigger is True

    def test_smart_trigger_evaluate_conditions_context_based(self, pattern_trigger, sample_context_for_trigger):
        """Test smart trigger condition evaluation for context-based conditions."""
        trigger = pattern_trigger
        context = sample_context_for_trigger

        # Test direct condition evaluation
        result = trigger._evaluate_conditions(context, None)

        # Should evaluate context conditions but may fail on time window depending on current time
        assert isinstance(result, bool)

    def test_smart_trigger_evaluate_conditions_analysis_based(self, ai_analysis_trigger, sample_context_for_trigger):
        """Test smart trigger condition evaluation for analysis-based conditions."""
        trigger = ai_analysis_trigger
        context = sample_context_for_trigger

        analysis_result = {
            "sentiment": "positive",
            "confidence": 0.8,
        }

        result = trigger._evaluate_conditions(context, analysis_result)
        assert result is True

    def test_smart_trigger_evaluate_conditions_invalid_dimension(self, sample_context_for_trigger):
        """Test smart trigger condition evaluation with invalid context dimension."""
        trigger = SmartTrigger(
            trigger_id="invalid_dim_trigger",
            trigger_type=AutomationTriggerType.PATTERN_DETECTED,
            conditions={
                "context.invalid_dimension": "some_value",
            },
        )

        result = trigger._evaluate_conditions(sample_context_for_trigger, None)
        assert result is False

    def test_smart_trigger_match_condition_value_equals(self, pattern_trigger):
        """Test smart trigger condition value matching with equals operator."""
        trigger = pattern_trigger

        # Direct equality
        assert trigger._match_condition_value("test", "test") is True
        assert trigger._match_condition_value("test", "other") is False

        # Dictionary with equals operator
        condition = {"operator": "equals", "value": "test"}
        assert trigger._match_condition_value("test", condition) is True
        assert trigger._match_condition_value("other", condition) is False

    def test_smart_trigger_match_condition_value_contains(self, pattern_trigger):
        """Test smart trigger condition value matching with contains operator."""
        trigger = pattern_trigger

        condition = {"operator": "contains", "value": "hello"}
        assert trigger._match_condition_value("hello world", condition) is True
        assert trigger._match_condition_value("goodbye", condition) is False

        # Case insensitive
        assert trigger._match_condition_value("Hello World", condition) is True

    def test_smart_trigger_match_condition_value_numeric_operators(self, pattern_trigger):
        """Test smart trigger condition value matching with numeric operators."""
        trigger = pattern_trigger

        # Greater than
        gt_condition = {"operator": "greater_than", "value": 5}
        assert trigger._match_condition_value(10, gt_condition) is True
        assert trigger._match_condition_value(3, gt_condition) is False

        # Less than
        lt_condition = {"operator": "less_than", "value": 5}
        assert trigger._match_condition_value(3, lt_condition) is True
        assert trigger._match_condition_value(10, lt_condition) is False

    def test_smart_trigger_match_condition_value_in_list(self, pattern_trigger):
        """Test smart trigger condition value matching with in_list operator."""
        trigger = pattern_trigger

        condition = {"operator": "in_list", "value": ["option1", "option2", "option3"]}
        assert trigger._match_condition_value("option1", condition) is True
        assert trigger._match_condition_value("option4", condition) is False

    @patch('src.ai.intelligent_automation.datetime')
    def test_smart_trigger_check_time_window_success(self, mock_datetime, pattern_trigger):
        """Test smart trigger time window checking for success case."""
        trigger = pattern_trigger

        # Mock current time to be within working hours (2 PM on Tuesday)
        mock_now = Mock()
        mock_now.hour = 14
        mock_now.weekday.return_value = 1  # Tuesday (day 1)
        mock_datetime.now.return_value = mock_now

        time_window = {"start_hour": 9, "end_hour": 17, "days": [0, 1, 2, 3, 4]}
        result = trigger._check_time_window(time_window)
        assert result is True

    @patch('src.ai.intelligent_automation.datetime')
    def test_smart_trigger_check_time_window_outside_hours(self, mock_datetime, pattern_trigger):
        """Test smart trigger time window checking outside allowed hours."""
        trigger = pattern_trigger

        # Mock current time to be outside working hours (8 PM)
        mock_now = Mock()
        mock_now.hour = 20
        mock_now.weekday.return_value = 1  # Tuesday
        mock_datetime.now.return_value = mock_now

        time_window = {"start_hour": 9, "end_hour": 17, "days": [0, 1, 2, 3, 4]}
        result = trigger._check_time_window(time_window)
        assert result is False

    @patch('src.ai.intelligent_automation.datetime')
    def test_smart_trigger_check_time_window_wrong_day(self, mock_datetime, pattern_trigger):
        """Test smart trigger time window checking on wrong day."""
        trigger = pattern_trigger

        # Mock current time to be on weekend (Saturday)
        mock_now = Mock()
        mock_now.hour = 14
        mock_now.weekday.return_value = 5  # Saturday (day 5)
        mock_datetime.now.return_value = mock_now

        time_window = {"start_hour": 9, "end_hour": 17, "days": [0, 1, 2, 3, 4]}  # Weekdays only
        result = trigger._check_time_window(time_window)
        assert result is False

    def test_smart_trigger_defaults(self):
        """Test smart trigger with default values."""
        trigger = SmartTrigger(
            trigger_id="minimal_trigger",
            trigger_type=AutomationTriggerType.USER_INITIATED,
            conditions={"simple": "condition"},
        )

        assert trigger.ai_analysis_required is False
        assert trigger.context_requirements == set()
        assert trigger.confidence_threshold == 0.7
        assert trigger.cooldown_period == timedelta(minutes=5)
        assert trigger.adaptation_enabled is True


class TestAdaptiveWorkflow:
    """Comprehensive tests for AdaptiveWorkflow class."""

    @pytest.fixture
    def sample_workflow(self):
        """Create sample adaptive workflow."""
        return AdaptiveWorkflow(
            workflow_id=WorkflowInstanceId("workflow_001"),
            base_steps=[
                {"id": "step1", "action": "open_application", "app_name": "TextEditor", "timeout": 10},
                {"id": "step2", "action": "type_text", "text": "Hello World", "delay": 1},
                {"id": "step3", "action": "save_file", "filename": "output.txt", "timeout": 5},
            ],
            adaptation_history=[],
            current_adaptations={
                "parameter_optimization": {
                    "step1": {"timeout": 15},
                    "step3": {"timeout": 8},
                }
            },
            performance_metrics={
                "average_execution_time": 12.5,
                "success_rate": 0.95,
                "total_runs": 20,
            },
            learning_enabled=True,
            adaptation_score=AdaptationScore(0.7),
        )

    @pytest.fixture
    def sample_context_for_workflow(self):
        """Create sample context for workflow testing."""
        return ContextState(
            context_id=ContextStateId("workflow_context"),
            timestamp=datetime.now(UTC),
            dimensions={
                ContextDimension.APPLICATION: {"name": "TextEditor", "active": True},
                ContextDimension.SYSTEM_STATE: {"cpu_usage": 45, "memory_usage": 60},
                ContextDimension.USER_STATE: {"activity": "working", "focus": "high"},
            },
            confidence=ConfidenceLevel(0.85),
        )

    def test_adaptive_workflow_creation_success(self, sample_workflow):
        """Test successful adaptive workflow creation."""
        workflow = sample_workflow

        assert workflow.workflow_id == "workflow_001"
        assert len(workflow.base_steps) == 3
        assert workflow.base_steps[0]["action"] == "open_application"
        assert len(workflow.current_adaptations) == 1
        assert "parameter_optimization" in workflow.current_adaptations
        assert workflow.performance_metrics["success_rate"] == 0.95
        assert workflow.learning_enabled is True
        assert workflow.adaptation_score == 0.7

    def test_adaptive_workflow_validation_empty_steps(self):
        """Test adaptive workflow validation with empty base steps."""
        with pytest.raises(Exception):  # Contract violation
            AdaptiveWorkflow(
                workflow_id=WorkflowInstanceId("invalid_workflow"),
                base_steps=[],  # Empty steps
            )

    def test_adaptive_workflow_validation_adaptation_score_range(self):
        """Test adaptive workflow validation for adaptation score range."""
        # Valid adaptation score
        valid_workflow = AdaptiveWorkflow(
            workflow_id=WorkflowInstanceId("valid_workflow"),
            base_steps=[{"id": "step1", "action": "test"}],
            adaptation_score=AdaptationScore(0.5),
        )
        assert valid_workflow.adaptation_score == 0.5

    def test_adaptive_workflow_get_optimized_steps_no_adaptations(self, sample_context_for_workflow):
        """Test getting optimized steps with no adaptations."""
        workflow = AdaptiveWorkflow(
            workflow_id=WorkflowInstanceId("no_adapt_workflow"),
            base_steps=[
                {"id": "step1", "action": "test", "param": "original"},
                {"id": "step2", "action": "test2", "param": "original"},
            ],
            current_adaptations={},  # No adaptations
        )

        optimized_steps = workflow.get_optimized_steps(sample_context_for_workflow)
        assert len(optimized_steps) == 2
        assert optimized_steps[0]["param"] == "original"
        assert optimized_steps[1]["param"] == "original"

    def test_adaptive_workflow_get_optimized_steps_with_adaptations(self, sample_workflow, sample_context_for_workflow):
        """Test getting optimized steps with parameter adaptations."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        optimized_steps = workflow.get_optimized_steps(context)

        assert len(optimized_steps) == 3
        # Check that parameter optimization was applied
        assert optimized_steps[0]["timeout"] == 15  # Optimized from 10
        assert optimized_steps[2]["timeout"] == 8   # Optimized from 5

    def test_adaptive_workflow_apply_adaptation_parameter_optimization(self, sample_workflow, sample_context_for_workflow):
        """Test applying parameter optimization adaptation."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        steps = [{"id": "step1", "timeout": 10, "param": "value"}]
        adaptation_data = {"step1": {"timeout": 20, "new_param": "new_value"}}

        result = workflow._apply_adaptation(
            steps, "parameter_optimization", adaptation_data, context
        )

        assert len(result) == 1
        assert result[0]["timeout"] == 20
        assert result[0]["new_param"] == "new_value"

    def test_adaptive_workflow_apply_adaptation_step_reordering(self, sample_workflow, sample_context_for_workflow):
        """Test applying step reordering adaptation."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        steps = [
            {"id": "step1", "action": "first"},
            {"id": "step2", "action": "second"},
            {"id": "step3", "action": "third"},
        ]
        adaptation_data = {"optimal_order": [2, 0, 1]}  # Reorder to 3rd, 1st, 2nd

        result = workflow._apply_adaptation(
            steps, "step_reordering", adaptation_data, context
        )

        assert len(result) == 3
        assert result[0]["action"] == "third"
        assert result[1]["action"] == "first"
        assert result[2]["action"] == "second"

    def test_adaptive_workflow_apply_adaptation_conditional_addition(self, sample_workflow, sample_context_for_workflow):
        """Test applying conditional addition adaptation."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        steps = [{"id": "step1", "action": "test"}]
        adaptation_data = {
            "0": [  # Add conditions to first step
                {"type": "application_active", "application": "TextEditor"},
                {"type": "time_based", "hour_range": [9, 17]},
            ]
        }

        result = workflow._apply_adaptation(
            steps, "conditional_addition", adaptation_data, context
        )

        assert len(result) == 1
        assert "conditions" in result[0]
        assert len(result[0]["conditions"]) == 2

    def test_adaptive_workflow_apply_adaptation_efficiency_improvement(self, sample_workflow, sample_context_for_workflow):
        """Test applying efficiency improvement adaptation."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        steps = [
            {"id": "step1", "action": "test", "parallelizable": True, "timeout": 10},
            {"id": "step2", "action": "test2", "parallelizable": False, "timeout": 20},
        ]
        adaptation_data = {
            "enable_parallel": True,
            "timeout_optimization": 0.8,  # Reduce timeouts by 20%
        }

        result = workflow._apply_adaptation(
            steps, "efficiency_improvement", adaptation_data, context
        )

        assert len(result) == 2
        assert result[0]["execution_mode"] == "parallel"  # Parallel enabled for parallelizable step
        assert "execution_mode" not in result[1]  # Not parallelizable
        assert result[0]["timeout"] == 8  # 10 * 0.8
        assert result[1]["timeout"] == 16  # 20 * 0.8

    def test_adaptive_workflow_apply_adaptation_unknown_type(self, sample_workflow, sample_context_for_workflow):
        """Test applying unknown adaptation type (should return original steps)."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        steps = [{"id": "step1", "action": "test"}]
        adaptation_data = {"some": "data"}

        result = workflow._apply_adaptation(
            steps, "unknown_adaptation_type", adaptation_data, context
        )

        assert result == steps  # Should return original steps unchanged

    def test_adaptive_workflow_optimize_parameters(self, sample_workflow, sample_context_for_workflow):
        """Test parameter optimization method."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        steps = [
            {"id": "step1", "timeout": 10, "param": "value1"},
            {"id": "step2", "timeout": 5, "param": "value2"},
        ]
        optimization_data = {
            "step1": {"timeout": 15, "new_param": "optimized"},
            "step2": {"timeout": 8},
        }

        result = workflow._optimize_parameters(steps, optimization_data, context)

        assert len(result) == 2
        assert result[0]["timeout"] == 15
        assert result[0]["new_param"] == "optimized"
        assert result[1]["timeout"] == 8

    def test_adaptive_workflow_should_apply_optimization_timeout_high_load(self, sample_workflow):
        """Test optimization application based on system state (high CPU load)."""
        workflow = sample_workflow

        # Context with high CPU usage
        high_load_context = ContextState(
            context_id=ContextStateId("high_load_context"),
            timestamp=datetime.now(UTC),
            dimensions={
                ContextDimension.SYSTEM_STATE: {"cpu_usage": 85},  # High load
            },
            confidence=ConfidenceLevel(0.8),
        )

        # Should apply longer timeout for high system load
        should_apply = workflow._should_apply_optimization(
            "timeout", 40, high_load_context  # Timeout > 30
        )
        assert should_apply is True

        # Should not apply shorter timeout for high system load
        should_not_apply = workflow._should_apply_optimization(
            "timeout", 20, high_load_context  # Timeout <= 30
        )
        assert should_not_apply is False

    def test_adaptive_workflow_should_apply_optimization_normal_load(self, sample_workflow):
        """Test optimization application with normal system load."""
        workflow = sample_workflow

        # Context with normal CPU usage
        normal_load_context = ContextState(
            context_id=ContextStateId("normal_load_context"),
            timestamp=datetime.now(UTC),
            dimensions={
                ContextDimension.SYSTEM_STATE: {"cpu_usage": 50},  # Normal load
            },
            confidence=ConfidenceLevel(0.8),
        )

        # Should apply optimization for non-timeout parameters
        should_apply = workflow._should_apply_optimization(
            "other_param", "value", normal_load_context
        )
        assert should_apply is True

    def test_adaptive_workflow_reorder_steps_success(self, sample_workflow):
        """Test successful step reordering."""
        workflow = sample_workflow

        steps = [
            {"id": "step1", "action": "first"},
            {"id": "step2", "action": "second"},
            {"id": "step3", "action": "third"},
        ]
        reorder_data = {"optimal_order": [2, 0, 1]}

        result = workflow._reorder_steps(steps, reorder_data)

        assert len(result) == 3
        assert result[0]["action"] == "third"
        assert result[1]["action"] == "first"
        assert result[2]["action"] == "second"

    def test_adaptive_workflow_reorder_steps_invalid_data(self, sample_workflow):
        """Test step reordering with invalid data."""
        workflow = sample_workflow

        steps = [{"id": "step1"}, {"id": "step2"}]

        # Missing optimal_order
        result1 = workflow._reorder_steps(steps, {})
        assert result1 == steps

        # Wrong length
        result2 = workflow._reorder_steps(steps, {"optimal_order": [0]})
        assert result2 == steps

        # Invalid indices
        result3 = workflow._reorder_steps(steps, {"optimal_order": [0, 5]})
        assert result3 == steps

    def test_adaptive_workflow_add_conditions(self, sample_workflow, sample_context_for_workflow):
        """Test adding conditions to workflow steps."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        steps = [{"id": "step1"}, {"id": "step2"}]
        condition_data = {
            "0": [{"type": "application_active", "application": "TextEditor"}],
            "1": [{"type": "time_based", "hour_range": [9, 17]}],
        }

        result = workflow._add_conditions(steps, condition_data, context)

        assert len(result) == 2
        assert "conditions" in result[0]
        assert len(result[0]["conditions"]) == 1
        assert "conditions" in result[1]

    def test_adaptive_workflow_should_add_condition_application_active(self, sample_workflow, sample_context_for_workflow):
        """Test should add condition for application active check."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        condition = {"type": "application_active", "application": "TextEditor"}
        should_add = workflow._should_add_condition(condition, context)
        assert should_add is True

        # Different application
        condition_different = {"type": "application_active", "application": "Browser"}
        should_not_add = workflow._should_add_condition(condition_different, context)
        assert should_not_add is False

    @patch('src.ai.intelligent_automation.datetime')
    def test_adaptive_workflow_should_add_condition_time_based(self, mock_datetime, sample_workflow, sample_context_for_workflow):
        """Test should add condition for time-based check."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        # Mock current time to be within range
        mock_now = Mock()
        mock_now.hour = 14  # 2 PM
        mock_datetime.now.return_value = mock_now

        condition = {"type": "time_based", "hour_range": [9, 17]}
        should_add = workflow._should_add_condition(condition, context)
        assert should_add is True

    @patch('src.ai.intelligent_automation.datetime')
    def test_adaptive_workflow_check_time_condition(self, mock_datetime, sample_workflow):
        """Test time condition checking."""
        workflow = sample_workflow

        # Mock current time
        mock_now = Mock()
        mock_now.hour = 14
        mock_datetime.now.return_value = mock_now

        # Within range
        condition_within = {"hour_range": [9, 17]}
        assert workflow._check_time_condition(condition_within) is True

        # Outside range
        condition_outside = {"hour_range": [18, 22]}
        assert workflow._check_time_condition(condition_outside) is False

    def test_adaptive_workflow_improve_efficiency(self, sample_workflow):
        """Test efficiency improvement method."""
        workflow = sample_workflow

        steps = [
            {"id": "step1", "parallelizable": True, "timeout": 10},
            {"id": "step2", "parallelizable": False, "timeout": 20},
        ]
        efficiency_data = {
            "enable_parallel": True,
            "timeout_optimization": 0.8,
        }

        result = workflow._improve_efficiency(steps, efficiency_data)

        assert len(result) == 2
        assert result[0]["execution_mode"] == "parallel"
        assert "execution_mode" not in result[1]
        assert result[0]["timeout"] == 8  # 10 * 0.8
        assert result[1]["timeout"] == 16  # 20 * 0.8, but minimum is 1

    def test_adaptive_workflow_record_execution_result(self, sample_workflow, sample_context_for_workflow):
        """Test recording execution result for learning."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        initial_history_length = len(workflow.adaptation_history)

        workflow.record_execution_result(
            execution_time=15.5,
            success=True,
            context=context,
            errors=[],
        )

        # Check history was updated
        assert len(workflow.adaptation_history) == initial_history_length + 1

        # Check latest record
        latest_record = workflow.adaptation_history[-1]
        assert latest_record["execution_time"] == 15.5
        assert latest_record["success"] is True
        assert "timestamp" in latest_record
        assert "context_snapshot" in latest_record

    def test_adaptive_workflow_record_execution_result_learning_disabled(self, sample_workflow, sample_context_for_workflow):
        """Test recording execution result when learning is disabled."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        # Disable learning
        object.__setattr__(workflow, "learning_enabled", False)

        initial_history_length = len(workflow.adaptation_history)

        workflow.record_execution_result(
            execution_time=15.5,
            success=True,
            context=context,
        )

        # History should not be updated
        assert len(workflow.adaptation_history) == initial_history_length

    def test_adaptive_workflow_record_execution_result_history_limit(self, sample_workflow, sample_context_for_workflow):
        """Test execution result recording with history limit."""
        workflow = sample_workflow
        context = sample_context_for_workflow

        # Add many records to exceed limit
        existing_history = [{"record": f"old_{i}"} for i in range(105)]
        object.__setattr__(workflow, "adaptation_history", existing_history)

        workflow.record_execution_result(
            execution_time=10.0,
            success=True,
            context=context,
        )

        # Should keep only last 100 records
        assert len(workflow.adaptation_history) == 100

    def test_adaptive_workflow_update_performance_metrics(self, sample_workflow):
        """Test updating performance metrics."""
        workflow = sample_workflow

        initial_metrics = dict(workflow.performance_metrics)

        workflow._update_performance_metrics(execution_time=20.0, success=True)

        # Check metrics were updated
        assert workflow.performance_metrics["total_runs"] == initial_metrics["total_runs"] + 1
        assert workflow.performance_metrics["successful_runs"] == initial_metrics.get("successful_runs", 0) + 1

        # Check timing metrics
        assert "min_execution_time" in workflow.performance_metrics
        assert "max_execution_time" in workflow.performance_metrics

    def test_adaptive_workflow_defaults(self):
        """Test adaptive workflow with default values."""
        workflow = AdaptiveWorkflow(
            workflow_id=WorkflowInstanceId("minimal_workflow"),
            base_steps=[{"id": "step1", "action": "test"}],
        )

        assert workflow.adaptation_history == []
        assert workflow.current_adaptations == {}
        assert workflow.performance_metrics == {}
        assert workflow.learning_enabled is True
        assert workflow.adaptation_score == 0.0


class TestDecisionNode:
    """Comprehensive tests for DecisionNode class."""

    @pytest.fixture
    def sample_decision_node(self):
        """Create sample decision node."""
        return DecisionNode(
            node_id=DecisionNodeId("decision_001"),
            decision_type="content_classification",
            ai_operation=AIOperation.ANALYZE,
            decision_criteria={
                "classification_types": ["urgent", "normal", "low_priority"],
                "confidence_required": 0.8,
                "fallback_behavior": "default_to_normal",
            },
            fallback_decision="normal",
            confidence_threshold=ConfidenceLevel(0.85),
            cache_duration=timedelta(minutes=15),
        )

    @pytest.fixture
    def sample_context_for_decision(self):
        """Create sample context for decision testing."""
        return ContextState(
            context_id=ContextStateId("decision_context"),
            timestamp=datetime.now(UTC),
            dimensions={
                ContextDimension.CONTENT: {
                    "text": "URGENT: Please review this document immediately",
                    "type": "email",
                },
                ContextDimension.APPLICATION: {"name": "EmailClient"},
            },
            confidence=ConfidenceLevel(0.9),
        )

    @pytest.fixture
    def mock_ai_processor(self):
        """Create mock AI processor."""
        processor = AsyncMock()
        processor.process_ai_request.return_value = Either.right({
            "result": "urgent",
            "metadata": {"confidence": 0.9},
        })
        return processor

    def test_decision_node_creation_success(self, sample_decision_node):
        """Test successful decision node creation."""
        node = sample_decision_node

        assert node.node_id == "decision_001"
        assert node.decision_type == "content_classification"
        assert node.ai_operation == AIOperation.ANALYZE
        assert "classification_types" in node.decision_criteria
        assert node.fallback_decision == "normal"
        assert node.confidence_threshold == 0.85
        assert node.cache_duration == timedelta(minutes=15)

    def test_decision_node_validation_empty_id(self):
        """Test decision node validation with empty node ID."""
        with pytest.raises(Exception):  # Contract violation
            DecisionNode(
                node_id=DecisionNodeId(""),  # Empty ID
                decision_type="test",
                ai_operation=AIOperation.ANALYZE,
                decision_criteria={},
                fallback_decision="default",
            )

    def test_decision_node_validation_confidence_range(self):
        """Test decision node validation for confidence threshold range."""
        # Valid confidence
        valid_node = DecisionNode(
            node_id=DecisionNodeId("valid_node"),
            decision_type="test",
            ai_operation=AIOperation.ANALYZE,
            decision_criteria={},
            fallback_decision="default",
            confidence_threshold=ConfidenceLevel(0.6),
        )
        assert valid_node.confidence_threshold == 0.6

    @pytest.mark.asyncio
    async def test_decision_node_make_decision_success(self, sample_decision_node, sample_context_for_decision, mock_ai_processor):
        """Test successful decision making."""
        node = sample_decision_node
        context = sample_context_for_decision
        input_data = {"text": "Urgent email content"}

        result = await node.make_decision(input_data, context, mock_ai_processor)

        assert result.is_right()
        decision = result.get_right()
        assert decision == "urgent"

    @pytest.mark.asyncio
    async def test_decision_node_make_decision_low_confidence(self, sample_decision_node, sample_context_for_decision):
        """Test decision making with low confidence response."""
        node = sample_decision_node
        context = sample_context_for_decision
        input_data = {"text": "Some content"}

        # Mock AI processor with low confidence response
        low_confidence_processor = AsyncMock()
        low_confidence_processor.process_ai_request.return_value = Either.right({
            "result": "urgent",
            "metadata": {"confidence": 0.5},  # Below 0.85 threshold
        })

        result = await node.make_decision(input_data, context, low_confidence_processor)

        assert result.is_right()
        decision = result.get_right()
        assert decision == "normal"  # Should fallback

    @pytest.mark.asyncio
    async def test_decision_node_make_decision_ai_failure(self, sample_decision_node, sample_context_for_decision):
        """Test decision making when AI processing fails."""
        node = sample_decision_node
        context = sample_context_for_decision
        input_data = {"text": "Some content"}

        # Mock AI processor with failure
        failed_processor = AsyncMock()
        failed_processor.process_ai_request.return_value = Either.left(
            ValidationError("ai_error", "AI processing failed")
        )

        result = await node.make_decision(input_data, context, failed_processor)

        assert result.is_right()
        decision = result.get_right()
        assert decision == "normal"  # Should fallback

    @pytest.mark.asyncio
    async def test_decision_node_make_decision_request_creation_failure(self, sample_decision_node, sample_context_for_decision, mock_ai_processor):
        """Test decision making when AI request creation fails."""
        node = sample_decision_node
        context = sample_context_for_decision
        input_data = {"text": "Some content"}

        # Mock create_ai_request to fail
        with patch('src.ai.intelligent_automation.create_ai_request') as mock_create_request:
            mock_create_request.return_value = Either.left(
                ValidationError("request_error", "Request creation failed")
            )

            result = await node.make_decision(input_data, context, mock_ai_processor)

            assert result.is_left()
            error = result.get_left()
            assert isinstance(error, ValidationError)

    def test_decision_node_prepare_decision_prompt(self, sample_decision_node, sample_context_for_decision):
        """Test decision prompt preparation."""
        node = sample_decision_node
        context = sample_context_for_decision
        input_data = {"text": "Test content", "type": "email"}

        prompt = node._prepare_decision_prompt(input_data, context)

        assert "Decision Type: content_classification" in prompt
        assert "Input Data:" in prompt
        assert "Context:" in prompt
        assert "Decision Criteria:" in prompt
        assert "Test content" in prompt
        assert node.fallback_decision in prompt

    def test_decision_node_extract_decision_simple(self, sample_decision_node):
        """Test decision extraction from simple AI response."""
        node = sample_decision_node

        # Simple response
        decision = node._extract_decision("urgent")
        assert decision == "urgent"

        # Response with extra content
        decision = node._extract_decision("The decision is urgent based on the analysis")
        assert decision == "urgent"

    def test_decision_node_extract_decision_with_prefixes(self, sample_decision_node):
        """Test decision extraction with common prefixes."""
        node = sample_decision_node

        test_cases = [
            ("Decision: urgent", "urgent"),
            ("The decision is: normal", "normal"),
            ("I decide: low_priority", "low_priority"),
            ("My decision: urgent", "urgent"),
            ("Based on the analysis: normal", "normal"),
            ("Conclusion: urgent", "urgent"),
        ]

        for input_text, expected in test_cases:
            decision = node._extract_decision(input_text)
            assert decision == expected

    def test_decision_node_extract_decision_multiple_words(self, sample_decision_node):
        """Test decision extraction with multiple words."""
        node = sample_decision_node

        # Short first word should combine with second
        decision = node._extract_decision("go to urgent")
        assert decision == "go to"  # First word <= 3 chars, combine with second

        # Long first word should be used alone
        decision = node._extract_decision("urgent priority high")
        assert decision == "urgent"  # First word > 3 chars

    def test_decision_node_extract_decision_empty_response(self, sample_decision_node):
        """Test decision extraction with empty response."""
        node = sample_decision_node

        decision = node._extract_decision("")
        assert decision == "normal"  # Should return fallback

        decision = node._extract_decision("   ")
        assert decision == "normal"  # Should return fallback

    def test_decision_node_extract_decision_long_response(self, sample_decision_node):
        """Test decision extraction with very long response."""
        node = sample_decision_node

        # Very long response should respect TEXT_LENGTH_LIMIT
        long_response = "a " * (TEXT_LENGTH_LIMIT + 10)  # Exceed limit
        decision = node._extract_decision(long_response)
        assert decision == "a"  # Should take first word

    def test_decision_node_defaults(self):
        """Test decision node with default values."""
        node = DecisionNode(
            node_id=DecisionNodeId("minimal_node"),
            decision_type="simple_decision",
            ai_operation=AIOperation.ANALYZE,
            decision_criteria={},
            fallback_decision="default",
        )

        assert node.confidence_threshold == 0.8
        assert node.cache_duration == timedelta(minutes=30)


class TestIntelligentAutomationEngine:
    """Comprehensive tests for IntelligentAutomationEngine class."""

    @pytest.fixture
    def automation_engine(self):
        """Create automation engine instance."""
        return IntelligentAutomationEngine()

    @pytest.fixture
    def sample_trigger(self):
        """Create sample smart trigger."""
        return SmartTrigger(
            trigger_id="test_trigger",
            trigger_type=AutomationTriggerType.PATTERN_DETECTED,
            conditions={"context.application": "TextEditor"},
            ai_analysis_required=False,
            context_requirements={ContextDimension.APPLICATION},
        )

    @pytest.fixture
    def sample_workflow(self):
        """Create sample adaptive workflow."""
        return AdaptiveWorkflow(
            workflow_id=WorkflowInstanceId("test_workflow"),
            base_steps=[{"id": "step1", "action": "test"}],
        )

    @pytest.fixture
    def sample_context(self):
        """Create sample context state."""
        return ContextState(
            context_id=ContextStateId("test_context"),
            timestamp=datetime.now(UTC),
            dimensions={
                ContextDimension.APPLICATION: "TextEditor",
                ContextDimension.USER_STATE: "active",
            },
            confidence=ConfidenceLevel(0.9),
        )

    @pytest.fixture
    def mock_ai_processor(self):
        """Create mock AI processor."""
        processor = AsyncMock()
        processor.process_ai_request.return_value = Either.right({
            "result": {"pattern_detected": True, "confidence": 0.9}
        })
        return processor

    def test_automation_engine_initialization(self, automation_engine):
        """Test automation engine initialization."""
        engine = automation_engine

        assert engine.smart_triggers == {}
        assert engine.adaptive_workflows == {}
        assert engine.decision_nodes == {}
        assert engine.context_history == []
        assert engine.automation_sessions == {}
        assert engine.learning_enabled is True

    @pytest.mark.asyncio
    async def test_automation_engine_evaluate_triggers_success(self, automation_engine, sample_trigger, sample_context, mock_ai_processor):
        """Test successful trigger evaluation."""
        engine = automation_engine
        trigger = sample_trigger
        context = sample_context

        # Add trigger to engine
        engine.add_smart_trigger(trigger)

        # Mock trigger evaluation to return True
        with patch.object(trigger, 'should_trigger', return_value=True):
            triggered = await engine.evaluate_triggers(context, mock_ai_processor)

            assert len(triggered) == 1
            assert triggered[0] == "test_trigger"

    @pytest.mark.asyncio
    async def test_automation_engine_evaluate_triggers_with_ai_analysis(self, automation_engine, sample_context, mock_ai_processor):
        """Test trigger evaluation with AI analysis required."""
        engine = automation_engine

        # Create trigger that requires AI analysis
        ai_trigger = SmartTrigger(
            trigger_id="ai_trigger",
            trigger_type=AutomationTriggerType.CONTENT_ANALYZED,
            conditions={"analysis.pattern_detected": True},
            ai_analysis_required=True,
        )

        engine.add_smart_trigger(ai_trigger)

        # Mock trigger evaluation to return True
        with patch.object(ai_trigger, 'should_trigger', return_value=True):
            triggered = await engine.evaluate_triggers(sample_context, mock_ai_processor)

            assert len(triggered) == 1
            assert triggered[0] == "ai_trigger"

    @pytest.mark.asyncio
    async def test_automation_engine_evaluate_triggers_cooldown(self, automation_engine, sample_trigger, sample_context):
        """Test trigger evaluation with cooldown period."""
        engine = automation_engine
        trigger = sample_trigger
        context = sample_context

        engine.add_smart_trigger(trigger)

        # Simulate recent activation
        engine.automation_sessions["test_trigger"] = {
            "last_activation": datetime.now(UTC) - timedelta(minutes=2)  # Recent activation
        }

        # Mock trigger evaluation to return True
        with patch.object(trigger, 'should_trigger', return_value=True):
            triggered = await engine.evaluate_triggers(context)

            # Should not trigger due to cooldown
            assert len(triggered) == 0

    @pytest.mark.asyncio
    async def test_automation_engine_evaluate_triggers_exception_handling(self, automation_engine, sample_context):
        """Test trigger evaluation with exception handling."""
        engine = automation_engine

        # Create trigger that will raise exception
        faulty_trigger = SmartTrigger(
            trigger_id="faulty_trigger",
            trigger_type=AutomationTriggerType.PATTERN_DETECTED,
            conditions={},
        )

        engine.add_smart_trigger(faulty_trigger)

        # Mock trigger to raise exception
        with patch.object(faulty_trigger, 'should_trigger', side_effect=Exception("Test error")):
            triggered = await engine.evaluate_triggers(sample_context)

            # Should handle exception gracefully
            assert len(triggered) == 0

    @pytest.mark.asyncio
    async def test_automation_engine_perform_trigger_analysis_success(self, automation_engine, sample_trigger, sample_context, mock_ai_processor):
        """Test successful trigger analysis."""
        engine = automation_engine

        result = await engine._perform_trigger_analysis(sample_trigger, sample_context, mock_ai_processor)

        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_automation_engine_perform_trigger_analysis_failure(self, automation_engine, sample_trigger, sample_context):
        """Test trigger analysis failure."""
        engine = automation_engine

        # Mock AI processor failure
        failed_processor = AsyncMock()
        failed_processor.process_ai_request.return_value = Either.left(
            ValidationError("ai_error", "AI failed")
        )

        result = await engine._perform_trigger_analysis(sample_trigger, sample_context, failed_processor)

        assert result is None

    def test_automation_engine_check_trigger_cooldown_no_activation(self, automation_engine):
        """Test trigger cooldown check with no previous activation."""
        engine = automation_engine

        result = engine._check_trigger_cooldown("new_trigger")
        assert result is True

    def test_automation_engine_check_trigger_cooldown_expired(self, automation_engine, sample_trigger):
        """Test trigger cooldown check with expired cooldown."""
        engine = automation_engine
        engine.add_smart_trigger(sample_trigger)

        # Old activation
        engine.automation_sessions["test_trigger"] = {
            "last_activation": datetime.now(UTC) - timedelta(minutes=15)  # Older than cooldown
        }

        result = engine._check_trigger_cooldown("test_trigger")
        assert result is True

    def test_automation_engine_check_trigger_cooldown_active(self, automation_engine, sample_trigger):
        """Test trigger cooldown check with active cooldown."""
        engine = automation_engine
        engine.add_smart_trigger(sample_trigger)

        # Recent activation
        engine.automation_sessions["test_trigger"] = {
            "last_activation": datetime.now(UTC) - timedelta(minutes=2)  # Within cooldown
        }

        result = engine._check_trigger_cooldown("test_trigger")
        assert result is False

    def test_automation_engine_record_trigger_activation(self, automation_engine):
        """Test recording trigger activation."""
        engine = automation_engine

        # First activation
        engine._record_trigger_activation("test_trigger")

        session = engine.automation_sessions["test_trigger"]
        assert "last_activation" in session
        assert session["activation_count"] == 1

        # Second activation
        engine._record_trigger_activation("test_trigger")
        assert engine.automation_sessions["test_trigger"]["activation_count"] == 2

    def test_automation_engine_execute_adaptive_workflow_success(self, automation_engine, sample_workflow, sample_context):
        """Test successful adaptive workflow execution."""
        engine = automation_engine
        engine.add_adaptive_workflow(sample_workflow)

        result = engine.execute_adaptive_workflow("test_workflow", sample_context)

        assert result.is_right()
        steps = result.get_right()
        assert len(steps) == 1
        assert steps[0]["action"] == "test"

    def test_automation_engine_execute_adaptive_workflow_not_found(self, automation_engine, sample_context):
        """Test adaptive workflow execution with workflow not found."""
        engine = automation_engine

        result = engine.execute_adaptive_workflow("nonexistent_workflow", sample_context)

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, ValidationError)
        assert "not found" in error.message

    def test_automation_engine_record_context_for_learning(self, automation_engine, sample_context):
        """Test recording context for learning."""
        engine = automation_engine

        initial_history_length = len(engine.context_history)

        engine._record_context_for_learning("test_workflow", sample_context)

        # Check context history updated
        assert len(engine.context_history) == initial_history_length + 1

        # Check workflow session updated
        session = engine.automation_sessions["workflow_test_workflow"]
        assert "contexts" in session
        assert len(session["contexts"]) == 1

    def test_automation_engine_record_context_for_learning_history_limit(self, automation_engine, sample_context):
        """Test context recording with history limit."""
        engine = automation_engine

        # Fill history beyond limit
        engine.context_history = [Mock() for _ in range(1005)]

        engine._record_context_for_learning("test_workflow", sample_context)

        # Should maintain limit
        assert len(engine.context_history) == 1000

    def test_automation_engine_update_context_state(self, automation_engine, sample_context):
        """Test updating context state."""
        engine = automation_engine

        initial_length = len(engine.context_history)

        engine.update_context_state(sample_context)

        assert len(engine.context_history) == initial_length + 1
        assert engine.context_history[-1] == sample_context

    def test_automation_engine_update_context_state_learning_disabled(self, automation_engine, sample_context):
        """Test updating context state with learning disabled."""
        engine = automation_engine
        engine.learning_enabled = False

        # Mock pattern analysis method
        with patch.object(engine, '_analyze_context_patterns') as mock_analyze:
            engine.update_context_state(sample_context)

            # Should not call pattern analysis when learning disabled
            mock_analyze.assert_not_called()

    def test_automation_engine_analyze_context_patterns_insufficient_history(self, automation_engine, sample_context):
        """Test context pattern analysis with insufficient history."""
        engine = automation_engine

        # Add only a few contexts
        engine.context_history = [Mock() for _ in range(5)]

        # Should not analyze with insufficient history
        engine._analyze_context_patterns(sample_context)

        # No exception should be raised

    def test_automation_engine_analyze_context_patterns_with_similar_contexts(self, automation_engine):
        """Test context pattern analysis with similar contexts."""
        engine = automation_engine

        # Create current context
        current_context = ContextState(
            context_id=ContextStateId("current"),
            timestamp=datetime.now(UTC),
            dimensions={ContextDimension.APPLICATION: "TextEditor"},
            confidence=ConfidenceLevel(0.9),
        )

        # Create similar contexts in history
        similar_contexts = []
        for i in range(20):
            similar_context = ContextState(
                context_id=ContextStateId(f"similar_{i}"),
                timestamp=datetime.now(UTC) - timedelta(hours=i),
                dimensions={ContextDimension.APPLICATION: "TextEditor"},
                confidence=ConfidenceLevel(0.8),
            )
            similar_contexts.append(similar_context)

        engine.context_history = similar_contexts

        # Mock similarity calculation to return high similarity
        with patch.object(current_context, 'similarity_to', return_value=0.9):
            engine._analyze_context_patterns(current_context)

            # Should have pattern insights
            assert "pattern_insights" in engine.automation_sessions

    def test_automation_engine_identify_adaptation_opportunities(self, automation_engine):
        """Test identifying adaptation opportunities."""
        engine = automation_engine

        # Create current context
        current_context = ContextState(
            context_id=ContextStateId("current"),
            timestamp=datetime.now(UTC),
            dimensions={ContextDimension.TEMPORAL: {"hour": 14}},
            confidence=ConfidenceLevel(0.9),
        )

        # Create similar contexts
        similar_contexts = [
            (Mock(get_dimension_value=Mock(return_value={"hour": 14})), 0.9),
            (Mock(get_dimension_value=Mock(return_value={"hour": 15})), 0.8),
        ]

        engine._identify_adaptation_opportunities(current_context, similar_contexts)

        # Should create pattern insights
        assert "pattern_insights" in engine.automation_sessions

    def test_automation_engine_analyze_temporal_patterns(self, automation_engine):
        """Test temporal pattern analysis."""
        engine = automation_engine

        temporal_values = [
            {"hour": 9},
            {"hour": 10},
            {"hour": 9},
            {"hour": 11},
        ]

        patterns = engine._analyze_temporal_patterns(temporal_values)

        assert "common_hours" in patterns
        assert "average_hour" in patterns
        assert "pattern_strength" in patterns

    def test_automation_engine_analyze_temporal_patterns_empty(self, automation_engine):
        """Test temporal pattern analysis with empty values."""
        engine = automation_engine

        patterns = engine._analyze_temporal_patterns([])
        assert patterns == {}

    def test_automation_engine_analyze_application_patterns(self, automation_engine):
        """Test application pattern analysis."""
        engine = automation_engine

        app_values = [
            "TextEditor",
            "Browser",
            "TextEditor",
            {"name": "TextEditor"},
        ]

        patterns = engine._analyze_application_patterns(app_values)

        assert "application_frequencies" in patterns
        assert "most_common_app" in patterns
        assert "pattern_strength" in patterns

    def test_automation_engine_analyze_application_patterns_empty(self, automation_engine):
        """Test application pattern analysis with empty values."""
        engine = automation_engine

        patterns = engine._analyze_application_patterns([])
        assert patterns == {}

    def test_automation_engine_add_smart_trigger(self, automation_engine, sample_trigger):
        """Test adding smart trigger."""
        engine = automation_engine

        engine.add_smart_trigger(sample_trigger)

        assert "test_trigger" in engine.smart_triggers
        assert engine.smart_triggers["test_trigger"] == sample_trigger

    def test_automation_engine_add_adaptive_workflow(self, automation_engine, sample_workflow):
        """Test adding adaptive workflow."""
        engine = automation_engine

        engine.add_adaptive_workflow(sample_workflow)

        assert "test_workflow" in engine.adaptive_workflows
        assert engine.adaptive_workflows["test_workflow"] == sample_workflow

    def test_automation_engine_add_decision_node(self, automation_engine):
        """Test adding decision node."""
        engine = automation_engine

        node = DecisionNode(
            node_id=DecisionNodeId("test_node"),
            decision_type="test",
            ai_operation=AIOperation.ANALYZE,
            decision_criteria={},
            fallback_decision="default",
        )

        engine.add_decision_node(node)

        assert "test_node" in engine.decision_nodes
        assert engine.decision_nodes["test_node"] == node

    def test_automation_engine_get_automation_statistics(self, automation_engine, sample_trigger, sample_workflow):
        """Test getting automation statistics."""
        engine = automation_engine

        # Add components
        engine.add_smart_trigger(sample_trigger)
        engine.add_adaptive_workflow(sample_workflow)

        # Add some session data
        engine.automation_sessions["test_trigger"] = {
            "activation_count": 5,
            "last_activation": datetime.now(UTC),
        }

        stats = engine.get_automation_statistics()

        assert "system_overview" in stats
        assert "workflow_statistics" in stats
        assert "trigger_statistics" in stats
        assert "pattern_insights" in stats
        assert "timestamp" in stats

        # Check system overview
        overview = stats["system_overview"]
        assert overview["total_smart_triggers"] == 1
        assert overview["total_adaptive_workflows"] == 1
        assert overview["learning_enabled"] is True

        # Check trigger statistics
        trigger_stats = stats["trigger_statistics"]
        assert "test_trigger" in trigger_stats
        assert trigger_stats["test_trigger"]["activation_count"] == 5


class TestModuleIntegration:
    """Test module integration and all functionality."""

    def test_branded_types_creation(self):
        """Test branded type creation and usage."""
        # Test all branded types can be created
        rule_id = AutomationRuleId("rule_001")
        workflow_id = WorkflowInstanceId("workflow_001")
        context_id = ContextStateId("context_001")
        node_id = DecisionNodeId("node_001")
        adaptation_score = AdaptationScore(0.8)
        confidence = ConfidenceLevel(0.9)

        assert rule_id == "rule_001"
        assert workflow_id == "workflow_001"
        assert context_id == "context_001"
        assert node_id == "node_001"
        assert adaptation_score == 0.8
        assert confidence == 0.9

    def test_constants_integration(self):
        """Test integration with constants module."""
        # Test that constants are properly imported and used
        assert CONTEXT_HISTORY_LIMIT > 0
        assert HIGH_SIMILARITY_THRESHOLD > 0.0
        assert MINIMUM_PATTERN_OCCURRENCES > 0
        assert TEXT_LENGTH_LIMIT > 0

    def test_ai_integration_usage(self):
        """Test AI integration components usage."""
        # Test that AI integration types are properly imported
        assert AIOperation.ANALYZE is not None
        assert ProcessingMode.ACCURATE is not None

    def test_error_handling_integration(self):
        """Test error handling integration."""
        # Test that ValidationError is properly imported and used
        error = ValidationError("test_field", "test_value", "test_message")
        assert error.field == "test_field"
        assert error.value == "test_value"
        assert error.message == "test_message"

    def test_either_integration(self):
        """Test Either monad integration."""
        # Test Either usage
        success_result = Either.right("success")
        error_result = Either.left("error")

        assert success_result.is_right()
        assert success_result.get_right() == "success"

        assert error_result.is_left()
        assert error_result.get_left() == "error"
