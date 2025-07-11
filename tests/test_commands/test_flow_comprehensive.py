"""Comprehensive tests for src/commands/flow.py.

This module provides targeted tests for the flow control commands module to achieve high coverage
toward the mandatory 95% threshold by covering all uncovered methods and execution paths.
"""

import pytest
from src.commands.flow import (
    BreakCommand,
    ConditionalCommand,
    ConditionType,
    LoopCommand,
    LoopType,
)
from src.core.types import (
    CommandId,
    CommandParameters,
    ExecutionContext,
    Permission,
)


class TestConditionalCommandMethods:
    """Test ConditionalCommand method functionality comprehensively."""

    @pytest.fixture
    def sample_context(self):
        """Create sample ExecutionContext for testing."""
        return ExecutionContext.create_test_context(
            permissions=frozenset(
                [
                    Permission.FLOW_CONTROL,
                    Permission.TEXT_INPUT,
                    Permission.SYSTEM_SOUND,
                ]
            )
        )

    def test_conditional_command_get_condition_type_invalid(self):
        """Test get_condition_type with invalid condition type."""
        command = ConditionalCommand(
            command_id=CommandId("cond-invalid"),
            parameters=CommandParameters(
                data={
                    "condition_type": "invalid_condition_type",
                    "left_operand": "test",
                    "right_operand": "test",
                }
            ),
        )

        # Should default to EQUALS for invalid condition type
        assert command.get_condition_type() == ConditionType.EQUALS

    def test_conditional_command_get_left_operand(self):
        """Test get_left_operand method."""
        command = ConditionalCommand(
            command_id=CommandId("cond-left"),
            parameters=CommandParameters(
                data={
                    "left_operand": "left_test_value",
                    "right_operand": "right_test_value",
                }
            ),
        )

        assert command.get_left_operand() == "left_test_value"

    def test_conditional_command_get_right_operand(self):
        """Test get_right_operand method."""
        command = ConditionalCommand(
            command_id=CommandId("cond-right"),
            parameters=CommandParameters(
                data={
                    "left_operand": "left_test_value",
                    "right_operand": "right_test_value",
                }
            ),
        )

        assert command.get_right_operand() == "right_test_value"

    def test_conditional_command_get_case_sensitive(self):
        """Test get_case_sensitive method."""
        # Test default case sensitive
        command1 = ConditionalCommand(
            command_id=CommandId("cond-case1"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_case_sensitive() is True

        # Test explicit case sensitive setting
        command2 = ConditionalCommand(
            command_id=CommandId("cond-case2"),
            parameters=CommandParameters(data={"case_sensitive": False}),
        )
        assert command2.get_case_sensitive() is False

    def test_conditional_command_get_then_action(self):
        """Test get_then_action method."""
        then_action = {"type": "log", "message": "then executed"}
        command = ConditionalCommand(
            command_id=CommandId("cond-then"),
            parameters=CommandParameters(
                data={
                    "then_action": then_action,
                }
            ),
        )

        assert command.get_then_action() == then_action

    def test_conditional_command_get_else_action(self):
        """Test get_else_action method."""
        else_action = {"type": "log", "message": "else executed"}
        command = ConditionalCommand(
            command_id=CommandId("cond-else"),
            parameters=CommandParameters(
                data={
                    "else_action": else_action,
                }
            ),
        )

        assert command.get_else_action() == else_action

    def test_conditional_command_get_timeout(self):
        """Test get_timeout method with various inputs."""
        # Test default timeout
        command1 = ConditionalCommand(
            command_id=CommandId("cond-timeout1"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_timeout().seconds == 5

        # Test custom timeout
        command2 = ConditionalCommand(
            command_id=CommandId("cond-timeout2"),
            parameters=CommandParameters(data={"timeout": 10}),
        )
        assert command2.get_timeout().seconds == 10

        # Test timeout exceeding maximum (should cap at 30)
        command3 = ConditionalCommand(
            command_id=CommandId("cond-timeout3"),
            parameters=CommandParameters(data={"timeout": 60}),
        )
        assert command3.get_timeout().seconds == 30

        # Test invalid timeout (should default to 5)
        command4 = ConditionalCommand(
            command_id=CommandId("cond-timeout4"),
            parameters=CommandParameters(data={"timeout": "invalid"}),
        )
        assert command4.get_timeout().seconds == 5

    def test_conditional_command_validation_detailed(self):
        """Test detailed _validate_impl functionality."""
        # Test validation failure with no actions
        command1 = ConditionalCommand(
            command_id=CommandId("cond-no-actions"),
            parameters=CommandParameters(
                data={
                    "left_operand": "test",
                    "right_operand": "test",
                }
            ),
        )
        assert command1.validate() is False

        # Test validation failure with invalid regex
        command2 = ConditionalCommand(
            command_id=CommandId("cond-invalid-regex"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.REGEX_MATCH,
                    "left_operand": "test",
                    "right_operand": "[invalid_regex",  # Invalid regex pattern
                    "then_action": {"type": "log", "message": "test"},
                }
            ),
        )
        assert command2.validate() is False

        # Test validation failure with invalid action type
        command3 = ConditionalCommand(
            command_id=CommandId("cond-invalid-action"),
            parameters=CommandParameters(
                data={
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"type": "dangerous_action", "message": "test"},
                }
            ),
        )
        assert command3.validate() is False

        # Test validation failure with invalid action structure
        command4 = ConditionalCommand(
            command_id=CommandId("cond-invalid-structure"),
            parameters=CommandParameters(
                data={
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": "not_a_dict",  # Invalid structure
                }
            ),
        )
        assert command4.validate() is False

    def test_conditional_command_execution_with_actions(self, sample_context):
        """Test conditional command execution with actual actions."""
        # Test execution with then action
        command1 = ConditionalCommand(
            command_id=CommandId("cond-exec-then"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.EQUALS,
                    "left_operand": "test",
                    "right_operand": "test",
                    "case_sensitive": True,
                    "then_action": {"type": "log", "message": "condition true"},
                }
            ),
        )

        result1 = command1.execute(sample_context)
        assert result1.success is True
        assert "Condition evaluated to True" in result1.output

        # Test execution with else action
        command2 = ConditionalCommand(
            command_id=CommandId("cond-exec-else"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.EQUALS,
                    "left_operand": "test1",
                    "right_operand": "test2",
                    "case_sensitive": True,
                    "else_action": {"type": "log", "message": "condition false"},
                }
            ),
        )

        result2 = command2.execute(sample_context)
        assert result2.success is True
        assert "Condition evaluated to False" in result2.output

    def test_conditional_command_condition_evaluation_edge_cases(self, sample_context):
        """Test edge cases in condition evaluation."""
        # Test case insensitive comparison
        command1 = ConditionalCommand(
            command_id=CommandId("cond-case-insensitive"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.EQUALS,
                    "left_operand": "TEST",
                    "right_operand": "test",
                    "case_sensitive": False,
                    "then_action": {"type": "log", "message": "matched"},
                }
            ),
        )

        result1 = command1.execute(sample_context)
        assert result1.success is True
        assert "Condition evaluated to True" in result1.output

        # Test numeric comparison with invalid numbers (fallback to string)
        command2 = ConditionalCommand(
            command_id=CommandId("cond-invalid-numbers"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.GREATER_THAN,
                    "left_operand": "not_a_number",
                    "right_operand": "also_not_a_number",
                    "case_sensitive": True,
                    "then_action": {"type": "log", "message": "compared as strings"},
                }
            ),
        )

        result2 = command2.execute(sample_context)
        assert result2.success is True

        # Test regex with case insensitive
        command3 = ConditionalCommand(
            command_id=CommandId("cond-regex-case"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.REGEX_MATCH,
                    "left_operand": "Hello World",
                    "right_operand": "hello",
                    "case_sensitive": False,
                    "then_action": {"type": "log", "message": "regex matched"},
                }
            ),
        )

        result3 = command3.execute(sample_context)
        assert result3.success is True

    def test_conditional_command_action_execution_types(self, sample_context):
        """Test different action execution types."""
        # Test pause action
        command1 = ConditionalCommand(
            command_id=CommandId("cond-pause"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.EQUALS,
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"type": "pause", "duration": 0.1},
                }
            ),
        )

        result1 = command1.execute(sample_context)
        assert result1.success is True

        # Test beep action
        command2 = ConditionalCommand(
            command_id=CommandId("cond-beep"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.EQUALS,
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"type": "beep"},
                }
            ),
        )

        result2 = command2.execute(sample_context)
        assert result2.success is True

        # Test display_message action
        command3 = ConditionalCommand(
            command_id=CommandId("cond-message"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.EQUALS,
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"type": "display_message", "message": "Hello!"},
                }
            ),
        )

        result3 = command3.execute(sample_context)
        assert result3.success is True

        # Test recognized but not implemented action types
        command4 = ConditionalCommand(
            command_id=CommandId("cond-recognized"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.EQUALS,
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {
                        "type": "set_variable",
                        "variable": "test",
                        "value": "value",
                    },
                }
            ),
        )

        result4 = command4.execute(sample_context)
        assert result4.success is True


class TestLoopCommandMethods:
    """Test LoopCommand method functionality comprehensively."""

    @pytest.fixture
    def sample_context(self):
        """Create sample ExecutionContext for testing."""
        return ExecutionContext.create_test_context(
            permissions=frozenset(
                [
                    Permission.FLOW_CONTROL,
                    Permission.TEXT_INPUT,
                    Permission.SYSTEM_SOUND,
                ]
            )
        )

    def test_loop_command_get_loop_type_invalid(self):
        """Test get_loop_type with invalid loop type."""
        command = LoopCommand(
            command_id=CommandId("loop-invalid"),
            parameters=CommandParameters(
                data={
                    "loop_type": "invalid_loop_type",
                    "count": 1,
                }
            ),
        )

        # Should default to FOR_COUNT for invalid loop type
        assert command.get_loop_type() == LoopType.FOR_COUNT

    def test_loop_command_get_count_edge_cases(self):
        """Test get_count method with edge cases."""
        # Test default count
        command1 = LoopCommand(
            command_id=CommandId("loop-default-count"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_count() == 1

        # Test count exceeding maximum (should cap at MAX_LOOP_ITERATIONS)
        command2 = LoopCommand(
            command_id=CommandId("loop-max-count"),
            parameters=CommandParameters(data={"count": 10000}),
        )
        assert command2.get_count() == 1000  # MAX_LOOP_ITERATIONS

        # Test negative count (should be at least 1)
        command3 = LoopCommand(
            command_id=CommandId("loop-negative-count"),
            parameters=CommandParameters(data={"count": -5}),
        )
        assert command3.get_count() == 1

        # Test invalid count (should default to 1)
        command4 = LoopCommand(
            command_id=CommandId("loop-invalid-count"),
            parameters=CommandParameters(data={"count": "not_a_number"}),
        )
        assert command4.get_count() == 1

    def test_loop_command_get_condition(self):
        """Test get_condition method."""
        condition = {
            "condition_type": "equals",
            "left_operand": "x",
            "right_operand": "10",
        }
        command = LoopCommand(
            command_id=CommandId("loop-condition"),
            parameters=CommandParameters(data={"condition": condition}),
        )

        assert command.get_condition() == condition

    def test_loop_command_get_items_with_limits(self):
        """Test get_items method with item limits."""
        # Test normal items list
        items = ["a", "b", "c"]
        command1 = LoopCommand(
            command_id=CommandId("loop-items1"),
            parameters=CommandParameters(data={"items": items}),
        )
        assert command1.get_items() == items

        # Test empty items
        command2 = LoopCommand(
            command_id=CommandId("loop-items2"),
            parameters=CommandParameters(data={}),
        )
        assert command2.get_items() == []

        # Test non-list items
        command3 = LoopCommand(
            command_id=CommandId("loop-items3"),
            parameters=CommandParameters(data={"items": "not_a_list"}),
        )
        assert command3.get_items() == []

        # Test items exceeding limit (should truncate to MAX_LOOP_ITERATIONS)
        large_items = [str(i) for i in range(1500)]  # More than MAX_LOOP_ITERATIONS
        command4 = LoopCommand(
            command_id=CommandId("loop-items4"),
            parameters=CommandParameters(data={"items": large_items}),
        )
        result_items = command4.get_items()
        assert len(result_items) == 1000  # MAX_LOOP_ITERATIONS

    def test_loop_command_get_loop_action(self):
        """Test get_loop_action method."""
        action = {"type": "log", "message": "iteration"}
        command = LoopCommand(
            command_id=CommandId("loop-action"),
            parameters=CommandParameters(data={"loop_action": action}),
        )

        assert command.get_loop_action() == action

    def test_loop_command_get_max_duration(self):
        """Test get_max_duration method with various inputs."""
        # Test default duration
        command1 = LoopCommand(
            command_id=CommandId("loop-duration1"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_max_duration().seconds == 60

        # Test custom duration
        command2 = LoopCommand(
            command_id=CommandId("loop-duration2"),
            parameters=CommandParameters(data={"max_duration": 120}),
        )
        assert command2.get_max_duration().seconds == 120

        # Test duration exceeding maximum (should cap at 300)
        command3 = LoopCommand(
            command_id=CommandId("loop-duration3"),
            parameters=CommandParameters(data={"max_duration": 600}),
        )
        assert command3.get_max_duration().seconds == 300

        # Test invalid duration (should default to 60)
        command4 = LoopCommand(
            command_id=CommandId("loop-duration4"),
            parameters=CommandParameters(data={"max_duration": "invalid"}),
        )
        assert command4.get_max_duration().seconds == 60

    def test_loop_command_get_break_on_error(self):
        """Test get_break_on_error method."""
        # Test default (should be True)
        command1 = LoopCommand(
            command_id=CommandId("loop-break1"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_break_on_error() is True

        # Test explicit setting
        command2 = LoopCommand(
            command_id=CommandId("loop-break2"),
            parameters=CommandParameters(data={"break_on_error": False}),
        )
        assert command2.get_break_on_error() is False

    def test_loop_command_validation_detailed(self):
        """Test detailed _validate_impl functionality."""
        # Test validation success with zero count (gets normalized to 1)
        command1 = LoopCommand(
            command_id=CommandId("loop-zero-count"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.FOR_COUNT,
                    "count": 0,  # Gets normalized to 1, so validation passes
                    "loop_action": {"type": "log", "message": "test"},
                }
            ),
        )
        assert command1.validate() is True
        assert command1.get_count() == 1  # Verify normalization

        # Test validation failure with invalid while condition
        command2 = LoopCommand(
            command_id=CommandId("loop-invalid-while"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.WHILE_CONDITION,
                    "condition": "not_a_dict",  # Invalid condition structure
                    "loop_action": {"type": "log", "message": "test"},
                }
            ),
        )
        assert command2.validate() is False

        # Test validation failure with missing condition fields
        command3 = LoopCommand(
            command_id=CommandId("loop-missing-fields"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.WHILE_CONDITION,
                    "condition": {"incomplete": "condition"},  # Missing required fields
                    "loop_action": {"type": "log", "message": "test"},
                }
            ),
        )
        assert command3.validate() is False

        # Test validation failure with invalid for_each items
        command4 = LoopCommand(
            command_id=CommandId("loop-invalid-items"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.FOR_EACH,
                    "items": [],  # Empty items
                    "loop_action": {"type": "log", "message": "test"},
                }
            ),
        )
        assert command4.validate() is False

        # Test validation failure with no loop action
        command5 = LoopCommand(
            command_id=CommandId("loop-no-action"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.FOR_COUNT,
                    "count": 1,
                    # No loop_action
                }
            ),
        )
        assert command5.validate() is False

    def test_loop_command_execution_for_count(self, sample_context):
        """Test loop execution with FOR_COUNT type."""
        command = LoopCommand(
            command_id=CommandId("loop-for-count-exec"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.FOR_COUNT,
                    "count": 3,
                    "loop_action": {"type": "log", "message": "iteration"},
                    "max_duration": 10,
                    "break_on_error": True,
                }
            ),
        )

        result = command.execute(sample_context)
        assert result.success is True
        assert "3 iterations" in result.output

    def test_loop_command_execution_while_condition(self, sample_context):
        """Test loop execution with WHILE_CONDITION type."""
        command = LoopCommand(
            command_id=CommandId("loop-while-exec"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.WHILE_CONDITION,
                    "condition": {
                        "condition_type": "equals",
                        "left_operand": "false",
                        "right_operand": "true",
                        "case_sensitive": True,
                    },
                    "loop_action": {"type": "log", "message": "while iteration"},
                    "max_duration": 10,
                    "break_on_error": True,
                }
            ),
        )

        result = command.execute(sample_context)
        assert result.success is True
        assert "0 iterations" in result.output  # Condition should be false immediately

    def test_loop_command_execution_for_each(self, sample_context):
        """Test loop execution with FOR_EACH type."""
        command = LoopCommand(
            command_id=CommandId("loop-for-each-exec"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.FOR_EACH,
                    "items": ["item1", "item2", "item3"],
                    "loop_action": {"type": "log", "message": "processing item"},
                    "max_duration": 10,
                    "break_on_error": True,
                }
            ),
        )

        result = command.execute(sample_context)
        assert result.success is True
        assert "3 iterations" in result.output

    def test_loop_command_action_types(self, sample_context):
        """Test different loop action types."""
        # Test increment_counter action
        command1 = LoopCommand(
            command_id=CommandId("loop-counter"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.FOR_COUNT,
                    "count": 2,
                    "loop_action": {
                        "type": "increment_counter",
                        "counter_name": "test_counter",
                    },
                }
            ),
        )

        result1 = command1.execute(sample_context)
        assert result1.success is True

        # Test pause action with limited duration
        command2 = LoopCommand(
            command_id=CommandId("loop-pause"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.FOR_COUNT,
                    "count": 2,
                    "loop_action": {"type": "pause", "duration": 0.05},  # Short pause
                }
            ),
        )

        result2 = command2.execute(sample_context)
        assert result2.success is True

        # Test beep action
        command3 = LoopCommand(
            command_id=CommandId("loop-beep"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.FOR_COUNT,
                    "count": 2,
                    "loop_action": {"type": "beep"},
                }
            ),
        )

        result3 = command3.execute(sample_context)
        assert result3.success is True


class TestBreakCommandMethods:
    """Test BreakCommand method functionality comprehensively."""

    @pytest.fixture
    def sample_context(self):
        """Create sample ExecutionContext for testing."""
        return ExecutionContext.create_test_context(
            permissions=frozenset(
                [
                    Permission.FLOW_CONTROL,
                    Permission.TEXT_INPUT,
                    Permission.SYSTEM_SOUND,
                ]
            )
        )

    def test_break_command_get_break_type(self):
        """Test get_break_type method."""
        # Test default break type
        command1 = BreakCommand(
            command_id=CommandId("break-default"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_break_type() == "loop"

        # Test explicit break type
        command2 = BreakCommand(
            command_id=CommandId("break-explicit"),
            parameters=CommandParameters(data={"break_type": "all"}),
        )
        assert command2.get_break_type() == "all"

    def test_break_command_get_break_label(self):
        """Test get_break_label method."""
        # Test no label
        command1 = BreakCommand(
            command_id=CommandId("break-no-label"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_break_label() is None

        # Test with label
        command2 = BreakCommand(
            command_id=CommandId("break-with-label"),
            parameters=CommandParameters(data={"break_label": "main_loop"}),
        )
        assert command2.get_break_label() == "main_loop"

    def test_break_command_get_break_message(self):
        """Test get_break_message method."""
        # Test no message
        command1 = BreakCommand(
            command_id=CommandId("break-no-message"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_break_message() is None

        # Test with message
        command2 = BreakCommand(
            command_id=CommandId("break-with-message"),
            parameters=CommandParameters(
                data={"break_message": "Breaking out of loop"}
            ),
        )
        assert command2.get_break_message() == "Breaking out of loop"

    def test_break_command_validation_detailed(self):
        """Test detailed _validate_impl functionality."""
        # Test validation failure with invalid break type
        command1 = BreakCommand(
            command_id=CommandId("break-invalid-type"),
            parameters=CommandParameters(data={"break_type": "invalid_break_type"}),
        )
        assert command1.validate() is False

        # Test validation success with valid break types
        valid_types = ["loop", "conditional", "all", "function", "script"]
        for break_type in valid_types:
            command = BreakCommand(
                command_id=CommandId(f"break-{break_type}"),
                parameters=CommandParameters(data={"break_type": break_type}),
            )
            assert command.validate() is True

    def test_break_command_execution_variations(self, sample_context):
        """Test break command execution with various parameters."""
        # Test execution with just break type
        command1 = BreakCommand(
            command_id=CommandId("break-simple"),
            parameters=CommandParameters(data={"break_type": "loop"}),
        )

        result1 = command1.execute(sample_context)
        assert result1.success is True
        assert "Break executed for loop" in result1.output

        # Test execution with break label
        command2 = BreakCommand(
            command_id=CommandId("break-labeled"),
            parameters=CommandParameters(
                data={
                    "break_type": "loop",
                    "break_label": "outer_loop",
                }
            ),
        )

        result2 = command2.execute(sample_context)
        assert result2.success is True
        assert "Break executed for loop (label: outer_loop)" in result2.output

        # Test execution with break message
        command3 = BreakCommand(
            command_id=CommandId("break-message"),
            parameters=CommandParameters(
                data={
                    "break_type": "all",
                    "break_message": "Emergency break!",
                }
            ),
        )

        result3 = command3.execute(sample_context)
        assert result3.success is True
        assert "Break executed for all" in result3.output

        # Test execution with both label and message
        command4 = BreakCommand(
            command_id=CommandId("break-full"),
            parameters=CommandParameters(
                data={
                    "break_type": "conditional",
                    "break_label": "condition_check",
                    "break_message": "Condition break executed",
                }
            ),
        )

        result4 = command4.execute(sample_context)
        assert result4.success is True
        assert (
            "Break executed for conditional (label: condition_check)" in result4.output
        )


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.fixture
    def sample_context(self):
        """Create sample ExecutionContext for testing."""
        return ExecutionContext.create_test_context(
            permissions=frozenset(
                [
                    Permission.FLOW_CONTROL,
                    Permission.TEXT_INPUT,
                    Permission.SYSTEM_SOUND,
                ]
            )
        )

    def test_conditional_command_exception_handling(self, sample_context):
        """Test exception handling in conditional command execution."""
        # Create a command that might cause an exception during evaluation
        command = ConditionalCommand(
            command_id=CommandId("cond-exception"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.REGEX_MATCH,
                    "left_operand": "test",
                    "right_operand": "[invalid",  # This will cause regex compilation error during execution
                    "then_action": {"type": "log", "message": "matched"},
                }
            ),
        )

        # Should handle the exception gracefully
        result = command.execute(sample_context)
        assert result.success is False
        assert "failed" in result.output.lower()

    def test_loop_command_exception_handling(self, sample_context):
        """Test exception handling in loop command execution."""
        # Create a command with an action that might cause issues
        command = LoopCommand(
            command_id=CommandId("loop-exception"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.FOR_COUNT,
                    "count": 2,
                    "loop_action": {
                        "type": "pause",
                        "duration": "invalid_duration",
                    },  # This might cause issues
                }
            ),
        )

        # Should handle any exceptions gracefully
        result = command.execute(sample_context)
        # The result should indicate completion, as the action handler should deal with invalid duration
        assert result.success is True

    def test_break_command_exception_handling(self, sample_context):
        """Test exception handling in break command execution."""
        # Test with normal parameters - should not cause exceptions
        command = BreakCommand(
            command_id=CommandId("break-normal"),
            parameters=CommandParameters(
                data={
                    "break_type": "loop",
                    "break_label": "test_label",
                    "break_message": "Test break message",
                }
            ),
        )

        result = command.execute(sample_context)
        assert result.success is True
        assert "Break executed" in result.output

    def test_condition_evaluation_edge_cases(self, sample_context):
        """Test edge cases in condition evaluation methods."""
        command = ConditionalCommand(
            command_id=CommandId("cond-edge-cases"),
            parameters=CommandParameters(
                data={
                    "condition_type": ConditionType.EQUALS,
                    "left_operand": "",  # Empty string
                    "right_operand": "",  # Empty string
                    "case_sensitive": True,
                    "then_action": {"type": "log", "message": "empty strings match"},
                }
            ),
        )

        result = command.execute(sample_context)
        assert result.success is True
        assert "condition evaluated to True" in result.output

    def test_loop_condition_evaluation_edge_cases(self, sample_context):
        """Test edge cases in loop condition evaluation."""
        # Test with valid condition that should evaluate to false
        command = LoopCommand(
            command_id=CommandId("loop-condition-edge"),
            parameters=CommandParameters(
                data={
                    "loop_type": LoopType.WHILE_CONDITION,
                    "condition": {
                        "condition_type": "equals",
                        "left_operand": "1",
                        "right_operand": "2",
                        "case_sensitive": True,
                    },
                    "loop_action": {"type": "log", "message": "should not execute"},
                    "max_duration": 5,
                }
            ),
        )

        result = command.execute(sample_context)
        assert result.success is True
        # Should have 0 iterations since condition is false
        assert "0 iterations" in result.output
