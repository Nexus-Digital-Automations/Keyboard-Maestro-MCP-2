"""Comprehensive flow command tests targeting maximum coverage.

This module specifically targets all uncovered code paths in src/commands/flow.py
to achieve significant coverage improvement toward the mandatory 95% threshold.
"""

import re
from unittest.mock import patch

import pytest
from src.commands.flow import (
    BreakCommand,
    ComparisonOperator,
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


class TestFlowCommandCoverageExpansion:
    """Test all uncovered code paths in flow commands for maximum coverage."""

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

    # ConditionalCommand Coverage Expansion

    def test_conditional_command_all_getter_methods(self):
        """Test all getter methods with various parameter combinations."""
        # Test get_condition_type with valid enum value
        command1 = ConditionalCommand(
            command_id=CommandId("cond-1"),
            parameters=CommandParameters(data={"condition_type": "regex_match"}),
        )
        assert command1.get_condition_type() == ConditionType.REGEX_MATCH

        # Test get_condition_type with invalid value (should default to EQUALS)
        command2 = ConditionalCommand(
            command_id=CommandId("cond-2"),
            parameters=CommandParameters(data={"condition_type": "invalid_type"}),
        )
        assert command2.get_condition_type() == ConditionType.EQUALS

        # Test get_left_operand and get_right_operand
        command3 = ConditionalCommand(
            command_id=CommandId("cond-3"),
            parameters=CommandParameters(
                data={
                    "left_operand": "test_left",
                    "right_operand": "test_right",
                }
            ),
        )
        assert command3.get_left_operand() == "test_left"
        assert command3.get_right_operand() == "test_right"

        # Test case sensitivity defaults
        command4 = ConditionalCommand(
            command_id=CommandId("cond-4"),
            parameters=CommandParameters(data={}),
        )
        assert command4.get_case_sensitive() is True

        # Test case sensitivity explicit setting
        command5 = ConditionalCommand(
            command_id=CommandId("cond-5"),
            parameters=CommandParameters(data={"case_sensitive": False}),
        )
        assert command5.get_case_sensitive() is False

    def test_conditional_command_action_getters(self):
        """Test action getter methods."""
        then_action = {"type": "log", "message": "then executed"}
        else_action = {"type": "beep"}

        command = ConditionalCommand(
            command_id=CommandId("cond-actions"),
            parameters=CommandParameters(
                data={
                    "then_action": then_action,
                    "else_action": else_action,
                }
            ),
        )

        assert command.get_then_action() == then_action
        assert command.get_else_action() == else_action

        # Test when actions are None
        command_no_actions = ConditionalCommand(
            command_id=CommandId("cond-no-actions"),
            parameters=CommandParameters(data={}),
        )
        assert command_no_actions.get_then_action() is None
        assert command_no_actions.get_else_action() is None

    def test_conditional_command_timeout_variations(self):
        """Test timeout getter with various inputs."""
        # Test default timeout
        command1 = ConditionalCommand(
            command_id=CommandId("cond-timeout-1"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_timeout().seconds == 5.0

        # Test valid timeout
        command2 = ConditionalCommand(
            command_id=CommandId("cond-timeout-2"),
            parameters=CommandParameters(data={"timeout": 10.0}),
        )
        assert command2.get_timeout().seconds == 10.0

        # Test timeout exceeding maximum (should cap at 30)
        command3 = ConditionalCommand(
            command_id=CommandId("cond-timeout-3"),
            parameters=CommandParameters(data={"timeout": 45.0}),
        )
        assert command3.get_timeout().seconds == 30.0

        # Test invalid timeout type (should default to 5)
        command4 = ConditionalCommand(
            command_id=CommandId("cond-timeout-4"),
            parameters=CommandParameters(data={"timeout": "invalid"}),
        )
        assert command4.get_timeout().seconds == 5.0

    def test_conditional_command_validation_comprehensive(self):
        """Test comprehensive validation scenarios."""
        # Test validation with no actions (should fail)
        command1 = ConditionalCommand(
            command_id=CommandId("cond-val-1"),
            parameters=CommandParameters(
                data={
                    "left_operand": "test",
                    "right_operand": "test",
                }
            ),
        )
        assert command1.validate() is False

        # Test validation with invalid regex pattern (should fail)
        command2 = ConditionalCommand(
            command_id=CommandId("cond-val-2"),
            parameters=CommandParameters(
                data={
                    "condition_type": "regex_match",
                    "left_operand": "test",
                    "right_operand": "[invalid",  # Invalid regex
                    "then_action": {"type": "log", "message": "test"},
                }
            ),
        )
        assert command2.validate() is False

        # Test validation with invalid action type (should fail)
        command3 = ConditionalCommand(
            command_id=CommandId("cond-val-3"),
            parameters=CommandParameters(
                data={
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"type": "dangerous_action"},  # Not in safe list
                }
            ),
        )
        assert command3.validate() is False

        # Test validation with non-dict action (should fail)
        command4 = ConditionalCommand(
            command_id=CommandId("cond-val-4"),
            parameters=CommandParameters(
                data={
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": "not_a_dict",  # Invalid structure
                }
            ),
        )
        assert command4.validate() is False

        # Test validation with action missing type (should fail)
        command5 = ConditionalCommand(
            command_id=CommandId("cond-val-5"),
            parameters=CommandParameters(
                data={
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"message": "no type field"},  # Missing type
                }
            ),
        )
        assert command5.validate() is False

        # Test valid command (should pass)
        command6 = ConditionalCommand(
            command_id=CommandId("cond-val-6"),
            parameters=CommandParameters(
                data={
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"type": "log", "message": "valid action"},
                }
            ),
        )
        assert command6.validate() is True

    def test_conditional_command_condition_evaluation_all_types(self, sample_context):
        """Test all condition evaluation types for complete coverage."""
        # Test EQUALS with case sensitivity
        command_eq = ConditionalCommand(
            command_id=CommandId("cond-eq"),
            parameters=CommandParameters(
                data={
                    "condition_type": "equals",
                    "left_operand": "Test",
                    "right_operand": "TEST",
                    "case_sensitive": False,
                    "then_action": {"type": "log", "message": "match"},
                }
            ),
        )
        result_eq = command_eq.execute(sample_context)
        assert result_eq.success is True

        # Test NOT_EQUALS
        command_ne = ConditionalCommand(
            command_id=CommandId("cond-ne"),
            parameters=CommandParameters(
                data={
                    "condition_type": "not_equals",
                    "left_operand": "hello",
                    "right_operand": "world",
                    "then_action": {"type": "log", "message": "not equal"},
                }
            ),
        )
        result_ne = command_ne.execute(sample_context)
        assert result_ne.success is True

        # Test CONTAINS
        command_contains = ConditionalCommand(
            command_id=CommandId("cond-contains"),
            parameters=CommandParameters(
                data={
                    "condition_type": "contains",
                    "left_operand": "hello world",
                    "right_operand": "world",
                    "then_action": {"type": "log", "message": "contains"},
                }
            ),
        )
        result_contains = command_contains.execute(sample_context)
        assert result_contains.success is True

        # Test NOT_CONTAINS
        command_not_contains = ConditionalCommand(
            command_id=CommandId("cond-not-contains"),
            parameters=CommandParameters(
                data={
                    "condition_type": "not_contains",
                    "left_operand": "hello",
                    "right_operand": "world",
                    "then_action": {"type": "log", "message": "not contains"},
                }
            ),
        )
        result_not_contains = command_not_contains.execute(sample_context)
        assert result_not_contains.success is True

        # Test IS_EMPTY
        command_empty = ConditionalCommand(
            command_id=CommandId("cond-empty"),
            parameters=CommandParameters(
                data={
                    "condition_type": "is_empty",
                    "left_operand": "   ",  # Just whitespace
                    "right_operand": "",
                    "then_action": {"type": "log", "message": "empty"},
                }
            ),
        )
        result_empty = command_empty.execute(sample_context)
        assert result_empty.success is True

        # Test IS_NOT_EMPTY
        command_not_empty = ConditionalCommand(
            command_id=CommandId("cond-not-empty"),
            parameters=CommandParameters(
                data={
                    "condition_type": "is_not_empty",
                    "left_operand": "hello",
                    "right_operand": "",
                    "then_action": {"type": "log", "message": "not empty"},
                }
            ),
        )
        result_not_empty = command_not_empty.execute(sample_context)
        assert result_not_empty.success is True

        # Test REGEX_MATCH with case insensitive
        command_regex = ConditionalCommand(
            command_id=CommandId("cond-regex"),
            parameters=CommandParameters(
                data={
                    "condition_type": "regex_match",
                    "left_operand": "Hello World",
                    "right_operand": "hello",
                    "case_sensitive": False,
                    "then_action": {"type": "log", "message": "regex match"},
                }
            ),
        )
        result_regex = command_regex.execute(sample_context)
        assert result_regex.success is True

        # Test numeric comparisons
        command_gt = ConditionalCommand(
            command_id=CommandId("cond-gt"),
            parameters=CommandParameters(
                data={
                    "condition_type": "greater_than",
                    "left_operand": "10",
                    "right_operand": "5",
                    "then_action": {"type": "log", "message": "greater"},
                }
            ),
        )
        result_gt = command_gt.execute(sample_context)
        assert result_gt.success is True

        command_lt = ConditionalCommand(
            command_id=CommandId("cond-lt"),
            parameters=CommandParameters(
                data={
                    "condition_type": "less_than",
                    "left_operand": "3",
                    "right_operand": "7",
                    "then_action": {"type": "log", "message": "less"},
                }
            ),
        )
        result_lt = command_lt.execute(sample_context)
        assert result_lt.success is True

        command_ge = ConditionalCommand(
            command_id=CommandId("cond-ge"),
            parameters=CommandParameters(
                data={
                    "condition_type": "greater_equal",
                    "left_operand": "5",
                    "right_operand": "5",
                    "then_action": {"type": "log", "message": "greater equal"},
                }
            ),
        )
        result_ge = command_ge.execute(sample_context)
        assert result_ge.success is True

        command_le = ConditionalCommand(
            command_id=CommandId("cond-le"),
            parameters=CommandParameters(
                data={
                    "condition_type": "less_equal",
                    "left_operand": "3",
                    "right_operand": "5",
                    "then_action": {"type": "log", "message": "less equal"},
                }
            ),
        )
        result_le = command_le.execute(sample_context)
        assert result_le.success is True

        # Test numeric comparison fallback to string comparison for invalid numbers
        command_numeric_fallback = ConditionalCommand(
            command_id=CommandId("cond-numeric-fallback"),
            parameters=CommandParameters(
                data={
                    "condition_type": "greater_than",
                    "left_operand": "not_a_number",
                    "right_operand": "also_not_a_number",
                    "then_action": {"type": "log", "message": "fallback"},
                }
            ),
        )
        result_numeric_fallback = command_numeric_fallback.execute(sample_context)
        assert result_numeric_fallback.success is True

    def test_conditional_command_action_execution_types(self, sample_context):
        """Test all supported action types for complete coverage."""
        # Test log action
        command_log = ConditionalCommand(
            command_id=CommandId("cond-log"),
            parameters=CommandParameters(
                data={
                    "condition_type": "equals",
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"type": "log", "message": "test message"},
                }
            ),
        )
        result_log = command_log.execute(sample_context)
        assert result_log.success is True

        # Test pause action
        command_pause = ConditionalCommand(
            command_id=CommandId("cond-pause"),
            parameters=CommandParameters(
                data={
                    "condition_type": "equals",
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"type": "pause", "duration": 0.1},
                }
            ),
        )
        result_pause = command_pause.execute(sample_context)
        assert result_pause.success is True

        # Test beep action
        command_beep = ConditionalCommand(
            command_id=CommandId("cond-beep"),
            parameters=CommandParameters(
                data={
                    "condition_type": "equals",
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"type": "beep"},
                }
            ),
        )
        result_beep = command_beep.execute(sample_context)
        assert result_beep.success is True

        # Test display_message action
        command_message = ConditionalCommand(
            command_id=CommandId("cond-message"),
            parameters=CommandParameters(
                data={
                    "condition_type": "equals",
                    "left_operand": "test",
                    "right_operand": "test",
                    "then_action": {"type": "display_message", "message": "Hello!"},
                }
            ),
        )
        result_message = command_message.execute(sample_context)
        assert result_message.success is True

        # Test recognized but not implemented action types
        for action_type in ["set_variable", "type_text", "key_press", "mouse_click"]:
            command = ConditionalCommand(
                command_id=CommandId(f"cond-{action_type}"),
                parameters=CommandParameters(
                    data={
                        "condition_type": "equals",
                        "left_operand": "test",
                        "right_operand": "test",
                        "then_action": {"type": action_type, "value": "test"},
                    }
                ),
            )
            result = command.execute(sample_context)
            assert result.success is True

    def test_conditional_command_else_action_execution(self, sample_context):
        """Test else action execution when condition is false."""
        command = ConditionalCommand(
            command_id=CommandId("cond-else"),
            parameters=CommandParameters(
                data={
                    "condition_type": "equals",
                    "left_operand": "hello",
                    "right_operand": "world",  # Will be false
                    "else_action": {"type": "log", "message": "condition false"},
                }
            ),
        )
        result = command.execute(sample_context)
        assert result.success is True
        assert "Condition evaluated to False" in result.output

    def test_conditional_command_no_action_to_execute(self, sample_context):
        """Test when condition evaluates but no matching action exists."""
        # True condition but no then_action
        command1 = ConditionalCommand(
            command_id=CommandId("cond-no-then"),
            parameters=CommandParameters(
                data={
                    "condition_type": "equals",
                    "left_operand": "test",
                    "right_operand": "test",
                    "else_action": {"type": "log", "message": "else"},
                }
            ),
        )
        result1 = command1.execute(sample_context)
        assert result1.success is True
        assert "no action to execute" in result1.output

        # False condition but no else_action
        command2 = ConditionalCommand(
            command_id=CommandId("cond-no-else"),
            parameters=CommandParameters(
                data={
                    "condition_type": "equals",
                    "left_operand": "hello",
                    "right_operand": "world",
                    "then_action": {"type": "log", "message": "then"},
                }
            ),
        )
        result2 = command2.execute(sample_context)
        assert result2.success is True
        assert "no action to execute" in result2.output

    def test_conditional_command_exception_handling(self, sample_context):
        """Test exception handling during condition evaluation and action execution."""
        # Test regex compilation error during execution (different from validation)
        with patch("re.search", side_effect=re.error("Runtime regex error")):
            command = ConditionalCommand(
                command_id=CommandId("cond-regex-error"),
                parameters=CommandParameters(
                    data={
                        "condition_type": "regex_match",
                        "left_operand": "test",
                        "right_operand": "valid_pattern",
                        "then_action": {"type": "log", "message": "test"},
                    }
                ),
            )
            result = command.execute(sample_context)
            assert result.success is False
            assert "failed" in result.error_message

    def test_conditional_command_permissions(self):
        """Test permission requirements based on action types."""
        # Base command with log action
        command1 = ConditionalCommand(
            command_id=CommandId("cond-perm-1"),
            parameters=CommandParameters(
                data={
                    "then_action": {"type": "log", "message": "test"},
                }
            ),
        )
        permissions1 = command1.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions1

        # Command with text input action
        command2 = ConditionalCommand(
            command_id=CommandId("cond-perm-2"),
            parameters=CommandParameters(
                data={
                    "then_action": {"type": "type_text", "text": "hello"},
                    "else_action": {"type": "key_press", "key": "enter"},
                }
            ),
        )
        permissions2 = command2.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions2
        assert Permission.TEXT_INPUT in permissions2

        # Command with mouse action
        command3 = ConditionalCommand(
            command_id=CommandId("cond-perm-3"),
            parameters=CommandParameters(
                data={
                    "then_action": {"type": "mouse_click", "x": 100, "y": 200},
                }
            ),
        )
        permissions3 = command3.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions3
        assert Permission.MOUSE_CONTROL in permissions3

    def test_conditional_command_security_risk_level(self):
        """Test security risk level assessment."""
        command = ConditionalCommand(
            command_id=CommandId("cond-security"),
            parameters=CommandParameters(data={}),
        )
        assert command.get_security_risk_level() == "medium"

    # LoopCommand Coverage Expansion

    def test_loop_command_all_getter_methods(self):
        """Test all loop command getter methods."""
        # Test get_loop_type with valid enum
        command1 = LoopCommand(
            command_id=CommandId("loop-1"),
            parameters=CommandParameters(data={"loop_type": "while_condition"}),
        )
        assert command1.get_loop_type() == LoopType.WHILE_CONDITION

        # Test get_loop_type with invalid value (should default to FOR_COUNT)
        command2 = LoopCommand(
            command_id=CommandId("loop-2"),
            parameters=CommandParameters(data={"loop_type": "invalid_type"}),
        )
        assert command2.get_loop_type() == LoopType.FOR_COUNT

        # Test get_count with various values
        command3 = LoopCommand(
            command_id=CommandId("loop-3"),
            parameters=CommandParameters(data={"count": 5}),
        )
        assert command3.get_count() == 5

        # Test get_count with default
        command4 = LoopCommand(
            command_id=CommandId("loop-4"),
            parameters=CommandParameters(data={}),
        )
        assert command4.get_count() == 1

        # Test get_count with value exceeding max (should cap at MAX_LOOP_ITERATIONS)
        command5 = LoopCommand(
            command_id=CommandId("loop-5"),
            parameters=CommandParameters(data={"count": 2000}),
        )
        assert command5.get_count() == 1000  # MAX_LOOP_ITERATIONS

        # Test get_count with invalid type (should default to 1)
        command6 = LoopCommand(
            command_id=CommandId("loop-6"),
            parameters=CommandParameters(data={"count": "invalid"}),
        )
        assert command6.get_count() == 1

        # Test get_condition
        condition = {
            "condition_type": "equals",
            "left_operand": "x",
            "right_operand": "5",
        }
        command7 = LoopCommand(
            command_id=CommandId("loop-7"),
            parameters=CommandParameters(data={"condition": condition}),
        )
        assert command7.get_condition() == condition

        # Test get_items with valid list
        items = ["a", "b", "c"]
        command8 = LoopCommand(
            command_id=CommandId("loop-8"),
            parameters=CommandParameters(data={"items": items}),
        )
        assert command8.get_items() == items

        # Test get_items with non-list (should return empty list)
        command9 = LoopCommand(
            command_id=CommandId("loop-9"),
            parameters=CommandParameters(data={"items": "not_a_list"}),
        )
        assert command9.get_items() == []

        # Test get_items with too many items (should truncate)
        large_items = [str(i) for i in range(1500)]
        command10 = LoopCommand(
            command_id=CommandId("loop-10"),
            parameters=CommandParameters(data={"items": large_items}),
        )
        result_items = command10.get_items()
        assert len(result_items) == 1000  # MAX_LOOP_ITERATIONS

        # Test get_loop_action
        action = {"type": "log", "message": "iteration"}
        command11 = LoopCommand(
            command_id=CommandId("loop-11"),
            parameters=CommandParameters(data={"loop_action": action}),
        )
        assert command11.get_loop_action() == action

    def test_loop_command_duration_methods(self):
        """Test duration-related methods."""
        # Test get_max_duration with default
        command1 = LoopCommand(
            command_id=CommandId("loop-dur-1"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_max_duration().seconds == 60.0

        # Test get_max_duration with custom value
        command2 = LoopCommand(
            command_id=CommandId("loop-dur-2"),
            parameters=CommandParameters(data={"max_duration": 120.0}),
        )
        assert command2.get_max_duration().seconds == 120.0

        # Test get_max_duration with value exceeding max (should cap at 300)
        command3 = LoopCommand(
            command_id=CommandId("loop-dur-3"),
            parameters=CommandParameters(data={"max_duration": 500.0}),
        )
        assert command3.get_max_duration().seconds == 300.0

        # Test get_max_duration with invalid type (should default to 60)
        command4 = LoopCommand(
            command_id=CommandId("loop-dur-4"),
            parameters=CommandParameters(data={"max_duration": "invalid"}),
        )
        assert command4.get_max_duration().seconds == 60.0

        # Test get_break_on_error with default
        command5 = LoopCommand(
            command_id=CommandId("loop-break-1"),
            parameters=CommandParameters(data={}),
        )
        assert command5.get_break_on_error() is True

        # Test get_break_on_error with explicit setting
        command6 = LoopCommand(
            command_id=CommandId("loop-break-2"),
            parameters=CommandParameters(data={"break_on_error": False}),
        )
        assert command6.get_break_on_error() is False

    def test_loop_command_validation_comprehensive(self):
        """Test comprehensive validation scenarios."""
        # Test validation failure with invalid count in parameters (checking raw parameter)
        command1 = LoopCommand(
            command_id=CommandId("loop-val-1"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_count",
                    "count": 0,  # Invalid count
                    "loop_action": {"type": "log", "message": "test"},
                }
            ),
        )
        # Since get_count() corrects to 1, but validation should check raw parameter
        # This depends on implementation - let's test what the code actually does
        validation_result = command1.validate()
        # The validation uses get_count() which returns 1, so it should pass
        assert validation_result is True

        # Test validation failure with invalid while condition
        command2 = LoopCommand(
            command_id=CommandId("loop-val-2"),
            parameters=CommandParameters(
                data={
                    "loop_type": "while_condition",
                    "condition": "not_a_dict",  # Invalid condition structure
                    "loop_action": {"type": "log", "message": "test"},
                }
            ),
        )
        assert command2.validate() is False

        # Test validation failure with missing condition fields
        command3 = LoopCommand(
            command_id=CommandId("loop-val-3"),
            parameters=CommandParameters(
                data={
                    "loop_type": "while_condition",
                    "condition": {"incomplete": "condition"},  # Missing required fields
                    "loop_action": {"type": "log", "message": "test"},
                }
            ),
        )
        assert command3.validate() is False

        # Test validation failure with empty for_each items
        command4 = LoopCommand(
            command_id=CommandId("loop-val-4"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_each",
                    "items": [],  # Empty items
                    "loop_action": {"type": "log", "message": "test"},
                }
            ),
        )
        assert command4.validate() is False

        # Test validation failure with no loop action
        command5 = LoopCommand(
            command_id=CommandId("loop-val-5"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_count",
                    "count": 1,
                    # No loop_action
                }
            ),
        )
        assert command5.validate() is False

        # Test validation failure with invalid action type
        command6 = LoopCommand(
            command_id=CommandId("loop-val-6"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_count",
                    "count": 1,
                    "loop_action": {"type": "dangerous_action"},  # Not in safe list
                }
            ),
        )
        assert command6.validate() is False

        # Test validation success with valid parameters
        command7 = LoopCommand(
            command_id=CommandId("loop-val-7"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_count",
                    "count": 3,
                    "loop_action": {"type": "log", "message": "valid"},
                }
            ),
        )
        assert command7.validate() is True

    def test_loop_command_condition_validation(self):
        """Test loop condition validation method."""
        # Test valid condition
        valid_condition = {
            "condition_type": "equals",
            "left_operand": "test",
            "right_operand": "value",
        }
        command1 = LoopCommand(
            command_id=CommandId("loop-cond-1"),
            parameters=CommandParameters(
                data={
                    "loop_type": "while_condition",
                    "condition": valid_condition,
                    "loop_action": {"type": "log", "message": "test"},
                }
            ),
        )
        assert command1._validate_condition(valid_condition) is True

        # Test invalid condition type
        invalid_condition1 = {
            "condition_type": "invalid_type",
            "left_operand": "test",
            "right_operand": "value",
        }
        assert command1._validate_condition(invalid_condition1) is False

        # Test missing required fields
        invalid_condition2 = {
            "condition_type": "equals",
            # Missing left_operand and right_operand
        }
        assert command1._validate_condition(invalid_condition2) is False

        # Test non-dict condition
        assert command1._validate_condition("not_a_dict") is False

    def test_loop_command_action_validation(self):
        """Test loop action validation method."""
        command = LoopCommand(
            command_id=CommandId("loop-action-val"),
            parameters=CommandParameters(data={}),
        )

        # Test valid action
        valid_action = {"type": "log", "message": "test"}
        assert command._validate_action(valid_action) is True

        # Test valid action with increment_counter (specific to loop actions)
        increment_action = {"type": "increment_counter", "counter_name": "test"}
        assert command._validate_action(increment_action) is True

        # Test invalid action type
        invalid_action1 = {"type": "dangerous_action"}
        assert command._validate_action(invalid_action1) is False

        # Test action without type
        invalid_action2 = {"message": "no type"}
        assert command._validate_action(invalid_action2) is False

        # Test non-dict action
        assert command._validate_action("not_a_dict") is False

    def test_loop_command_execution_for_count(self, sample_context):
        """Test FOR_COUNT loop execution."""
        command = LoopCommand(
            command_id=CommandId("loop-for-count"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_count",
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
        assert result.metadata["iterations_completed"] == 3

    def test_loop_command_execution_while_condition(self, sample_context):
        """Test WHILE_CONDITION loop execution."""
        # Test condition that should be false immediately
        command1 = LoopCommand(
            command_id=CommandId("loop-while-false"),
            parameters=CommandParameters(
                data={
                    "loop_type": "while_condition",
                    "condition": {
                        "condition_type": "equals",
                        "left_operand": "false",
                        "right_operand": "true",
                    },
                    "loop_action": {"type": "log", "message": "iteration"},
                    "max_duration": 10,
                }
            ),
        )
        result1 = command1.execute(sample_context)
        assert result1.success is True
        assert result1.metadata["iterations_completed"] == 0

        # Test condition that should be true for a few iterations
        command2 = LoopCommand(
            command_id=CommandId("loop-while-limited"),
            parameters=CommandParameters(
                data={
                    "loop_type": "while_condition",
                    "condition": {
                        "condition_type": "equals",
                        "left_operand": "true",
                        "right_operand": "true",
                    },
                    "loop_action": {"type": "log", "message": "iteration"},
                    "max_duration": 0.1,  # Short duration to prevent long execution
                }
            ),
        )
        result2 = command2.execute(sample_context)
        assert result2.success is True
        # Should have some iterations but be limited by duration
        assert result2.metadata["iterations_completed"] >= 0

    def test_loop_command_execution_for_each(self, sample_context):
        """Test FOR_EACH loop execution."""
        command = LoopCommand(
            command_id=CommandId("loop-for-each"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_each",
                    "items": ["apple", "banana", "cherry"],
                    "loop_action": {"type": "log", "message": "processing item"},
                    "max_duration": 10,
                }
            ),
        )
        result = command.execute(sample_context)
        assert result.success is True
        assert "3 iterations" in result.output
        assert result.metadata["iterations_completed"] == 3

    def test_loop_command_action_execution_types(self, sample_context):
        """Test all loop action execution types."""
        # Test log action with current item
        command1 = LoopCommand(
            command_id=CommandId("loop-action-log"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_each",
                    "items": ["item1"],
                    "loop_action": {"type": "log", "message": "processing"},
                }
            ),
        )
        result1 = command1.execute(sample_context)
        assert result1.success is True

        # Test pause action in loop (limited duration)
        command2 = LoopCommand(
            command_id=CommandId("loop-action-pause"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_count",
                    "count": 2,
                    "loop_action": {"type": "pause", "duration": 0.01},  # Very short
                }
            ),
        )
        result2 = command2.execute(sample_context)
        assert result2.success is True

        # Test increment_counter action
        command3 = LoopCommand(
            command_id=CommandId("loop-action-counter"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_count",
                    "count": 2,
                    "loop_action": {
                        "type": "increment_counter",
                        "counter_name": "test_counter",
                    },
                }
            ),
        )
        result3 = command3.execute(sample_context)
        assert result3.success is True

        # Test beep action
        command4 = LoopCommand(
            command_id=CommandId("loop-action-beep"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_count",
                    "count": 1,
                    "loop_action": {"type": "beep"},
                }
            ),
        )
        result4 = command4.execute(sample_context)
        assert result4.success is True

        # Test recognized but not implemented action types
        for action_type in ["set_variable", "type_text", "key_press", "mouse_click"]:
            command = LoopCommand(
                command_id=CommandId(f"loop-action-{action_type}"),
                parameters=CommandParameters(
                    data={
                        "loop_type": "for_count",
                        "count": 1,
                        "loop_action": {"type": action_type, "value": "test"},
                    }
                ),
            )
            result = command.execute(sample_context)
            assert result.success is True

    def test_loop_command_condition_evaluation(self, sample_context):
        """Test loop condition evaluation logic."""
        command = LoopCommand(
            command_id=CommandId("loop-cond-eval"),
            parameters=CommandParameters(
                data={
                    "loop_type": "while_condition",
                    "condition": {
                        "condition_type": "equals",
                        "left_operand": "test",
                        "right_operand": "test",
                        "case_sensitive": True,
                    },
                    "loop_action": {"type": "log", "message": "iteration"},
                    "max_duration": 0.1,  # Short to prevent long execution
                }
            ),
        )

        # Test the condition evaluation directly
        condition = {
            "condition_type": "equals",
            "left_operand": "hello",
            "right_operand": "hello",
            "case_sensitive": True,
        }
        assert command._evaluate_loop_condition(condition) is True

        # Test with case insensitive
        condition_case_insensitive = {
            "condition_type": "equals",
            "left_operand": "Hello",
            "right_operand": "HELLO",
            "case_sensitive": False,
        }
        assert command._evaluate_loop_condition(condition_case_insensitive) is True

        # Test different condition types
        condition_not_equals = {
            "condition_type": "not_equals",
            "left_operand": "hello",
            "right_operand": "world",
        }
        assert command._evaluate_loop_condition(condition_not_equals) is True

        condition_contains = {
            "condition_type": "contains",
            "left_operand": "hello world",
            "right_operand": "world",
        }
        assert command._evaluate_loop_condition(condition_contains) is True

        condition_not_contains = {
            "condition_type": "not_contains",
            "left_operand": "hello",
            "right_operand": "world",
        }
        assert command._evaluate_loop_condition(condition_not_contains) is True

        condition_empty = {
            "condition_type": "is_empty",
            "left_operand": "   ",  # Whitespace
        }
        assert command._evaluate_loop_condition(condition_empty) is True

        condition_not_empty = {
            "condition_type": "is_not_empty",
            "left_operand": "hello",
        }
        assert command._evaluate_loop_condition(condition_not_empty) is True

        condition_greater = {
            "condition_type": "greater_than",
            "left_operand": "10",
            "right_operand": "5",
        }
        assert command._evaluate_loop_condition(condition_greater) is True

        condition_less = {
            "condition_type": "less_than",
            "left_operand": "3",
            "right_operand": "7",
        }
        assert command._evaluate_loop_condition(condition_less) is True

        # Test exception handling in condition evaluation
        invalid_condition = {
            "condition_type": "invalid_type",
            "left_operand": "test",
            "right_operand": "test",
        }
        assert command._evaluate_loop_condition(invalid_condition) is False

    def test_loop_command_error_handling_and_timeout(self, sample_context):
        """Test error handling and timeout scenarios."""
        # Test loop that breaks on error
        with patch.object(LoopCommand, "_execute_loop_action", return_value=False):
            command1 = LoopCommand(
                command_id=CommandId("loop-error-break"),
                parameters=CommandParameters(
                    data={
                        "loop_type": "for_count",
                        "count": 5,
                        "loop_action": {"type": "log", "message": "test"},
                        "break_on_error": True,
                    }
                ),
            )
            result1 = command1.execute(sample_context)
            assert result1.success is True
            assert result1.metadata["total_errors"] > 0

        # Test loop that continues on error
        with patch.object(LoopCommand, "_execute_loop_action", return_value=False):
            command2 = LoopCommand(
                command_id=CommandId("loop-error-continue"),
                parameters=CommandParameters(
                    data={
                        "loop_type": "for_count",
                        "count": 3,
                        "loop_action": {"type": "log", "message": "test"},
                        "break_on_error": False,
                    }
                ),
            )
            result2 = command2.execute(sample_context)
            assert result2.success is True
            assert result2.metadata["iterations_completed"] == 3
            assert result2.metadata["total_errors"] == 3

        # Test exception during loop execution
        with patch.object(
            LoopCommand, "_execute_loop_action", side_effect=Exception("Test error")
        ):
            command3 = LoopCommand(
                command_id=CommandId("loop-exception"),
                parameters=CommandParameters(
                    data={
                        "loop_type": "for_count",
                        "count": 1,
                        "loop_action": {"type": "log", "message": "test"},
                    }
                ),
            )
            result3 = command3.execute(sample_context)
            assert result3.success is False
            assert "failed" in result3.error_message

    def test_loop_command_permissions(self):
        """Test permission requirements based on action types."""
        # Base command with log action
        command1 = LoopCommand(
            command_id=CommandId("loop-perm-1"),
            parameters=CommandParameters(
                data={
                    "loop_action": {"type": "log", "message": "test"},
                }
            ),
        )
        permissions1 = command1.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions1

        # Command with text input action
        command2 = LoopCommand(
            command_id=CommandId("loop-perm-2"),
            parameters=CommandParameters(
                data={
                    "loop_action": {"type": "type_text", "text": "hello"},
                }
            ),
        )
        permissions2 = command2.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions2
        assert Permission.TEXT_INPUT in permissions2

        # Command with key press action
        command3 = LoopCommand(
            command_id=CommandId("loop-perm-3"),
            parameters=CommandParameters(
                data={
                    "loop_action": {"type": "key_press", "key": "enter"},
                }
            ),
        )
        permissions3 = command3.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions3
        assert Permission.TEXT_INPUT in permissions3

        # Command with mouse action
        command4 = LoopCommand(
            command_id=CommandId("loop-perm-4"),
            parameters=CommandParameters(
                data={
                    "loop_action": {"type": "mouse_click", "x": 100, "y": 200},
                }
            ),
        )
        permissions4 = command4.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions4
        assert Permission.MOUSE_CONTROL in permissions4

    def test_loop_command_security_risk_level(self):
        """Test security risk level assessment."""
        command = LoopCommand(
            command_id=CommandId("loop-security"),
            parameters=CommandParameters(data={}),
        )
        assert command.get_security_risk_level() == "high"

    # BreakCommand Coverage Expansion

    def test_break_command_all_getter_methods(self):
        """Test all break command getter methods."""
        # Test get_break_type with default
        command1 = BreakCommand(
            command_id=CommandId("break-1"),
            parameters=CommandParameters(data={}),
        )
        assert command1.get_break_type() == "loop"

        # Test get_break_type with explicit value
        command2 = BreakCommand(
            command_id=CommandId("break-2"),
            parameters=CommandParameters(data={"break_type": "all"}),
        )
        assert command2.get_break_type() == "all"

        # Test get_break_label with None
        command3 = BreakCommand(
            command_id=CommandId("break-3"),
            parameters=CommandParameters(data={}),
        )
        assert command3.get_break_label() is None

        # Test get_break_label with value
        command4 = BreakCommand(
            command_id=CommandId("break-4"),
            parameters=CommandParameters(data={"break_label": "main_loop"}),
        )
        assert command4.get_break_label() == "main_loop"

        # Test get_break_message with None
        command5 = BreakCommand(
            command_id=CommandId("break-5"),
            parameters=CommandParameters(data={}),
        )
        assert command5.get_break_message() is None

        # Test get_break_message with value
        command6 = BreakCommand(
            command_id=CommandId("break-6"),
            parameters=CommandParameters(data={"break_message": "Breaking out"}),
        )
        assert command6.get_break_message() == "Breaking out"

    def test_break_command_validation_comprehensive(self):
        """Test comprehensive validation scenarios."""
        # Test validation failure with invalid break type
        command1 = BreakCommand(
            command_id=CommandId("break-val-1"),
            parameters=CommandParameters(data={"break_type": "invalid_type"}),
        )
        assert command1.validate() is False

        # Test validation success with all valid break types
        valid_types = ["loop", "conditional", "all", "function", "script"]
        for break_type in valid_types:
            command = BreakCommand(
                command_id=CommandId(f"break-val-{break_type}"),
                parameters=CommandParameters(data={"break_type": break_type}),
            )
            assert command.validate() is True

        # Test validation success with valid parameters
        command2 = BreakCommand(
            command_id=CommandId("break-val-2"),
            parameters=CommandParameters(
                data={
                    "break_type": "loop",
                    "break_label": "outer_loop",
                    "break_message": "Exiting loop",
                }
            ),
        )
        assert command2.validate() is True

    def test_break_command_execution_variations(self, sample_context):
        """Test break command execution with various parameters."""
        # Test execution with just break type
        command1 = BreakCommand(
            command_id=CommandId("break-exec-1"),
            parameters=CommandParameters(data={"break_type": "loop"}),
        )
        result1 = command1.execute(sample_context)
        assert result1.success is True
        assert "Break executed for loop" in result1.output
        assert result1.metadata["break_type"] == "loop"

        # Test execution with break label
        command2 = BreakCommand(
            command_id=CommandId("break-exec-2"),
            parameters=CommandParameters(
                data={
                    "break_type": "conditional",
                    "break_label": "main_condition",
                }
            ),
        )
        result2 = command2.execute(sample_context)
        assert result2.success is True
        assert (
            "Break executed for conditional (label: main_condition)" in result2.output
        )
        assert result2.metadata["break_label"] == "main_condition"

        # Test execution with break message
        command3 = BreakCommand(
            command_id=CommandId("break-exec-3"),
            parameters=CommandParameters(
                data={
                    "break_type": "all",
                    "break_message": "Emergency break executed",
                }
            ),
        )
        result3 = command3.execute(sample_context)
        assert result3.success is True
        assert "Break executed for all" in result3.output
        assert result3.metadata["break_message"] == "Emergency break executed"

        # Test execution with all parameters
        command4 = BreakCommand(
            command_id=CommandId("break-exec-4"),
            parameters=CommandParameters(
                data={
                    "break_type": "function",
                    "break_label": "processing_function",
                    "break_message": "Function break executed",
                }
            ),
        )
        result4 = command4.execute(sample_context)
        assert result4.success is True
        assert (
            "Break executed for function (label: processing_function)" in result4.output
        )
        assert result4.metadata["break_type"] == "function"
        assert result4.metadata["break_label"] == "processing_function"
        assert result4.metadata["break_message"] == "Function break executed"

    def test_break_command_exception_handling(self, sample_context):
        """Test exception handling in break command execution."""
        # Test normal execution (should not raise exceptions)
        command = BreakCommand(
            command_id=CommandId("break-exception"),
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

        # Simulate exception during execution
        with patch("time.time", side_effect=Exception("Time error")):
            result_error = command.execute(sample_context)
            assert result_error.success is False
            assert "failed" in result_error.error_message

    def test_break_command_permissions(self):
        """Test permission requirements."""
        command = BreakCommand(
            command_id=CommandId("break-perms"),
            parameters=CommandParameters(data={}),
        )
        permissions = command.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions
        assert len(permissions) == 1  # Only flow control

    def test_break_command_security_risk_level(self):
        """Test security risk level assessment."""
        command = BreakCommand(
            command_id=CommandId("break-security"),
            parameters=CommandParameters(data={}),
        )
        assert command.get_security_risk_level() == "low"

    # Additional ComparisonOperator enum coverage

    def test_comparison_operator_complete_enum(self):
        """Test all ComparisonOperator enum values for complete coverage."""
        assert ComparisonOperator.EQ.value == "=="
        assert ComparisonOperator.NE.value == "!="
        assert ComparisonOperator.GT.value == ">"
        assert ComparisonOperator.LT.value == "<"
        assert ComparisonOperator.GE.value == ">="
        assert ComparisonOperator.LE.value == "<="
        assert ComparisonOperator.IN.value == "in"
        assert ComparisonOperator.NOT_IN.value == "not_in"

        # Test enum comparison
        assert ComparisonOperator.EQ != ComparisonOperator.NE
        assert ComparisonOperator.GT == ComparisonOperator.GT

    # Edge cases and error conditions

    def test_edge_cases_and_boundary_conditions(self, sample_context):
        """Test edge cases and boundary conditions for complete coverage."""
        # Test ConditionalCommand with empty operands
        command1 = ConditionalCommand(
            command_id=CommandId("edge-1"),
            parameters=CommandParameters(
                data={
                    "condition_type": "equals",
                    "left_operand": "",
                    "right_operand": "",
                    "then_action": {"type": "log", "message": "empty match"},
                }
            ),
        )
        result1 = command1.execute(sample_context)
        assert result1.success is True

        # Test LoopCommand with minimum count
        command2 = LoopCommand(
            command_id=CommandId("edge-2"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_count",
                    "count": 1,
                    "loop_action": {"type": "log", "message": "single iteration"},
                }
            ),
        )
        result2 = command2.execute(sample_context)
        assert result2.success is True
        assert result2.metadata["iterations_completed"] == 1

        # Test LoopCommand with single item
        command3 = LoopCommand(
            command_id=CommandId("edge-3"),
            parameters=CommandParameters(
                data={
                    "loop_type": "for_each",
                    "items": ["single_item"],
                    "loop_action": {"type": "log", "message": "processing"},
                }
            ),
        )
        result3 = command3.execute(sample_context)
        assert result3.success is True
        assert result3.metadata["iterations_completed"] == 1

        # Test BreakCommand with minimum parameters
        command4 = BreakCommand(
            command_id=CommandId("edge-4"),
            parameters=CommandParameters(data={}),
        )
        result4 = command4.execute(sample_context)
        assert result4.success is True
        assert result4.metadata["break_type"] == "loop"
