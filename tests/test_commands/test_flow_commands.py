"""
Tests for flow control commands.

Tests conditional, loop, and break commands with security validation
and proper contract enforcement.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.core.types import (
    CommandId, CommandParameters, ExecutionContext, Permission, Duration
)
from src.commands.flow import (
    ConditionalCommand, LoopCommand, BreakCommand,
    ConditionType, LoopType
)


class TestConditionalCommand:
    """Test conditional command functionality."""
    
    def test_conditional_command_creation(self):
        """Test basic conditional command creation."""
        params = CommandParameters({
            "condition_type": "equals",
            "left_operand": "hello",
            "right_operand": "hello",
            "then_action": {"type": "log", "message": "Equal!"}
        })
        cmd = ConditionalCommand(CommandId("test"), params)
        
        assert cmd.get_condition_type() == ConditionType.EQUALS
        assert cmd.get_left_operand() == "hello"
        assert cmd.get_right_operand() == "hello"
        assert cmd.get_then_action() is not None
    
    def test_conditional_validation_valid(self):
        """Test conditional command validation with valid parameters."""
        params = CommandParameters({
            "condition_type": "contains",
            "left_operand": "hello world",
            "right_operand": "world",
            "then_action": {"type": "log", "message": "Contains world!"},
            "else_action": {"type": "log", "message": "Doesn't contain world"}
        })
        cmd = ConditionalCommand(CommandId("test"), params)
        
        assert cmd.validate() is True
    
    def test_conditional_validation_no_actions(self):
        """Test conditional command validation with no actions."""
        params = CommandParameters({
            "condition_type": "equals",
            "left_operand": "test",
            "right_operand": "test"
        })
        cmd = ConditionalCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    def test_conditional_validation_invalid_regex(self):
        """Test conditional command validation with invalid regex."""
        params = CommandParameters({
            "condition_type": "regex_match",
            "left_operand": "test",
            "right_operand": "[invalid regex",
            "then_action": {"type": "log", "message": "Match!"}
        })
        cmd = ConditionalCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    def test_conditional_execution_equals_true(self):
        """Test conditional execution with equals condition (true)."""
        params = CommandParameters({
            "condition_type": "equals",
            "left_operand": "test",
            "right_operand": "test",
            "then_action": {"type": "log", "message": "Equal!"}
        })
        cmd = ConditionalCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.FLOW_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert "Condition evaluated to True" in result.output
        assert result.metadata["condition_result"] is True
        assert result.metadata["action_executed"] == "then"
    
    def test_conditional_execution_equals_false(self):
        """Test conditional execution with equals condition (false)."""
        params = CommandParameters({
            "condition_type": "equals",
            "left_operand": "test",
            "right_operand": "different",
            "else_action": {"type": "log", "message": "Not equal!"}
        })
        cmd = ConditionalCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.FLOW_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert "Condition evaluated to False" in result.output
        assert result.metadata["condition_result"] is False
        assert result.metadata["action_executed"] == "else"
    
    def test_conditional_execution_contains(self):
        """Test conditional execution with contains condition."""
        params = CommandParameters({
            "condition_type": "contains",
            "left_operand": "hello world",
            "right_operand": "world",
            "then_action": {"type": "log", "message": "Contains world!"}
        })
        cmd = ConditionalCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.FLOW_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert result.metadata["condition_result"] is True
    
    def test_conditional_execution_numeric_comparison(self):
        """Test conditional execution with numeric comparison."""
        params = CommandParameters({
            "condition_type": "greater_than",
            "left_operand": "10",
            "right_operand": "5",
            "then_action": {"type": "log", "message": "Greater!"}
        })
        cmd = ConditionalCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.FLOW_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert result.metadata["condition_result"] is True
    
    def test_conditional_permissions(self):
        """Test conditional command permission requirements."""
        params = CommandParameters({
            "condition_type": "equals",
            "left_operand": "test",
            "right_operand": "test",
            "then_action": {"type": "log", "message": "Equal!"}
        })
        cmd = ConditionalCommand(CommandId("test"), params)
        
        permissions = cmd.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions
    
    def test_conditional_security_risk(self):
        """Test conditional command security risk level."""
        params = CommandParameters({
            "condition_type": "equals",
            "left_operand": "test",
            "right_operand": "test",
            "then_action": {"type": "log", "message": "Equal!"}
        })
        cmd = ConditionalCommand(CommandId("test"), params)
        
        assert cmd.get_security_risk_level() == "medium"


class TestLoopCommand:
    """Test loop command functionality."""
    
    def test_loop_command_creation(self):
        """Test basic loop command creation."""
        params = CommandParameters({
            "loop_type": "for_count",
            "count": 3,
            "loop_action": {"type": "log", "message": "Iteration"}
        })
        cmd = LoopCommand(CommandId("test"), params)
        
        assert cmd.get_loop_type() == LoopType.FOR_COUNT
        assert cmd.get_count() == 3
        assert cmd.get_loop_action() is not None
    
    def test_loop_validation_valid_for_count(self):
        """Test loop command validation with valid for_count parameters."""
        params = CommandParameters({
            "loop_type": "for_count",
            "count": 5,
            "loop_action": {"type": "log", "message": "Iteration"},
            "max_duration": 30.0
        })
        cmd = LoopCommand(CommandId("test"), params)
        
        assert cmd.validate() is True
    
    def test_loop_validation_valid_for_each(self):
        """Test loop command validation with valid for_each parameters."""
        params = CommandParameters({
            "loop_type": "for_each",
            "items": ["apple", "banana", "cherry"],
            "loop_action": {"type": "log", "message": "Processing item"}
        })
        cmd = LoopCommand(CommandId("test"), params)
        
        assert cmd.validate() is True
    
    def test_loop_validation_invalid_count(self):
        """Test loop command validation with invalid count."""
        params = CommandParameters({
            "loop_type": "for_count",
            "count": 2000,  # Too many iterations
            "loop_action": {"type": "log", "message": "Iteration"}
        })
        cmd = LoopCommand(CommandId("test"), params)
        
        # Count should be clamped but validation should still pass
        # because the clamped value is valid
        assert cmd.get_count() == 1000  # MAX_LOOP_ITERATIONS
        assert cmd.validate() is True  # Should pass with clamped value
    
    def test_loop_validation_no_action(self):
        """Test loop command validation with no action."""
        params = CommandParameters({
            "loop_type": "for_count",
            "count": 3
        })
        cmd = LoopCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_loop_execution_for_count(self, mock_sleep):
        """Test loop execution with for_count type."""
        params = CommandParameters({
            "loop_type": "for_count",
            "count": 3,
            "loop_action": {"type": "log", "message": "Iteration"},
            "break_on_error": True
        })
        cmd = LoopCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.FLOW_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert "Loop completed 3 iterations" in result.output
        assert result.metadata["iterations_completed"] == 3
        assert result.metadata["total_errors"] == 0
    
    @patch('time.sleep')
    def test_loop_execution_for_each(self, mock_sleep):
        """Test loop execution with for_each type."""
        params = CommandParameters({
            "loop_type": "for_each",
            "items": ["apple", "banana"],
            "loop_action": {"type": "log", "message": "Processing"},
            "break_on_error": False
        })
        cmd = LoopCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.FLOW_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert result.metadata["iterations_completed"] == 2
    
    @patch('time.sleep')
    def test_loop_execution_while_condition(self, mock_sleep):
        """Test loop execution with while_condition type."""
        params = CommandParameters({
            "loop_type": "while_condition",
            "condition": {
                "condition_type": "less_than",
                "left_operand": "1",
                "right_operand": "3"
            },
            "loop_action": {"type": "increment_counter", "counter_name": "test"}
        })
        cmd = LoopCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.FLOW_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        # While condition should execute at least once
        assert result.metadata["iterations_completed"] >= 1
    
    def test_loop_permissions(self):
        """Test loop command permission requirements."""
        params = CommandParameters({
            "loop_type": "for_count",
            "count": 3,
            "loop_action": {"type": "log", "message": "Iteration"}
        })
        cmd = LoopCommand(CommandId("test"), params)
        
        permissions = cmd.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions
    
    def test_loop_security_risk(self):
        """Test loop command security risk level."""
        params = CommandParameters({
            "loop_type": "for_count",
            "count": 3,
            "loop_action": {"type": "log", "message": "Iteration"}
        })
        cmd = LoopCommand(CommandId("test"), params)
        
        assert cmd.get_security_risk_level() == "high"


class TestBreakCommand:
    """Test break command functionality."""
    
    def test_break_command_creation(self):
        """Test basic break command creation."""
        params = CommandParameters({
            "break_type": "loop",
            "break_message": "Breaking out of loop"
        })
        cmd = BreakCommand(CommandId("test"), params)
        
        assert cmd.get_break_type() == "loop"
        assert cmd.get_break_message() == "Breaking out of loop"
        assert cmd.get_break_label() is None
    
    def test_break_validation_valid(self):
        """Test break command validation with valid parameters."""
        params = CommandParameters({
            "break_type": "conditional",
            "break_label": "outer_loop",
            "break_message": "Condition met, breaking"
        })
        cmd = BreakCommand(CommandId("test"), params)
        
        assert cmd.validate() is True
    
    def test_break_validation_invalid_type(self):
        """Test break command validation with invalid break type."""
        params = CommandParameters({
            "break_type": "invalid_type"
        })
        cmd = BreakCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    def test_break_execution(self):
        """Test break command execution."""
        params = CommandParameters({
            "break_type": "loop",
            "break_label": "main_loop",
            "break_message": "Task completed early"
        })
        cmd = BreakCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.FLOW_CONTROL]),
            timeout=Duration.from_seconds(30)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert "Break executed for loop" in result.output
        assert result.metadata["break_type"] == "loop"
        assert result.metadata["break_label"] == "main_loop"
        assert result.metadata["break_message"] == "Task completed early"
    
    def test_break_permissions(self):
        """Test break command permission requirements."""
        params = CommandParameters({"break_type": "loop"})
        cmd = BreakCommand(CommandId("test"), params)
        
        permissions = cmd.get_required_permissions()
        assert Permission.FLOW_CONTROL in permissions
    
    def test_break_security_risk(self):
        """Test break command security risk level."""
        params = CommandParameters({"break_type": "loop"})
        cmd = BreakCommand(CommandId("test"), params)
        
        assert cmd.get_security_risk_level() == "low"


if __name__ == "__main__":
    pytest.main([__file__])