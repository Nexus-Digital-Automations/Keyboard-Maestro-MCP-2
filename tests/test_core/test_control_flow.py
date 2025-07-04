"""
Unit tests for control flow types and AST representation.

Tests the core control flow functionality including type safety, security validation,
and builder patterns with comprehensive edge case coverage.
"""

import pytest
import uuid
from datetime import datetime
from typing import List, Dict, Any

from src.core.control_flow import (
    ControlFlowBuilder, ControlFlowValidator, SecurityLimits,
    ComparisonOperator, ControlFlowType, ConditionExpression, ActionBlock,
    LoopConfiguration, SwitchCase, IfThenElseNode, ForLoopNode, WhileLoopNode,
    SwitchCaseNode, TryCatchNode, IteratorVariable,
    create_simple_if, create_for_loop, create_while_loop
)
from src.core.errors import SecurityError, ValidationError, ContractViolationError


class TestSecurityLimits:
    """Test security limits validation."""
    
    def test_valid_security_limits(self):
        """Test valid security limits creation."""
        limits = SecurityLimits(
            max_iterations=5000,
            max_nesting_depth=5,
            max_timeout_seconds=60,
            max_action_count=50,
            max_condition_length=200
        )
        
        assert limits.max_iterations == 5000
        assert limits.max_nesting_depth == 5
        assert limits.max_timeout_seconds == 60
        assert limits.max_action_count == 50
        assert limits.max_condition_length == 200
    
    def test_invalid_security_limits(self):
        """Test invalid security limits rejection."""
        with pytest.raises(ContractViolationError):
            SecurityLimits(max_iterations=0)
        
        with pytest.raises(ContractViolationError):
            SecurityLimits(max_iterations=20000)
        
        with pytest.raises(ContractViolationError):
            SecurityLimits(max_nesting_depth=0)
        
        with pytest.raises(ContractViolationError):
            SecurityLimits(max_nesting_depth=30)


class TestConditionExpression:
    """Test condition expression validation and creation."""
    
    def test_valid_condition_creation(self):
        """Test valid condition expression creation."""
        condition = ConditionExpression.create_safe(
            expression="variable_name",
            operator=ComparisonOperator.EQUALS,
            operand="test_value"
        )
        
        assert condition.expression == "variable_name"
        assert condition.operator == ComparisonOperator.EQUALS
        assert condition.operand == "test_value"
        assert condition.case_sensitive is True
        assert condition.negate is False
        assert condition.timeout_seconds == 10
    
    def test_condition_with_options(self):
        """Test condition creation with custom options."""
        condition = ConditionExpression.create_safe(
            expression="clipboard_content",
            operator=ComparisonOperator.CONTAINS,
            operand="password",
            case_sensitive=False,
            negate=True,
            timeout_seconds=30
        )
        
        assert condition.expression == "clipboard_content"
        assert condition.operator == ComparisonOperator.CONTAINS
        assert condition.operand == "password"
        assert condition.case_sensitive is False
        assert condition.negate is True
        assert condition.timeout_seconds == 30
    
    def test_condition_input_sanitization(self):
        """Test condition input sanitization."""
        condition = ConditionExpression.create_safe(
            expression="  whitespace_test  ",
            operator=ComparisonOperator.EQUALS,
            operand="x" * 2000  # Over limit
        )
        
        assert condition.expression == "whitespace_test"
        assert len(condition.operand) == 1000  # Should be truncated
    
    def test_invalid_condition_parameters(self):
        """Test invalid condition parameter rejection."""
        with pytest.raises(ContractViolationError):
            ConditionExpression(
                expression="",
                operator=ComparisonOperator.EQUALS,
                operand="test"
            )
        
        with pytest.raises(ContractViolationError):
            ConditionExpression(
                expression="x" * 600,  # Too long
                operator=ComparisonOperator.EQUALS,
                operand="test"
            )
        
        with pytest.raises(ContractViolationError):
            ConditionExpression(
                expression="test",
                operator=ComparisonOperator.EQUALS,
                operand="test",
                timeout_seconds=0
            )


class TestActionBlock:
    """Test action block validation and creation."""
    
    def test_valid_action_block(self):
        """Test valid action block creation."""
        actions = [
            {"type": "type_text", "text": "Hello World"},
            {"type": "pause", "duration": 1.0}
        ]
        
        block = ActionBlock.from_actions(actions)
        
        assert len(block.actions) == 2
        assert block.actions[0]["type"] == "type_text"
        assert block.actions[1]["type"] == "pause"
        assert block.parallel is False
        assert block.timeout_seconds == 30
    
    def test_action_block_with_options(self):
        """Test action block with custom options."""
        actions = [{"type": "test_action", "param": "value"}]
        
        block = ActionBlock.from_actions(
            actions,
            parallel=True,
            error_handling="continue",
            timeout_seconds=60
        )
        
        assert block.parallel is True
        assert block.error_handling == "continue"
        assert block.timeout_seconds == 60
    
    def test_empty_action_block(self):
        """Test empty action block creation."""
        block = ActionBlock.empty()
        
        assert len(block.actions) == 1
        assert block.actions[0]["type"] == "noop"
    
    def test_action_count_limit(self):
        """Test action count security limit."""
        actions = [{"type": "test", "id": i} for i in range(150)]
        
        block = ActionBlock.from_actions(actions)
        
        # Should be limited to 100 actions
        assert len(block.actions) == 100
    
    def test_invalid_action_block(self):
        """Test invalid action block rejection."""
        with pytest.raises(ContractViolationError):
            ActionBlock(actions=[])  # Empty actions
        
        with pytest.raises(ContractViolationError):
            ActionBlock(
                actions=[{"type": "test"}],
                timeout_seconds=0
            )


class TestLoopConfiguration:
    """Test loop configuration validation."""
    
    def test_valid_loop_config(self):
        """Test valid loop configuration."""
        config = LoopConfiguration(
            iterator_variable=IteratorVariable("item"),
            collection_expression="selected_files",
            max_iterations=500,
            timeout_seconds=30
        )
        
        assert config.iterator_variable == "item"
        assert config.collection_expression == "selected_files"
        assert config.max_iterations == 500
        assert config.timeout_seconds == 30
        assert config.break_on_error is True
    
    def test_invalid_loop_config(self):
        """Test invalid loop configuration rejection."""
        with pytest.raises(ContractViolationError):
            LoopConfiguration(
                iterator_variable=IteratorVariable(""),
                collection_expression="test"
            )
        
        with pytest.raises(ContractViolationError):
            LoopConfiguration(
                iterator_variable=IteratorVariable("test"),
                collection_expression=""
            )
        
        with pytest.raises(ContractViolationError):
            LoopConfiguration(
                iterator_variable=IteratorVariable("test"),
                collection_expression="test",
                max_iterations=0
            )


class TestSwitchCase:
    """Test switch case validation."""
    
    def test_valid_switch_case(self):
        """Test valid switch case creation."""
        actions = ActionBlock.from_actions([{"type": "test_action"}])
        case = SwitchCase(
            case_value="option1",
            actions=actions
        )
        
        assert case.case_value == "option1"
        assert case.actions == actions
        assert case.is_default is False
        assert case.case_id is not None
    
    def test_default_switch_case(self):
        """Test default switch case creation."""
        actions = ActionBlock.from_actions([{"type": "default_action"}])
        case = SwitchCase(
            case_value="",  # Empty for default
            actions=actions,
            is_default=True
        )
        
        assert case.case_value == ""
        assert case.is_default is True
    
    def test_invalid_switch_case(self):
        """Test invalid switch case rejection."""
        actions = ActionBlock.from_actions([{"type": "test"}])
        
        with pytest.raises(ContractViolationError):
            SwitchCase(
                case_value="",  # Empty but not default
                actions=actions,
                is_default=False
            )


class TestControlFlowValidator:
    """Test control flow security validator."""
    
    def setup_method(self):
        """Setup test validator."""
        self.validator = ControlFlowValidator()
    
    def test_nesting_depth_validation(self):
        """Test nesting depth validation."""
        nodes = []
        for depth in range(5):
            node = IfThenElseNode(
                condition=ConditionExpression.create_safe("test", ComparisonOperator.EQUALS, "value"),
                then_actions=ActionBlock.empty(),
                depth=depth
            )
            nodes.append(node)
        
        assert self.validator.validate_nesting_depth(nodes) is True
        
        # Add node with excessive depth
        deep_node = IfThenElseNode(
            condition=ConditionExpression.create_safe("test", ComparisonOperator.EQUALS, "value"),
            then_actions=ActionBlock.empty(),
            depth=15
        )
        nodes.append(deep_node)
        
        assert self.validator.validate_nesting_depth(nodes) is False
    
    def test_loop_bounds_validation(self):
        """Test loop bounds validation."""
        # Valid for loop
        for_loop = ForLoopNode(
            loop_config=LoopConfiguration(
                iterator_variable=IteratorVariable("i"),
                collection_expression="items",
                max_iterations=500
            ),
            loop_actions=ActionBlock.empty()
        )
        
        assert self.validator.validate_loop_bounds(for_loop) is True
        
        # Invalid for loop (too many iterations)
        invalid_for_loop = ForLoopNode(
            loop_config=LoopConfiguration(
                iterator_variable=IteratorVariable("i"),
                collection_expression="items",
                max_iterations=20000
            ),
            loop_actions=ActionBlock.empty()
        )
        
        assert self.validator.validate_loop_bounds(invalid_for_loop) is False
        
        # Valid while loop
        while_loop = WhileLoopNode(
            condition=ConditionExpression.create_safe("test", ComparisonOperator.EQUALS, "value"),
            loop_actions=ActionBlock.empty(),
            max_iterations=800
        )
        
        assert self.validator.validate_loop_bounds(while_loop) is True
        
        # Invalid while loop
        invalid_while_loop = WhileLoopNode(
            condition=ConditionExpression.create_safe("test", ComparisonOperator.EQUALS, "value"),
            loop_actions=ActionBlock.empty(),
            max_iterations=15000
        )
        
        assert self.validator.validate_loop_bounds(invalid_while_loop) is False
    
    def test_action_count_validation(self):
        """Test action count validation."""
        # Create node with acceptable action count
        actions = [{"type": "test", "id": i} for i in range(50)]
        action_block = ActionBlock.from_actions(actions)
        
        if_node = IfThenElseNode(
            condition=ConditionExpression.create_safe("test", ComparisonOperator.EQUALS, "value"),
            then_actions=action_block
        )
        
        assert self.validator.validate_action_count(if_node) is True
        
        # Create node with too many actions
        many_actions = [{"type": "test", "id": i} for i in range(200)]
        large_action_block = ActionBlock.from_actions(many_actions)
        
        large_if_node = IfThenElseNode(
            condition=ConditionExpression.create_safe("test", ComparisonOperator.EQUALS, "value"),
            then_actions=large_action_block
        )
        
        assert self.validator.validate_action_count(large_if_node) is False
    
    def test_condition_security_validation(self):
        """Test condition security validation."""
        # Safe condition
        safe_condition = ConditionExpression.create_safe(
            "variable_name",
            ComparisonOperator.EQUALS,
            "safe_value"
        )
        
        assert self.validator.validate_condition_security(safe_condition) is True
        
        # Dangerous condition - command injection
        dangerous_condition = ConditionExpression.create_safe(
            "rm -rf /",
            ComparisonOperator.EQUALS,
            "dangerous"
        )
        
        assert self.validator.validate_condition_security(dangerous_condition) is False
        
        # Dangerous condition - script execution
        script_condition = ConditionExpression.create_safe(
            "exec('malicious code')",
            ComparisonOperator.EQUALS,
            "value"
        )
        
        assert self.validator.validate_condition_security(script_condition) is False
    
    def test_regex_security_validation(self):
        """Test regex pattern security validation."""
        # Safe regex
        safe_regex = ConditionExpression.create_safe(
            "text_input",
            ComparisonOperator.MATCHES_REGEX,
            r"^\d{4}-\d{2}-\d{2}$"  # Date pattern
        )
        
        assert self.validator.validate_condition_security(safe_regex) is True
        
        # ReDoS vulnerable regex
        redos_regex = ConditionExpression.create_safe(
            "text_input",
            ComparisonOperator.MATCHES_REGEX,
            r"(.+)+"  # Vulnerable pattern
        )
        
        assert self.validator.validate_condition_security(redos_regex) is False


class TestControlFlowBuilder:
    """Test control flow builder functionality."""
    
    def setup_method(self):
        """Setup test builder."""
        self.builder = ControlFlowBuilder()
    
    def test_simple_if_then_else(self):
        """Test simple if/then/else construction."""
        then_actions = [{"type": "type_text", "text": "True"}]
        else_actions = [{"type": "type_text", "text": "False"}]
        
        self.builder.if_condition(
            "clipboard_content",
            ComparisonOperator.CONTAINS,
            "password"
        ).then_actions(then_actions).else_actions(else_actions)
        
        nodes = self.builder.build()
        
        assert len(nodes) == 1
        assert isinstance(nodes[0], IfThenElseNode)
        
        if_node = nodes[0]
        assert if_node.condition.expression == "clipboard_content"
        assert if_node.condition.operator == ComparisonOperator.CONTAINS
        assert if_node.condition.operand == "password"
        assert len(if_node.then_actions.actions) == 1
        assert len(if_node.else_actions.actions) == 1
    
    def test_for_each_loop(self):
        """Test for-each loop construction."""
        loop_actions = [
            {"type": "open_file", "file": "%Variable%file%"},
            {"type": "process_document"}
        ]
        
        self.builder.for_each(
            "file",
            "selected_files_in_finder",
            loop_actions,
            max_iterations=50
        )
        
        nodes = self.builder.build()
        
        assert len(nodes) == 1
        assert isinstance(nodes[0], ForLoopNode)
        
        for_node = nodes[0]
        assert for_node.loop_config.iterator_variable == "file"
        assert for_node.loop_config.collection_expression == "selected_files_in_finder"
        assert for_node.loop_config.max_iterations == 50
        assert len(for_node.loop_actions.actions) == 2
    
    def test_while_loop(self):
        """Test while loop construction."""
        loop_actions = [
            {"type": "check_condition"},
            {"type": "update_variable"}
        ]
        
        self.builder.while_condition(
            "counter",
            ComparisonOperator.LESS_THAN,
            "10",
            loop_actions,
            max_iterations=20
        )
        
        nodes = self.builder.build()
        
        assert len(nodes) == 1
        assert isinstance(nodes[0], WhileLoopNode)
        
        while_node = nodes[0]
        assert while_node.condition.expression == "counter"
        assert while_node.condition.operator == ComparisonOperator.LESS_THAN
        assert while_node.condition.operand == "10"
        assert while_node.max_iterations == 20
        assert len(while_node.loop_actions.actions) == 2
    
    def test_switch_case(self):
        """Test switch/case construction."""
        cases = [
            ("Safari", [{"type": "screenshot"}]),
            ("Chrome", [{"type": "export_bookmarks"}]),
            ("Firefox", [{"type": "clear_cache"}])
        ]
        default_actions = [{"type": "show_notification", "text": "Unsupported app"}]
        
        self.builder.switch_on(
            "frontmost_application",
            cases,
            default_actions
        )
        
        nodes = self.builder.build()
        
        assert len(nodes) == 1
        assert isinstance(nodes[0], SwitchCaseNode)
        
        switch_node = nodes[0]
        assert switch_node.switch_variable == "frontmost_application"
        assert len(switch_node.cases) == 3
        assert switch_node.has_default_case() is True
        
        # Check individual cases
        assert switch_node.cases[0].case_value == "Safari"
        assert switch_node.cases[1].case_value == "Chrome"
        assert switch_node.cases[2].case_value == "Firefox"
    
    def test_try_catch(self):
        """Test try/catch construction."""
        try_actions = [{"type": "risky_operation"}]
        catch_actions = [{"type": "error_handler"}]
        finally_actions = [{"type": "cleanup"}]
        
        self.builder.try_catch(try_actions, catch_actions, finally_actions)
        
        nodes = self.builder.build()
        
        assert len(nodes) == 1
        assert isinstance(nodes[0], TryCatchNode)
        
        try_node = nodes[0]
        assert len(try_node.try_actions.actions) == 1
        assert len(try_node.catch_actions.actions) == 1
        assert try_node.finally_actions is not None
        assert len(try_node.finally_actions.actions) == 1
    
    def test_builder_security_validation(self):
        """Test builder security validation."""
        # Test dangerous condition rejection
        with pytest.raises(ValueError, match="security validation"):
            self.builder.if_condition(
                "exec('rm -rf /')",
                ComparisonOperator.EQUALS,
                "dangerous"
            )
        
        # Test excessive loop iterations rejection
        with pytest.raises(ValueError, match="security validation"):
            self.builder.while_condition(
                "condition",
                ComparisonOperator.EQUALS,
                "value",
                [{"type": "action"}],
                max_iterations=20000
            )
    
    def test_builder_reset(self):
        """Test builder reset functionality."""
        self.builder.if_condition("test", ComparisonOperator.EQUALS, "value")
        nodes = self.builder.build()
        
        assert len(nodes) == 1
        
        self.builder.reset()
        empty_nodes = self.builder.build()
        
        assert len(empty_nodes) == 0


class TestHelperFunctions:
    """Test helper functions for common patterns."""
    
    def test_create_simple_if(self):
        """Test simple if creation helper."""
        then_actions = [{"type": "action_true"}]
        else_actions = [{"type": "action_false"}]
        
        if_node = create_simple_if(
            "variable",
            ComparisonOperator.EQUALS,
            "value",
            then_actions,
            else_actions
        )
        
        assert isinstance(if_node, IfThenElseNode)
        assert if_node.condition.expression == "variable"
        assert if_node.has_else_branch() is True
    
    def test_create_for_loop(self):
        """Test for loop creation helper."""
        actions = [{"type": "process_item"}]
        
        for_node = create_for_loop(
            "item",
            "collection",
            actions,
            max_iterations=100
        )
        
        assert isinstance(for_node, ForLoopNode)
        assert for_node.loop_config.iterator_variable == "item"
        assert for_node.loop_config.max_iterations == 100
    
    def test_create_while_loop(self):
        """Test while loop creation helper."""
        actions = [{"type": "loop_action"}]
        
        while_node = create_while_loop(
            "counter",
            ComparisonOperator.LESS_THAN,
            "10",
            actions,
            max_iterations=50
        )
        
        assert isinstance(while_node, WhileLoopNode)
        assert while_node.condition.expression == "counter"
        assert while_node.max_iterations == 50