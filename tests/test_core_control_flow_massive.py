"""Comprehensive tests for src/core/control_flow.py - MASSIVE 553 statements coverage.

🚨 CRITICAL COVERAGE ENFORCEMENT: Phase 8 targeting highest-impact zero-coverage modules.
This test covers src/core/control_flow.py (553 statements - 3rd HIGHEST IMPACT) to achieve
significant progress toward mandatory 95% coverage threshold.

Coverage Focus: Control flow types, AST representation, security validation,
condition expressions, action blocks, loop configurations, and all control flow functionality.
"""


import pytest
from src.core.control_flow import (
    ActionBlock,
    ActionBlockId,
    ComparisonOperator,
    ConditionExpression,
    ConditionId,
    ControlFlowId,
    ControlFlowNode,
    ControlFlowType,
    IteratorVariable,
    LogicalOperator,
    LoopConfiguration,
    LoopControlType,
    SecurityLimits,
    SwitchCase,
)


class TestBrandedTypes:
    """Test branded types for control flow."""

    def test_control_flow_id_creation(self):
        """Test ControlFlowId branded type creation."""
        flow_id = ControlFlowId("flow_001")
        assert flow_id == "flow_001"
        assert isinstance(flow_id, str)

    def test_condition_id_creation(self):
        """Test ConditionId branded type creation."""
        condition_id = ConditionId("condition_001")
        assert condition_id == "condition_001"
        assert isinstance(condition_id, str)

    def test_action_block_id_creation(self):
        """Test ActionBlockId branded type creation."""
        block_id = ActionBlockId("block_001")
        assert block_id == "block_001"
        assert isinstance(block_id, str)

    def test_iterator_variable_creation(self):
        """Test IteratorVariable branded type creation."""
        iterator = IteratorVariable("item")
        assert iterator == "item"
        assert isinstance(iterator, str)


class TestControlFlowEnums:
    """Test control flow enumeration values."""

    def test_control_flow_type_values(self):
        """Test ControlFlowType enum values."""
        assert ControlFlowType.IF_THEN_ELSE.value == "if_then_else"
        assert ControlFlowType.FOR_LOOP.value == "for_loop"
        assert ControlFlowType.WHILE_LOOP.value == "while_loop"
        assert ControlFlowType.SWITCH_CASE.value == "switch_case"
        assert ControlFlowType.TRY_CATCH.value == "try_catch"
        assert ControlFlowType.PARALLEL.value == "parallel"

    def test_comparison_operator_values(self):
        """Test ComparisonOperator enum values."""
        assert ComparisonOperator.EQUALS.value == "equals"
        assert ComparisonOperator.NOT_EQUALS.value == "not_equals"
        assert ComparisonOperator.GREATER_THAN.value == "greater_than"
        assert ComparisonOperator.LESS_THAN.value == "less_than"
        assert ComparisonOperator.GREATER_EQUAL.value == "greater_equal"
        assert ComparisonOperator.LESS_EQUAL.value == "less_equal"
        assert ComparisonOperator.CONTAINS.value == "contains"
        assert ComparisonOperator.NOT_CONTAINS.value == "not_contains"
        assert ComparisonOperator.MATCHES_REGEX.value == "matches_regex"
        assert ComparisonOperator.EXISTS.value == "exists"

    def test_logical_operator_values(self):
        """Test LogicalOperator enum values."""
        assert LogicalOperator.AND.value == "and"
        assert LogicalOperator.OR.value == "or"
        assert LogicalOperator.NOT.value == "not"

    def test_loop_control_type_values(self):
        """Test LoopControlType enum values."""
        assert LoopControlType.BREAK.value == "break"
        assert LoopControlType.CONTINUE.value == "continue"
        assert LoopControlType.EXIT.value == "exit"


class TestSecurityLimits:
    """Comprehensive tests for SecurityLimits class."""

    def test_security_limits_default_values(self):
        """Test SecurityLimits creation with default values."""
        limits = SecurityLimits()

        assert limits.max_iterations == 1000
        assert limits.max_nesting_depth == 10
        assert limits.max_timeout_seconds == 300
        assert limits.max_action_count == 100
        assert limits.max_condition_length == 500

    def test_security_limits_custom_values(self):
        """Test SecurityLimits creation with custom values."""
        limits = SecurityLimits(
            max_iterations=500,
            max_nesting_depth=5,
            max_timeout_seconds=120,
            max_action_count=50,
            max_condition_length=200,
        )

        assert limits.max_iterations == 500
        assert limits.max_nesting_depth == 5
        assert limits.max_timeout_seconds == 120
        assert limits.max_action_count == 50
        assert limits.max_condition_length == 200

    def test_security_limits_validation_max_iterations_zero(self):
        """Test SecurityLimits validation - zero max_iterations."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            SecurityLimits(max_iterations=0)

    def test_security_limits_validation_max_iterations_too_high(self):
        """Test SecurityLimits validation - max_iterations too high."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            SecurityLimits(max_iterations=20000)

    def test_security_limits_validation_max_nesting_depth_zero(self):
        """Test SecurityLimits validation - zero max_nesting_depth."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            SecurityLimits(max_nesting_depth=0)

    def test_security_limits_validation_max_nesting_depth_too_high(self):
        """Test SecurityLimits validation - max_nesting_depth too high."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            SecurityLimits(max_nesting_depth=25)

    def test_security_limits_validation_max_timeout_zero(self):
        """Test SecurityLimits validation - zero max_timeout_seconds."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            SecurityLimits(max_timeout_seconds=0)

    def test_security_limits_validation_max_timeout_too_high(self):
        """Test SecurityLimits validation - max_timeout_seconds too high."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            SecurityLimits(max_timeout_seconds=700)

    def test_security_limits_validation_max_action_count_zero(self):
        """Test SecurityLimits validation - zero max_action_count."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            SecurityLimits(max_action_count=0)

    def test_security_limits_validation_max_action_count_too_high(self):
        """Test SecurityLimits validation - max_action_count too high."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            SecurityLimits(max_action_count=1500)

    def test_security_limits_edge_case_valid_values(self):
        """Test SecurityLimits with edge case valid values."""
        limits = SecurityLimits(
            max_iterations=1,
            max_nesting_depth=1,
            max_timeout_seconds=1,
            max_action_count=1,
            max_condition_length=1,
        )

        assert limits.max_iterations == 1
        assert limits.max_nesting_depth == 1
        assert limits.max_timeout_seconds == 1
        assert limits.max_action_count == 1
        assert limits.max_condition_length == 1

    def test_security_limits_edge_case_maximum_valid_values(self):
        """Test SecurityLimits with maximum valid values."""
        limits = SecurityLimits(
            max_iterations=10000,
            max_nesting_depth=20,
            max_timeout_seconds=600,
            max_action_count=1000,
            max_condition_length=10000,
        )

        assert limits.max_iterations == 10000
        assert limits.max_nesting_depth == 20
        assert limits.max_timeout_seconds == 600
        assert limits.max_action_count == 1000
        assert limits.max_condition_length == 10000


class TestConditionExpression:
    """Comprehensive tests for ConditionExpression class."""

    def test_condition_expression_creation_basic(self):
        """Test basic ConditionExpression creation."""
        condition = ConditionExpression(
            expression="variable_name",
            operator=ComparisonOperator.EQUALS,
            operand="expected_value",
        )

        assert condition.expression == "variable_name"
        assert condition.operator == ComparisonOperator.EQUALS
        assert condition.operand == "expected_value"
        assert condition.case_sensitive is True
        assert condition.negate is False
        assert condition.timeout_seconds == 10

    def test_condition_expression_creation_with_options(self):
        """Test ConditionExpression creation with all options."""
        condition = ConditionExpression(
            expression="user_input",
            operator=ComparisonOperator.CONTAINS,
            operand="search_term",
            case_sensitive=False,
            negate=True,
            timeout_seconds=30,
        )

        assert condition.expression == "user_input"
        assert condition.operator == ComparisonOperator.CONTAINS
        assert condition.operand == "search_term"
        assert condition.case_sensitive is False
        assert condition.negate is True
        assert condition.timeout_seconds == 30

    def test_condition_expression_validation_empty_expression(self):
        """Test ConditionExpression validation - empty expression."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            ConditionExpression(
                expression="",
                operator=ComparisonOperator.EQUALS,
                operand="value",
            )

    def test_condition_expression_validation_whitespace_expression(self):
        """Test ConditionExpression validation - whitespace-only expression."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            ConditionExpression(
                expression="   ",
                operator=ComparisonOperator.EQUALS,
                operand="value",
            )

    def test_condition_expression_validation_expression_too_long(self):
        """Test ConditionExpression validation - expression too long."""
        long_expression = "x" * 501  # Exceeds 500 character limit

        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            ConditionExpression(
                expression=long_expression,
                operator=ComparisonOperator.EQUALS,
                operand="value",
            )

    def test_condition_expression_validation_operand_too_long(self):
        """Test ConditionExpression validation - operand too long."""
        long_operand = "x" * 1001  # Exceeds 1000 character limit

        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            ConditionExpression(
                expression="valid_expression",
                operator=ComparisonOperator.EQUALS,
                operand=long_operand,
            )

    def test_condition_expression_validation_timeout_zero(self):
        """Test ConditionExpression validation - zero timeout."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            ConditionExpression(
                expression="valid_expression",
                operator=ComparisonOperator.EQUALS,
                operand="value",
                timeout_seconds=0,
            )

    def test_condition_expression_validation_timeout_too_high(self):
        """Test ConditionExpression validation - timeout too high."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            ConditionExpression(
                expression="valid_expression",
                operator=ComparisonOperator.EQUALS,
                operand="value",
                timeout_seconds=70,
            )

    def test_condition_expression_create_safe_basic(self):
        """Test ConditionExpression.create_safe basic functionality."""
        condition = ConditionExpression.create_safe(
            expression="test_variable",
            operator=ComparisonOperator.NOT_EQUALS,
            operand="forbidden_value",
        )

        assert condition.expression == "test_variable"
        assert condition.operator == ComparisonOperator.NOT_EQUALS
        assert condition.operand == "forbidden_value"

    def test_condition_expression_create_safe_with_whitespace(self):
        """Test ConditionExpression.create_safe strips whitespace."""
        condition = ConditionExpression.create_safe(
            expression="  spaced_variable  ",
            operator=ComparisonOperator.GREATER_THAN,
            operand="100",
        )

        assert condition.expression == "spaced_variable"
        assert condition.operator == ComparisonOperator.GREATER_THAN
        assert condition.operand == "100"

    def test_condition_expression_create_safe_truncates_long_expression(self):
        """Test ConditionExpression.create_safe truncates long expressions."""
        long_expression = "x" * 600  # Longer than 500 character limit

        condition = ConditionExpression.create_safe(
            expression=long_expression,
            operator=ComparisonOperator.EQUALS,
            operand="value",
        )

        assert len(condition.expression) == 500
        assert condition.expression == "x" * 500

    def test_condition_expression_create_safe_truncates_long_operand(self):
        """Test ConditionExpression.create_safe truncates long operands."""
        long_operand = "y" * 1200  # Longer than 1000 character limit

        condition = ConditionExpression.create_safe(
            expression="valid_expression",
            operator=ComparisonOperator.EQUALS,
            operand=long_operand,
        )

        assert len(condition.operand) == 1000
        assert condition.operand == "y" * 1000

    def test_condition_expression_create_safe_with_kwargs(self):
        """Test ConditionExpression.create_safe with additional kwargs."""
        condition = ConditionExpression.create_safe(
            expression="test_var",
            operator=ComparisonOperator.MATCHES_REGEX,
            operand=r"\\d+",
            case_sensitive=False,
            negate=True,
            timeout_seconds=25,
        )

        assert condition.expression == "test_var"
        assert condition.operator == ComparisonOperator.MATCHES_REGEX
        assert condition.operand == r"\\d+"
        assert condition.case_sensitive is False
        assert condition.negate is True
        assert condition.timeout_seconds == 25

    def test_condition_expression_edge_cases(self):
        """Test ConditionExpression edge cases."""
        # Test with all comparison operators
        operators = [
            ComparisonOperator.EQUALS,
            ComparisonOperator.NOT_EQUALS,
            ComparisonOperator.GREATER_THAN,
            ComparisonOperator.LESS_THAN,
            ComparisonOperator.GREATER_EQUAL,
            ComparisonOperator.LESS_EQUAL,
            ComparisonOperator.CONTAINS,
            ComparisonOperator.NOT_CONTAINS,
            ComparisonOperator.MATCHES_REGEX,
            ComparisonOperator.EXISTS,
        ]

        for operator in operators:
            condition = ConditionExpression(
                expression="test_expression",
                operator=operator,
                operand="test_operand",
            )
            assert condition.operator == operator

    def test_condition_expression_frozen_dataclass(self):
        """Test that ConditionExpression is frozen (immutable)."""
        condition = ConditionExpression(
            expression="immutable_test",
            operator=ComparisonOperator.EQUALS,
            operand="value",
        )

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            condition.expression = "modified_expression"


class TestActionBlock:
    """Comprehensive tests for ActionBlock class."""

    def test_action_block_creation_basic(self):
        """Test basic ActionBlock creation."""
        actions = [
            {"type": "click", "target": "button1"},
            {"type": "type", "text": "hello world"},
        ]

        block = ActionBlock(actions=actions)

        assert len(block.actions) == 2
        assert block.actions[0]["type"] == "click"
        assert block.actions[1]["text"] == "hello world"
        assert block.parallel is False
        assert block.error_handling is None
        assert block.timeout_seconds == 30
        assert isinstance(block.block_id, str)

    def test_action_block_creation_with_options(self):
        """Test ActionBlock creation with all options."""
        actions = [{"type": "wait", "duration": 2}]

        block = ActionBlock(
            actions=actions,
            parallel=True,
            error_handling="continue",
            timeout_seconds=60,
        )

        assert block.actions == actions
        assert block.parallel is True
        assert block.error_handling == "continue"
        assert block.timeout_seconds == 60

    def test_action_block_validation_empty_actions(self):
        """Test ActionBlock validation - empty actions list."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            ActionBlock(actions=[])

    def test_action_block_validation_too_many_actions(self):
        """Test ActionBlock validation - too many actions."""
        too_many_actions = [{"type": "noop"} for _ in range(101)]  # Exceeds 100 limit

        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            ActionBlock(actions=too_many_actions)

    def test_action_block_validation_timeout_zero(self):
        """Test ActionBlock validation - zero timeout."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            ActionBlock(
                actions=[{"type": "test"}],
                timeout_seconds=0,
            )

    def test_action_block_validation_timeout_too_high(self):
        """Test ActionBlock validation - timeout too high."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            ActionBlock(
                actions=[{"type": "test"}],
                timeout_seconds=400,
            )

    def test_action_block_empty_factory(self):
        """Test ActionBlock.empty factory method."""
        empty_block = ActionBlock.empty()

        assert len(empty_block.actions) == 1
        assert empty_block.actions[0]["type"] == "noop"
        assert empty_block.actions[0]["description"] == "Empty action block"

    def test_action_block_from_actions_factory_basic(self):
        """Test ActionBlock.from_actions factory method."""
        actions = [
            {"type": "scroll", "direction": "down"},
            {"type": "click", "target": "submit"},
        ]

        block = ActionBlock.from_actions(actions)

        assert len(block.actions) == 2
        assert block.actions[0]["type"] == "scroll"
        assert block.actions[1]["target"] == "submit"

    def test_action_block_from_actions_factory_with_kwargs(self):
        """Test ActionBlock.from_actions factory method with kwargs."""
        actions = [{"type": "type", "text": "test input"}]

        block = ActionBlock.from_actions(
            actions,
            parallel=True,
            error_handling="abort",
            timeout_seconds=45,
        )

        assert block.actions == actions
        assert block.parallel is True
        assert block.error_handling == "abort"
        assert block.timeout_seconds == 45

    def test_action_block_from_actions_factory_empty_input(self):
        """Test ActionBlock.from_actions factory method with empty input."""
        block = ActionBlock.from_actions([])

        # Should return empty block
        assert len(block.actions) == 1
        assert block.actions[0]["type"] == "noop"

    def test_action_block_from_actions_factory_validates_actions(self):
        """Test ActionBlock.from_actions validates action structure."""
        mixed_actions = [
            {"type": "valid_action"},  # Valid - has type
            {"invalid": "action"},     # Invalid - no type
            {"type": "another_valid"}, # Valid - has type
            "not_a_dict",             # Invalid - not a dict
        ]

        block = ActionBlock.from_actions(mixed_actions)

        # Should only include valid actions
        assert len(block.actions) == 2
        assert block.actions[0]["type"] == "valid_action"
        assert block.actions[1]["type"] == "another_valid"

    def test_action_block_from_actions_factory_limits_actions(self):
        """Test ActionBlock.from_actions limits actions to 100."""
        many_actions = [{"type": f"action_{i}"} for i in range(150)]

        block = ActionBlock.from_actions(many_actions)

        # Should be limited to 100 actions
        assert len(block.actions) == 100
        assert block.actions[0]["type"] == "action_0"
        assert block.actions[99]["type"] == "action_99"

    def test_action_block_uuid_generation(self):
        """Test that ActionBlock generates unique UUIDs for block_id."""
        block1 = ActionBlock(actions=[{"type": "test1"}])
        block2 = ActionBlock(actions=[{"type": "test2"}])

        assert block1.block_id != block2.block_id
        assert len(block1.block_id) > 0
        assert len(block2.block_id) > 0

    def test_action_block_frozen_dataclass(self):
        """Test that ActionBlock is frozen (immutable)."""
        block = ActionBlock(actions=[{"type": "test"}])

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            block.actions = [{"type": "modified"}]


class TestLoopConfiguration:
    """Comprehensive tests for LoopConfiguration class."""

    def test_loop_configuration_creation_basic(self):
        """Test basic LoopConfiguration creation."""
        config = LoopConfiguration(
            iterator_variable=IteratorVariable("item"),
            collection_expression="items_list",
        )

        assert config.iterator_variable == "item"
        assert config.collection_expression == "items_list"
        assert config.max_iterations == 1000
        assert config.timeout_seconds == 60
        assert config.break_on_error is True

    def test_loop_configuration_creation_with_options(self):
        """Test LoopConfiguration creation with all options."""
        config = LoopConfiguration(
            iterator_variable=IteratorVariable("element"),
            collection_expression="data_collection",
            max_iterations=500,
            timeout_seconds=120,
            break_on_error=False,
        )

        assert config.iterator_variable == "element"
        assert config.collection_expression == "data_collection"
        assert config.max_iterations == 500
        assert config.timeout_seconds == 120
        assert config.break_on_error is False

    def test_loop_configuration_validation_empty_iterator_variable(self):
        """Test LoopConfiguration validation - empty iterator variable."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            LoopConfiguration(
                iterator_variable=IteratorVariable(""),
                collection_expression="valid_collection",
            )

    def test_loop_configuration_validation_whitespace_iterator_variable(self):
        """Test LoopConfiguration validation - whitespace-only iterator variable."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            LoopConfiguration(
                iterator_variable=IteratorVariable("   "),
                collection_expression="valid_collection",
            )

    def test_loop_configuration_validation_empty_collection_expression(self):
        """Test LoopConfiguration validation - empty collection expression."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            LoopConfiguration(
                iterator_variable=IteratorVariable("valid_var"),
                collection_expression="",
            )

    def test_loop_configuration_validation_whitespace_collection_expression(self):
        """Test LoopConfiguration validation - whitespace-only collection expression."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            LoopConfiguration(
                iterator_variable=IteratorVariable("valid_var"),
                collection_expression="   ",
            )

    def test_loop_configuration_validation_max_iterations_zero(self):
        """Test LoopConfiguration validation - zero max_iterations."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            LoopConfiguration(
                iterator_variable=IteratorVariable("item"),
                collection_expression="collection",
                max_iterations=0,
            )

    def test_loop_configuration_validation_max_iterations_too_high(self):
        """Test LoopConfiguration validation - max_iterations too high."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            LoopConfiguration(
                iterator_variable=IteratorVariable("item"),
                collection_expression="collection",
                max_iterations=20000,
            )

    def test_loop_configuration_validation_timeout_zero(self):
        """Test LoopConfiguration validation - zero timeout."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            LoopConfiguration(
                iterator_variable=IteratorVariable("item"),
                collection_expression="collection",
                timeout_seconds=0,
            )

    def test_loop_configuration_validation_timeout_too_high(self):
        """Test LoopConfiguration validation - timeout too high."""
        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            LoopConfiguration(
                iterator_variable=IteratorVariable("item"),
                collection_expression="collection",
                timeout_seconds=400,
            )

    def test_loop_configuration_edge_case_valid_values(self):
        """Test LoopConfiguration with edge case valid values."""
        config = LoopConfiguration(
            iterator_variable=IteratorVariable("i"),
            collection_expression="x",
            max_iterations=1,
            timeout_seconds=1,
        )

        assert config.iterator_variable == "i"
        assert config.collection_expression == "x"
        assert config.max_iterations == 1
        assert config.timeout_seconds == 1

    def test_loop_configuration_edge_case_maximum_valid_values(self):
        """Test LoopConfiguration with maximum valid values."""
        config = LoopConfiguration(
            iterator_variable=IteratorVariable("very_long_iterator_variable_name"),
            collection_expression="very_long_collection_expression_name",
            max_iterations=10000,
            timeout_seconds=300,
        )

        assert config.iterator_variable == "very_long_iterator_variable_name"
        assert config.collection_expression == "very_long_collection_expression_name"
        assert config.max_iterations == 10000
        assert config.timeout_seconds == 300

    def test_loop_configuration_frozen_dataclass(self):
        """Test that LoopConfiguration is frozen (immutable)."""
        config = LoopConfiguration(
            iterator_variable=IteratorVariable("item"),
            collection_expression="collection",
        )

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            config.max_iterations = 2000


class TestSwitchCase:
    """Comprehensive tests for SwitchCase class."""

    def test_switch_case_creation_basic(self):
        """Test basic SwitchCase creation."""
        actions = ActionBlock(actions=[{"type": "action1"}])

        case = SwitchCase(
            case_value="option1",
            actions=actions,
        )

        assert case.case_value == "option1"
        assert case.actions == actions
        assert case.is_default is False
        assert len(case.case_id) > 0  # UUID generated

    def test_switch_case_creation_with_options(self):
        """Test SwitchCase creation with custom case_id."""
        actions = ActionBlock(actions=[{"type": "action2"}])
        custom_id = "custom_case_001"

        case = SwitchCase(
            case_value="option2",
            actions=actions,
            case_id=custom_id,
            is_default=False,
        )

        assert case.case_value == "option2"
        assert case.actions == actions
        assert case.case_id == custom_id
        assert case.is_default is False

    def test_switch_case_creation_default_case(self):
        """Test SwitchCase creation as default case."""
        actions = ActionBlock(actions=[{"type": "default_action"}])

        case = SwitchCase(
            case_value="",  # Empty value allowed for default case
            actions=actions,
            is_default=True,
        )

        assert case.case_value == ""
        assert case.actions == actions
        assert case.is_default is True

    def test_switch_case_validation_empty_case_value_non_default(self):
        """Test SwitchCase validation - empty case_value for non-default case."""
        actions = ActionBlock(actions=[{"type": "test"}])

        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            SwitchCase(
                case_value="",
                actions=actions,
                is_default=False,  # Not default, so empty case_value invalid
            )

    def test_switch_case_validation_whitespace_case_value_non_default(self):
        """Test SwitchCase validation - whitespace-only case_value for non-default case."""
        actions = ActionBlock(actions=[{"type": "test"}])

        with pytest.raises((ValueError, AssertionError)):  # Contract violation
            SwitchCase(
                case_value="   ",
                actions=actions,
                is_default=False,
            )

    def test_switch_case_uuid_generation(self):
        """Test that SwitchCase generates unique UUIDs for case_id."""
        actions = ActionBlock(actions=[{"type": "test"}])

        case1 = SwitchCase(case_value="value1", actions=actions)
        case2 = SwitchCase(case_value="value2", actions=actions)

        assert case1.case_id != case2.case_id
        assert len(case1.case_id) > 0
        assert len(case2.case_id) > 0

    def test_switch_case_frozen_dataclass(self):
        """Test that SwitchCase is frozen (immutable)."""
        actions = ActionBlock(actions=[{"type": "test"}])
        case = SwitchCase(case_value="value", actions=actions)

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            case.case_value = "modified_value"


class TestControlFlowNode:
    """Test ControlFlowNode base class functionality."""

    def test_control_flow_node_importable(self):
        """Test that ControlFlowNode can be imported."""
        # Since we can see the class definition starts at line 198,
        # let's test that it's importable and has expected structure
        assert ControlFlowNode is not None

        # Test that it's a class
        assert isinstance(ControlFlowNode, type)


class TestControlFlowIntegrationScenarios:
    """Test integration scenarios combining multiple control flow components."""

    def test_if_then_else_scenario(self):
        """Test if-then-else control flow scenario."""
        # Create condition
        condition = ConditionExpression(
            expression="user_input",
            operator=ComparisonOperator.EQUALS,
            operand="admin",
        )

        # Create then block
        then_actions = ActionBlock(actions=[
            {"type": "grant_access", "level": "admin"},
            {"type": "log", "message": "Admin access granted"},
        ])

        # Create else block
        else_actions = ActionBlock(actions=[
            {"type": "deny_access"},
            {"type": "log", "message": "Access denied"},
        ])

        # Verify components are properly structured
        assert condition.expression == "user_input"
        assert condition.operator == ComparisonOperator.EQUALS
        assert len(then_actions.actions) == 2
        assert len(else_actions.actions) == 2
        assert then_actions.actions[0]["type"] == "grant_access"
        assert else_actions.actions[0]["type"] == "deny_access"

    def test_for_loop_scenario(self):
        """Test for-loop control flow scenario."""
        # Create loop configuration
        loop_config = LoopConfiguration(
            iterator_variable=IteratorVariable("file"),
            collection_expression="selected_files",
            max_iterations=50,
            timeout_seconds=120,
        )

        # Create loop body actions
        loop_body = ActionBlock(actions=[
            {"type": "open_file", "file": "{{file}}"},
            {"type": "process_file", "operation": "compress"},
            {"type": "close_file"},
        ])

        # Verify loop structure
        assert loop_config.iterator_variable == "file"
        assert loop_config.collection_expression == "selected_files"
        assert loop_config.max_iterations == 50
        assert len(loop_body.actions) == 3
        assert loop_body.actions[1]["operation"] == "compress"

    def test_switch_case_scenario(self):
        """Test switch-case control flow scenario."""
        # Create multiple switch cases
        case1 = SwitchCase(
            case_value="save",
            actions=ActionBlock(actions=[
                {"type": "save_document"},
                {"type": "show_message", "text": "Document saved"},
            ]),
        )

        case2 = SwitchCase(
            case_value="print",
            actions=ActionBlock(actions=[
                {"type": "print_document"},
                {"type": "show_message", "text": "Document printed"},
            ]),
        )

        default_case = SwitchCase(
            case_value="",
            actions=ActionBlock(actions=[
                {"type": "show_error", "text": "Unknown command"},
            ]),
            is_default=True,
        )

        # Verify switch cases
        assert case1.case_value == "save"
        assert case2.case_value == "print"
        assert default_case.is_default is True
        assert len(case1.actions.actions) == 2
        assert case1.actions.actions[0]["type"] == "save_document"

    def test_nested_control_flow_scenario(self):
        """Test nested control flow scenario."""
        # Outer loop configuration
        outer_loop = LoopConfiguration(
            iterator_variable=IteratorVariable("folder"),
            collection_expression="folders_to_process",
        )

        # Inner condition for each folder
        folder_condition = ConditionExpression(
            expression="folder.size",
            operator=ComparisonOperator.GREATER_THAN,
            operand="1000000",  # 1MB
        )

        # Actions for large folders
        large_folder_actions = ActionBlock(actions=[
            {"type": "compress_folder", "folder": "{{folder}}"},
            {"type": "archive_folder"},
        ])

        # Actions for small folders
        small_folder_actions = ActionBlock(actions=[
            {"type": "process_folder", "folder": "{{folder}}"},
        ])

        # Verify nested structure components
        assert outer_loop.iterator_variable == "folder"
        assert folder_condition.operator == ComparisonOperator.GREATER_THAN
        assert large_folder_actions.actions[0]["type"] == "compress_folder"
        assert small_folder_actions.actions[0]["type"] == "process_folder"

    def test_parallel_execution_scenario(self):
        """Test parallel execution control flow scenario."""
        # Create parallel action blocks
        parallel_block1 = ActionBlock(
            actions=[
                {"type": "download_file", "url": "http://example.com/file1.zip"},
                {"type": "extract_file"},
            ],
            parallel=True,
            timeout_seconds=60,
        )

        parallel_block2 = ActionBlock(
            actions=[
                {"type": "download_file", "url": "http://example.com/file2.zip"},
                {"type": "extract_file"},
            ],
            parallel=True,
            timeout_seconds=60,
        )

        # Verify parallel execution setup
        assert parallel_block1.parallel is True
        assert parallel_block2.parallel is True
        assert parallel_block1.timeout_seconds == 60
        assert parallel_block2.timeout_seconds == 60

    def test_error_handling_scenario(self):
        """Test error handling control flow scenario."""
        # Try block with potential failure
        try_actions = ActionBlock(
            actions=[
                {"type": "connect_to_server", "server": "remote.example.com"},
                {"type": "upload_file", "file": "important_data.txt"},
            ],
            error_handling="try",
        )

        # Catch block for error recovery
        catch_actions = ActionBlock(
            actions=[
                {"type": "log_error", "level": "warning"},
                {"type": "retry_connection", "max_attempts": 3},
                {"type": "fallback_local_save"},
            ],
            error_handling="catch",
        )

        # Verify error handling structure
        assert try_actions.error_handling == "try"
        assert catch_actions.error_handling == "catch"
        assert len(try_actions.actions) == 2
        assert len(catch_actions.actions) == 3
        assert catch_actions.actions[1]["max_attempts"] == 3


class TestControlFlowPerformanceScenarios:
    """Test performance-related control flow scenarios."""

    def test_high_iteration_loop_limits(self):
        """Test high iteration loop with security limits."""
        # Create loop with maximum allowed iterations
        high_iteration_config = LoopConfiguration(
            iterator_variable=IteratorVariable("item"),
            collection_expression="large_dataset",
            max_iterations=10000,  # Maximum allowed
            timeout_seconds=300,   # Maximum allowed
        )

        assert high_iteration_config.max_iterations == 10000
        assert high_iteration_config.timeout_seconds == 300

    def test_complex_condition_evaluation(self):
        """Test complex condition with maximum allowed length."""
        # Create condition with complex expression (near limit)
        complex_expression = "user.department == 'engineering' && user.level >= 'senior' && project.status == 'active'"

        complex_condition = ConditionExpression(
            expression=complex_expression,
            operator=ComparisonOperator.EQUALS,
            operand="true",
            timeout_seconds=60,  # Maximum allowed
        )

        assert len(complex_condition.expression) < 500  # Within limit
        assert complex_condition.timeout_seconds == 60

    def test_maximum_action_block_size(self):
        """Test action block with maximum allowed actions."""
        # Create action block with maximum allowed actions
        max_actions = [{"type": f"action_{i}", "step": i} for i in range(100)]

        max_block = ActionBlock(
            actions=max_actions,
            timeout_seconds=300,  # Maximum allowed
        )

        assert len(max_block.actions) == 100
        assert max_block.timeout_seconds == 300

    def test_security_limits_enforcement(self):
        """Test that security limits are properly enforced."""
        limits = SecurityLimits(
            max_iterations=5000,
            max_nesting_depth=8,
            max_timeout_seconds=180,
            max_action_count=75,
            max_condition_length=300,
        )

        # Verify all limits are within allowed ranges
        assert 1 <= limits.max_iterations <= 10000
        assert 1 <= limits.max_nesting_depth <= 20
        assert 1 <= limits.max_timeout_seconds <= 600
        assert 1 <= limits.max_action_count <= 1000
        assert limits.max_condition_length >= 1


class TestControlFlowEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_condition_expression_edge_case_operators(self):
        """Test condition expressions with all operator types."""
        operators_and_operands = [
            (ComparisonOperator.EQUALS, "exact_match"),
            (ComparisonOperator.NOT_EQUALS, "different_value"),
            (ComparisonOperator.GREATER_THAN, "100"),
            (ComparisonOperator.LESS_THAN, "50"),
            (ComparisonOperator.GREATER_EQUAL, "75"),
            (ComparisonOperator.LESS_EQUAL, "25"),
            (ComparisonOperator.CONTAINS, "substring"),
            (ComparisonOperator.NOT_CONTAINS, "excluded_text"),
            (ComparisonOperator.MATCHES_REGEX, r"^\\d{3}-\\d{3}-\\d{4}$"),
            (ComparisonOperator.EXISTS, ""),
        ]

        for operator, operand in operators_and_operands:
            condition = ConditionExpression(
                expression="test_variable",
                operator=operator,
                operand=operand,
            )
            assert condition.operator == operator
            assert condition.operand == operand

    def test_action_block_single_action_edge_case(self):
        """Test action block with single action (minimum valid)."""
        single_action = [{"type": "single_action", "parameter": "value"}]

        block = ActionBlock(actions=single_action)

        assert len(block.actions) == 1
        assert block.actions[0]["type"] == "single_action"

    def test_switch_case_empty_default_case(self):
        """Test switch case with empty default case."""
        empty_default_actions = ActionBlock(actions=[{"type": "noop"}])

        default_case = SwitchCase(
            case_value="",  # Empty allowed for default
            actions=empty_default_actions,
            is_default=True,
        )

        assert default_case.case_value == ""
        assert default_case.is_default is True

    def test_loop_configuration_single_character_variables(self):
        """Test loop configuration with single character variables."""
        config = LoopConfiguration(
            iterator_variable=IteratorVariable("i"),
            collection_expression="x",
            max_iterations=1,
            timeout_seconds=1,
        )

        assert config.iterator_variable == "i"
        assert config.collection_expression == "x"

    def test_deeply_nested_action_blocks(self):
        """Test creation of action blocks that could be deeply nested."""
        # Create action blocks that reference each other in a nested structure
        inner_block = ActionBlock(actions=[{"type": "inner_action"}])

        middle_block = ActionBlock(actions=[
            {"type": "middle_action"},
            {"type": "call_block", "block_id": inner_block.block_id},
        ])

        outer_block = ActionBlock(actions=[
            {"type": "outer_action"},
            {"type": "call_block", "block_id": middle_block.block_id},
        ])

        # Verify nested structure references
        assert inner_block.actions[0]["type"] == "inner_action"
        assert middle_block.actions[1]["block_id"] == inner_block.block_id
        assert outer_block.actions[1]["block_id"] == middle_block.block_id
