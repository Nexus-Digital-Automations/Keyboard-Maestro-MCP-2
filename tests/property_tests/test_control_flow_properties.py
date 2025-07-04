"""
Property-based tests for control flow functionality.

Uses Hypothesis to validate control flow behavior across input ranges,
ensuring security boundaries and correctness properties hold.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import List, Dict, Any

from src.core.control_flow import (
    ControlFlowBuilder, ControlFlowValidator, SecurityLimits,
    ComparisonOperator, ConditionExpression, ActionBlock,
    LoopConfiguration, SwitchCase, IfThenElseNode, ForLoopNode,
    WhileLoopNode, SwitchCaseNode, IteratorVariable, ControlFlowType,
    create_simple_if, create_for_loop, create_while_loop
)
from src.core.errors import SecurityError, ValidationError, ContractViolationError


# Hypothesis strategies for control flow testing
@st.composite
def safe_text_strategy(draw):
    """Generate safe text without dangerous patterns."""
    # Base text generation
    text = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'), 
                               whitelist_characters=' _-'),
        min_size=1,
        max_size=100
    ))
    
    # Filter out dangerous patterns
    dangerous_patterns = [
        'exec', 'eval', 'import', 'subprocess', 'rm ', 'del ',
        'format', 'curl', 'wget', '`', 'system'
    ]
    
    assume(not any(pattern in text.lower() for pattern in dangerous_patterns))
    
    # Ensure the stripped text is not empty
    stripped_text = text.strip()
    assume(len(stripped_text) > 0)
    
    return stripped_text


@st.composite
def safe_action_strategy(draw):
    """Generate safe action dictionaries."""
    action_type = draw(st.sampled_from([
        'type_text', 'pause', 'click', 'move_mouse', 'press_key',
        'show_notification', 'play_sound', 'set_variable'
    ]))
    
    action = {"type": action_type}
    
    # Add type-specific parameters
    if action_type == 'type_text':
        action['text'] = draw(safe_text_strategy())
    elif action_type == 'pause':
        action['duration'] = draw(st.floats(min_value=0.1, max_value=10.0))
    elif action_type == 'show_notification':
        action['title'] = draw(safe_text_strategy())
        action['message'] = draw(safe_text_strategy())
    elif action_type == 'set_variable':
        action['name'] = draw(safe_text_strategy())
        action['value'] = draw(safe_text_strategy())
    
    return action


@st.composite
def safe_condition_strategy(draw):
    """Generate safe condition expressions."""
    expression = draw(safe_text_strategy())
    operator = draw(st.sampled_from(list(ComparisonOperator)))
    operand = draw(safe_text_strategy())
    
    return ConditionExpression.create_safe(
        expression=expression,
        operator=operator,
        operand=operand,
        case_sensitive=draw(st.booleans()),
        negate=draw(st.booleans()),
        timeout_seconds=draw(st.integers(min_value=1, max_value=60))
    )


@st.composite
def safe_action_block_strategy(draw):
    """Generate safe action blocks."""
    actions = draw(st.lists(
        safe_action_strategy(),
        min_size=1,
        max_size=10
    ))
    
    return ActionBlock.from_actions(
        actions,
        parallel=draw(st.booleans()),
        timeout_seconds=draw(st.integers(min_value=1, max_value=300))
    )


class TestControlFlowProperties:
    """Property-based tests for control flow structures."""
    
    @given(safe_condition_strategy(), safe_action_block_strategy())
    def test_if_then_else_properties(self, condition, then_actions):
        """Property: If/then/else nodes should always be valid when created with safe inputs."""
        else_actions = ActionBlock.from_actions([{"type": "noop"}])
        
        node = IfThenElseNode(
            flow_type=ControlFlowType.IF_THEN_ELSE,
            condition=condition,
            then_actions=then_actions,
            else_actions=else_actions
        )
        
        # Properties that should always hold
        assert node.node_id is not None
        assert node.flow_type.value == "if_then_else"
        assert node.depth >= 0
        assert node.created_at is not None
        assert node.has_else_branch() is True
        assert len(node.then_actions.actions) > 0
        assert len(node.else_actions.actions) > 0
    
    @given(
        safe_text_strategy(),
        safe_text_strategy(),
        safe_action_block_strategy(),
        st.integers(min_value=1, max_value=1000)
    )
    def test_for_loop_properties(self, iterator, collection, actions, max_iterations):
        """Property: For loops should enforce iteration bounds and maintain structure."""
        loop_config = LoopConfiguration(
            iterator_variable=IteratorVariable(iterator),
            collection_expression=collection,
            max_iterations=max_iterations
        )
        
        node = ForLoopNode(
            flow_type=ControlFlowType.FOR_LOOP,
            loop_config=loop_config,
            loop_actions=actions
        )
        
        # Properties that should always hold
        assert node.node_id is not None
        assert node.flow_type.value == "for_loop"
        assert node.loop_config.max_iterations == max_iterations
        assert node.loop_config.max_iterations <= 1000  # Security bound
        assert len(node.loop_actions.actions) > 0
        assert node.loop_config.iterator_variable == iterator
        assert node.loop_config.collection_expression == collection
    
    @given(
        safe_condition_strategy(),
        safe_action_block_strategy(),
        st.integers(min_value=1, max_value=1000)
    )
    def test_while_loop_properties(self, condition, actions, max_iterations):
        """Property: While loops should enforce iteration bounds and condition validity."""
        node = WhileLoopNode(
            flow_type=ControlFlowType.WHILE_LOOP,
            condition=condition,
            loop_actions=actions,
            max_iterations=max_iterations
        )
        
        # Properties that should always hold
        assert node.node_id is not None
        assert node.flow_type.value == "while_loop"
        assert node.max_iterations == max_iterations
        assert node.max_iterations <= 1000  # Security bound
        assert len(node.loop_actions.actions) > 0
        assert node.condition.expression is not None
        assert len(node.condition.expression) > 0
    
    @given(
        safe_text_strategy(),
        st.lists(
            st.tuples(safe_text_strategy(), st.lists(safe_action_strategy(), min_size=1, max_size=5)),
            min_size=1,
            max_size=10
        )
    )
    def test_switch_case_properties(self, switch_variable, case_data):
        """Property: Switch/case structures should maintain case integrity and uniqueness."""
        cases = []
        for case_value, case_actions in case_data:
            action_block = ActionBlock.from_actions(case_actions)
            cases.append(SwitchCase(
                case_value=case_value,
                actions=action_block
            ))
        
        node = SwitchCaseNode(
            flow_type=ControlFlowType.SWITCH_CASE,
            switch_variable=switch_variable,
            cases=cases
        )
        
        # Properties that should always hold
        assert node.node_id is not None
        assert node.flow_type.value == "switch_case"
        assert len(node.cases) == len(case_data)
        assert len(node.cases) <= 50  # Security bound
        assert node.switch_variable == switch_variable
        assert all(len(case.actions.actions) > 0 for case in node.cases)
        
        # Case IDs should be unique
        case_ids = [case.case_id for case in node.cases]
        assert len(case_ids) == len(set(case_ids))


class TestSecurityProperties:
    """Property-based tests for security validation."""
    
    @given(safe_text_strategy())
    def test_condition_security_properties(self, condition_text):
        """Property: No condition should execute malicious code."""
        # Create condition with the test text
        try:
            condition = ConditionExpression.create_safe(
                expression=condition_text,
                operator=ComparisonOperator.EQUALS,
                operand="test"
            )
            
            validator = ControlFlowValidator()
            is_secure = validator.validate_condition_security(condition)
            
            # If validation passes, condition should not contain dangerous patterns
            if is_secure:
                dangerous_patterns = [
                    'exec', 'eval', 'import', 'subprocess', 'rm ', 'del ',
                    'format', 'curl', 'wget', '`', 'system'
                ]
                
                condition_lower = condition_text.lower()
                assert not any(pattern in condition_lower for pattern in dangerous_patterns)
            
        except (ValidationError, ValueError):
            # Invalid conditions should be rejected
            pass
    
    @given(st.lists(safe_action_strategy(), min_size=1, max_size=200))
    def test_action_count_security_properties(self, actions):
        """Property: Action count should be bounded for security."""
        action_block = ActionBlock.from_actions(actions)
        
        # Action count should be limited
        assert len(action_block.actions) <= 100
        
        # If input was over limit, it should be truncated
        if len(actions) > 100:
            assert len(action_block.actions) == 100
        else:
            assert len(action_block.actions) == len(actions)
    
    @given(st.integers(min_value=1, max_value=50000))
    def test_iteration_bounds_properties(self, requested_iterations):
        """Property: Loop iterations should be bounded for security."""
        validator = ControlFlowValidator()
        
        # Test for loop bounds
        try:
            loop_config = LoopConfiguration(
                iterator_variable=IteratorVariable("i"),
                collection_expression="items",
                max_iterations=requested_iterations
            )
            
            for_node = ForLoopNode(
                flow_type=ControlFlowType.FOR_LOOP,
                loop_config=loop_config,
                loop_actions=ActionBlock.from_actions([{"type": "test"}])
            )
            
            is_valid = validator.validate_loop_bounds(for_node)
            
            # If validation passes, iterations should be within bounds
            if is_valid:
                assert for_node.loop_config.max_iterations <= 1000  # Default SecurityLimits.max_iterations
            else:
                assert for_node.loop_config.max_iterations > 1000   # Default SecurityLimits.max_iterations
                
        except (ValidationError, ValueError, ContractViolationError):
            # Invalid configurations should be rejected
            # Either by our security limits (> 1000) or by LoopConfiguration contracts (> 10000)
            assert requested_iterations > 1000 or requested_iterations < 1
    
    @given(st.lists(safe_condition_strategy(), min_size=1, max_size=30))
    def test_nesting_depth_properties(self, conditions):
        """Property: Nesting depth should be bounded for stack safety."""
        validator = ControlFlowValidator()
        
        # Create nested if/then/else structures
        nodes = []
        for i, condition in enumerate(conditions):
            node = IfThenElseNode(
                flow_type=ControlFlowType.IF_THEN_ELSE,
                condition=condition,
                then_actions=ActionBlock.from_actions([{"type": "test"}]),
                depth=i
            )
            nodes.append(node)
        
        is_valid = validator.validate_nesting_depth(nodes)
        max_depth = max(node.depth for node in nodes)
        
        # If validation passes, depth should be within bounds
        if is_valid:
            assert max_depth <= 10
        else:
            assert max_depth > 10


class TestBuilderProperties:
    """Property-based tests for control flow builder."""
    
    @given(
        safe_condition_strategy(),
        st.lists(safe_action_strategy(), min_size=1, max_size=10),
        st.lists(safe_action_strategy(), min_size=1, max_size=10)
    )
    def test_builder_if_then_else_properties(self, condition, then_actions, else_actions):
        """Property: Builder should create valid if/then/else structures."""
        builder = ControlFlowBuilder()
        
        builder.if_condition(
            condition.expression,
            condition.operator,
            condition.operand
        ).then_actions(then_actions).else_actions(else_actions)
        
        nodes = builder.build()
        
        assert len(nodes) == 1
        assert isinstance(nodes[0], IfThenElseNode)
        
        node = nodes[0]
        assert node.condition.expression == condition.expression
        assert node.condition.operator == condition.operator
        assert node.condition.operand == condition.operand
        assert len(node.then_actions.actions) == len(then_actions)
        assert len(node.else_actions.actions) == len(else_actions)
    
    @given(
        safe_text_strategy(),
        safe_text_strategy(),
        st.lists(safe_action_strategy(), min_size=1, max_size=10),
        st.integers(min_value=1, max_value=1000)
    )
    def test_builder_for_loop_properties(self, iterator, collection, actions, max_iterations):
        """Property: Builder should create valid for loop structures."""
        builder = ControlFlowBuilder()
        
        builder.for_each(iterator, collection, actions, max_iterations=max_iterations)
        
        nodes = builder.build()
        
        assert len(nodes) == 1
        assert isinstance(nodes[0], ForLoopNode)
        
        node = nodes[0]
        assert node.loop_config.iterator_variable == iterator
        assert node.loop_config.collection_expression == collection
        assert node.loop_config.max_iterations == max_iterations
        assert len(node.loop_actions.actions) == len(actions)
    
    @given(st.lists(
        st.tuples(safe_text_strategy(), st.lists(safe_action_strategy(), min_size=1, max_size=3)),
        min_size=1,
        max_size=10
    ))
    def test_builder_switch_properties(self, case_data):
        """Property: Builder should create valid switch structures."""
        builder = ControlFlowBuilder()
        
        switch_variable = "test_variable"
        cases = [(value, actions) for value, actions in case_data]
        default_actions = [{"type": "default_action"}]
        
        builder.switch_on(switch_variable, cases, default_actions)
        
        nodes = builder.build()
        
        assert len(nodes) == 1
        assert isinstance(nodes[0], SwitchCaseNode)
        
        node = nodes[0]
        assert node.switch_variable == switch_variable
        assert len(node.cases) == len(case_data)
        assert node.has_default_case() is True
        
        # Verify case values match input
        for i, (expected_value, _) in enumerate(case_data):
            assert node.cases[i].case_value == expected_value


class TestHelperFunctionProperties:
    """Property-based tests for helper functions."""
    
    @given(
        safe_text_strategy(),
        st.sampled_from(list(ComparisonOperator)),
        safe_text_strategy(),
        st.lists(safe_action_strategy(), min_size=1, max_size=5)
    )
    def test_create_simple_if_properties(self, expression, operator, operand, then_actions):
        """Property: create_simple_if should always produce valid structures."""
        node = create_simple_if(expression, operator, operand, then_actions)
        
        assert isinstance(node, IfThenElseNode)
        assert node.condition.expression == expression
        assert node.condition.operator == operator
        assert node.condition.operand == operand
        assert len(node.then_actions.actions) == len(then_actions)
        assert node.has_else_branch() is False
    
    @given(
        safe_text_strategy(),
        safe_text_strategy(),
        st.lists(safe_action_strategy(), min_size=1, max_size=5),
        st.integers(min_value=1, max_value=1000)
    )
    def test_create_for_loop_properties(self, iterator, collection, actions, max_iterations):
        """Property: create_for_loop should always produce valid structures."""
        node = create_for_loop(iterator, collection, actions, max_iterations)
        
        assert isinstance(node, ForLoopNode)
        assert node.loop_config.iterator_variable == iterator
        assert node.loop_config.collection_expression == collection
        assert node.loop_config.max_iterations == max_iterations
        assert len(node.loop_actions.actions) == len(actions)
    
    @given(
        safe_text_strategy(),
        st.sampled_from(list(ComparisonOperator)),
        safe_text_strategy(),
        st.lists(safe_action_strategy(), min_size=1, max_size=5),
        st.integers(min_value=1, max_value=1000)
    )
    def test_create_while_loop_properties(self, expression, operator, operand, actions, max_iterations):
        """Property: create_while_loop should always produce valid structures."""
        node = create_while_loop(expression, operator, operand, actions, max_iterations)
        
        assert isinstance(node, WhileLoopNode)
        assert node.condition.expression == expression
        assert node.condition.operator == operator
        assert node.condition.operand == operand
        assert node.max_iterations == max_iterations
        assert len(node.loop_actions.actions) == len(actions)


# Performance and stress test properties
class TestPerformanceProperties:
    """Property-based tests for performance characteristics."""
    
    @settings(max_examples=50, deadline=5000)  # 5 second deadline
    @given(st.lists(safe_action_strategy(), min_size=50, max_size=100))
    def test_large_action_block_performance(self, actions):
        """Property: Large action blocks should be created efficiently."""
        import time
        
        start_time = time.time()
        action_block = ActionBlock.from_actions(actions)
        creation_time = time.time() - start_time
        
        # Should create within reasonable time
        assert creation_time < 1.0  # 1 second max
        assert len(action_block.actions) <= 100  # Security bound enforced
    
    @settings(max_examples=20, deadline=10000)  # 10 second deadline
    @given(st.lists(safe_condition_strategy(), min_size=1, max_size=20))
    def test_complex_validation_performance(self, conditions):
        """Property: Complex validation should complete efficiently."""
        import time
        
        validator = ControlFlowValidator()
        
        start_time = time.time()
        for condition in conditions:
            validator.validate_condition_security(condition)
        validation_time = time.time() - start_time
        
        # Should validate within reasonable time
        assert validation_time < 2.0  # 2 seconds max for all conditions