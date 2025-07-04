"""
Control flow types and AST representation for Keyboard Maestro MCP macro engine.

This module provides type-safe control flow constructs (if/then/else, loops, switch/case)
with comprehensive security validation and functional programming patterns.
"""

from __future__ import annotations
from typing import NewType, Protocol, TypeVar, Generic, Union, Any, Dict, List, Optional, Literal, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from datetime import datetime, timedelta

from .types import (
    MacroId, ExecutionToken, Duration, Permission, CommandParameters,
    ExecutionContext, CommandResult
)
from .contracts import require, ensure

# Branded Types for Control Flow
ControlFlowId = NewType('ControlFlowId', str)
ConditionId = NewType('ConditionId', str)
ActionBlockId = NewType('ActionBlockId', str)
IteratorVariable = NewType('IteratorVariable', str)


class ControlFlowType(Enum):
    """Supported control flow types with security boundaries."""
    IF_THEN_ELSE = "if_then_else"
    FOR_LOOP = "for_loop"
    WHILE_LOOP = "while_loop"
    SWITCH_CASE = "switch_case"
    TRY_CATCH = "try_catch"
    PARALLEL = "parallel"


class ComparisonOperator(Enum):
    """Safe comparison operators for conditions."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES_REGEX = "matches_regex"
    EXISTS = "exists"


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"


class LoopControlType(Enum):
    """Loop control flow actions."""
    BREAK = "break"
    CONTINUE = "continue"
    EXIT = "exit"


@dataclass
class SecurityLimits:
    """Security constraints for control flow execution."""
    max_iterations: int = 1000
    max_nesting_depth: int = 10
    max_timeout_seconds: int = 300
    max_action_count: int = 100
    max_condition_length: int = 500
    
    @require(lambda self: self.max_iterations > 0 and self.max_iterations <= 10000)
    @require(lambda self: self.max_nesting_depth > 0 and self.max_nesting_depth <= 20)
    @require(lambda self: self.max_timeout_seconds > 0 and self.max_timeout_seconds <= 600)
    @require(lambda self: self.max_action_count > 0 and self.max_action_count <= 1000)
    def __post_init__(self):
        """Validate security limits."""
        pass


@dataclass(frozen=True)
class ConditionExpression:
    """Type-safe condition expression with security validation."""
    expression: str
    operator: ComparisonOperator
    operand: str
    case_sensitive: bool = True
    negate: bool = False
    timeout_seconds: int = 10
    
    @require(lambda self: len(self.expression.strip()) > 0)
    @require(lambda self: len(self.expression) <= 500)
    @require(lambda self: len(self.operand) <= 1000)
    @require(lambda self: self.timeout_seconds > 0 and self.timeout_seconds <= 60)
    def __post_init__(self):
        """Validate condition expression."""
        pass
    
    @classmethod
    def create_safe(
        cls,
        expression: str,
        operator: ComparisonOperator,
        operand: str,
        **kwargs
    ) -> ConditionExpression:
        """Create a validated condition expression."""
        # Sanitize inputs
        safe_expression = expression.strip()[:500]
        safe_operand = operand[:1000]
        
        return cls(
            expression=safe_expression,
            operator=operator,
            operand=safe_operand,
            **kwargs
        )


@dataclass(frozen=True)
class ActionBlock:
    """Container for actions within control flow structures."""
    actions: List[Dict[str, Any]]
    block_id: ActionBlockId = field(default_factory=lambda: ActionBlockId(str(uuid.uuid4())))
    parallel: bool = False
    error_handling: Optional[str] = None
    timeout_seconds: int = 30
    
    @require(lambda self: len(self.actions) > 0)
    @require(lambda self: len(self.actions) <= 100)
    @require(lambda self: self.timeout_seconds > 0 and self.timeout_seconds <= 300)
    def __post_init__(self):
        """Validate action block."""
        pass
    
    @classmethod
    def empty(cls) -> ActionBlock:
        """Create an empty action block."""
        return cls(actions=[{"type": "noop", "description": "Empty action block"}])
    
    @classmethod
    def from_actions(cls, actions: List[Dict[str, Any]], **kwargs) -> ActionBlock:
        """Create action block from action list with validation."""
        if not actions:
            return cls.empty()
        
        # Validate each action has required structure
        validated_actions = []
        for action in actions[:100]:  # Limit to 100 actions
            if isinstance(action, dict) and 'type' in action:
                validated_actions.append(action)
        
        return cls(actions=validated_actions, **kwargs)


@dataclass(frozen=True)
class LoopConfiguration:
    """Configuration for loop control flow."""
    iterator_variable: IteratorVariable
    collection_expression: str
    max_iterations: int = 1000
    timeout_seconds: int = 60
    break_on_error: bool = True
    
    @require(lambda self: len(self.iterator_variable.strip()) > 0)
    @require(lambda self: len(self.collection_expression.strip()) > 0)
    @require(lambda self: self.max_iterations > 0 and self.max_iterations <= 10000)
    @require(lambda self: self.timeout_seconds > 0 and self.timeout_seconds <= 300)
    def __post_init__(self):
        """Validate loop configuration."""
        pass


@dataclass(frozen=True)
class SwitchCase:
    """Individual case in a switch statement."""
    case_value: str
    actions: ActionBlock
    case_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_default: bool = False
    
    @require(lambda self: len(self.case_value.strip()) > 0 or self.is_default)
    def __post_init__(self):
        """Validate switch case."""
        pass


@dataclass(frozen=True)
class ControlFlowNode:
    """Base control flow node in the AST."""
    flow_type: ControlFlowType
    node_id: ControlFlowId = field(default_factory=lambda: ControlFlowId(str(uuid.uuid4())))
    parent_id: Optional[ControlFlowId] = None
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    @require(lambda self: self.depth >= 0 and self.depth <= 20)
    def __post_init__(self):
        """Validate control flow node."""
        pass


@dataclass(frozen=True)
class IfThenElseNode:
    """If/Then/Else control flow node."""
    flow_type: ControlFlowType
    condition: ConditionExpression
    then_actions: ActionBlock
    else_actions: Optional[ActionBlock] = None
    node_id: ControlFlowId = field(default_factory=lambda: ControlFlowId(str(uuid.uuid4())))
    parent_id: Optional[ControlFlowId] = None
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def has_else_branch(self) -> bool:
        """Check if this node has an else branch."""
        return self.else_actions is not None


@dataclass(frozen=True)
class ForLoopNode:
    """For loop control flow node."""
    flow_type: ControlFlowType
    loop_config: LoopConfiguration
    loop_actions: ActionBlock
    node_id: ControlFlowId = field(default_factory=lambda: ControlFlowId(str(uuid.uuid4())))
    parent_id: Optional[ControlFlowId] = None
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class WhileLoopNode:
    """While loop control flow node."""
    flow_type: ControlFlowType
    condition: ConditionExpression
    loop_actions: ActionBlock
    max_iterations: int = 1000
    node_id: ControlFlowId = field(default_factory=lambda: ControlFlowId(str(uuid.uuid4())))
    parent_id: Optional[ControlFlowId] = None
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    @require(lambda self: self.max_iterations > 0 and self.max_iterations <= 10000)
    def __post_init__(self):
        """Validate while loop configuration."""
        pass


@dataclass(frozen=True)
class SwitchCaseNode:
    """Switch/Case control flow node."""
    flow_type: ControlFlowType
    switch_variable: str
    cases: List[SwitchCase]
    default_case: Optional[ActionBlock] = None
    node_id: ControlFlowId = field(default_factory=lambda: ControlFlowId(str(uuid.uuid4())))
    parent_id: Optional[ControlFlowId] = None
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    @require(lambda self: len(self.switch_variable.strip()) > 0)
    @require(lambda self: len(self.cases) > 0)
    @require(lambda self: len(self.cases) <= 50)  # Prevent excessive case counts
    def __post_init__(self):
        """Validate switch/case configuration."""
        pass
    
    def has_default_case(self) -> bool:
        """Check if this switch has a default case."""
        return self.default_case is not None


@dataclass(frozen=True)
class TryCatchNode:
    """Try/Catch control flow node for error handling."""
    flow_type: ControlFlowType
    try_actions: ActionBlock
    catch_actions: ActionBlock
    finally_actions: Optional[ActionBlock] = None
    node_id: ControlFlowId = field(default_factory=lambda: ControlFlowId(str(uuid.uuid4())))
    parent_id: Optional[ControlFlowId] = None
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)



class ControlFlowValidator:
    """Security-first control flow validation."""
    
    def __init__(self, limits: SecurityLimits = SecurityLimits()):
        """Initialize validator with security limits."""
        self.limits = limits
    
    def validate_nesting_depth(self, nodes: List[ControlFlowNodeType]) -> bool:
        """Validate control flow nesting depth."""
        max_depth = max((node.depth for node in nodes), default=0)
        return max_depth <= self.limits.max_nesting_depth
    
    def validate_loop_bounds(self, node: Union[ForLoopNode, WhileLoopNode]) -> bool:
        """Validate loop iteration limits."""
        if isinstance(node, ForLoopNode):
            return node.loop_config.max_iterations <= self.limits.max_iterations
        elif isinstance(node, WhileLoopNode):
            return node.max_iterations <= self.limits.max_iterations
        return False
    
    def validate_action_count(self, node: ControlFlowNodeType) -> bool:
        """Validate total action count in control flow."""
        total_actions = 0
        
        if isinstance(node, IfThenElseNode):
            total_actions += len(node.then_actions.actions)
            if node.else_actions:
                total_actions += len(node.else_actions.actions)
        elif isinstance(node, (ForLoopNode, WhileLoopNode)):
            if isinstance(node, ForLoopNode):
                total_actions += len(node.loop_actions.actions)
            else:
                total_actions += len(node.loop_actions.actions)
        elif isinstance(node, SwitchCaseNode):
            for case in node.cases:
                total_actions += len(case.actions.actions)
            if node.default_case:
                total_actions += len(node.default_case.actions)
        elif isinstance(node, TryCatchNode):
            total_actions += len(node.try_actions.actions)
            total_actions += len(node.catch_actions.actions)
            if node.finally_actions:
                total_actions += len(node.finally_actions.actions)
        
        return total_actions <= self.limits.max_action_count
    
    def validate_condition_security(self, condition: ConditionExpression) -> bool:
        """Validate condition expression for security."""
        # Check for dangerous patterns
        dangerous_patterns = [
            'exec', 'eval', 'import', '__import__', 'subprocess',
            'os.system', 'shell', 'cmd', '`', 'rm ', 'del ',
            'format', 'curl', 'wget', 'http'
        ]
        
        expr_lower = condition.expression.lower()
        operand_lower = condition.operand.lower()
        
        for pattern in dangerous_patterns:
            if pattern in expr_lower or pattern in operand_lower:
                return False
        
        # Validate regex patterns for ReDoS prevention
        if condition.operator == ComparisonOperator.MATCHES_REGEX:
            return self._validate_regex_safety(condition.operand)
        
        return True
    
    def _validate_regex_safety(self, pattern: str) -> bool:
        """Validate regex pattern for ReDoS prevention."""
        # Basic ReDoS pattern detection
        dangerous_regex = [
            r'(.+)+', r'(.*).*', r'(.*)+', r'(.+).*',
            r'(\w+)+', r'(\w*)*', r'(\d+)+', r'(\d*)*'
        ]
        
        for danger in dangerous_regex:
            if danger in pattern:
                return False
        
        return len(pattern) <= self.limits.max_condition_length


class ControlFlowBuilder:
    """Fluent API for building control flow structures."""
    
    def __init__(self, validator: Optional[ControlFlowValidator] = None):
        """Initialize builder with optional validator."""
        self._nodes: List[ControlFlowNodeType] = []
        self._validator = validator or ControlFlowValidator()
        self._current_depth = 0
    
    def if_condition(
        self,
        expression: str,
        operator: ComparisonOperator,
        operand: str,
        **kwargs
    ) -> ControlFlowBuilder:
        """Add if condition to control flow."""
        condition = ConditionExpression.create_safe(expression, operator, operand, **kwargs)
        
        if not self._validator.validate_condition_security(condition):
            raise ValueError("Condition failed security validation")
        
        # Placeholder for then/else actions - will be set by subsequent calls
        then_actions = ActionBlock.empty()
        
        node = IfThenElseNode(
            flow_type=ControlFlowType.IF_THEN_ELSE,
            condition=condition,
            then_actions=then_actions,
            depth=self._current_depth
        )
        
        self._nodes.append(node)
        return self
    
    def then_actions(self, actions: List[Dict[str, Any]]) -> ControlFlowBuilder:
        """Add then actions to the most recent if condition."""
        if not self._nodes or not isinstance(self._nodes[-1], IfThenElseNode):
            raise ValueError("then_actions requires a preceding if_condition")
        
        last_node = self._nodes[-1]
        action_block = ActionBlock.from_actions(actions)
        
        # Replace the last node with updated then actions
        updated_node = IfThenElseNode(
            flow_type=ControlFlowType.IF_THEN_ELSE,
            node_id=last_node.node_id,
            condition=last_node.condition,
            then_actions=action_block,
            else_actions=last_node.else_actions,
            parent_id=last_node.parent_id,
            depth=last_node.depth,
            created_at=last_node.created_at
        )
        
        self._nodes[-1] = updated_node
        return self
    
    def else_actions(self, actions: List[Dict[str, Any]]) -> ControlFlowBuilder:
        """Add else actions to the most recent if condition."""
        if not self._nodes or not isinstance(self._nodes[-1], IfThenElseNode):
            raise ValueError("else_actions requires a preceding if_condition")
        
        last_node = self._nodes[-1]
        action_block = ActionBlock.from_actions(actions)
        
        # Replace the last node with updated else actions
        updated_node = IfThenElseNode(
            flow_type=ControlFlowType.IF_THEN_ELSE,
            node_id=last_node.node_id,
            condition=last_node.condition,
            then_actions=last_node.then_actions,
            else_actions=action_block,
            parent_id=last_node.parent_id,
            depth=last_node.depth,
            created_at=last_node.created_at
        )
        
        self._nodes[-1] = updated_node
        return self
    
    def for_each(
        self,
        iterator: str,
        collection: str,
        actions: List[Dict[str, Any]],
        **kwargs
    ) -> ControlFlowBuilder:
        """Add for-each loop to control flow."""
        loop_config = LoopConfiguration(
            iterator_variable=IteratorVariable(iterator),
            collection_expression=collection,
            **kwargs
        )
        
        action_block = ActionBlock.from_actions(actions)
        
        node = ForLoopNode(
            flow_type=ControlFlowType.FOR_LOOP,
            loop_config=loop_config,
            loop_actions=action_block,
            depth=self._current_depth
        )
        
        if not self._validator.validate_loop_bounds(node):
            raise ValueError("Loop configuration failed security validation")
        
        self._nodes.append(node)
        return self
    
    def while_condition(
        self,
        expression: str,
        operator: ComparisonOperator,
        operand: str,
        actions: List[Dict[str, Any]],
        max_iterations: int = 1000,
        **kwargs
    ) -> ControlFlowBuilder:
        """Add while loop to control flow."""
        condition = ConditionExpression.create_safe(expression, operator, operand, **kwargs)
        
        if not self._validator.validate_condition_security(condition):
            raise ValueError("Condition failed security validation")
        
        action_block = ActionBlock.from_actions(actions)
        
        node = WhileLoopNode(
            flow_type=ControlFlowType.WHILE_LOOP,
            condition=condition,
            loop_actions=action_block,
            max_iterations=max_iterations,
            depth=self._current_depth
        )
        
        if not self._validator.validate_loop_bounds(node):
            raise ValueError("Loop configuration failed security validation")
        
        self._nodes.append(node)
        return self
    
    def switch_on(
        self,
        variable: str,
        cases: List[tuple[str, List[Dict[str, Any]]]],
        default_actions: Optional[List[Dict[str, Any]]] = None
    ) -> ControlFlowBuilder:
        """Add switch/case to control flow."""
        switch_cases = []
        for case_value, case_actions in cases:
            case_block = ActionBlock.from_actions(case_actions)
            switch_cases.append(SwitchCase(
                case_value=case_value,
                actions=case_block
            ))
        
        default_block = None
        if default_actions:
            default_block = ActionBlock.from_actions(default_actions)
        
        node = SwitchCaseNode(
            switch_variable=variable,
            cases=switch_cases,
            default_case=default_block,
            depth=self._current_depth
        )
        
        if not self._validator.validate_action_count(node):
            raise ValueError("Switch statement failed security validation")
        
        self._nodes.append(node)
        return self
    
    def try_catch(
        self,
        try_actions: List[Dict[str, Any]],
        catch_actions: List[Dict[str, Any]],
        finally_actions: Optional[List[Dict[str, Any]]] = None
    ) -> ControlFlowBuilder:
        """Add try/catch error handling to control flow."""
        try_block = ActionBlock.from_actions(try_actions)
        catch_block = ActionBlock.from_actions(catch_actions)
        finally_block = None
        
        if finally_actions:
            finally_block = ActionBlock.from_actions(finally_actions)
        
        node = TryCatchNode(
            try_actions=try_block,
            catch_actions=catch_block,
            finally_actions=finally_block,
            depth=self._current_depth
        )
        
        if not self._validator.validate_action_count(node):
            raise ValueError("Try/catch statement failed security validation")
        
        self._nodes.append(node)
        return self
    
    def build(self) -> List[ControlFlowNodeType]:
        """Build and validate the complete control flow structure."""
        if not self._validator.validate_nesting_depth(self._nodes):
            raise ValueError("Control flow nesting depth exceeds security limits")
        
        return self._nodes.copy()
    
    def reset(self) -> ControlFlowBuilder:
        """Reset builder for reuse."""
        self._nodes.clear()
        self._current_depth = 0
        return self


# Helper functions for common patterns
def create_simple_if(
    condition_expr: str,
    operator: ComparisonOperator,
    operand: str,
    then_actions: List[Dict[str, Any]],
    else_actions: Optional[List[Dict[str, Any]]] = None
) -> IfThenElseNode:
    """Create a simple if/then/else structure."""
    condition = ConditionExpression.create_safe(condition_expr, operator, operand)
    then_block = ActionBlock.from_actions(then_actions)
    else_block = ActionBlock.from_actions(else_actions) if else_actions else None
    
    return IfThenElseNode(
        flow_type=ControlFlowType.IF_THEN_ELSE,
        condition=condition,
        then_actions=then_block,
        else_actions=else_block
    )


def create_for_loop(
    iterator: str,
    collection: str,
    actions: List[Dict[str, Any]],
    max_iterations: int = 1000
) -> ForLoopNode:
    """Create a for loop structure."""
    loop_config = LoopConfiguration(
        iterator_variable=IteratorVariable(iterator),
        collection_expression=collection,
        max_iterations=max_iterations
    )
    action_block = ActionBlock.from_actions(actions)
    
    return ForLoopNode(
        flow_type=ControlFlowType.FOR_LOOP,
        loop_config=loop_config,
        loop_actions=action_block
    )


def create_while_loop(
    condition_expr: str,
    operator: ComparisonOperator,
    operand: str,
    actions: List[Dict[str, Any]],
    max_iterations: int = 1000
) -> WhileLoopNode:
    """Create a while loop structure."""
    condition = ConditionExpression.create_safe(condition_expr, operator, operand)
    action_block = ActionBlock.from_actions(actions)
    
    return WhileLoopNode(
        flow_type=ControlFlowType.WHILE_LOOP,
        condition=condition,
        loop_actions=action_block,
        max_iterations=max_iterations
    )


# Advanced Control Flow Features

@dataclass(frozen=True)
class NestedControlFlow:
    """Container for nested control flow structures."""
    parent_node: ControlFlowNodeType
    child_nodes: List[ControlFlowNodeType]
    nesting_level: int
    
    @require(lambda self: self.nesting_level >= 0 and self.nesting_level <= 20)
    @require(lambda self: len(self.child_nodes) <= 50)
    def __post_init__(self):
        """Validate nested structure."""
        pass
    
    def get_total_depth(self) -> int:
        """Calculate total nesting depth."""
        if not self.child_nodes:
            return self.nesting_level
        
        max_child_depth = max(
            child.depth if hasattr(child, 'depth') else 0
            for child in self.child_nodes
        )
        return self.nesting_level + max_child_depth


@dataclass(frozen=True)
class LoopControl:
    """Loop control flow actions (break, continue, exit)."""
    control_type: LoopControlType
    condition: Optional[ConditionExpression] = None
    target_loop_id: Optional[ControlFlowId] = None
    
    def is_conditional(self) -> bool:
        """Check if this is a conditional loop control."""
        return self.condition is not None


@dataclass(frozen=True)
class ParallelExecutionNode:
    """Parallel execution control flow node."""
    flow_type: ControlFlowType
    parallel_branches: List[ActionBlock]
    max_concurrent: int = 4
    timeout_per_branch: int = 30
    fail_fast: bool = True
    node_id: ControlFlowId = field(default_factory=lambda: ControlFlowId(str(uuid.uuid4())))
    parent_id: Optional[ControlFlowId] = None
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    @require(lambda self: len(self.parallel_branches) > 0)
    @require(lambda self: len(self.parallel_branches) <= 10)
    @require(lambda self: self.max_concurrent > 0 and self.max_concurrent <= 8)
    @require(lambda self: self.timeout_per_branch > 0 and self.timeout_per_branch <= 300)
    def __post_init__(self):
        """Validate parallel execution configuration."""
        pass


# Type alias for all control flow nodes
ControlFlowNodeType = Union[
    IfThenElseNode, ForLoopNode, WhileLoopNode, 
    SwitchCaseNode, TryCatchNode, ParallelExecutionNode
]


class AdvancedControlFlowBuilder(ControlFlowBuilder):
    """Extended builder with advanced control flow features."""
    
    def __init__(self, validator: Optional[ControlFlowValidator] = None):
        """Initialize advanced builder."""
        super().__init__(validator)
        self._nested_structures: List[NestedControlFlow] = []
        self._loop_controls: List[LoopControl] = []
    
    def begin_nested_block(self, parent_node: ControlFlowNodeType) -> AdvancedControlFlowBuilder:
        """Begin a nested control flow block."""
        if self._current_depth >= 10:
            raise ValueError("Nesting depth exceeds security limits")
        
        self._current_depth += 1
        return self
    
    def end_nested_block(self) -> AdvancedControlFlowBuilder:
        """End a nested control flow block."""
        if self._current_depth > 0:
            self._current_depth -= 1
        return self
    
    def add_loop_control(
        self,
        control_type: LoopControlType,
        condition: Optional[ConditionExpression] = None,
        target_loop_id: Optional[ControlFlowId] = None
    ) -> AdvancedControlFlowBuilder:
        """Add loop control (break, continue, exit)."""
        control = LoopControl(
            control_type=control_type,
            condition=condition,
            target_loop_id=target_loop_id
        )
        
        self._loop_controls.append(control)
        return self
    
    def parallel_execution(
        self,
        branches: List[List[Dict[str, Any]]],
        max_concurrent: int = 4,
        timeout_per_branch: int = 30,
        fail_fast: bool = True
    ) -> AdvancedControlFlowBuilder:
        """Add parallel execution block."""
        if len(branches) > 10:
            raise ValueError("Too many parallel branches (max 10)")
        
        parallel_branches = []
        for branch_actions in branches:
            branch_block = ActionBlock.from_actions(branch_actions)
            parallel_branches.append(branch_block)
        
        node = ParallelExecutionNode(
            flow_type=ControlFlowType.PARALLEL,
            parallel_branches=parallel_branches,
            max_concurrent=max_concurrent,
            timeout_per_branch=timeout_per_branch,
            fail_fast=fail_fast,
            depth=self._current_depth
        )
        
        self._nodes.append(node)
        return self
    
    def nested_if_then_else(
        self,
        condition: ConditionExpression,
        then_flow: List[ControlFlowNodeType],
        else_flow: Optional[List[ControlFlowNodeType]] = None
    ) -> AdvancedControlFlowBuilder:
        """Create nested if/then/else with sub-flows."""
        # Create action blocks that contain nested structures
        then_actions = self._create_nested_action_block(then_flow)
        else_actions = None
        
        if else_flow:
            else_actions = self._create_nested_action_block(else_flow)
        
        node = IfThenElseNode(
            flow_type=ControlFlowType.IF_THEN_ELSE,
            condition=condition,
            then_actions=then_actions,
            else_actions=else_actions,
            depth=self._current_depth
        )
        
        self._nodes.append(node)
        return self
    
    def _create_nested_action_block(self, nested_nodes: List[ControlFlowNodeType]) -> ActionBlock:
        """Create action block that represents nested control structures."""
        # Convert nested control flow nodes to action representations
        nested_actions = []
        for node in nested_nodes:
            action = {
                "type": "nested_control_flow",
                "control_flow_type": node.flow_type.value,
                "node_id": node.node_id,
                "metadata": self._extract_node_metadata(node)
            }
            nested_actions.append(action)
        
        return ActionBlock.from_actions(nested_actions)
    
    def _extract_node_metadata(self, node: ControlFlowNodeType) -> Dict[str, Any]:
        """Extract metadata from control flow node."""
        metadata = {
            "node_type": type(node).__name__,
            "depth": node.depth,
            "created_at": node.created_at.isoformat()
        }
        
        if isinstance(node, IfThenElseNode):
            metadata.update({
                "condition_operator": node.condition.operator.value,
                "has_else": node.has_else_branch()
            })
        elif isinstance(node, (ForLoopNode, WhileLoopNode)):
            if isinstance(node, ForLoopNode):
                metadata.update({
                    "iterator": node.loop_config.iterator_variable,
                    "max_iterations": node.loop_config.max_iterations
                })
            else:
                metadata.update({
                    "condition_operator": node.condition.operator.value,
                    "max_iterations": node.max_iterations
                })
        elif isinstance(node, SwitchCaseNode):
            metadata.update({
                "switch_variable": node.switch_variable,
                "case_count": len(node.cases),
                "has_default": node.has_default_case()
            })
        elif isinstance(node, ParallelExecutionNode):
            metadata.update({
                "branch_count": len(node.parallel_branches),
                "max_concurrent": node.max_concurrent,
                "fail_fast": node.fail_fast
            })
        
        return metadata
    
    def build_advanced(self) -> tuple[List[ControlFlowNodeType], List[LoopControl], List[NestedControlFlow]]:
        """Build advanced control flow with all features."""
        nodes = self.build()
        return nodes, self._loop_controls.copy(), self._nested_structures.copy()


class ControlFlowOptimizer:
    """Optimizer for control flow structures."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.optimization_stats = {
            "redundant_conditions_removed": 0,
            "empty_branches_removed": 0,
            "nested_structures_flattened": 0
        }
    
    def optimize_control_flow(self, nodes: List[ControlFlowNodeType]) -> List[ControlFlowNodeType]:
        """Optimize control flow structures for performance and clarity."""
        optimized_nodes = []
        
        for node in nodes:
            optimized_node = self._optimize_node(node)
            if optimized_node is not None:
                optimized_nodes.append(optimized_node)
        
        return optimized_nodes
    
    def _optimize_node(self, node: ControlFlowNodeType) -> Optional[ControlFlowNodeType]:
        """Optimize individual control flow node."""
        if isinstance(node, IfThenElseNode):
            return self._optimize_if_then_else(node)
        elif isinstance(node, (ForLoopNode, WhileLoopNode)):
            return self._optimize_loop(node)
        elif isinstance(node, SwitchCaseNode):
            return self._optimize_switch_case(node)
        else:
            return node
    
    def _optimize_if_then_else(self, node: IfThenElseNode) -> Optional[IfThenElseNode]:
        """Optimize if/then/else node."""
        # Remove empty else branches
        else_actions = node.else_actions
        if else_actions and len(else_actions.actions) == 1:
            if else_actions.actions[0].get("type") == "noop":
                else_actions = None
                self.optimization_stats["empty_branches_removed"] += 1
        
        # Check for redundant conditions
        if self._is_always_true_condition(node.condition):
            # Convert to simple action block
            self.optimization_stats["redundant_conditions_removed"] += 1
            # Return the then actions as the optimized result
            # In practice, this would need to be handled differently
        
        return IfThenElseNode(
            flow_type=node.flow_type,
            node_id=node.node_id,
            condition=node.condition,
            then_actions=node.then_actions,
            else_actions=else_actions,
            parent_id=node.parent_id,
            depth=node.depth,
            created_at=node.created_at
        )
    
    def _optimize_loop(self, node: Union[ForLoopNode, WhileLoopNode]) -> Optional[Union[ForLoopNode, WhileLoopNode]]:
        """Optimize loop node."""
        # Check for empty loop bodies
        if isinstance(node, ForLoopNode):
            if len(node.loop_actions.actions) == 1 and node.loop_actions.actions[0].get("type") == "noop":
                return None  # Remove empty loop
        elif isinstance(node, WhileLoopNode):
            if len(node.loop_actions.actions) == 1 and node.loop_actions.actions[0].get("type") == "noop":
                return None  # Remove empty loop
        
        return node
    
    def _optimize_switch_case(self, node: SwitchCaseNode) -> Optional[SwitchCaseNode]:
        """Optimize switch/case node."""
        # Remove cases with empty action blocks
        optimized_cases = []
        for case in node.cases:
            if not (len(case.actions.actions) == 1 and case.actions.actions[0].get("type") == "noop"):
                optimized_cases.append(case)
            else:
                self.optimization_stats["empty_branches_removed"] += 1
        
        if not optimized_cases:
            return None  # Remove empty switch
        
        return SwitchCaseNode(
            node_id=node.node_id,
            switch_variable=node.switch_variable,
            cases=optimized_cases,
            default_case=node.default_case,
            parent_id=node.parent_id,
            depth=node.depth,
            created_at=node.created_at
        )
    
    def _is_always_true_condition(self, condition: ConditionExpression) -> bool:
        """Check if condition is always true."""
        # Simple optimization for obvious always-true conditions
        if condition.operator == ComparisonOperator.EQUALS:
            return condition.expression == condition.operand
        return False
    
    def get_optimization_stats(self) -> Dict[str, int]:
        """Get optimization statistics."""
        return self.optimization_stats.copy()


# Enhanced helper functions for advanced features
def create_nested_if_structure(
    conditions: List[ConditionExpression],
    action_chains: List[List[Dict[str, Any]]]
) -> List[IfThenElseNode]:
    """Create nested if structure from multiple conditions."""
    if len(conditions) != len(action_chains):
        raise ValueError("Conditions and action chains must have same length")
    
    builder = AdvancedControlFlowBuilder()
    
    for i, (condition, actions) in enumerate(zip(conditions, action_chains)):
        builder.if_condition(
            condition.expression,
            condition.operator,
            condition.operand
        ).then_actions(actions)
        
        if i < len(conditions) - 1:
            builder.begin_nested_block(builder._nodes[-1])
    
    # Close nested blocks
    for _ in range(len(conditions) - 1):
        builder.end_nested_block()
    
    nodes, _, _ = builder.build_advanced()
    return [node for node in nodes if isinstance(node, IfThenElseNode)]


def create_parallel_execution(
    branches: List[List[Dict[str, Any]]],
    max_concurrent: int = 4,
    fail_fast: bool = True
) -> ParallelExecutionNode:
    """Create parallel execution structure."""
    builder = AdvancedControlFlowBuilder()
    builder.parallel_execution(branches, max_concurrent=max_concurrent, fail_fast=fail_fast)
    
    nodes, _, _ = builder.build_advanced()
    return nodes[0]  # type: ignore


def create_loop_with_controls(
    loop_type: str,
    condition_or_config: Union[ConditionExpression, LoopConfiguration],
    actions: List[Dict[str, Any]],
    break_conditions: Optional[List[ConditionExpression]] = None,
    continue_conditions: Optional[List[ConditionExpression]] = None
) -> Union[ForLoopNode, WhileLoopNode]:
    """Create loop with break/continue controls."""
    builder = AdvancedControlFlowBuilder()
    
    # Add break conditions
    if break_conditions:
        for break_condition in break_conditions:
            builder.add_loop_control(LoopControlType.BREAK, condition=break_condition)
    
    # Add continue conditions
    if continue_conditions:
        for continue_condition in continue_conditions:
            builder.add_loop_control(LoopControlType.CONTINUE, condition=continue_condition)
    
    # Create the appropriate loop type
    if loop_type == "for" and isinstance(condition_or_config, LoopConfiguration):
        node = ForLoopNode(
            flow_type=ControlFlowType.FOR_LOOP,
            loop_config=condition_or_config,
            loop_actions=ActionBlock.from_actions(actions)
        )
    elif loop_type == "while" and isinstance(condition_or_config, ConditionExpression):
        node = WhileLoopNode(
            flow_type=ControlFlowType.WHILE_LOOP,
            condition=condition_or_config,
            loop_actions=ActionBlock.from_actions(actions)
        )
    else:
        raise ValueError("Invalid loop type or configuration mismatch")
    
    return node