"""
Flow Control Commands

Provides secure flow control including conditionals, loops, and breaks
with comprehensive validation and safety limits to prevent infinite loops
and resource exhaustion.
"""

from __future__ import annotations
from typing import Optional, FrozenSet, Union, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import re

from ..core.types import ExecutionContext, CommandResult, Permission, Duration
from ..core.contracts import require, ensure
from .base import BaseCommand, create_command_result, MAX_LOOP_ITERATIONS
from .validation import SecurityValidator


class ConditionType(Enum):
    """Types of conditions for flow control."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    REGEX_MATCH = "regex_match"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"


class LoopType(Enum):
    """Types of loops."""
    FOR_COUNT = "for_count"
    WHILE_CONDITION = "while_condition"
    FOR_EACH = "for_each"


class ComparisonOperator(Enum):
    """Comparison operators for conditions."""
    EQ = "=="
    NE = "!="
    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="
    IN = "in"
    NOT_IN = "not_in"


@dataclass(frozen=True)
class ConditionalCommand(BaseCommand):
    """
    Execute commands conditionally based on evaluated conditions.
    
    Provides secure conditional execution with validation to prevent
    code injection and ensure safe condition evaluation.
    """
    
    def get_condition_type(self) -> ConditionType:
        """Get the type of condition to evaluate."""
        condition_str = self.parameters.get("condition_type", "equals")
        try:
            return ConditionType(condition_str)
        except ValueError:
            return ConditionType.EQUALS
    
    def get_left_operand(self) -> str:
        """Get the left operand for comparison."""
        return str(self.parameters.get("left_operand", ""))
    
    def get_right_operand(self) -> str:
        """Get the right operand for comparison."""
        return str(self.parameters.get("right_operand", ""))
    
    def get_case_sensitive(self) -> bool:
        """Check if string comparisons should be case sensitive."""
        return self.parameters.get("case_sensitive", True)
    
    def get_then_action(self) -> Optional[Dict[str, Any]]:
        """Get the action to execute if condition is true."""
        return self.parameters.get("then_action")
    
    def get_else_action(self) -> Optional[Dict[str, Any]]:
        """Get the action to execute if condition is false."""
        return self.parameters.get("else_action")
    
    def get_timeout(self) -> Duration:
        """Get timeout for condition evaluation."""
        timeout_seconds = self.parameters.get("timeout", 5.0)
        try:
            timeout = Duration.from_seconds(float(timeout_seconds))
            # Limit timeout to reasonable range
            if timeout.seconds > 30:
                return Duration.from_seconds(30)
            return timeout
        except (ValueError, TypeError):
            return Duration.from_seconds(5)
    
    def _validate_impl(self) -> bool:
        """Validate conditional parameters."""
        condition_type = self.get_condition_type()
        left_operand = self.get_left_operand()
        right_operand = self.get_right_operand()
        
        # Validate operands for security
        validator = SecurityValidator()
        if not validator.validate_text_input(left_operand, "left_operand"):
            return False
        
        if not validator.validate_text_input(right_operand, "right_operand"):
            return False
        
        # Validate regex pattern if using regex match
        if condition_type == ConditionType.REGEX_MATCH:
            try:
                re.compile(right_operand)
            except re.error:
                return False
        
        # Validate that at least one action is provided
        then_action = self.get_then_action()
        else_action = self.get_else_action()
        
        if not then_action and not else_action:
            return False
        
        # Validate action structures
        if then_action and not self._validate_action(then_action):
            return False
        
        if else_action and not self._validate_action(else_action):
            return False
        
        # Validate timeout
        timeout = self.get_timeout()
        if timeout.seconds <= 0 or timeout.seconds > 30:
            return False
        
        return True
    
    def _validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate action structure."""
        if not isinstance(action, dict):
            return False
        
        # Basic action validation - must have a type
        if "type" not in action:
            return False
        
        # Validate action type is safe
        action_type = action.get("type", "")
        safe_action_types = {
            "log", "set_variable", "pause", "beep", "display_message",
            "type_text", "key_press", "mouse_click"
        }
        
        return action_type in safe_action_types
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute conditional logic with safe evaluation."""
        condition_type = self.get_condition_type()
        left_operand = self.get_left_operand()
        right_operand = self.get_right_operand()
        case_sensitive = self.get_case_sensitive()
        then_action = self.get_then_action()
        else_action = self.get_else_action()
        
        start_time = time.time()
        
        try:
            # Evaluate condition
            condition_result = self._evaluate_condition(
                condition_type, left_operand, right_operand, case_sensitive
            )
            
            # Determine which action to execute
            action_to_execute = then_action if condition_result else else_action
            action_type = "then" if condition_result else "else"
            
            if action_to_execute:
                # Execute the chosen action
                action_result = self._execute_action(action_to_execute, context)
                
                execution_time = Duration.from_seconds(time.time() - start_time)
                
                return create_command_result(
                    success=action_result,
                    output=f"Condition evaluated to {condition_result}, executed {action_type} action",
                    execution_time=execution_time,
                    condition_result=condition_result,
                    action_executed=action_type,
                    left_operand=left_operand,
                    right_operand=right_operand,
                    condition_type=condition_type.value
                )
            else:
                execution_time = Duration.from_seconds(time.time() - start_time)
                
                return create_command_result(
                    success=True,
                    output=f"Condition evaluated to {condition_result}, no action to execute",
                    execution_time=execution_time,
                    condition_result=condition_result,
                    action_executed=None
                )
                
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Conditional execution failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def _evaluate_condition(
        self, 
        condition_type: ConditionType, 
        left: str, 
        right: str, 
        case_sensitive: bool
    ) -> bool:
        """Safely evaluate condition without code injection."""
        try:
            # Prepare operands for comparison
            left_val = left if case_sensitive else left.lower()
            right_val = right if case_sensitive else right.lower()
            
            if condition_type == ConditionType.EQUALS:
                return left_val == right_val
            elif condition_type == ConditionType.NOT_EQUALS:
                return left_val != right_val
            elif condition_type == ConditionType.CONTAINS:
                return right_val in left_val
            elif condition_type == ConditionType.NOT_CONTAINS:
                return right_val not in left_val
            elif condition_type == ConditionType.IS_EMPTY:
                return len(left.strip()) == 0
            elif condition_type == ConditionType.IS_NOT_EMPTY:
                return len(left.strip()) > 0
            elif condition_type == ConditionType.REGEX_MATCH:
                flags = 0 if case_sensitive else re.IGNORECASE
                return bool(re.search(right, left, flags))
            else:
                # Numeric comparisons
                try:
                    left_num = float(left)
                    right_num = float(right)
                    
                    if condition_type == ConditionType.GREATER_THAN:
                        return left_num > right_num
                    elif condition_type == ConditionType.LESS_THAN:
                        return left_num < right_num
                    elif condition_type == ConditionType.GREATER_EQUAL:
                        return left_num >= right_num
                    elif condition_type == ConditionType.LESS_EQUAL:
                        return left_num <= right_num
                    else:
                        return False
                except ValueError:
                    # If numeric conversion fails, default to string comparison
                    return left_val == right_val
                    
        except Exception:
            return False
    
    def _execute_action(self, action: Dict[str, Any], context: ExecutionContext) -> bool:
        """Execute a simple action safely."""
        try:
            action_type = action.get("type", "")
            
            if action_type == "log":
                message = action.get("message", "")
                print(f"[LOG] {message}")
                return True
            elif action_type == "pause":
                duration = float(action.get("duration", 1.0))
                duration = max(0.1, min(10.0, duration))  # Limit pause duration
                time.sleep(duration)
                return True
            elif action_type == "beep":
                print("\a", end="", flush=True)
                return True
            elif action_type == "display_message":
                message = action.get("message", "")
                print(f"[MESSAGE] {message}")
                return True
            else:
                # For other action types, we would integrate with the command registry
                # For now, just return success for recognized types
                return action_type in {"set_variable", "type_text", "key_press", "mouse_click"}
                
        except Exception:
            return False
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Conditional execution may require various permissions based on actions."""
        # Base permission for flow control
        permissions = {Permission.FLOW_CONTROL}
        
        # Add permissions based on actions
        then_action = self.get_then_action()
        else_action = self.get_else_action()
        
        for action in [then_action, else_action]:
            if action:
                action_type = action.get("type", "")
                if action_type in {"type_text", "key_press"}:
                    permissions.add(Permission.TEXT_INPUT)
                elif action_type == "mouse_click":
                    permissions.add(Permission.MOUSE_CONTROL)
        
        return frozenset(permissions)
    
    def get_security_risk_level(self) -> str:
        """Conditional execution has medium risk due to dynamic execution."""
        return "medium"


@dataclass(frozen=True)
class LoopCommand(BaseCommand):
    """
    Execute commands in a loop with safety limits and break conditions.
    
    Provides secure loop execution with validation to prevent infinite loops
    and resource exhaustion.
    """
    
    def get_loop_type(self) -> LoopType:
        """Get the type of loop to execute."""
        loop_str = self.parameters.get("loop_type", "for_count")
        try:
            return LoopType(loop_str)
        except ValueError:
            return LoopType.FOR_COUNT
    
    def get_count(self) -> int:
        """Get the number of iterations for count-based loops."""
        count = self.parameters.get("count", 1)
        try:
            return max(1, min(MAX_LOOP_ITERATIONS, int(count)))
        except (ValueError, TypeError):
            return 1
    
    def get_condition(self) -> Optional[Dict[str, Any]]:
        """Get the condition for while loops."""
        return self.parameters.get("condition")
    
    def get_items(self) -> List[str]:
        """Get items for for-each loops."""
        items = self.parameters.get("items", [])
        if isinstance(items, list):
            # Limit number of items to prevent resource exhaustion
            return [str(item) for item in items[:MAX_LOOP_ITERATIONS]]
        return []
    
    def get_loop_action(self) -> Optional[Dict[str, Any]]:
        """Get the action to execute in each iteration."""
        return self.parameters.get("loop_action")
    
    def get_max_duration(self) -> Duration:
        """Get maximum duration for loop execution."""
        duration_seconds = self.parameters.get("max_duration", 60.0)
        try:
            duration = Duration.from_seconds(float(duration_seconds))
            # Limit duration to reasonable range
            if duration.seconds > 300:  # 5 minutes max
                return Duration.from_seconds(300)
            return duration
        except (ValueError, TypeError):
            return Duration.from_seconds(60)
    
    def get_break_on_error(self) -> bool:
        """Check if loop should break on action errors."""
        return self.parameters.get("break_on_error", True)
    
    def _validate_impl(self) -> bool:
        """Validate loop parameters."""
        loop_type = self.get_loop_type()
        
        # Validate based on loop type
        if loop_type == LoopType.FOR_COUNT:
            count = self.get_count()
            if count < 1 or count > MAX_LOOP_ITERATIONS:
                return False
        elif loop_type == LoopType.WHILE_CONDITION:
            condition = self.get_condition()
            if not condition or not self._validate_condition(condition):
                return False
        elif loop_type == LoopType.FOR_EACH:
            items = self.get_items()
            if not items or len(items) > MAX_LOOP_ITERATIONS:
                return False
        
        # Validate loop action
        loop_action = self.get_loop_action()
        if not loop_action or not self._validate_action(loop_action):
            return False
        
        # Validate max duration
        max_duration = self.get_max_duration()
        if max_duration.seconds <= 0 or max_duration.seconds > 300:
            return False
        
        return True
    
    def _validate_condition(self, condition: Dict[str, Any]) -> bool:
        """Validate loop condition structure."""
        if not isinstance(condition, dict):
            return False
        
        required_fields = ["condition_type", "left_operand", "right_operand"]
        for field in required_fields:
            if field not in condition:
                return False
        
        # Validate condition type
        condition_type = condition.get("condition_type", "")
        try:
            ConditionType(condition_type)
        except ValueError:
            return False
        
        return True
    
    def _validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate loop action structure."""
        if not isinstance(action, dict):
            return False
        
        # Basic action validation - must have a type
        if "type" not in action:
            return False
        
        # Validate action type is safe
        action_type = action.get("type", "")
        safe_action_types = {
            "log", "set_variable", "pause", "beep", "display_message",
            "type_text", "key_press", "mouse_click", "increment_counter"
        }
        
        return action_type in safe_action_types
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute loop with safety limits and monitoring."""
        loop_type = self.get_loop_type()
        max_duration = self.get_max_duration()
        break_on_error = self.get_break_on_error()
        loop_action = self.get_loop_action()
        
        start_time = time.time()
        iterations_completed = 0
        total_errors = 0
        
        try:
            if loop_type == LoopType.FOR_COUNT:
                count = self.get_count()
                
                for i in range(count):
                    # Check timeout
                    if time.time() - start_time > max_duration.seconds:
                        break
                    
                    # Execute action
                    success = self._execute_loop_action(loop_action, context, i)
                    iterations_completed += 1
                    
                    if not success:
                        total_errors += 1
                        if break_on_error:
                            break
                    
                    # Small delay to prevent resource exhaustion
                    time.sleep(0.001)
                    
            elif loop_type == LoopType.WHILE_CONDITION:
                condition = self.get_condition()
                
                while iterations_completed < MAX_LOOP_ITERATIONS:
                    # Check timeout
                    if time.time() - start_time > max_duration.seconds:
                        break
                    
                    # Evaluate condition
                    if not self._evaluate_loop_condition(condition):
                        break
                    
                    # Execute action
                    success = self._execute_loop_action(loop_action, context, iterations_completed)
                    iterations_completed += 1
                    
                    if not success:
                        total_errors += 1
                        if break_on_error:
                            break
                    
                    # Small delay to prevent resource exhaustion
                    time.sleep(0.001)
                    
            elif loop_type == LoopType.FOR_EACH:
                items = self.get_items()
                
                for i, item in enumerate(items):
                    # Check timeout
                    if time.time() - start_time > max_duration.seconds:
                        break
                    
                    # Execute action with current item
                    success = self._execute_loop_action(loop_action, context, i, item)
                    iterations_completed += 1
                    
                    if not success:
                        total_errors += 1
                        if break_on_error:
                            break
                    
                    # Small delay to prevent resource exhaustion
                    time.sleep(0.001)
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            return create_command_result(
                success=True,
                output=f"Loop completed {iterations_completed} iterations with {total_errors} errors",
                execution_time=execution_time,
                loop_type=loop_type.value,
                iterations_completed=iterations_completed,
                total_errors=total_errors,
                timed_out=time.time() - start_time > max_duration.seconds
            )
            
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Loop execution failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time),
                iterations_completed=iterations_completed,
                total_errors=total_errors
            )
    
    def _evaluate_loop_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate loop condition safely."""
        try:
            condition_type = ConditionType(condition.get("condition_type", "equals"))
            left_operand = str(condition.get("left_operand", ""))
            right_operand = str(condition.get("right_operand", ""))
            case_sensitive = condition.get("case_sensitive", True)
            
            # Use the same condition evaluation logic as ConditionalCommand
            return self._evaluate_condition_logic(condition_type, left_operand, right_operand, case_sensitive)
            
        except Exception:
            return False
    
    def _evaluate_condition_logic(
        self, 
        condition_type: ConditionType, 
        left: str, 
        right: str, 
        case_sensitive: bool
    ) -> bool:
        """Evaluate condition logic (shared with ConditionalCommand)."""
        try:
            left_val = left if case_sensitive else left.lower()
            right_val = right if case_sensitive else right.lower()
            
            if condition_type == ConditionType.EQUALS:
                return left_val == right_val
            elif condition_type == ConditionType.NOT_EQUALS:
                return left_val != right_val
            elif condition_type == ConditionType.CONTAINS:
                return right_val in left_val
            elif condition_type == ConditionType.NOT_CONTAINS:
                return right_val not in left_val
            elif condition_type == ConditionType.IS_EMPTY:
                return len(left.strip()) == 0
            elif condition_type == ConditionType.IS_NOT_EMPTY:
                return len(left.strip()) > 0
            elif condition_type == ConditionType.GREATER_THAN:
                return float(left) > float(right)
            elif condition_type == ConditionType.LESS_THAN:
                return float(left) < float(right)
            else:
                return False
                
        except Exception:
            return False
    
    def _execute_loop_action(
        self, 
        action: Dict[str, Any], 
        context: ExecutionContext, 
        iteration: int, 
        current_item: Optional[str] = None
    ) -> bool:
        """Execute action within loop iteration."""
        try:
            action_type = action.get("type", "")
            
            if action_type == "log":
                message = action.get("message", "")
                item_info = f" (item: {current_item})" if current_item else ""
                print(f"[LOOP {iteration}] {message}{item_info}")
                return True
            elif action_type == "pause":
                duration = float(action.get("duration", 0.1))
                duration = max(0.01, min(1.0, duration))  # Limit pause duration in loops
                time.sleep(duration)
                return True
            elif action_type == "increment_counter":
                counter_name = action.get("counter_name", "default")
                # In a real implementation, this would update a variable store
                print(f"[COUNTER] {counter_name} incremented at iteration {iteration}")
                return True
            elif action_type == "beep":
                print("\a", end="", flush=True)
                return True
            else:
                # For other action types, assume success for recognized types
                return action_type in {"set_variable", "type_text", "key_press", "mouse_click"}
                
        except Exception:
            return False
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Loop execution requires flow control and action-specific permissions."""
        permissions = {Permission.FLOW_CONTROL}
        
        # Add permissions based on loop action
        loop_action = self.get_loop_action()
        if loop_action:
            action_type = loop_action.get("type", "")
            if action_type in {"type_text", "key_press"}:
                permissions.add(Permission.TEXT_INPUT)
            elif action_type == "mouse_click":
                permissions.add(Permission.MOUSE_CONTROL)
        
        return frozenset(permissions)
    
    def get_security_risk_level(self) -> str:
        """Loop execution has high risk due to potential resource exhaustion."""
        return "high"


@dataclass(frozen=True)
class BreakCommand(BaseCommand):
    """
    Break out of loops or conditional structures.
    
    Provides controlled flow interruption with validation to prevent
    misuse and ensure proper control flow.
    """
    
    def get_break_type(self) -> str:
        """Get the type of break (loop, conditional, or all)."""
        return self.parameters.get("break_type", "loop")
    
    def get_break_label(self) -> Optional[str]:
        """Get optional label for targeted breaks."""
        return self.parameters.get("break_label")
    
    def get_break_message(self) -> Optional[str]:
        """Get optional message to display when breaking."""
        return self.parameters.get("break_message")
    
    def _validate_impl(self) -> bool:
        """Validate break parameters."""
        break_type = self.get_break_type()
        
        # Validate break type
        valid_break_types = {"loop", "conditional", "all", "function", "script"}
        if break_type not in valid_break_types:
            return False
        
        # Validate break label if provided
        break_label = self.get_break_label()
        if break_label:
            validator = SecurityValidator()
            if not validator.validate_text_input(break_label, "break_label"):
                return False
        
        # Validate break message if provided
        break_message = self.get_break_message()
        if break_message:
            validator = SecurityValidator()
            if not validator.validate_text_input(break_message, "break_message"):
                return False
        
        return True
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute break command."""
        break_type = self.get_break_type()
        break_label = self.get_break_label()
        break_message = self.get_break_message()
        
        start_time = time.time()
        
        try:
            # Display break message if provided
            if break_message:
                print(f"[BREAK] {break_message}")
            
            # In a real implementation, this would signal the execution engine
            # to break out of the appropriate control structure
            # For now, we'll just log the break action
            
            label_info = f" (label: {break_label})" if break_label else ""
            output_message = f"Break executed for {break_type}{label_info}"
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            return create_command_result(
                success=True,
                output=output_message,
                execution_time=execution_time,
                break_type=break_type,
                break_label=break_label,
                break_message=break_message
            )
            
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Break execution failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Break command requires flow control permission."""
        return frozenset([Permission.FLOW_CONTROL])
    
    def get_security_risk_level(self) -> str:
        """Break command has low risk as it only affects control flow."""
        return "low"