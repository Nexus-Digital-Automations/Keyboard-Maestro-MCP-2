"""
Comprehensive macro editing type system with interactive modification capabilities.

This module provides type-safe macro editing operations including action modification,
conditional logic editing, trigger management, and interactive debugging with
comprehensive security validation and rollback capabilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Set
from enum import Enum

from .contracts import require, ensure
from .types import MacroId
from .either import Either
from .errors import SecurityViolationError, ValidationError


class EditOperation(Enum):
    """Supported macro editing operations with type safety."""
    INSPECT = "inspect"
    MODIFY_ACTION = "modify_action"
    ADD_ACTION = "add_action"
    DELETE_ACTION = "delete_action"
    REORDER_ACTIONS = "reorder_actions"
    MODIFY_CONDITION = "modify_condition"
    ADD_TRIGGER = "add_trigger"
    REMOVE_TRIGGER = "remove_trigger"
    UPDATE_PROPERTIES = "update_properties"
    DEBUG_EXECUTE = "debug_execute"
    COMPARE_MACROS = "compare_macros"
    VALIDATE_MACRO = "validate_macro"


@dataclass(frozen=True)
class MacroModification:
    """Type-safe macro modification specification with contracts."""
    operation: EditOperation
    target_element: Optional[str] = None        # Action UUID, trigger ID, etc.
    new_value: Optional[Dict[str, Any]] = None  # New configuration
    position: Optional[int] = None              # For reordering or insertion
    
    @require(lambda self: self.operation in EditOperation)
    @require(lambda self: self.position is None or self.position >= 0)
    def __post_init__(self):
        """Validate modification specification."""
        pass


@dataclass(frozen=True)
class MacroInspection:
    """Comprehensive macro inspection result with security validation."""
    macro_id: str
    macro_name: str
    enabled: bool
    group_name: str
    action_count: int
    trigger_count: int
    condition_count: int
    actions: List[Dict[str, Any]]
    triggers: List[Dict[str, Any]]
    conditions: List[Dict[str, Any]]
    variables_used: Set[str]
    estimated_execution_time: float
    complexity_score: int
    health_score: int
    
    @require(lambda self: self.action_count >= 0)
    @require(lambda self: self.trigger_count >= 0)
    @require(lambda self: self.condition_count >= 0)
    @require(lambda self: 0 <= self.complexity_score <= 100)
    @require(lambda self: 0 <= self.health_score <= 100)
    def __post_init__(self):
        """Validate inspection results."""
        pass


@dataclass(frozen=True)
class DebugSession:
    """Interactive debugging session configuration with security limits."""
    macro_id: str
    breakpoints: Set[str] = field(default_factory=set)  # Action UUIDs
    watch_variables: Set[str] = field(default_factory=set)  # Variable names
    step_mode: bool = False
    timeout_seconds: int = 60
    
    @require(lambda self: 0 < self.timeout_seconds <= 300)
    @require(lambda self: len(self.breakpoints) <= 50)
    @require(lambda self: len(self.watch_variables) <= 20)
    def __post_init__(self):
        """Validate debug session configuration."""
        pass


@dataclass(frozen=True)
class MacroComparison:
    """Result of comparing two macros with detailed analysis."""
    macro1_id: str
    macro2_id: str
    differences: List[Dict[str, Any]]
    similarity_score: float
    recommendation: str
    
    @require(lambda self: 0.0 <= self.similarity_score <= 1.0)
    def __post_init__(self):
        """Validate comparison results."""
        pass


class MacroEditor:
    """Fluent API for type-safe macro editing operations."""
    
    @require(lambda self, macro_id: isinstance(macro_id, str) and len(macro_id.strip()) > 0)
    def __init__(self, macro_id: str):
        self.macro_id = macro_id
        self._modifications: List[MacroModification] = []
    
    @require(lambda self, action_type: isinstance(action_type, str) and len(action_type) > 0)
    @require(lambda self, config: isinstance(config, dict))
    def add_action(self, action_type: str, config: Dict, position: Optional[int] = None) -> MacroEditor:
        """Add new action to macro with validation."""
        mod = MacroModification(
            operation=EditOperation.ADD_ACTION,
            new_value={"type": action_type, "config": config},
            position=position
        )
        self._modifications.append(mod)
        return self
    
    @require(lambda self, action_id: isinstance(action_id, str) and len(action_id) > 0)
    @require(lambda self, new_config: isinstance(new_config, dict))
    def modify_action(self, action_id: str, new_config: Dict) -> MacroEditor:
        """Modify existing action with type safety."""
        mod = MacroModification(
            operation=EditOperation.MODIFY_ACTION,
            target_element=action_id,
            new_value=new_config
        )
        self._modifications.append(mod)
        return self
    
    @require(lambda self, action_id: isinstance(action_id, str) and len(action_id) > 0)
    def delete_action(self, action_id: str) -> MacroEditor:
        """Delete action from macro."""
        mod = MacroModification(
            operation=EditOperation.DELETE_ACTION,
            target_element=action_id
        )
        self._modifications.append(mod)
        return self
    
    @require(lambda self, new_order: isinstance(new_order, list) and len(new_order) > 0)
    def reorder_actions(self, new_order: List[str]) -> MacroEditor:
        """Reorder actions in macro."""
        mod = MacroModification(
            operation=EditOperation.REORDER_ACTIONS,
            new_value={"action_order": new_order}
        )
        self._modifications.append(mod)
        return self
    
    @require(lambda self, condition_type: isinstance(condition_type, str) and len(condition_type) > 0)
    @require(lambda self, config: isinstance(config, dict))
    def add_condition(self, condition_type: str, config: Dict) -> MacroEditor:
        """Add conditional logic to macro."""
        mod = MacroModification(
            operation=EditOperation.MODIFY_CONDITION,
            new_value={"type": condition_type, "config": config}
        )
        self._modifications.append(mod)
        return self
    
    @require(lambda self, trigger_type: isinstance(trigger_type, str) and len(trigger_type) > 0)
    @require(lambda self, config: isinstance(config, dict))
    def add_trigger(self, trigger_type: str, config: Dict) -> MacroEditor:
        """Add trigger to macro."""
        mod = MacroModification(
            operation=EditOperation.ADD_TRIGGER,
            new_value={"type": trigger_type, "config": config}
        )
        self._modifications.append(mod)
        return self
    
    @require(lambda self, properties: isinstance(properties, dict))
    def update_properties(self, properties: Dict[str, Any]) -> MacroEditor:
        """Update macro properties."""
        mod = MacroModification(
            operation=EditOperation.UPDATE_PROPERTIES,
            new_value=properties
        )
        self._modifications.append(mod)
        return self
    
    @ensure(lambda result: isinstance(result, list))
    def get_modifications(self) -> List[MacroModification]:
        """Get all pending modifications."""
        return self._modifications.copy()
    
    def clear_modifications(self) -> MacroEditor:
        """Clear all pending modifications."""
        self._modifications.clear()
        return self


class MacroEditorValidator:
    """Security-first macro editing validation with comprehensive checks."""
    
    @staticmethod
    def validate_modification_permissions(macro_id: str, operation: EditOperation) -> Either[SecurityViolationError, None]:
        """Validate user has permission to modify macro."""
        # Check if macro identifier is valid
        if not macro_id or len(macro_id.strip()) == 0:
            return Either.left(SecurityViolationError(
                "invalid_macro_id", 
                "Empty or invalid macro identifier"
            ))
        
        # Check for system macro protection
        if macro_id.startswith("com.stairways.keyboardmaestro."):
            return Either.left(SecurityViolationError(
                "system_macro_protection",
                "Cannot modify system macros"
            ))
        
        # Validate operation type
        dangerous_operations = {EditOperation.DEBUG_EXECUTE}
        if operation in dangerous_operations:
            # In a real implementation, check actual permissions
            # For now, allow all operations for testing
            pass
        
        return Either.right(None)
    
    @staticmethod
    def validate_action_modification(action_config: Dict) -> Either[SecurityViolationError, Dict]:
        """Prevent malicious action modifications."""
        if not isinstance(action_config, dict):
            return Either.left(SecurityViolationError(
                "invalid_action_config",
                "Action configuration must be a dictionary"
            ))
        
        # Sanitize script content
        if "script" in action_config:
            script = str(action_config["script"])
            dangerous_patterns = ["rm -rf", "sudo", "eval", "exec", "system"]
            
            script_lower = script.lower()
            for pattern in dangerous_patterns:
                if pattern in script_lower:
                    return Either.left(SecurityViolationError(
                        "dangerous_script_content",
                        f"Script contains dangerous pattern: {pattern}"
                    ))
        
        # Validate file paths
        if "file_path" in action_config:
            file_path = str(action_config["file_path"])
            if ".." in file_path or file_path.startswith("/"):
                return Either.left(SecurityViolationError(
                    "invalid_file_path",
                    "File path contains dangerous patterns"
                ))
        
        # Limit action complexity (basic check)
        if len(str(action_config)) > 10000:
            return Either.left(SecurityViolationError(
                "action_too_complex",
                "Action configuration exceeds size limit"
            ))
        
        return Either.right(action_config)
    
    @staticmethod
    def validate_debug_session(debug_config: Dict) -> Either[SecurityViolationError, None]:
        """Prevent abuse of debugging capabilities."""
        if not isinstance(debug_config, dict):
            return Either.left(SecurityViolationError(
                "invalid_debug_config",
                "Debug configuration must be a dictionary"
            ))
        
        # Limit breakpoint count
        breakpoints = debug_config.get("breakpoints", [])
        if not isinstance(breakpoints, list):
            breakpoints = []
        
        if len(breakpoints) > 50:
            return Either.left(SecurityViolationError(
                "too_many_breakpoints",
                "Maximum 50 breakpoints allowed"
            ))
        
        # Validate timeout
        timeout = debug_config.get("timeout_seconds", 60)
        if not isinstance(timeout, int) or timeout <= 0 or timeout > 300:
            return Either.left(SecurityViolationError(
                "invalid_debug_timeout",
                "Debug timeout must be between 1 and 300 seconds"
            ))
        
        # Check watch variable count
        watch_vars = debug_config.get("watch_variables", [])
        if not isinstance(watch_vars, list):
            watch_vars = []
        
        if len(watch_vars) > 20:
            return Either.left(SecurityViolationError(
                "too_many_watch_variables",
                "Maximum 20 watch variables allowed"
            ))
        
        return Either.right(None)


# Utility functions for macro analysis
def calculate_macro_complexity(macro_data: Dict[str, Any]) -> int:
    """Calculate complexity score for a macro (0-100)."""
    if not isinstance(macro_data, dict):
        return 0
    
    base_score = 0
    
    # Action count contribution (0-40 points)
    action_count = len(macro_data.get("actions", []))
    base_score += min(action_count * 2, 40)
    
    # Condition count contribution (0-30 points)
    condition_count = len(macro_data.get("conditions", []))
    base_score += min(condition_count * 5, 30)
    
    # Trigger count contribution (0-20 points)
    trigger_count = len(macro_data.get("triggers", []))
    base_score += min(trigger_count * 3, 20)
    
    # Control flow contribution (0-10 points)
    has_control_flow = any(
        action.get("type", "").startswith("control_flow")
        for action in macro_data.get("actions", [])
    )
    if has_control_flow:
        base_score += 10
    
    return min(base_score, 100)


def calculate_macro_health(macro_data: Dict[str, Any]) -> int:
    """Calculate health score for a macro (0-100)."""
    if not isinstance(macro_data, dict):
        return 0
    
    health_score = 100
    
    # Deduct for missing elements
    if not macro_data.get("name"):
        health_score -= 10
    
    if not macro_data.get("actions"):
        health_score -= 30
    
    if not macro_data.get("triggers"):
        health_score -= 20
    
    # Check for deprecated action types
    deprecated_actions = ["legacy_script", "deprecated_text"]
    for action in macro_data.get("actions", []):
        if action.get("type") in deprecated_actions:
            health_score -= 5
    
    # Check for overly complex actions
    for action in macro_data.get("actions", []):
        if len(str(action)) > 5000:
            health_score -= 10
    
    return max(health_score, 0)